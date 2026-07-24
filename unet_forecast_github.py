"""
Deterministic U-Net next-day forecaster over the study region (15-35N, 90-110E).

Forecast model F_xi: R^{4 x H x W} -> R^{4 x H x W}
    A multi-scale encoder-decoder U-Net with residual blocks (group
    normalization + SiLU activation), max-pool downsampling in the encoder
    and bilinear upsampling in the decoder, with skip connections between
    matching encoder/decoder stages.

Training pairs (x_t, x_{t+dt}) are consecutive daily CMFD v2.0 fields.
Train: 1981-2015.  Validation: 2016-2020.

Loss: L = E_t || F_xi(x_t) - x_{t+dt} ||^2  (mean squared error)

Run:
    python unet_forecast_github.py 2>&1 | tee unet_forecast_github.log
"""

import logging
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
from glob import glob
import netCDF4 as nc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ----------------------------- Config -----------------------------
DATA_DIR = "/data/CMFD/1dy010deg"
VARS = ["prec", "pres", "wind", "temp"]
TRAIN_YEARS = list(range(1981, 2016))      # 1981-2015
VAL_YEARS = list(range(2016, 2021))        # 2016-2020

LAT_MIN, LAT_MAX = 15.0, 35.0
LON_MIN, LON_MAX = 90.0, 110.0

CHANNEL = 4
FIELD_SHAPE = (CHANNEL, 200, 200)
BASE_DIM = 64
CHANNEL_MULTS = (1, 2, 4, 8)              # 64, 128, 256, 512
NUM_RES_GROUPS = 8

LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 50
PRINT_EVERY = 50
SAVE_TOP_K = 5

MEAN_STD_FILE = "/data/cmfd_mean_std.pkl"
MODEL_DIR = "/data/model"
LOG_FILE = "/data/unet_forecast_github.log"
MODEL_PREFIX = "forecast_unet"


# ----------------------------- Logging -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ----------------------------- Model -----------------------------
class ResBlock(nn.Module):
    """Residual block: two 3x3 convs with GroupNorm + SiLU."""

    def __init__(self, ch_in, ch_out, groups=NUM_RES_GROUPS):
        super().__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, ch_out)
        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, ch_out)
        self.act2 = nn.SiLU()
        self.skip = nn.Conv2d(ch_in, ch_out, 1) if ch_in != ch_out else nn.Identity()

    def forward(self, x):
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act2(h + self.skip(x))


class Downsample(nn.Module):
    """Strided-conv downsampling (preserves a learnable channel mapping)."""

    def __init__(self, ch_in, ch_out, scale=2):
        super().__init__()
        self.pool = nn.MaxPool2d(scale)
        self.proj = nn.Conv2d(ch_in, ch_out, 1)

    def forward(self, x):
        return self.proj(self.pool(x))


class Upsample(nn.Module):
    """Bilinear upsampling followed by 3x3 conv (channel mapping)."""

    def __init__(self, ch_in, ch_out, scale=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode="bilinear",
                              align_corners=True)
        self.conv = nn.Conv2d(ch_in, ch_out, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class ForecastUNet(nn.Module):
    """Deterministic multi-scale encoder-decoder U-Net forecaster.

    Encoder progressively downsamples via max-pooling to capture large-scale
    spatial patterns; decoder restores resolution via bilinear upsampling.
    Skip connections between matching encoder and decoder stages preserve
    fine-scale spatial information. Input and output are both four-variable
    fields with identical spatial extent.
    """

    def __init__(self, ch_in=CHANNEL, ch_out=CHANNEL, base=BASE_DIM,
                 channel_mults=CHANNEL_MULTS, num_res_blocks=2):
        super().__init__()
        dims = [base * m for m in channel_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.stem = nn.Conv2d(ch_in, dims[0], 3, padding=1)

        # Encoder
        self.downs = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(in_out):
            is_last = i == len(in_out) - 1
            blocks = nn.ModuleList(
                [ResBlock(d_in if j == 0 else d_in, d_in) for j in range(num_res_blocks)]
            )
            down = (nn.Conv2d(d_in, d_out, 3, padding=1) if is_last
                    else Downsample(d_in, d_out))
            self.downs.append(nn.ModuleList([blocks, down]))

        # Bottleneck
        mid = dims[-1]
        self.mid1 = ResBlock(mid, mid)
        self.mid2 = ResBlock(mid, mid)

        # Decoder
        self.ups = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(reversed(in_out)):
            is_last = i == len(in_out) - 1
            blocks = nn.ModuleList([
                ResBlock(d_out + d_in, d_out),
                ResBlock(d_out + d_in, d_out),
            ])
            up = (nn.Conv2d(d_out, d_in, 3, padding=1) if is_last
                  else Upsample(d_out, d_in))
            self.ups.append(nn.ModuleList([blocks, up]))

        self.final = nn.Conv2d(dims[0], ch_out, 1)

    def forward(self, x):
        x = self.stem(x)
        skips = []

        for blocks, down in self.downs:
            for blk in blocks:
                x = blk(x)
                skips.append(x)
            x = down(x)

        x = self.mid1(x)
        x = self.mid2(x)

        for blocks, up in self.ups:
            for blk in blocks:
                x = torch.cat([x, skips.pop()], dim=1)
                x = blk(x)
            x = up(x)

        return self.final(x)


# ----------------------------- Dataset -----------------------------
def _build_date_list(years):
    dates = []
    for y in years:
        d = datetime(y, 1, 1)
        while d.year == y:
            if not (d.month == 2 and d.day == 29):
                dates.append(d)
            d += timedelta(days=1)
    return dates


def _read_day(var, y, m, d, mean_std, lat_range, lon_range):
    """Load + transform + normalize a single CMFD day for one variable."""
    path = f"{DATA_DIR}/{var}_CMFD_V0200_B-01_01dy_010deg_{y}01-{y}12.nc"
    with nc.Dataset(path) as ds:
        lat = ds.variables["lat"][:]
        lon = ds.variables["lon"][:]
        lat_idx = np.where((lat >= lat_range[0]) & (lat <= lat_range[1]))[0]
        lon_idx = np.where((lon >= lon_range[0]) & (lon <= lon_range[1]))[0]
        arr = ds.variables[var][:, lat_idx.min():lat_idx.max() + 1,
                                  lon_idx.min():lon_idx.max() + 1]
        fill_value = getattr(ds.variables[var], "_FillValue", 1e20)
        arr = np.ma.masked_where(arr >= fill_value, arr)
        if var == "prec":
            arr = arr * 3600.0
        if var in ("prec", "wind"):
            arr = np.ma.log(arr + 0.01)
        doy = (datetime(y, m, d) - datetime(y, 1, 1)).days
        arr_day = arr[doy, :, :]
        arr_day = (arr_day - mean_std[var]["mean"]) / mean_std[var]["std"]
        return arr_day.filled(0).astype(np.float32), (~arr_day.mask)


class ConsecutivePairDataset(Dataset):
    """Consecutive (x_t, x_{t+dt}) pairs from CMFD for a fixed year list.

    Each item returns the field at day D and the field at day D+1, both
    stacked over the four variables. Pairs that cross a year boundary
    are skipped.
    """

    def __init__(self, years, mean_std):
        self.mean_std = mean_std
        self.pairs = []
        dates = _build_date_list(years)
        for i in range(len(dates) - 1):
            d0, d1 = dates[i], dates[i + 1]
            if d1 != d0 + timedelta(days=1):
                continue
            self.pairs.append((d0, d1))
        if not self.pairs:
            raise RuntimeError(f"No valid consecutive pairs in years {years}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        d0, d1 = self.pairs[idx]
        x0_list, m0_list = [], []
        x1_list, m1_list = [], []
        for v in VARS:
            arr0, m0 = _read_day(v, d0.year, d0.month, d0.day,
                                 self.mean_std,
                                 (LAT_MIN, LAT_MAX), (LON_MIN, LON_MAX))
            arr1, m1 = _read_day(v, d1.year, d1.month, d1.day,
                                 self.mean_std,
                                 (LAT_MIN, LAT_MAX), (LON_MIN, LON_MAX))
            x0_list.append(arr0)
            m0_list.append(m0)
            x1_list.append(arr1)
            m1_list.append(m1)
        x_t = torch.from_numpy(np.stack(x0_list, axis=0))
        x_next = torch.from_numpy(np.stack(x1_list, axis=0))
        mask = torch.from_numpy(np.stack(m0_list, axis=0))
        return x_t, x_next, mask


# ----------------------------- Helpers -----------------------------
def maybe_resume(model, optimizer, scheduler, device):
    files = sorted(glob(os.path.join(MODEL_DIR, f"{MODEL_PREFIX}_*.pt")),
                   key=os.path.getmtime)
    if not files:
        logger.info("No checkpoint found, training from scratch.")
        return 0, float("inf")
    path = files[-1]
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    logger.info(f"Resumed from {path}")
    return ckpt.get("global_step", 0), ckpt.get("best_val", float("inf"))


def prune_checkpoints(prefix, top_k):
    files = sorted(glob(os.path.join(MODEL_DIR, f"{prefix}_*.pt")),
                   key=os.path.getmtime)
    for f in files[:-top_k]:
        os.remove(f)
        logger.info(f"Removed old checkpoint: {os.path.basename(f)}")


def ckpt_path(prefix, val_loss, suffix=""):
    ts = datetime.now().strftime("%Y%m%d%H%M")
    return os.path.join(MODEL_DIR, f"{prefix}_{ts}_{val_loss:.5f}{suffix}.pt")


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, count = 0.0, 0
    for x_t, x_next, _mask in loader:
        x_t = x_t.to(device, non_blocking=True)
        x_next = x_next.to(device, non_blocking=True)
        pred = model(x_t)
        loss = criterion(pred, x_next)
        total += loss.item() * x_t.size(0)
        count += x_t.size(0)
    model.train()
    return total / max(count, 1)


# ----------------------------- Train -----------------------------
def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("U-Net next-day forecaster over the study region")
    logger.info(f"Train years: {TRAIN_YEARS}  Val years: {VAL_YEARS}")
    logger.info(f"Field: {FIELD_SHAPE}  Base dim: {BASE_DIM}  "
                f"Mults: {CHANNEL_MULTS}")
    logger.info(f"Batch: {BATCH_SIZE}  LR: {LEARNING_RATE}  Epochs: {EPOCHS}")
    logger.info("=" * 60)

    with open(MEAN_STD_FILE, "rb") as f:
        mean_std = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_ds = ConsecutivePairDataset(TRAIN_YEARS, mean_std)
    val_ds = ConsecutivePairDataset(VAL_YEARS, mean_std)
    logger.info(f"Train pairs: {len(train_ds)}  Val pairs: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            drop_last=False)

    model = ForecastUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: ForecastUNet  Params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    start_step, best_val = maybe_resume(model, optimizer, scheduler, device)
    global_step = start_step

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, count = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}",
                    leave=False, mininterval=1.0)
        for x_t, x_next, _mask in pbar:
            x_t = x_t.to(device, non_blocking=True)
            x_next = x_next.to(device, non_blocking=True)

            pred = model(x_t)
            loss = criterion(pred, x_next)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * x_t.size(0)
            count += x_t.size(0)
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.5f}",
                             avg=f"{running/count:.5f}",
                             step=global_step)

            if global_step % PRINT_EVERY == 0:
                val_loss = evaluate(model, val_loader, criterion, device)
                logger.info(f"  [step {global_step}] "
                            f"train={running/count:.5f}  val={val_loss:.5f}")

                torch.save({
                    "global_step": global_step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "best_val": best_val,
                }, ckpt_path(MODEL_PREFIX, val_loss))
                prune_checkpoints(MODEL_PREFIX, SAVE_TOP_K)

                if val_loss < best_val:
                    best_val = val_loss
                    best_path = os.path.join(MODEL_DIR, f"{MODEL_PREFIX}_best.pt")
                    torch.save(model.state_dict(), best_path)
                    logger.info(f"  New best val={best_val:.5f} -> best.pt")

        # End-of-epoch validation
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        logger.info(
            f"Epoch {epoch}/{EPOCHS}  train={running/max(count,1):.5f}  "
            f"val={val_loss:.5f}  best_val={best_val:.5f}  "
            f"elapsed={(time.time()-t_start)/60:.1f}min"
        )

    logger.info("Training complete.")
    logger.info(f"Best val={best_val:.5f}  total={(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()