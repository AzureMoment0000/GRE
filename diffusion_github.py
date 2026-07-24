"""
Diffusion-based CMFD field generation over the study region (15-35N, 90-110E).

V-prediction diffusion on four-variable daily fields (prec, pres, wind, temp)
with static conditioning (solar altitude, elevation). Trained on CMFD v2.0
daily fields with v-prediction parameterization (v = alpha*eps - sigma*x).

Run:
    python diffusion_github.py 2>&1 | tee diffusion_github.log
"""

import gc
import logging
import math
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
from functools import partial
from glob import glob
from pathlib import Path
import netCDF4 as nc
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ----------------------------- Model components -----------------------------
def Downsample_1deg(dim_in, dim_out, scale=2):
    class DownsampleModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.maxpool = nn.MaxPool2d(scale)
            self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=1)

        def forward(self, x):
            return self.conv(self.maxpool(x))

    return DownsampleModule()


def Upsample_1deg(dim_in, dim_out, scale=2):
    class UpsampleModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.upsample = nn.Upsample(scale_factor=scale, mode="bilinear",
                                        align_corners=True)
            self.conv = nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)

        def forward(self, x):
            return self.conv(self.upsample(x))

    return UpsampleModule()


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :] * 1000.0
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class Block(nn.Module):
    def __init__(self, dim_in, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim_in, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.norm(self.proj(x))
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp_t = (nn.Sequential(nn.SiLU(),
                                    nn.Linear(time_emb_dim, dim_out * 2))
                      if time_emb_dim is not None else None)
        self.block1 = Block(dim_in, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = (nn.Conv2d(dim_in, dim_out, 1)
                          if dim_in != dim_out else nn.Identity())

    def forward(self, x, time_emb=None):
        scale_shift_t = None
        if self.mlp_t is not None and time_emb is not None:
            time_emb = rearrange(self.mlp_t(time_emb), "b c -> b c 1 1")
            scale_shift_t = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift_t)
        h = self.block2(h)
        return h + self.res_conv(x)


class UNet_1degV1(nn.Module):
    def __init__(self, dim_in, dim_out, c, c_mults=(1, 2, 4, 8),
                 resnet_block_groups=4, scale=(2, 2, 2)):
        super().__init__()
        self.init_conv = nn.Conv2d(dim_in, c, 1, padding=0)
        dims = [c * x for x in c_mults]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = c * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(c),
            nn.Linear(c, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (d_in, d_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(d_in, d_in, time_emb_dim=time_dim),
                block_klass(d_in, d_in, time_emb_dim=time_dim),
                (Downsample_1deg(d_in, d_out, scale=scale[ind])
                 if not is_last
                 else nn.Conv2d(d_in, d_out, 3, padding=1)),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (d_in, d_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(d_out + d_in, d_out, time_emb_dim=time_dim),
                block_klass(d_out + d_in, d_out, time_emb_dim=time_dim),
                (Upsample_1deg(d_out, d_in, scale=scale[-ind - 1])
                 if not is_last
                 else nn.Conv2d(d_out, d_in, 3, padding=1)),
            ]))

        self.final_res_block = block_klass(c * 2, c, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(c, dim_out, 1)

    def forward(self, x, time, cond):
        x = torch.cat([x, cond], dim=1)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []

        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)


class Diffusion_for_UNet:
    def __init__(self, size, lambda_range=20, steps=1000, device="cuda"):
        self.size = size
        self.device = device
        self.lambda_range = lambda_range
        self.steps = steps
        self.b = math.atan(math.exp(-lambda_range / 2.0))
        self.a = (math.atan(math.exp(lambda_range / 2))
                  - math.atan(math.exp(-lambda_range / 2.0)))
        self.bound = 0.9999
        self.dim = [-1] + [1 for _ in size]

    def u2lambda(self, u):
        return -2 * torch.log(torch.tan(self.a * u + self.b)).to(self.device)

    def u2alpha(self, u):
        return torch.sqrt(1.0 / (1 + torch.exp(-self.u2lambda(u)))).to(self.device)

    def u2sigma(self, u):
        return torch.sqrt(1.0 - 1.0 / (1 + torch.exp(-self.u2lambda(u)))).to(self.device)

    def noisify(self, x, u, eps=None, mask=None):
        if eps is None:
            eps = torch.randn_like(x).to(self.device)
        if mask is not None:
            eps = eps * mask
            x = x * mask
        alpha = self.u2alpha(u).view(self.dim)
        sigma = self.u2sigma(u).view(self.dim)
        x_t = alpha * x + sigma * eps
        v = alpha * eps - sigma * x
        return x_t, eps, v

    def sample_timesteps(self, n):
        return torch.rand(n).to(self.device)


# ----------------------------- Config -----------------------------
DATA_DIR = "/data/CMFD/1dy010deg"
VARS = ["prec", "pres", "wind", "temp"]
YEARS = range(1981, 2021)

SOLAR_FILE = "/data/solar_altitude.nc"
ELEV_FILE = "/data/elevation.nc"
MEAN_STD_FILE = "/data/cmfd_mean_std.pkl"

LAT_MIN, LAT_MAX = 15.0, 35.0
LON_MIN, LON_MAX = 90.0, 110.0

CHANNEL = 4
FIELD_SHAPE = [CHANNEL, 200, 200]
BASE_DIM = 64

LEARNING_RATE = 1e-5
BATCH_SIZE = 4
NUM_WORKERS = 10
EPOCHS = 1000
CHECK_STEPS = 1000
MAX_CHECKPOINTS = 20
LOSS_SKIP_THRESHOLD = 3.0
GRAD_CLIP_NORM = 1.0

MODEL_DIR = Path("/data/model")
MODEL_PREFIX = "diffusion_model"

LOG_FILE = "/data/diffusion_github.log"
GRAD_ACCUM_STEPS = 1


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


# ----------------------------- Dataset -----------------------------
class WeatherDataset(Dataset):
    """CMFD daily field with static (solar, elevation) conditioning.

    Returns (data, mask, static) where data is the four-variable normalized
    field and static carries day-of-year solar altitude plus elevation.
    """

    def __init__(self, years, vars, data_dir, mean_std,
                 solar_file, elev_file,
                 lat_min, lat_max, lon_min, lon_max):
        self.years = list(years)
        self.vars = vars
        self.data_dir = data_dir
        self.mean_std = mean_std
        self.solar_file = solar_file
        self.elev_file = elev_file
        self.lat_range = (lat_min, lat_max)
        self.lon_range = (lon_min, lon_max)

        self.dates = []
        for y in self.years:
            d = datetime(y, 1, 1)
            while d.year == y:
                if not (d.month == 2 and d.day == 29):
                    self.dates.append(d.strftime("%Y%m%d"))
                d += timedelta(days=1)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]
        y = int(date[:4]); m = int(date[4:6]); d = int(date[6:])

        atm_list = []
        atm_mask_list = []
        for v in self.vars:
            path = f"{self.data_dir}/{v}_CMFD_V0200_B-01_01dy_010deg_{y}01-{y}12.nc"
            with nc.Dataset(path) as data:
                lat = data.variables["lat"][:]
                lon = data.variables["lon"][:]
                lat_idx = np.where((lat >= self.lat_range[0]) & (lat <= self.lat_range[1]))[0]
                lon_idx = np.where((lon >= self.lon_range[0]) & (lon <= self.lon_range[1]))[0]
                arr = data.variables[v][:, lat_idx.min():lat_idx.max() + 1,
                                          lon_idx.min():lon_idx.max() + 1]
                fill_value = getattr(data.variables[v], "_FillValue", 1e20)
                arr = np.ma.masked_where(arr >= fill_value, arr)

                if v == "prec":
                    arr = arr * 3600.0
                if v in ("prec", "wind"):
                    arr = np.ma.log(arr + 0.01)

                doy = (datetime(y, m, d) - datetime(y, 1, 1)).days
                arr_day = arr[doy, :, :]

                mean = self.mean_std[v]["mean"]
                std = self.mean_std[v]["std"]
                atm_list.append((arr_day - mean) / std)
                atm_mask_list.append(~arr_day.mask)

        atm = np.ma.stack(atm_list, axis=0)
        atm_mask = np.stack(atm_mask_list, axis=0)

        doy = (datetime(y, m, d) - datetime(y, 1, 1)).days
        with nc.Dataset(self.solar_file) as f:
            solar = np.ma.array(f.variables["solar_altitude"][doy, :, :])
        with nc.Dataset(self.elev_file) as f:
            elev = f.variables["elevation"][:, :]
            elev = (elev - elev.mean()) / elev.std()
            elev = np.ma.array(elev)

        static = np.ma.stack([solar, elev], axis=0)
        static_mask = np.stack([~solar.mask, ~elev.mask], axis=0)

        data = torch.tensor(atm.filled(0), dtype=torch.float32)
        data_mask = torch.tensor(atm_mask, dtype=torch.bool)
        static = torch.tensor(static.filled(0), dtype=torch.float32)
        static_mask = torch.tensor(static_mask, dtype=torch.bool)
        return data, data_mask, static, static_mask, date


# ----------------------------- Model / optimizer -----------------------------
def build_model():
    return UNet_1degV1(CHANNEL + 2, CHANNEL, BASE_DIM, c_mults=(1, 2, 4, 4))


def maybe_resume(model, optimizer, scheduler, accelerator):
    files = sorted(glob(str(MODEL_DIR / f"{MODEL_PREFIX}_*.pt")),
                   key=os.path.getmtime)
    if not files:
        logger.info("No checkpoint found, training from scratch.")
        return 0

    path = files[-1]
    ckpt = torch.load(path, map_location=accelerator.device, weights_only=True)
    state = ckpt["model"]
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.module.load_state_dict(state)
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    logger.info(f"Resumed from {path}")
    return ckpt.get("global_step", 0)


def prune_checkpoints():
    files = sorted(glob(str(MODEL_DIR / f"{MODEL_PREFIX}_*.pt")),
                   key=os.path.getmtime)
    while len(files) > MAX_CHECKPOINTS:
        os.remove(files[0])
        files = sorted(glob(str(MODEL_DIR / f"{MODEL_PREFIX}_*.pt")),
                       key=os.path.getmtime)


def save_checkpoint(model, optimizer, scheduler, global_step, loss_running,
                    tag=""):
    suffix = f"_{tag}" if tag else ""
    ts = datetime.now().strftime("%Y%m%d%H")
    loss_str = f"{loss_running:.4f}".replace(".", "")
    path = MODEL_DIR / f"{MODEL_PREFIX}_{ts}{suffix}_{loss_str}.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": global_step,
    }, path)
    logger.info(f"Saved checkpoint: {path.name}")


# ----------------------------- Train -----------------------------
def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Diffusion model training over the study region")
    logger.info(f"Years: {YEARS}  Vars: {VARS}")
    logger.info(f"Field: {FIELD_SHAPE}  Base dim: {BASE_DIM}")
    logger.info(f"Batch: {BATCH_SIZE}  LR: {LEARNING_RATE}  Epochs: {EPOCHS}")
    logger.info("=" * 60)

    with open(MEAN_STD_FILE, "rb") as f:
        mean_std = pickle.load(f)

    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM_STEPS)
    device = accelerator.device
    logger.info(f"Device: {device}")

    dataset = WeatherDataset(YEARS, VARS, DATA_DIR, mean_std,
                             SOLAR_FILE, ELEV_FILE,
                             LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True)
    logger.info(f"Train samples: {len(dataset)}")

    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: UNet_1degV1  Params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=1.0)
    mse = nn.MSELoss()
    diffusion = Diffusion_for_UNet(size=FIELD_SHAPE, lambda_range=20.0,
                                   steps=1000, device=device)

    model, optimizer, scheduler, dataloader = accelerator.prepare(
        model, optimizer, scheduler, dataloader)

    start_step = maybe_resume(model, optimizer, scheduler, accelerator)
    global_step = start_step
    best_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"Epoch {epoch}  lr={optimizer.param_groups[0]['lr']:.6f}")
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
        loss_running = 0.0
        n_seen = 0
        for i, (data, data_mask, static, _static_mask, _date) in enumerate(pbar):
            data = data.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            data_mask = data_mask.to(device, non_blocking=True)

            t = diffusion.sample_timesteps(data.shape[0])
            x_t, _noise, v = diffusion.noisify(data, t, None, mask=data_mask)
            predicted_v = model(x_t, t, cond=static)
            predicted_v = predicted_v * data_mask
            v = v * data_mask

            loss = mse(predicted_v, v)

            if loss.item() > LOSS_SKIP_THRESHOLD:
                logger.info(f"Skip step, loss={loss.item():.4f}")
                del data, static, t, loss, predicted_v, v, x_t
                continue

            loss_running = (loss_running * n_seen + loss.item()) / (n_seen + 1)
            n_seen += 1

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            pbar.set_postfix(MSE=f"{loss.item():.4f}",
                             running=f"{loss_running:.4f}",
                             step=global_step)

            if global_step % CHECK_STEPS == 0:
                gc.collect()
                torch.cuda.empty_cache()

                if accelerator.is_main_process:
                    if loss_running < best_loss:
                        best_loss = loss_running
                        save_checkpoint(model, optimizer, scheduler,
                                        global_step, loss_running, tag="best")
                    elif global_step % (3 * CHECK_STEPS) == 0:
                        save_checkpoint(model, optimizer, scheduler,
                                        global_step, loss_running)
                prune_checkpoints()

        scheduler.step()
        logger.info(f"Epoch {epoch} done  running={loss_running:.4f}  "
                    f"best={best_loss:.4f}  elapsed={(time.time()-t_start)/60:.1f}min")

    logger.info(f"Training complete in {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()