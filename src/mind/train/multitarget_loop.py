"""MIND pixel distillation from AEF, Climplicit, GeoCLIP, and SINR."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mind.data.dataset import AEFDataset
from mind.data.mindset import load_mindset
from mind.data.quantization import dequantize
from mind.encoders import ReSIRENLocationEncoder
from mind.train.losses import CosineMSELoss, InfoNCELoss, vicreg_regularizer
from mind.train.schedule import cosine_warmup_lr


class CombinedDataset:
    """Pixel coordinates and decoded AEF targets, with optional cached coordinate teachers."""

    def __init__(
        self,
        coords: Tensor,
        aef: Tensor,
        years: tuple[int, ...],
        device: str,
        teacher_targets: dict[str, Tensor] | None = None,
    ) -> None:
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must have shape [N, 2] in latitude/longitude order")
        if aef.ndim not in (2, 3) or len(aef) != len(coords):
            raise ValueError("aef must have shape [N, channels] or [N, years, channels]")
        if aef.ndim == 3 and aef.shape[1] != len(years):
            raise ValueError("aef year axis must match years")
        if not len(coords) or not years or not aef.is_floating_point():
            raise ValueError("Expected nonempty pixels with decoded floating-point AEF targets")
        self.coords = coords.float().to(device)
        self.aef = aef.to(device)
        self.years = years
        self.device = device
        self.teacher_targets = teacher_targets or {}
        for key, values in self.teacher_targets.items():
            if values.ndim != 2 or len(values) != len(coords) or not values.is_floating_point():
                raise ValueError(
                    f"Cached {key} targets must have shape [N, channels] and be decoded"
                )

    @classmethod
    def from_dir(cls, d: str | Path, device: str) -> "CombinedDataset":
        """Read released MINDSET GeoParquet, decoded NumPy targets, or original AEF shards."""
        p = Path(d)
        if (p / "manifest.json").exists():
            pixels = AEFDataset.from_dir(p)
            aef = torch.empty(pixels.targets.shape, dtype=torch.float16, device=device)
            for start in range(0, len(pixels), 8192):
                stop = start + 8192
                annual = dequantize(pixels.targets[start:stop].to(device), pixels.dequant)
                aef[start:stop] = F.normalize(annual, dim=-1).half()
            return cls(pixels.coords, aef, pixels.years, device)
        if (p / "coords.npy").exists():
            return cls(
                torch.from_numpy(np.load(p / "coords.npy")),
                torch.from_numpy(np.load(p / "aef.npy")),
                tuple(int(y) for y in np.load(p / "years.npy")),
                device,
            )
        coords, aef, years, teachers = load_mindset(p)
        return cls(coords, aef, years, device, teachers)

    def aef_batch(self, idx: Tensor) -> Tensor:
        """Return each sampled pixel's mean across decoded annual AEF embeddings."""
        values = self.aef[idx].float()
        return values.mean(1) if values.ndim == 3 else values

    def __len__(self) -> int:
        return self.coords.shape[0]


_CLIM_TEACHER: list = []
_GEO_TEACHER: list = []


def _load_climplicit(device: torch.device) -> nn.Module:
    """Cached Climplicit teacher: (lon, lat) degrees to 1024 features."""
    if not _CLIM_TEACHER:
        from rshf.climplicit import Climplicit

        m = Climplicit.from_pretrained("Jobedo/climplicit", config={"return_chelsa": False})
        _CLIM_TEACHER.append(m.to(device).eval().requires_grad_(False))
    return _CLIM_TEACHER[0]


def _load_geoclip(device: torch.device) -> nn.Module:
    """Cached GeoCLIP teacher: (lat, lon) degrees to 512 features."""
    if not _GEO_TEACHER:
        from geoclip import LocationEncoder

        _GEO_TEACHER.append(LocationEncoder().to(device).eval().requires_grad_(False))
    return _GEO_TEACHER[0]


_SINR_TEACHER: list = []


def _load_sinr(device: torch.device) -> nn.Module:
    """Cached SINR teacher (Cole et al., 2023), with 256 output features."""
    if not _SINR_TEACHER:
        import json

        from huggingface_hub import hf_hub_download
        from rshf.sinr import SINR, SINRConfig

        repo = "MVRL/sinr-location-encoder-1000-cls"
        cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
        conf = SINRConfig(
            num_inputs=cfg["num_inputs"],
            num_filts=cfg["num_filts"],
            depth=cfg["depth"],
            num_classes=cfg["num_classes"],
        )
        _SINR_TEACHER.append(
            SINR.from_pretrained(repo, config=conf).to(device).eval().requires_grad_(False)
        )
    return _SINR_TEACHER[0]


def _sinr_embed(teacher: nn.Module, coords_latlon: Tensor) -> Tensor:
    """Convert (lat, lon) coordinates for SINR; preprocessing mutates its input."""
    from rshf.sinr import preprocess_locs

    return teacher(preprocess_locs(coords_latlon[:, [1, 0]].clone()), return_feats=True)


def train_combined(
    encoder: ReSIRENLocationEncoder,
    dataset: CombinedDataset,
    steps: int,
    batch_size: int = 2048,
    w_aef: float = 1.0,
    w_clim: float = 1.0,
    w_geo: float = 1.0,
    w_sinr: float = 1.0,
    cos_weight: float = 1.0,
    mse_weight: float = 1.0,
    infonce_weight: float = 0.0,
    lr: float = 3e-4,
    weight_decay: float = 0.05,
    warmup_steps: int = 1000,
    log_every: int = 50,
    log_fn: Callable[[int, dict[str, float]], None] | None = None,
    ckpt_path: str | None = None,
    ckpt_every: int = 2000,
    standardize_teachers: bool = True,
    vicreg_var_weight: float = 0.0,
    vicreg_cov_weight: float = 0.0,
    matryoshka_dims: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048),
    mrl_teachers: tuple[str, ...] = (),
) -> ReSIRENLocationEncoder:
    """Distill AEF annual means and coordinate teachers into ReSIREN prefixes.

    Checkpoints contain the encoder state dict. Full-width teacher heads are saved
    separately; prefix heads are used only during training.
    """
    if encoder.use_year:
        raise ValueError("MIND training uses coordinates only; set use_year=False")
    if steps < 1 or batch_size < 1 or ckpt_every < 1 or log_every < 1:
        raise ValueError("steps, batch_size, ckpt_every and log_every must be positive")
    weights = (w_aef, w_clim, w_geo, w_sinr)
    if min(weights) < 0 or max(weights) == 0:
        raise ValueError("Teacher weights must be nonnegative with at least one active teacher")
    if any(m <= 0 or m > encoder.embed_dim for m in matryoshka_dims):
        raise ValueError("Matryoshka dimensions must be within the encoder width")
    dev = torch.device(dataset.device)
    encoder = encoder.to(dev).train()
    reg_loss = CosineMSELoss(mse_weight=mse_weight, cos_weight=cos_weight).to(dev)
    infonce = InfoNCELoss().to(dev) if infonce_weight > 0 else None
    n = len(dataset)
    params = list(encoder.parameters())
    if infonce is not None:  # learnable temperature
        params += [*infonce.parameters()]

    def loss_fn(pred: Tensor, target: Tensor) -> dict:
        """Cosine/MSE distillation loss with optional in-batch InfoNCE."""
        d = reg_loss(pred, target)
        if infonce is not None:
            d = {**d, "loss": d["loss"] + infonce_weight * infonce(pred, target)["loss"]}
        return d

    clim_teacher = clim_head = None
    if w_clim > 0:
        if "clim" in dataset.teacher_targets:
            cdim = dataset.teacher_targets["clim"].shape[-1]
        else:
            clim_teacher = _load_climplicit(dev)
            with torch.no_grad():
                cdim = clim_teacher(dataset.coords[:2][:, [1, 0]]).shape[-1]
        clim_head = nn.Linear(encoder.embed_dim, cdim).to(dev).train()
        params += [*clim_head.parameters()]

    geo_teacher = geo_head = None
    if w_geo > 0:
        if "geo" in dataset.teacher_targets:
            gdim = dataset.teacher_targets["geo"].shape[-1]
        else:
            geo_teacher = _load_geoclip(dev)
            with torch.no_grad():
                gdim = geo_teacher(dataset.coords[:2]).shape[-1]
        geo_head = nn.Linear(encoder.embed_dim, gdim).to(dev).train()
        params += [*geo_head.parameters()]

    sinr_teacher = sinr_head = None
    if w_sinr > 0:
        if "sinr" in dataset.teacher_targets:
            sdim = dataset.teacher_targets["sinr"].shape[-1]
        else:
            sinr_teacher = _load_sinr(dev)
            with torch.no_grad():
                sdim = _sinr_embed(sinr_teacher, dataset.coords[:2]).shape[-1]
        sinr_head = nn.Linear(encoder.embed_dim, sdim).to(dev).train()
        params += [*sinr_head.parameters()]

    @torch.no_grad()
    def coordinate_targets(idx: Tensor) -> dict[str, Tensor]:
        targets = {}
        for key, weight in (("clim", w_clim), ("geo", w_geo), ("sinr", w_sinr)):
            if weight > 0 and key in dataset.teacher_targets:
                values = dataset.teacher_targets[key]
                targets[key] = values[idx.to(values.device)].to(dev).float()
        if w_clim > 0 and "clim" not in targets:
            targets["clim"] = clim_teacher(dataset.coords[idx][:, [1, 0]]).float()
        if w_geo > 0 and "geo" not in targets:
            targets["geo"] = geo_teacher(dataset.coords[idx]).float()
        if w_sinr > 0 and "sinr" not in targets:
            targets["sinr"] = _sinr_embed(sinr_teacher, dataset.coords[idx]).float()
        return targets

    # Each prefix has separate training heads for teacher reconstruction.
    mrl_dims = tuple(m for m in matryoshka_dims if m < encoder.embed_dim)
    mrl_heads = nn.ModuleDict()
    if mrl_dims:
        tdims = {"aef": dataset.aef.shape[-1]}
        if w_clim > 0:
            tdims["clim"] = cdim
        if w_geo > 0:
            tdims["geo"] = gdim
        if w_sinr > 0:
            tdims["sinr"] = sdim
        # Unlisted teachers supervise only the full-width head.
        if mrl_teachers:
            tdims = {k: v for k, v in tdims.items() if k in mrl_teachers}
        for tk, td in tdims.items():
            mrl_heads[tk] = (
                nn.ModuleDict({str(m): nn.Linear(m, td) for m in mrl_dims}).to(dev).train()
            )
        params += [p for h in mrl_heads.values() for p in h.parameters()]

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=(dev.type == "cuda"))
    # PHI-S global standardization (arXiv:2410.01680): channel means and one RMS per teacher.
    tgt_stats: dict[str, tuple[Tensor, Tensor]] = {}
    if standardize_teachers:
        ns = min(50000, n)
        sidx = torch.randint(0, n, (ns,), device=dev)
        with torch.no_grad():
            samples = coordinate_targets(sidx)
            samples["aef"] = dataset.aef_batch(sidx)
            for k, v in samples.items():
                mu = v.mean(0, keepdim=True)
                scale = (v - mu).pow(2).mean().sqrt().clamp_min(1e-6)  # scalar RMS across all dims
                tgt_stats[k] = (mu, scale)

    def std_tgt(t: Tensor, key: str) -> Tensor:
        """Standardize a teacher target by its (per-channel mean, scalar RMS); identity if unset."""
        if key not in tgt_stats:
            return t
        mu, scale = tgt_stats[key]
        return (t - mu) / scale

    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), device=dev)
        for g in opt.param_groups:
            g["lr"] = cosine_warmup_lr(step, warmup_steps, steps, lr)
        opt.zero_grad(set_to_none=True)
        targets = coordinate_targets(idx)
        clim_tgt, geo_tgt, sinr_tgt = (targets.get(key) for key in ("clim", "geo", "sinr"))
        aef_tgt = dataset.aef_batch(idx)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            pooled = encoder(dataset.coords[idx], return_features=True)
            aef_pred = encoder.head(pooled)
            la = loss_fn(aef_pred, std_tgt(aef_tgt, "aef"))
            total = w_aef * la["loss"]
            logd = {"aef_cos": float(la["cosine"])}
            if w_clim > 0:
                lc = loss_fn(clim_head(pooled), std_tgt(clim_tgt, "clim"))
                total = total + w_clim * lc["loss"]
                logd["clim_cos"] = float(lc["cosine"])
            if w_geo > 0:
                lg = loss_fn(geo_head(pooled), std_tgt(geo_tgt, "geo"))
                total = total + w_geo * lg["loss"]
                logd["geo_cos"] = float(lg["cosine"])
            if w_sinr > 0:
                ls = loss_fn(sinr_head(pooled), std_tgt(sinr_tgt, "sinr"))
                total = total + w_sinr * ls["loss"]
                logd["sinr_cos"] = float(ls["cosine"])
            if mrl_dims:
                mrl_tgts = {"aef": (std_tgt(aef_tgt, "aef"), w_aef)}
                if w_clim > 0:
                    mrl_tgts["clim"] = (std_tgt(clim_tgt, "clim"), w_clim)
                if w_geo > 0:
                    mrl_tgts["geo"] = (std_tgt(geo_tgt, "geo"), w_geo)
                if w_sinr > 0:
                    mrl_tgts["sinr"] = (std_tgt(sinr_tgt, "sinr"), w_sinr)
                mrl_total = pooled.new_zeros(())
                mrl_tgts = {k: v for k, v in mrl_tgts.items() if k in mrl_heads}
                for tk, (tg, w) in mrl_tgts.items():
                    for m in mrl_dims:
                        mrl_total = (
                            mrl_total
                            + w * loss_fn(mrl_heads[tk][str(m)](pooled[:, :m]), tg)["loss"]
                        )
                mrl_total = mrl_total / len(mrl_dims)
                total = total + mrl_total
                logd["mrl"] = float(mrl_total.detach())
            if vicreg_var_weight > 0.0 or vicreg_cov_weight > 0.0:
                vic = vicreg_regularizer(pooled.float(), vicreg_var_weight, vicreg_cov_weight)
                total = total + vic
                logd["vicreg"] = float(vic.detach())
        total.backward()
        opt.step()
        if log_fn is not None and (step % log_every == 0 or step == steps - 1):
            log_fn(step, {"loss": float(total.detach()), **logd})
        if ckpt_path is not None and step > 0 and step % ckpt_every == 0:
            torch.save(encoder.state_dict(), ckpt_path)

    if ckpt_path is not None:
        torch.save(encoder.state_dict(), ckpt_path)
        if clim_head is not None:
            torch.save(clim_head.state_dict(), ckpt_path.replace(".pt", "") + ".climplicit.pt")
        if geo_head is not None:
            torch.save(geo_head.state_dict(), ckpt_path.replace(".pt", "") + ".geoclip.pt")
        if sinr_head is not None:
            torch.save(sinr_head.state_dict(), ckpt_path.replace(".pt", "") + ".sinr.pt")
    return encoder
