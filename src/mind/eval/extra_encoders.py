"""Pretrained location-encoder baselines for CoordBench."""

import urllib.request
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

CACHE_DIR = Path.home() / ".cache" / "mind" / "extra_encoders"
GAIR_SIGMAS = (2.0**0, 2.0**4, 2.0**8)
BATCH = 65536


def _fetch(url: str, name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / name
    if not p.exists():
        tmp = p.with_suffix(p.suffix + ".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(p)
    return p


# GAIR (Liu et al., ISPRS 2026, arXiv:2503.16683).
def _equal_earth(coords: torch.Tensor) -> torch.Tensor:
    """GAIR's Equal-Earth projection of (lon, lat) pairs in degrees."""
    a1, a2, a3, a4, sf = 1.340264, -0.081106, 0.000893, 0.003796, 66.50336
    lat, lon = torch.deg2rad(coords[:, 1]), torch.deg2rad(coords[:, 0])
    theta = torch.asin((3.0**0.5 / 2) * torch.sin(lat))
    denom = 3 * (9 * a4 * theta**8 + 7 * a3 * theta**6 + 3 * a2 * theta**2 + a1)
    x = (2 * 3.0**0.5 * lon * torch.cos(theta)) / denom
    y = a4 * theta**9 + a3 * theta**7 + a2 * theta**3 + a1 * theta
    return (torch.stack((x, y), dim=1) * sf) / 180


class _GairCapsule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.b: torch.Tensor
        self.register_buffer("b", torch.zeros(256, 2))
        self.capsule = nn.Sequential(
            nn.Identity(),  # preserve GAIR state-dict indices
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(1024, 768))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vp = 2 * np.pi * x @ self.b.T
        return self.head(self.capsule(torch.cat((vp.cos(), vp.sin()), dim=-1)))


class _GairLocationEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for i in range(len(GAIR_SIGMAS)):
            self.add_module(f"LocEnc{i}", _GairCapsule())

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = _equal_earth(coords)
        return sum(self._modules[f"LocEnc{i}"](x) for i in range(len(GAIR_SIGMAS)))


@cache
def gair_model(device: str) -> nn.Module:
    from huggingface_hub import hf_hub_download

    state = torch.load(
        hf_hub_download("PingL/GAIR", "checkpoint.pth"), map_location="cpu", weights_only=False
    )
    for key in ("state_dict", "model", "model_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    prefix = next(k for k in state if "LocEnc0.capsule.1.weight" in k).split("LocEnc0")[0]
    sub = {}
    for k, v in state.items():
        if not k.startswith(prefix):
            continue
        name = k[len(prefix) :]
        # GAIR stores the RFF matrix as LocEnc{i}.capsule.0.b; ours is a buffer LocEnc{i}.b
        sub[name.replace("capsule.0.b", "b")] = v
    enc = _GairLocationEncoder()
    enc.load_state_dict(sub)
    return enc.to(device).eval()


@torch.no_grad()
def embed_gair(lat: np.ndarray, lon: np.ndarray, device: str = "cuda") -> np.ndarray:
    """GAIR location-encoder embeddings [N, 768]."""
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    enc = gair_model(dev)
    coords = torch.stack(
        [torch.as_tensor(lon, dtype=torch.float32), torch.as_tensor(lat, dtype=torch.float32)], 1
    )
    out = [enc(coords[i : i + BATCH].to(dev)).cpu().numpy() for i in range(0, len(coords), BATCH)]
    return np.concatenate(out, axis=0)


# CSP (Mai et al., ICML 2023). Loading follows SatCLIP
# notebooks/C01_Simple_CSP_Usage. Use UNSUPER checkpoints without label fine-tuning.

CSP_SHA = "67f34075c944ad3845499892a23c7363691cf3bf"  # pinned gengchenmai/csp revision
CSP_REPO = "https://github.com/gengchenmai/csp.git"
CSP_CKPT_URL = "https://www.dropbox.com/s/qxr644rj1qxekn2/model_dir.zip?dl=1"
CSP_VARIANTS = {"csp_inat": "model_inat_2018", "csp_fmow": "model_fmow"}


def _csp_repo() -> Path:
    import subprocess

    d = CACHE_DIR / "csp"
    if not (d / "main" / "utils.py").exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", CSP_REPO, str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "--quiet", CSP_SHA], check=True)
    return d


def _csp_ckpt(variant: str) -> Path:
    import zipfile

    sub = CSP_VARIANTS[variant]
    root = CACHE_DIR / "model_dir"
    if not root.exists():
        z = _fetch(CSP_CKPT_URL, "csp_model_dir.zip")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(CACHE_DIR, members=[m for m in zf.namelist() if "__MACOSX" not in m])
    hits = sorted((root / sub).glob("*UNSUPER*.pth.tar"))
    if not hits:
        raise FileNotFoundError(f"no CSP UNSUPER checkpoint under {root / sub}")
    return hits[0]


@cache
def csp_model(variant: str, device: str) -> tuple:
    """Frozen CSP location encoder. Returns (model, convert_loc_to_tensor) for the given variant."""
    import sys

    repo = _csp_repo()
    main = str(repo / "main")
    if main not in sys.path:  # the repo's modules import each other by bare name
        sys.path.insert(0, main)
    from models import LocationImageEncoder  # ty: ignore[unresolved-import]
    from utils import convert_loc_to_tensor, get_model  # ty: ignore[unresolved-import]

    ck = torch.load(_csp_ckpt(variant), map_location="cpu", weights_only=False)
    p = dict(ck["params"])
    p["device"] = device
    loc_enc = get_model(
        train_locs=None,
        params=p,
        spa_enc_type=p["spa_enc_type"],
        num_inputs=p["num_loc_feats"],
        num_classes=p["num_classes"],
        num_filts=p["num_filts"],
        num_users=p["num_users"],
        device=device,
    )
    model = LocationImageEncoder(
        loc_enc=loc_enc,
        train_loss=p["train_loss"],
        unsuper_loss=p["unsuper_loss"],
        cnn_feat_dim=p["cnn_feat_dim"],
        spa_enc_type=p["spa_enc_type"],
    ).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, convert_loc_to_tensor


@torch.no_grad()
def embed_csp(
    lat: np.ndarray, lon: np.ndarray, device: str = "cuda", variant: str = "csp_inat"
) -> np.ndarray:
    """CSP embeddings [N, 256]; ``convert_loc_to_tensor`` expects (lon, lat) degrees."""
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    model, to_tensor = csp_model(variant, dev)
    lonlat = np.stack([np.asarray(lon), np.asarray(lat)], axis=1).astype(np.float32)
    out = []
    for i in range(0, len(lonlat), BATCH):
        feats = model.loc_enc(to_tensor(lonlat[i : i + BATCH], device=dev), return_feats=True)
        out.append(feats.detach().float().cpu().numpy())
    return np.concatenate(out, axis=0)


# TTE (Tessellating the Earth, ECCV 2026).

TTE_SHA = "2bce5e6c5a963f884ea01fe1cb8a1262a6b9efbf"  # pinned mvrl/TTE revision
TTE_REPO = "https://github.com/mvrl/TTE.git"
TTE_HF = "MVRL/TTE"


@cache
def tte_model(device: str) -> nn.Module:
    import subprocess
    import sys

    d = CACHE_DIR / "tte_repo"
    if not (d / "tte" / "model.py").exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", TTE_REPO, str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "--quiet", TTE_SHA], check=True)
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    from tte import TTE  # ty: ignore[unresolved-import]

    return TTE.from_pretrained(TTE_HF).to(device).eval()


@torch.no_grad()
def embed_tte(lat: np.ndarray, lon: np.ndarray, device: str = "cuda") -> np.ndarray:
    """TTE embeddings [N, 512], L2-normalized; the model takes (lat, lon) degrees."""
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    enc = tte_model(dev)
    coords = torch.stack(
        [torch.as_tensor(lat, dtype=torch.float32), torch.as_tensor(lon, dtype=torch.float32)], 1
    )
    out = [
        enc.encode(coords[i : i + BATCH].to(dev)).float().cpu().numpy()
        for i in range(0, len(coords), BATCH)
    ]
    return np.concatenate(out, axis=0)


# TaxaBind (Sastry et al., WACV 2025). Pass the config explicitly to retain sigma.

TAXABIND_SHA = "87fb301dd2a3a855f4aa5008a0ac734289a8b1ff"  # pinned mvrl/rshf revision
TAXABIND_REPO = "https://github.com/mvrl/rshf.git"
TAXABIND_HF = "MVRL/ecogeo"


@cache
def taxabind_model(device: str) -> nn.Module:
    import json
    import subprocess
    import sys

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    d = CACHE_DIR / "rshf"
    if not (d / "rshf" / "geoclip" / "model.py").exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", TAXABIND_REPO, str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "--quiet", TAXABIND_SHA], check=True)
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    from rshf.geoclip.model import GeoCLIPConfig, LocationEncoder

    with open(hf_hub_download(TAXABIND_HF, "config.json")) as fh:
        cfg = json.load(fh)
    model = LocationEncoder(
        GeoCLIPConfig(
            sigma=cfg["sigma"],
            input_size=cfg["input_size"],
            encoded_size=cfg["encoded_size"],
            dim=cfg["dim"],
        )
    )
    model.load_state_dict(load_file(hf_hub_download(TAXABIND_HF, "model.safetensors")))
    return model.to(device).eval()


@torch.no_grad()
def embed_taxabind(lat: np.ndarray, lon: np.ndarray, device: str = "cuda") -> np.ndarray:
    """TaxaBind embeddings [N, 512]; the model takes (lat, lon) degrees."""
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    enc = taxabind_model(dev)
    coords = torch.stack(
        [torch.as_tensor(lat, dtype=torch.float32), torch.as_tensor(lon, dtype=torch.float32)], 1
    )
    out = [
        enc(coords[i : i + BATCH].to(dev)).float().cpu().numpy()
        for i in range(0, len(coords), BATCH)
    ]
    return np.concatenate(out, axis=0)


# SLED (Lane et al., arXiv:2608.06612). Its projection expects (lat, lon).

SLED_SHA = "0e30d3e9e69200ab41560bdedfd3a6c71178dbd6"  # pinned geohai/sled revision
SLED_REPO = "https://github.com/geohai/sled.git"
SLED_VARIANTS = {"sled_s2ls": "geohai/sled-s2-ls", "sled_s1s2ls": "geohai/sled-s1-s2-ls"}


@cache
def sled_model(variant: str, device: str) -> nn.Module:
    import json
    import subprocess
    import sys

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    d = CACHE_DIR / "sled"
    if not (d / "sled_geo" / "position_encoders.py").exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", SLED_REPO, str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "checkout", "--quiet", SLED_SHA], check=True)
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    from sled_geo.position_encoders import (  # ty: ignore[unresolved-import]
        LocationEncoderGeoCLIP,
    )

    hf = SLED_VARIANTS[variant]
    with open(hf_hub_download(hf, "config.json")) as fh:
        cfg = json.load(fh)
    model = LocationEncoderGeoCLIP(
        sigma=cfg["sigma"],
        input_size=cfg["input_size"],
        encoded_size=cfg["encoded_size"],
        dim=cfg["dim"],
    )
    model.load_state_dict(load_file(hf_hub_download(hf, "model.safetensors")))
    return model.to(device).eval()


@torch.no_grad()
def embed_sled(
    lat: np.ndarray, lon: np.ndarray, device: str = "cuda", variant: str = "sled_s1s2ls"
) -> np.ndarray:
    """SLED location embeddings [N, 768]. Input order is (lat, lon) in degrees."""
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    enc = sled_model(variant, dev)
    coords = torch.stack(
        [torch.as_tensor(lat, dtype=torch.float32), torch.as_tensor(lon, dtype=torch.float32)], 1
    )
    out = [
        enc(coords[i : i + BATCH].to(dev)).float().cpu().numpy()
        for i in range(0, len(coords), BATCH)
    ]
    return np.concatenate(out, axis=0)
