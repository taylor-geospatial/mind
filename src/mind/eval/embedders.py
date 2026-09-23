"""Coordinate embeddings for MIND, AlphaEarth, and the comparison models."""

import hashlib
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mind.data.quantization import dequantize_signed_square
from mind.data.zarr_aef import INNER_CHUNK_PX, MOSAIC_YEARS, NODATA, lonlat_to_pixel, open_mosaic
from mind.encoders import ReSIRENLocationEncoder
from mind.inference import embed, load_mind
from mind.location_encoder import load_pretrained_location_encoder

AEF_CACHE_DIR = Path.home() / ".cache" / "mind" / "aef_embed"


def chunks(n: int, size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, n, size):
        yield start, min(start + size, n)


@cache
def load_ours(ckpt_path: str | Path, encoder_type: str, device: str) -> ReSIRENLocationEncoder:
    """Cache a MIND checkpoint with architecture inferred from its weights."""
    variants = {"resiren": "siren", "finer": "finer", "hsiren": "hsiren"}
    if encoder_type not in variants:
        raise ValueError(f"Unsupported MIND encoder: {encoder_type}")
    return load_mind(ckpt_path, device=device, variant=variants[encoder_type])


def embed_ours(
    ckpt_path: str | Path,
    lat: np.ndarray,
    lon: np.ndarray,
    year: int | np.ndarray = 2021,
    feature: str = "pooled",
    encoder_type: str = "resiren",
    context_k: int = 0,
    context_radius: float = 0.1,
    context_input: bool = False,
    device: str = "cuda",
    batch_size: int = 8192,
) -> np.ndarray:
    """Embed MIND coordinates, optionally averaging a ring of nearby queries.

    ``context_input`` is retained for compatibility; linear heads commute with pooling.
    """
    enc = load_ours(ckpt_path, encoder_type, device)
    n = len(lat)
    years = np.clip(np.broadcast_to(year, (n,)), 2017, 2025)
    if context_k > 0:
        lat, lon = ring_coords(lat, lon, context_k, context_radius)
        years = np.repeat(years, context_k + 1)
    features = embed(enc, lat, lon, dim=None, year=years, batch_size=batch_size, feature=feature)
    if context_k > 0:
        features = features.reshape(n, context_k + 1, features.shape[-1]).mean(axis=1)
    return features


def embed_ours_heads(
    ckpt_path: str | Path,
    lat: np.ndarray,
    lon: np.ndarray,
    year: int | np.ndarray = 2021,
    device: str = "cuda",
    **arch: Any,
) -> np.ndarray:
    """Concatenate AEF, Climplicit, GeoCLIP, and SINR head reconstructions.

    Requires the teacher head state dicts saved beside the encoder checkpoint.
    """
    pooled = embed_ours(ckpt_path, lat, lon, year=year, feature="pooled", device=device, **arch)
    aef = embed_ours(ckpt_path, lat, lon, year=year, feature="head", device=device, **arch)
    base = str(ckpt_path).removesuffix(".pt")
    pooled_t = torch.as_tensor(pooled, device=device, dtype=torch.float32)
    parts = [aef]
    for suffix in ("climplicit", "geoclip", "sinr"):
        head = Path(f"{base}.{suffix}.pt")
        if not head.exists():
            raise FileNotFoundError(f"distilled {suffix} head not found at {head}")
        sd = torch.load(head, map_location=device)
        parts.append((pooled_t @ sd["weight"].float().t() + sd["bias"].float()).cpu().numpy())
    return np.concatenate(parts, axis=1)


@cache
def load_satclip(weights_path: str | Path, device: str) -> torch.nn.Module:
    return load_pretrained_location_encoder(weights_path, device=device)


@torch.no_grad()
def embed_satclip(
    weights_path: str | Path, lat: np.ndarray, lon: np.ndarray, device: str = "cuda"
) -> np.ndarray:
    """Pretrained SatCLIP location encoder (expects (lon, lat) order)."""
    enc = load_satclip(weights_path, device)
    lonlat = torch.stack([torch.as_tensor(lon), torch.as_tensor(lat)], dim=1).double()
    return enc(lonlat.to(device), chunk_size=8192).float().cpu().numpy()


@cache
def geoclip_model(device: str) -> torch.nn.Module:
    from geoclip import LocationEncoder

    return LocationEncoder().to(device).eval()


@torch.no_grad()
def embed_geoclip(
    lat: np.ndarray, lon: np.ndarray, device: str = "cuda", batch_size: int = 8192
) -> np.ndarray:
    """Pretrained GeoCLIP encoder (Equal-Earth + RFF). Input (lat, lon) -> 512-d."""
    enc = geoclip_model(device)
    x = torch.stack([torch.as_tensor(np.asarray(lat)), torch.as_tensor(np.asarray(lon))], 1).float()
    return np.concatenate(
        [enc(x[a:b].to(device)).float().cpu().numpy() for a, b in chunks(len(x), batch_size)]
    )


@cache
def climplicit_model(device: str) -> torch.nn.Module:
    from rshf.climplicit import Climplicit

    return (
        Climplicit.from_pretrained("Jobedo/climplicit", config={"return_chelsa": False})
        .to(device)
        .eval()
    )


@torch.no_grad()
def embed_climplicit(
    lat: np.ndarray, lon: np.ndarray, device: str = "cuda", batch_size: int = 8192
) -> np.ndarray:
    """Pretrained Climplicit climate-specialist encoder (CHELSA, ReSIREN). Input (lon, lat) -> 1024-d."""
    enc = climplicit_model(device)
    x = torch.stack([torch.as_tensor(np.asarray(lon)), torch.as_tensor(np.asarray(lat))], 1).float()
    return np.concatenate(
        [enc(x[a:b].to(device)).float().cpu().numpy() for a, b in chunks(len(x), batch_size)]
    )


@cache
def sinr_model(device: str) -> torch.nn.Module:
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
    return SINR.from_pretrained(repo, config=conf).to(device).eval()


@torch.no_grad()
def embed_sinr(
    lat: np.ndarray, lon: np.ndarray, device: str = "cuda", batch_size: int = 8192
) -> np.ndarray:
    """Pretrained SINR location encoder (Cole et al. 2023). Input (lon, lat) -> 256-d features."""
    from rshf.sinr import preprocess_locs

    enc = sinr_model(device)
    x = torch.stack([torch.as_tensor(np.asarray(lon)), torch.as_tensor(np.asarray(lat))], 1).float()
    return np.concatenate(
        [
            enc(preprocess_locs(x[a:b].clone().to(device)), return_feats=True).float().cpu().numpy()
            for a, b in chunks(len(x), batch_size)
        ]
    )


def ring_coords(
    lat: np.ndarray, lon: np.ndarray, k: int, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return each query and ``k`` ring neighbors, flattened; ``radius`` is in degrees."""
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    ang = np.arange(k) * (2.0 * np.pi / k)
    rlat = np.concatenate([lat[:, None], lat[:, None] + (radius * np.sin(ang))[None, :]], axis=1)
    rlon = np.concatenate([lon[:, None], lon[:, None] + (radius * np.cos(ang))[None, :]], axis=1)
    return rlat.reshape(-1), rlon.reshape(-1)


def jitter_coords(
    lat: np.ndarray, lon: np.ndarray, k: int, sigma: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return each query and ``k`` jittered copies, flattened; ``sigma`` is in degrees."""
    rng = np.random.default_rng(seed)
    lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
    n = len(lat)
    jlat = np.concatenate([lat[:, None], lat[:, None] + rng.normal(0, sigma, (n, k))], axis=1)
    jlon = np.concatenate([lon[:, None], lon[:, None] + rng.normal(0, sigma, (n, k))], axis=1)
    return jlat.reshape(-1), jlon.reshape(-1)


def embed_raw(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Encode latitude and longitude with sine and cosine."""
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([np.sin(lat_r), np.cos(lat_r), np.sin(lon_r), np.cos(lon_r)], axis=1).astype(
        np.float32
    )


def embed_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Map latitude and longitude to unit-sphere Cartesian coordinates."""
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack(
        [np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)], axis=1
    ).astype(np.float32)


@lru_cache(maxsize=1)
def mosaic_handle() -> Any:
    return open_mosaic()


def embed_aef(
    lat: np.ndarray, lon: np.ndarray, year: int | np.ndarray = 2021, workers: int = 32
) -> np.ndarray:
    """Read 64-channel AlphaEarth embeddings from the mosaic.

    Returns dequantized, L2-normalized vectors, with nodata set to NaN. Reads are
    grouped by chunk and cached for each set of coordinates and years.
    """
    years_in = np.broadcast_to(np.asarray(year), (len(lat),))
    digest = hashlib.md5(
        np.concatenate([np.asarray(lat), np.asarray(lon), years_in]).astype(np.float64).tobytes()
    ).hexdigest()
    cache_path = AEF_CACHE_DIR / f"{digest}.npy"
    if cache_path.exists():
        return np.load(cache_path)

    arr = mosaic_handle()
    _, _, height, width = arr.shape
    n = len(lat)
    years = np.clip(np.broadcast_to(np.asarray(year), (n,)), MOSAIC_YEARS[0], MOSAIC_YEARS[-1])
    time_idx = (years - MOSAIC_YEARS[0]).astype(int)
    frow, fcol = lonlat_to_pixel(np.asarray(lat, float), np.asarray(lon, float))
    row = np.clip(np.round(frow).astype(int), 0, height - 1)
    col = np.clip(np.round(fcol).astype(int), 0, width - 1)

    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i in range(n):
        groups[(int(time_idx[i]), row[i] // INNER_CHUNK_PX, col[i] // INNER_CHUNK_PX)].append(i)
    out = np.full((n, 64), np.nan, dtype=np.float32)

    def read_group(item: tuple[tuple[int, int, int], list[int]]) -> None:
        (t, gy, gx), idxs = item
        y0, x0 = gy * INNER_CHUNK_PX, gx * INNER_CHUNK_PX
        block = np.asarray(arr[t, :, y0 : y0 + INNER_CHUNK_PX, x0 : x0 + INNER_CHUNK_PX])
        for i in idxs:
            vec = block[:, row[i] - y0, col[i] - x0]
            if np.all(vec != NODATA):
                out[i] = vec.astype(np.float32)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(read_group, groups.items()))

    embeddings = F.normalize(dequantize_signed_square(torch.from_numpy(out)), dim=-1).numpy()
    AEF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embeddings)
    return embeddings
