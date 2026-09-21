"""Sample annual AlphaEarth pixel vectors from an EPSG:4326 Zarr mosaic.

The int8 array has axes [year, band, y, x], 64 bands, and nodata -128. Vectors
remain quantized; decoding and normalization happen in the training loader.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

AEF_MOSAIC_URL = "s3://us-west-2.opendata.source.coop/tge-labs/aef-mosaic"
NODATA = -128
N_BANDS = 64
# Affine from the group metadata (pixel registration, top-down).
LON_ORIGIN = -180.0
LAT_ORIGIN = 83.68570533713473
PIXEL_DEG = 0.00008983111749910169
MOSAIC_YEARS = (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)
INNER_CHUNK_PX = 256


@dataclass(frozen=True)
class SampleBatch:
    lat: np.ndarray  # [N] float64
    lon: np.ndarray  # [N] float64
    years: np.ndarray  # [Y] int16
    targets: np.ndarray  # [N, Y, 64] int8


def open_mosaic(url: str = AEF_MOSAIC_URL, anon: bool = True, concurrency: int = 16) -> Any:
    """Open the mosaic read-only with the requested Zarr fetch concurrency."""
    import zarr
    import zarr.storage

    zarr.config.set({"async.concurrency": concurrency})
    storage_options = {
        "anon": anon,
        "client_kwargs": {"region_name": "us-west-2"},
        # Path-style addressing supports bucket names containing dots.
        "config_kwargs": {"s3": {"addressing_style": "path"}},
    }
    store = zarr.storage.FsspecStore.from_url(url, storage_options=storage_options)
    return zarr.open_group(store, mode="r")["embeddings"]


def pixel_to_lonlat(rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = LON_ORIGIN + (cols + 0.5) * PIXEL_DEG
    lat = LAT_ORIGIN - (rows + 0.5) * PIXEL_DEG
    return lon, lat


def lonlat_to_pixel(
    lat: float | np.ndarray, lon: float | np.ndarray
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Inverse of ``pixel_to_lonlat`` (returns fractional row, col); scalar or array."""
    col = (lon - LON_ORIGIN) / PIXEL_DEG - 0.5
    row = (LAT_ORIGIN - lat) / PIXEL_DEG - 0.5
    return row, col


def _block_origins(
    rng: np.random.Generator,
    n_blocks: int,
    height: int,
    width: int,
    block_px: int,
    snap_to_chunk: bool,
    bboxes: np.ndarray | None,
) -> list[tuple[int, int]]:
    """Choose block origins, optionally within WGS84 [west, south, east, north] boxes."""
    origins: list[tuple[int, int]] = []
    for _ in range(n_blocks):
        if bboxes is not None:
            west, south, east, north = bboxes[int(rng.integers(0, len(bboxes)))]
            row, col = lonlat_to_pixel(rng.uniform(south, north), rng.uniform(west, east))
            y0 = min(max(int(row), 0), height - block_px)
            x0 = min(max(int(col), 0), width - block_px)
        else:
            y0 = int(rng.integers(0, height - block_px))
            x0 = int(rng.integers(0, width - block_px))
        if snap_to_chunk:
            y0 = (y0 // INNER_CHUNK_PX) * INNER_CHUNK_PX
            x0 = (x0 // INNER_CHUNK_PX) * INNER_CHUNK_PX
        origins.append((y0, x0))
    return origins


def sample_blocks(
    arr: Any,
    rng: np.random.Generator,
    n_blocks: int = 8,
    block_px: int = INNER_CHUNK_PX,
    pixels_per_block: int = 64,
    snap_to_chunk: bool = True,
    bboxes: np.ndarray | None = None,
) -> SampleBatch:
    """Sample pixels with valid vectors in every year.

    ``bboxes`` is [M, 4] in WGS84 west/south/east/north order. When omitted,
    block origins are sampled uniformly across the array.
    """
    _, _, height, width = arr.shape
    lat_parts: list[np.ndarray] = []
    lon_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []

    for y0, x0 in _block_origins(rng, n_blocks, height, width, block_px, snap_to_chunk, bboxes):
        block = np.asarray(arr[:, :, y0 : y0 + block_px, x0 : x0 + block_px])  # [T,B,bp,bp]
        valid = np.all(block != NODATA, axis=(0, 1))  # [bp, bp]
        rows, cols = np.nonzero(valid)
        if rows.size == 0:
            continue
        take = min(pixels_per_block, rows.size)
        pick = rng.choice(rows.size, size=take, replace=False)
        local_r, local_c = rows[pick], cols[pick]
        vecs = block[:, :, local_r, local_c].transpose(2, 0, 1).copy()  # [take, T, B] int8
        lon, lat = pixel_to_lonlat(y0 + local_r, x0 + local_c)
        target_parts.append(vecs)
        lat_parts.append(lat)
        lon_parts.append(lon)

    years = np.asarray(MOSAIC_YEARS, np.int16)
    if not target_parts:
        return SampleBatch(
            np.empty(0), np.empty(0), years, np.empty((0, len(MOSAIC_YEARS), N_BANDS), np.int8)
        )
    return SampleBatch(
        lat=np.concatenate(lat_parts),
        lon=np.concatenate(lon_parts),
        years=years,
        targets=np.concatenate(target_parts, axis=0),
    )
