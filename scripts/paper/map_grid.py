"""Reconstruct the paper's map coordinates and WorldClim land masks."""

import os
from pathlib import Path

import numpy as np
import rasterio


def map_grid(level: int, bounds: tuple | None = None) -> tuple:
    """Coarsen the 0.01-degree WorldClim mask, retaining cells with any valid child.

    ``bounds`` is (west, south, east, north), selecting coarse cell centers.
    """
    from mind.eval.benchmarks import CACHE, load_worldclim

    raster = Path(os.environ.get("MIND_LAND_RASTER", CACHE / "worldclim10m/wc2.1_10m_bio_1.tif"))
    if not raster.exists() and "MIND_LAND_RASTER" not in os.environ:
        load_worldclim(bio_vars=(1,), source="live")
    with rasterio.open(raster) as src:
        values = src.read(1)
        valid = np.isfinite(values) & (values != src.nodata)
        transform = src.transform

    factor = 2**level
    fine_lat = np.arange(90 - 0.005, -90, -0.01)
    fine_lon = np.arange(-180 + 0.005, 180, 0.01)
    lat, lon = fine_lat, fine_lon
    for _ in range(level):
        lat = lat[: len(lat) // 2 * 2].reshape(-1, 2).mean(1)
        lon = lon[: len(lon) // 2 * 2].reshape(-1, 2).mean(1)
    ri, ci = np.arange(len(lat)), np.arange(len(lon))
    if bounds is not None:
        west, south, east, north = bounds
        ri = ri[(lat >= south) & (lat <= north)]
        ci = ci[(lon >= west) & (lon <= east)]
    columns = ci[:, None] * factor + np.arange(factor)
    source_cols = np.clip(
        ((fine_lon[columns.ravel()] - transform.c) / transform.a).astype(int),
        0,
        valid.shape[1] - 1,
    )
    land = np.empty((len(ri), len(ci)), dtype=bool)
    for i, row in enumerate(ri):
        rows = fine_lat[row * factor : (row + 1) * factor]
        source_rows = np.clip(
            ((rows - transform.f) / transform.e).astype(int), 0, valid.shape[0] - 1
        )
        children = valid[np.ix_(source_rows, source_cols)]
        land[i] = children.reshape(factor, len(ci), factor).any(axis=(0, 2))
    return land, lat[ri], lon[ci]
