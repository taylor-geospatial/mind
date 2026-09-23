"""Build the globe's local height texture from public Mapzen Terrain Tiles.

Run with ``uv run python scripts/data/build_globe_dem.py``. Source credits and
the texture encoding are recorded in ``docs/assets/terrain-attribution.txt``.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import rasterio
from PIL import Image
from rasterio.merge import merge
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

TILES = "https://elevation-tiles-prod.s3.amazonaws.com/geotiff/2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/mind-terrain-tiles"))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/globe-dem.png"))
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    def fetch(tile) -> Path:
        x, y = tile
        path = args.cache / f"2-{x}-{y}.tif"
        if not path.exists():
            with urlopen(f"{TILES}/{x}/{y}.tif", timeout=30) as response:
                path.write_bytes(response.read())
        return path

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(fetch, [(x, y) for x in range(4) for y in range(4)]))
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(path)) for path in paths]
        mosaic, transform = merge(sources)
        heights = np.zeros((1024, 2048), dtype=np.float32)
        reproject(
            source=mosaic[0],
            destination=heights,
            src_transform=transform,
            src_crs=sources[0].crs,
            src_nodata=sources[0].nodata,
            dst_transform=from_bounds(-180, -90, 180, 90, 2048, 1024),
            dst_crs="EPSG:4326",
            dst_nodata=0,
            resampling=Resampling.bilinear,
        )
    pixels = np.rint(np.clip(heights, 0, 9000) * (255 / 9000)).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(args.output, optimize=True)
    print(f"Saved {args.output}: {args.output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
