"""Sample MINDSET coordinates near cities and store annual AEF pixels as int8."""

import argparse
import io
import logging
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

import numpy as np

from mind.data.dataset import write_manifest, write_shard
from mind.data.zarr_aef import MOSAIC_YEARS, open_mosaic, sample_blocks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_aef_urban")

CITIES_URL = "https://download.geonames.org/export/dump/cities15000.zip"


def load_cities(cache: str) -> np.ndarray:
    """Load GeoNames cities as float64 (latitude, longitude, population) rows."""
    path = Path(cache)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading %s ...", CITIES_URL)
        raw = urlopen(CITIES_URL, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            path.write_bytes(zf.read("cities15000.txt"))
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        f = line.split("\t")  # geonames dump: 4=lat, 5=lon, 14=population
        rows.append((float(f[4]), float(f[5]), float(f[14])))
    out = np.asarray(rows, dtype=np.float64)
    out = out[out[:, 2] >= 15_000]
    logger.info("loaded %d cities (pop>=15k)", len(out))
    return out


def city_bboxes(cities: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample cities by sqrt(population); return WGS84 (west, south, east, north) bounds."""
    w = np.sqrt(cities[:, 2])
    pick = rng.choice(len(cities), size=n, p=w / w.sum())
    lat, lon, pop = cities[pick, 0], cities[pick, 1], cities[pick, 2]
    r_km = np.clip(2.0 * np.sqrt(pop / 1e5), 2.0, 25.0)
    dlat = r_km / 111.0
    dlon = r_km / (111.0 * np.clip(np.cos(np.radians(lat)), 0.2, 1.0))
    return np.stack([lon - dlon, lat - dlat, lon + dlon, lat + dlat], axis=1)


def _sample_chunk(
    seed: int, n_blocks: int, block_px: int, pixels_per_block: int, bboxes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    arr = open_mosaic()
    rng = np.random.default_rng(seed)
    sb = sample_blocks(arr, rng, n_blocks, block_px, pixels_per_block, bboxes=bboxes)
    coords = np.stack([sb.lat, sb.lon], axis=1).astype(np.float32)
    return coords, sb.targets


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n-blocks", type=int, default=16000, help="urban block locations to sample")
    p.add_argument("--blocks-per-task", type=int, default=16)
    p.add_argument("--block-px", type=int, default=256)
    p.add_argument(
        "--pixels-per-block",
        type=int,
        default=768,
        help="dense in-block sampling (~90m spacing at 768/256px): close pairs = hard negatives",
    )
    p.add_argument("--shard-size", type=int, default=1_000_000)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cities-cache", default="data/cities15000.txt")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    cities = load_cities(args.cities_cache)

    n_tasks = (args.n_blocks + args.blocks_per_task - 1) // args.blocks_per_task
    logger.info("sampling %d urban blocks over %d tasks", args.n_blocks, n_tasks)

    coord_buf: list[np.ndarray] = []
    target_buf: list[np.ndarray] = []
    shard_names: list[str] = []
    total = 0
    buffered = 0

    def flush() -> None:
        nonlocal buffered
        if not coord_buf:
            return
        name = f"shard_{len(shard_names):05d}"
        write_shard(args.out, name, np.concatenate(coord_buf), np.concatenate(target_buf))
        shard_names.append(name)
        logger.info("wrote %s (%d samples)", name, buffered)
        coord_buf.clear()
        target_buf.clear()
        buffered = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _sample_chunk,
                args.seed + 1 + i,
                args.blocks_per_task,
                args.block_px,
                args.pixels_per_block,
                city_bboxes(cities, args.blocks_per_task, rng),
            )
            for i in range(n_tasks)
        ]
        for done, fut in enumerate(as_completed(futures), 1):
            coords, targets = fut.result()
            if coords.shape[0] == 0:
                continue
            coord_buf.append(coords)
            target_buf.append(targets)
            buffered += coords.shape[0]
            total += coords.shape[0]
            if buffered >= args.shard_size:
                flush()
            if done % 20 == 0:
                logger.info("%d/%d tasks, %d samples", done, n_tasks, total)

    flush()
    if total == 0:
        raise RuntimeError(
            "no urban samples produced across all tasks -- check city bboxes / mosaic access; "
            "refusing to write an empty manifest"
        )
    write_manifest(args.out, MOSAIC_YEARS, shard_names, total, dequant="signed_square")
    logger.info("DONE: %d urban samples across %d shards", total, len(shard_names))


if __name__ == "__main__":
    main()
