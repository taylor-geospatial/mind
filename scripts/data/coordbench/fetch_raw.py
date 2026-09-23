"""Fetch CoordBench source tables into COORDBENCH_BUILD_ROOT/raw.

SatCLIP's official CSVs require a manual Google Drive download; pass --satclip-dir.
"""

import argparse
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

from mind.eval.benchmarks import (
    AIR_TEMP_URL,
    CDC_PLACES_URL,
    DATA_ROOT,
    DEEPMIND_EVAL_CONFIGS,
    PDFM_CONUS27_URL,
    SOILGRIDS_COG,
    USAVARS_BASE,
    USAVARS_LABELS,
    WORLDCLIM_BIO_URL,
)

ROOT = Path(os.environ.get("COORDBENCH_BUILD_ROOT", "data/coordbench"))
RAW = ROOT / "raw"
SATCLIP_URL = "https://drive.google.com/drive/folders/1tI2qo6iioRrv3P1OxSXwHKObpLCrinad"


def _copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    if not src.is_file():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  copied {src} -> {dst}")


def _download(url: str, dst: Path) -> None:
    if dst.exists():
        print(f"  skip (exists): {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} -> {dst}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dst)


def _extract_files(archive: Path, names: tuple[str, ...], dst: Path) -> None:
    """Extract requested files by basename."""
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in names:
            matches = [p for p in z.namelist() if Path(p).name == name]
            if len(matches) != 1:
                raise ValueError(f"Expected one {name} in {archive}, found {len(matches)}")
            target = dst / name
            if target.exists():
                continue
            tmp = target.with_suffix(target.suffix + ".part")
            with z.open(matches[0]) as source, tmp.open("wb") as out:
                shutil.copyfileobj(source, out)
            tmp.replace(target)


def fetch_pdfm_conus27() -> None:
    _download(PDFM_CONUS27_URL, RAW / "pdfm_conus27" / "conus27.csv")


def fetch_air_temp() -> None:
    _download(AIR_TEMP_URL, RAW / "air_temp" / "stationDataAll.csv")


def fetch_california_housing() -> None:
    from sklearn.datasets import fetch_california_housing

    dst = RAW / "california_housing" / "california_housing.csv"
    if dst.exists():
        print(f"  skip (exists): {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    fetch_california_housing(as_frame=True).frame.to_csv(dst, index=False)
    print(f"  wrote {dst}")


def fetch_satclip_official(src_dir: Path = Path(DATA_ROOT) / "satclip_official") -> None:
    names = (
        "lat_lon_country.csv",
        "lat_lon_ecoregion.csv",
        "labels_population.csv",
        "labels_elevation.csv",
    )
    missing = [
        n
        for n in names
        if not (RAW / "satclip_official" / n).exists() and not (src_dir / n).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Download SatCLIP's official CSVs from {SATCLIP_URL} and pass --satclip-dir PATH. "
            f"Missing in {src_dir}: {', '.join(missing)}"
        )
    for name in names:
        _copy(src_dir / name, RAW / "satclip_official" / name)


def fetch_usavars() -> None:
    for label in USAVARS_LABELS:
        _download(USAVARS_BASE.format(f"{label}.csv"), RAW / "usavars" / f"usavars_{label}.csv")


def fetch_worldclim() -> None:
    _download(WORLDCLIM_BIO_URL, RAW / "worldclim" / "wc2.1_10m_bio.zip")


def fetch_soilgrids() -> None:
    """Fetch the 4000x2000 SoilGrids overviews used by the benchmark."""
    import rasterio
    from rasterio.transform import Affine

    for prop in ("soc", "phh2o"):
        dst = RAW / "soilgrids" / f"soilgrids_overview_{prop}.tif"
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = "/vsicurl/" + SOILGRIDS_COG.format(prop=prop, depth="0-5cm")
        with rasterio.open(url) as src:
            out_w, out_h = 4000, 2000
            arr = src.read(1, out_shape=(out_h, out_w))
            transform = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
            profile = src.profile.copy()
            profile.update(driver="GTiff", width=out_w, height=out_h, transform=transform, count=1)
        tmp = dst.with_suffix(".part.tif")
        with rasterio.open(tmp, "w", **profile) as dst_ds:
            dst_ds.write(arr, 1)
        tmp.replace(dst)
        print(f"  wrote {dst}")


def fetch_sustainbench() -> None:
    names = ("dhs_trainval_labels.csv", "dhs_test_labels.csv")
    dst = RAW / "sustainbench"
    if all((dst / name).exists() for name in names):
        return
    src = Path(DATA_ROOT) / "sustainbench"
    if all((src / name).is_file() for name in names):
        for name in names:
            _copy(src / name, dst / name)
        return
    with urllib.request.urlopen(
        "https://api.figshare.com/v2/articles/26026798", timeout=120
    ) as response:
        metadata = json.load(response)
    source = next(f for f in metadata["files"] if f["name"] == "sustainbench_dhs_labels.zip")
    archive = dst / "sustainbench_dhs_labels.zip"
    _download(source["download_url"], archive)
    _extract_files(archive, names, dst)


def fetch_cdc_places() -> None:
    _download(CDC_PLACES_URL, RAW / "cdc_places" / "cdc_places_zcta_2023.csv")


def fetch_deepmind_eval() -> None:
    names = tuple(f"{stem}.csv" for stem in DEEPMIND_EVAL_CONFIGS)
    dst = RAW / "deepmind_eval" / "evaluation"
    if all((dst / name).is_file() for name in names):
        return
    archive = RAW / "deepmind_eval" / "evaluation.zip"
    _download("https://zenodo.org/api/records/16585402/files/evaluation.zip/content", archive)
    _extract_files(archive, names, dst)


FETCHERS = {
    "pdfm_conus27": fetch_pdfm_conus27,
    "air_temp": fetch_air_temp,
    "california_housing": fetch_california_housing,
    "satclip_official": fetch_satclip_official,
    "usavars": fetch_usavars,
    "worldclim": fetch_worldclim,
    "soilgrids": fetch_soilgrids,
    "sustainbench": fetch_sustainbench,
    "cdc_places": fetch_cdc_places,
    "deepmind_eval": fetch_deepmind_eval,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="+", choices=list(FETCHERS), help="restrict to these families")
    ap.add_argument("--satclip-dir", type=Path, default=Path(DATA_ROOT) / "satclip_official")
    args = ap.parse_args()
    for family in args.only or list(FETCHERS):
        print(f"[{family}]")
        if family == "satclip_official":
            fetch_satclip_official(args.satclip_dir)
        else:
            FETCHERS[family]()


if __name__ == "__main__":
    main()
