"""Check that normalized tables contain the source coordinates and label values."""

import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from mind.eval.benchmarks import DEEPMIND_EVAL_CONFIGS

ROOT = Path(os.environ.get("COORDBENCH_BUILD_ROOT", "data/coordbench"))
RAW = ROOT / "raw"
DATA = ROOT / "data"


def _load(config: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / config / "data.parquet")


def _check(
    config: str, norm: pd.DataFrame, lon: np.ndarray, lat: np.ndarray, col: str, vals, atol=1e-6
) -> None:
    """Match each source value to a distinct normalized value at the same coordinate.

    Extra normalized rows are allowed; duplicate coordinates use multiset matching.
    """
    bench_df = pd.DataFrame({"_lon": np.round(lon, 6), "_lat": np.round(lat, 6), "_bench": vals})
    n = norm.copy()
    n["_lon"] = np.round(n["lon"].to_numpy(np.float64), 6)
    n["_lat"] = np.round(n["lat"].to_numpy(np.float64), 6)

    bench_groups = bench_df.groupby(["_lon", "_lat"])["_bench"].apply(list)
    norm_groups = n.groupby(["_lon", "_lat"])[col].apply(list)

    missing_keys = bench_groups.index.difference(norm_groups.index)
    if len(missing_keys):
        raise AssertionError(
            f"{config}.{col}: {len(missing_keys)}/{len(bench_groups)} benchmark coordinates "
            "not found in normalized table"
        )

    is_numeric = pd.api.types.is_numeric_dtype(bench_df["_bench"])
    bad = 0
    for key, bvals in bench_groups.items():
        nvals = norm_groups.loc[key]
        if is_numeric:
            remaining = list(np.asarray(nvals, dtype=np.float64))
            ok = True
            for bv in bvals:
                hit = next(
                    (
                        i
                        for i, nv in enumerate(remaining)
                        if np.isclose(nv, bv, atol=atol, equal_nan=True)
                    ),
                    None,
                )
                if hit is None:
                    ok = False
                    break
                remaining.pop(hit)
        else:
            ok = Counter(map(str, bvals)) <= Counter(map(str, nvals))
        bad += not ok
    if bad:
        raise AssertionError(
            f"{config}.{col}: {bad}/{len(bench_groups)} coordinates have a value not present "
            "in normalized table"
        )
    print(f"  OK {config}.{col} ({len(bench_groups)} distinct coordinates, {len(bench_df)} rows)")


def validate_pdfm_conus27() -> None:
    from mind.eval.benchmarks import load_pdfm

    norm = _load("pdfm_conus27")
    b = load_pdfm(csv_path=str(RAW / "pdfm_conus27" / "conus27.csv"))
    for col, vals in b.tasks.items():
        _check("pdfm_conus27", norm, b.lon, b.lat, col, vals)


def validate_air_temp() -> None:
    raw = pd.read_csv(RAW / "air_temp" / "stationDataAll.csv")
    names = {c.lower(): c for c in raw.columns}
    lon = names[next(c for c in names if c.startswith("lon"))]
    lat = names[next(c for c in names if c.startswith("lat"))]
    col = names["meant"]
    _check("air_temp", _load("air_temp"), raw[lon].to_numpy(), raw[lat].to_numpy(), col, raw[col])


def validate_california_housing() -> None:
    raw = pd.read_csv(RAW / "california_housing" / "california_housing.csv")
    _check(
        "california_housing",
        _load("california_housing"),
        raw["Longitude"].to_numpy(),
        raw["Latitude"].to_numpy(),
        "MedHouseVal",
        raw["MedHouseVal"],
    )


def validate_satclip_official() -> None:
    from mind.eval.benchmarks import load_satclip_official

    norms = {
        "satclip-country": ("satclip_country", "country"),
        "satclip-ecoregion": ("satclip_ecoregion", "ecoregion_class_index"),
        "satclip-biome": ("satclip_ecoregion", "biome_class_index"),
        "satclip-population": ("satclip_population", "log_population"),
        "satclip-elevation": ("satclip_elevation", "elevation"),
    }
    for b in load_satclip_official(data_dir=str(RAW / "satclip_official")):
        config, col = norms[b.name]
        norm = _load(config)
        _check(config, norm, b.lon, b.lat, col, next(iter(b.tasks.values())))


def _usavars_value_col(label: str, raw_csv: Path) -> str:
    """Select the source value column using the benchmark loader's rules."""
    df = pd.read_csv(raw_csv, nrows=5)
    skip = {"id", "lat", "lon"}
    cands = [
        c
        for c in df.columns
        if c.lower() not in skip
        and not c.startswith("Unnamed")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    value_col = next((c for c in cands if label in c.lower()), None)
    return value_col or next((c for c in cands if c.lower() == "price"), None) or cands[0]


def validate_usavars() -> None:
    from mind.eval.benchmarks import USAVARS_LABELS

    frames = {}
    coords = []
    for label in USAVARS_LABELS:
        raw = pd.read_csv(RAW / "usavars" / f"usavars_{label}.csv")
        raw.columns = [str(c).strip() for c in raw.columns]
        frames[label] = raw
        names = {c.lower(): c for c in raw.columns}
        if {"id", "lat", "lon"} <= names.keys():
            xyz = raw[[names["id"], names["lat"], names["lon"]]].copy()
            xyz.columns = ["ID", "lat", "lon"]
            coords.append(xyz)
    locations = pd.concat(coords).dropna().drop_duplicates("ID").set_index("ID")
    for label, raw in frames.items():
        names = {c.lower(): c for c in raw.columns}
        if not {"lat", "lon"} <= names.keys():
            raw = raw.merge(locations, left_on=names["id"], right_index=True, how="inner")
            names = {c.lower(): c for c in raw.columns}
        col = _usavars_value_col(label, RAW / "usavars" / f"usavars_{label}.csv")
        _check(
            f"usavars_{label}",
            _load(f"usavars_{label}"),
            raw[names["lon"]].to_numpy(),
            raw[names["lat"]].to_numpy(),
            col,
            raw[col],
        )


def validate_sustainbench() -> None:
    from mind.eval.benchmarks import load_sustainbench

    norm = _load("sustainbench")
    for b in load_sustainbench(data_dir=str(RAW / "sustainbench")):
        (col, vals) = next(iter(b.tasks.items()))
        _check("sustainbench", norm, b.lon, b.lat, col, vals)


def validate_cdc_places() -> None:
    from mind.eval.benchmarks import CDC_PLACES_MEASURES

    raw = pd.read_csv(RAW / "cdc_places" / "cdc_places_zcta_2023.csv")
    raw.columns = [c.lower() for c in raw.columns]
    geo = raw["geolocation"].str.extract(r"POINT \(([-0-9.]+) ([-0-9.]+)\)").astype(float)
    norm = _load("cdc_places")
    for prefix in CDC_PLACES_MEASURES.values():
        col = f"{prefix.lower()}_crudeprev"
        _check("cdc_places", norm, geo[0].to_numpy(), geo[1].to_numpy(), col, raw[col])


def validate_worldclim() -> None:
    from mind.eval.benchmarks import load_worldclim

    norm = _load("worldclim_bio")
    for b in load_worldclim(bio_vars=(1, 12), source="live"):
        (col, vals) = next(iter(b.tasks.items()))
        _check("worldclim_bio", norm, b.lon, b.lat, col, vals)


def validate_soilgrids() -> None:
    from mind.eval.benchmarks import load_soilgrids

    norm = _load("soilgrids")
    for b in load_soilgrids(source="live"):
        (col, vals) = next(iter(b.tasks.items()))
        _check("soilgrids", norm, b.lon, b.lat, col, vals)


def validate_deepmind_eval() -> None:
    from mind.eval.benchmarks import load_deepmind_eval

    for b in load_deepmind_eval(
        root=str(RAW / "deepmind_eval" / "evaluation"), names=set(DEEPMIND_EVAL_CONFIGS)
    ):
        config = f"dm_{b.name.removeprefix('dm-')}"
        norm = _load(config)
        (_, vals) = next(iter(b.tasks.items()))
        _check(config, norm, b.lon, b.lat, "label", vals)


VALIDATORS = {
    "pdfm_conus27": validate_pdfm_conus27,
    "air_temp": validate_air_temp,
    "california_housing": validate_california_housing,
    "satclip_official": validate_satclip_official,
    "usavars": validate_usavars,
    "worldclim": validate_worldclim,
    "soilgrids": validate_soilgrids,
    "sustainbench": validate_sustainbench,
    "cdc_places": validate_cdc_places,
    "deepmind_eval": validate_deepmind_eval,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None, choices=list(VALIDATORS))
    args = ap.parse_args()
    failures = []
    for family in args.only or list(VALIDATORS):
        print(f"[{family}]")
        try:
            VALIDATORS[family]()
        except (AssertionError, KeyError, ValueError, OSError) as e:
            print(f"  FAILED: {e}")
            failures.append(family)
    if failures:
        raise SystemExit(f"validation failed for: {failures}")
    print("all configs validated OK")


if __name__ == "__main__":
    main()
