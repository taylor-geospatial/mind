"""Convert local raw source files to data/<config>/data.parquet."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from mind.eval.benchmarks import DEEPMIND_EVAL_CONFIGS, _random_land_points, _sample_raster
from scripts.data.coordbench import dm_eval
from scripts.data.coordbench.schema import build_canonical, validate_canonical

ROOT = Path(os.environ.get("COORDBENCH_BUILD_ROOT", "data/coordbench"))
RAW = ROOT / "raw"
DATA = ROOT / "data"


def _write(config: str, df: pd.DataFrame) -> None:
    validate_canonical(df, config)
    out = DATA / config / "data.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  {config}: {len(df)} rows -> {out}")


def normalize_pdfm_conus27() -> None:
    df = pd.read_csv(RAW / "pdfm_conus27" / "conus27.csv")
    _write("pdfm_conus27", build_canonical(df, lon_col="longitude", lat_col="latitude"))


def normalize_air_temp() -> None:
    df = pd.read_csv(RAW / "air_temp" / "stationDataAll.csv")
    cols = {c.lower(): c for c in df.columns}
    lat_col = next(cols[k] for k in cols if k.startswith("lat"))
    lon_col = next(cols[k] for k in cols if k.startswith("lon"))
    _write("air_temp", build_canonical(df, lon_col=lon_col, lat_col=lat_col))


def normalize_california_housing() -> None:
    df = pd.read_csv(RAW / "california_housing" / "california_housing.csv")
    _write("california_housing", build_canonical(df, lon_col="Longitude", lat_col="Latitude"))


def normalize_satclip_official() -> None:
    d = RAW / "satclip_official"
    c = pd.read_csv(d / "lat_lon_country.csv")
    _write("satclip_country", build_canonical(c, lon_col="lon", lat_col="lat"))

    e = pd.read_csv(d / "lat_lon_ecoregion.csv")
    _write("satclip_ecoregion", build_canonical(e, lon_col="lon", lat_col="lat"))

    p = pd.read_csv(d / "labels_population.csv")
    _write("satclip_population", build_canonical(p, lon_col="Lon", lat_col="Lat"))

    el = pd.read_csv(d / "labels_elevation.csv")
    _write("satclip_elevation", build_canonical(el, lon_col="Lon", lat_col="Lat"))


def normalize_usavars() -> None:
    """Join income/housing coordinates by USAVars ID, dropping unmatched rows."""
    labels = ("treecover", "elevation", "population", "nightlights", "income", "roads", "housing")
    frames = {}
    for label in labels:
        df = pd.read_csv(RAW / "usavars" / f"usavars_{label}.csv")
        df.columns = [str(c).strip() for c in df.columns]
        frames[label] = df

    id_maps = []
    for df in frames.values():
        cl = {c.lower(): c for c in df.columns}
        if {"id", "lat", "lon"} <= set(cl):
            m = df[[cl["id"], cl["lat"], cl["lon"]]].copy()
            m.columns = ["ID", "lat", "lon"]
            id_maps.append(m)
    id2ll = pd.concat(id_maps).dropna().drop_duplicates("ID").set_index("ID")

    for label, df in frames.items():
        cl = {c.lower(): c for c in df.columns}
        if {"lat", "lon"} <= set(cl):
            out = build_canonical(df, lon_col=cl["lon"], lat_col=cl["lat"], id_col=cl.get("id"))
        else:
            n_before = len(df)
            joined = df.merge(id2ll, left_on=cl["id"], right_index=True, how="inner")
            if len(joined) < n_before:
                print(f"  usavars_{label}: dropped {n_before - len(joined)} rows with no ID match")
            out = build_canonical(joined, lon_col="lon", lat_col="lat", id_col=cl["id"])
        _write(f"usavars_{label}", out)


def _worldclim_points() -> pd.DataFrame:
    """Recreate the seed-0 land sample from the downloaded WorldClim rasters."""
    import rasterio

    archive = (RAW / "worldclim" / "wc2.1_10m_bio.zip").resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"Missing {archive}; run fetch_raw --only worldclim first")
    lat, lon = _random_land_points(np.random.default_rng(0), 60000)
    cols = {"lat": lat, "lon": lon}
    for var in (1, 12):
        with rasterio.open(f"/vsizip/{archive}/wc2.1_10m_bio_{var}.tif") as src:
            cols[f"bio{var}"] = _sample_raster(src, lat, lon)
    df = pd.DataFrame(cols).dropna()
    return df.sample(20000, random_state=0) if len(df) > 20000 else df


def normalize_worldclim() -> None:
    df = _worldclim_points()
    _write("worldclim_bio", build_canonical(df, lon_col="lon", lat_col="lat"))


def normalize_soilgrids() -> None:
    import rasterio
    from rasterio.transform import rowcol
    from rasterio.warp import transform

    land = _worldclim_points()
    land = land.sample(min(6000, len(land)), random_state=0)
    lat, lon = land["lat"].to_numpy(), land["lon"].to_numpy()
    cols = {"lat": lat, "lon": lon}
    for prop in ("soc", "phh2o"):
        with rasterio.open(RAW / "soilgrids" / f"soilgrids_overview_{prop}.tif") as src:
            xs, ys = transform("EPSG:4326", src.crs, lon.tolist(), lat.tolist())
            rows, columns = rowcol(src.transform, xs, ys)
            rows = np.clip(np.asarray(rows), 0, src.height - 1)
            columns = np.clip(np.asarray(columns), 0, src.width - 1)
            values = src.read(1)[rows, columns].astype(np.float64)
            if src.nodata is not None:
                values[values == src.nodata] = np.nan
        values[values <= 0] = np.nan
        cols[prop] = values
    _write("soilgrids", build_canonical(pd.DataFrame(cols).dropna(), lon_col="lon", lat_col="lat"))


def normalize_sustainbench() -> None:
    d = RAW / "sustainbench"
    tr = pd.read_csv(d / "dhs_trainval_labels.csv")
    te = pd.read_csv(d / "dhs_test_labels.csv")
    tr_out = build_canonical(
        tr,
        lon_col="lon",
        lat_col="lat",
        timestamp_col="year",
        timestamp_unit="year",
        split_value="trainval",
        id_col="DHSID_EA",
    )
    te_out = build_canonical(
        te,
        lon_col="lon",
        lat_col="lat",
        timestamp_col="year",
        timestamp_unit="year",
        split_value="test",
        id_col="DHSID_EA",
    )
    _write("sustainbench", pd.concat([tr_out, te_out], ignore_index=True))


def normalize_cdc_places() -> None:
    df = pd.read_csv(RAW / "cdc_places" / "cdc_places_zcta_2023.csv")
    df.columns = [c.lower() for c in df.columns]
    geo = df["geolocation"].str.extract(r"POINT \(([-0-9.]+) ([-0-9.]+)\)").astype(float)
    df["lon"], df["lat"] = geo[0], geo[1]
    _write("cdc_places", build_canonical(df, lon_col="lon", lat_col="lat"))


def normalize_deepmind_eval() -> None:
    raw_dir = RAW / "deepmind_eval" / "evaluation"
    csvs = {csv.stem: csv for csv in dm_eval.discover(raw_dir)}
    missing = set(DEEPMIND_EVAL_CONFIGS) - csvs.keys()
    if missing:
        raise FileNotFoundError(
            f"Missing DeepMind evaluation CSVs under {raw_dir}: {sorted(missing)}; run fetch_raw first"
        )
    for stem in DEEPMIND_EVAL_CONFIGS:
        csv = csvs[stem]
        df = pd.read_csv(csv)
        if not {"label", "x", "y"}.issubset(df.columns):
            raise ValueError(f"{csv}: expected label/x/y columns")
        kwargs = {"lon_col": "x", "lat_col": "y"}
        if "valid_time_start_ms" in df.columns:
            kwargs["timestamp_col"] = "valid_time_start_ms"
        if "valid_time_end_ms" in df.columns:
            kwargs["timestamp_end_col"] = "valid_time_end_ms"
        if "split" in df.columns:
            kwargs["split_col"] = "split"
        _write(f"dm_{csv.stem}", build_canonical(df, **kwargs))


NORMALIZERS = {
    "pdfm_conus27": normalize_pdfm_conus27,
    "air_temp": normalize_air_temp,
    "california_housing": normalize_california_housing,
    "satclip_official": normalize_satclip_official,
    "usavars": normalize_usavars,
    "worldclim": normalize_worldclim,
    "soilgrids": normalize_soilgrids,
    "sustainbench": normalize_sustainbench,
    "cdc_places": normalize_cdc_places,
    "deepmind_eval": normalize_deepmind_eval,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None, choices=list(NORMALIZERS))
    args = ap.parse_args()
    for family in args.only or list(NORMALIZERS):
        print(f"[{family}]")
        NORMALIZERS[family]()


if __name__ == "__main__":
    main()
