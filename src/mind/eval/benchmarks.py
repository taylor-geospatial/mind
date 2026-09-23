"""CoordBench loaders returning coordinates, task labels, and optional splits.

Defaults download ``taylor-geospatial/CoordBench`` from Hugging Face. Set
``COORDBENCH_ROOT`` to a checkout containing ``data/<config>/data.parquet``.
Loader-specific path arguments read the original source files instead.
"""

import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path.home() / ".cache" / "mind" / "benchmarks"
DATA_ROOT = os.environ.get("MIND_DATA_ROOT", "data")

COORDBENCH_REPO = "taylor-geospatial/CoordBench"
# canonical (non-task) schema columns, excluded when scanning for task columns
_CANONICAL_EXTRA = {"timestamp", "timestamp_end", "split", "id"}
DEEPMIND_EVAL_CONFIGS = (
    "africa_crop_mask",
    "aster_ged",
    "canada_crops_coarse",
    "canada_crops_fine",
    "descals",
    "ethiopia_crops",
    "glance",
    "lcmap_lc",
    "lcmap_lcc",
    "lcmap_lu",
    "lcmap_luc",
    "lucas_lc",
    "lucas_lu",
    "openet_ensemble",
    "us_trees",
)


def _load_coordbench(config: str) -> pd.DataFrame:
    """Read one canonical table from a local CoordBench checkout or the Hub."""
    root = os.environ.get("COORDBENCH_ROOT")
    if root is not None:
        path = Path(root).expanduser() / "data" / config / "data.parquet"
        if not path.is_file():
            raise FileNotFoundError(
                f"COORDBENCH_ROOT is set, but the local table is missing: {path}. "
                "Download this table or unset COORDBENCH_ROOT to use the Hub."
            )
        return pd.read_parquet(path)

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(COORDBENCH_REPO, f"data/{config}/data.parquet", repo_type="dataset")
    return pd.read_parquet(path)


def _lonlat_cols(df: pd.DataFrame) -> tuple[str, str]:
    """Find the lon/lat columns (canonical in CoordBench, source-named in raw overrides)."""
    lon_col = next(c for c in ("lon", "longitude", "Lon", "x") if c in df.columns)
    lat_col = next(c for c in ("lat", "latitude", "Lat", "y") if c in df.columns)
    return lon_col, lat_col


def _first_col(df: pd.DataFrame, *names: str) -> str | None:
    return next((n for n in names if n in df.columns), None)


WORLDCLIM_BIO_URL = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip"
# SoilGrids v2 COGs (Goode Interrupted Homolosine projection); point-sampled with reprojection.
SOILGRIDS_COG = "https://files.isric.org/soilgrids/latest/data/{prop}/{prop}_{depth}_mean.vrt"

PDFM_CONUS27_URL = (
    "https://raw.githubusercontent.com/google-research/population-dynamics/master/"
    "data/benchmarks/conus27.csv"
)
PDFM_NON_TASK = {
    "place",
    "county",
    "state",
    "latitude",
    "longitude",
    "Count_Person",
    "imputation_split",
    "superresolution_split",
    "extrapolation_split",
}
# stationDataAll.csv (Hooker et al., 2018), used by SatCLIP.
AIR_TEMP_URL = "https://api.figshare.com/v2/file/download/12609182"
USAVARS_BASE = (
    "https://hf.co/datasets/torchgeo/usavars/resolve/01377abfaf50c0cc8548aaafb79533666bbf288f/{}"
)
USAVARS_LABELS = (
    "treecover",
    "elevation",
    "population",
    "nightlights",
    "income",
    "roads",
    "housing",
)
USAVARS_NODATA = -999.0  # source nodata sentinel
# Apply log1p to population, nightlights, income, and housing targets.
USAVARS_LOG_LABELS = frozenset({"population", "income", "nightlights", "housing"})
NATURAL_EARTH_COUNTRIES = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
    "ne_110m_admin_0_countries.geojson"
)
ECOREGIONS_ZIP = "https://storage.googleapis.com/teow2016/Ecoregions2017.zip"


@dataclass
class Benchmark:
    name: str
    lat: np.ndarray
    lon: np.ndarray
    tasks: dict[str, np.ndarray] = field(default_factory=dict)
    task_type: str = "regression"  # "regression" | "classification"
    year: np.ndarray | None = None  # per-point year for the year-conditioned encoder
    test_mask: np.ndarray | None = None  # held-out test rows (else k-fold CV)


def _download(url: str, filename: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / filename
    if not path.exists():
        with urllib.request.urlopen(url) as resp:
            path.write_bytes(resp.read())
    return path


def _random_land_points(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform-on-sphere random points (lon in [-180,180], equal-area in lat)."""
    lon = rng.uniform(-180, 180, size=n)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, size=n)))
    return lat, lon


def load_pdfm(csv_path: str | None = None, split: str | None = None) -> Benchmark:
    """Load PDFM conus27. ``split`` selects an official held-out split; ``None`` uses CV."""
    df = pd.read_csv(csv_path) if csv_path else _load_coordbench("pdfm_conus27")
    lon_col, lat_col = _lonlat_cols(df)
    non_task = PDFM_NON_TASK | {lon_col, lat_col} | _CANONICAL_EXTRA
    task_cols = [c for c in df.columns if c not in non_task]
    test_mask = None
    if split is not None:
        test_mask = (df[f"{split}_split"].astype(str) == "test").to_numpy()
    return Benchmark(
        name=f"pdfm-conus27{'-' + split if split else ''}",
        lat=df[lat_col].to_numpy(np.float64),
        lon=df[lon_col].to_numpy(np.float64),
        tasks={c: df[c].to_numpy(np.float64) for c in task_cols},
        test_mask=test_mask,
    )


def load_air_temp(url: str | None = None) -> Benchmark:
    """``url`` fetches the original figshare file instead of the CoordBench source."""
    if url:
        with urllib.request.urlopen(url) as resp:
            df = pd.read_csv(io.BytesIO(resp.read()))
    else:
        df = _load_coordbench("air_temp")
    lon_col, lat_col = _lonlat_cols(df)
    cols = {c.lower(): c for c in df.columns}
    # prefer an explicit temperature column (meanT) over other mean* columns (meanP = pressure)
    temp_keys = [k for k in cols if "temp" in k or k == "meant"] or [
        k for k in cols if k.startswith("mean")
    ]
    temp_col = cols[temp_keys[0]]
    return Benchmark(
        name="satclip-air-temp",
        lat=df[lat_col].to_numpy(np.float64),
        lon=df[lon_col].to_numpy(np.float64),
        tasks={"air_temperature": df[temp_col].to_numpy(np.float64)},
    )


def load_california_housing() -> Benchmark:
    df = _load_coordbench("california_housing")
    lon_col, lat_col = _lonlat_cols(df)
    return Benchmark(
        name="california-housing",
        lat=df[lat_col].to_numpy(np.float64),
        lon=df[lon_col].to_numpy(np.float64),
        tasks={"median_house_value": df["MedHouseVal"].to_numpy(np.float64)},
    )


# SatCLIP downstream data (Klemmer et al., microsoft/satclip issue #6).
# Ocean sentinels: country 0; ecoregion and biome -1.
def load_satclip_official(data_dir: str | None = None) -> list[Benchmark]:
    """SatCLIP's released country/ecoregion/biome (classification) + population/elevation
    (regression). ``data_dir`` reads the 4 raw CSVs instead of the CoordBench source."""

    def _get(config: str, filename: str) -> pd.DataFrame:
        return pd.read_csv(f"{data_dir}/{filename}") if data_dir else _load_coordbench(config)

    out: list[Benchmark] = []

    c = _get("satclip_country", "lat_lon_country.csv")
    lon_c, lat_c = _lonlat_cols(c)
    c = c[c["country"] != 0]  # index 0 = ocean/nodata (67% of the global sample)
    out.append(
        Benchmark(
            name="satclip-country",
            lat=c[lat_c].to_numpy(np.float64),
            lon=c[lon_c].to_numpy(np.float64),
            tasks={"country": c["country"].to_numpy()},
            task_type="classification",
        )
    )

    e = _get("satclip_ecoregion", "lat_lon_ecoregion.csv")
    lon_e, lat_e = _lonlat_cols(e)
    e = e[e["ecoregion_class_index"] != -1]  # -1 = ocean/nodata
    for col, name in [
        ("ecoregion_class_index", "satclip-ecoregion"),
        ("biome_class_index", "satclip-biome"),
    ]:
        out.append(
            Benchmark(
                name=name,
                lat=e[lat_e].to_numpy(np.float64),
                lon=e[lon_e].to_numpy(np.float64),
                tasks={col: e[col].to_numpy()},
                task_type="classification",
            )
        )

    p = _get("satclip_population", "labels_population.csv")  # target is already log_population
    lon_p, lat_p = _lonlat_cols(p)
    out.append(
        Benchmark(
            name="satclip-population",
            lat=p[lat_p].to_numpy(np.float64),
            lon=p[lon_p].to_numpy(np.float64),
            tasks={"log_population": p["log_population"].to_numpy(np.float64)},
        )
    )

    el = _get("satclip_elevation", "labels_elevation.csv")
    lon_el, lat_el = _lonlat_cols(el)
    out.append(
        Benchmark(
            name="satclip-elevation",
            lat=el[lat_el].to_numpy(np.float64),
            lon=el[lon_el].to_numpy(np.float64),
            tasks={"elevation": el["elevation"].to_numpy(np.float64)},
        )
    )
    return out


# DHS indices from the TorchSpatial/LocBench label pack.
SUSTAINBENCH_TASKS = {
    "asset": "asset_index",
    "water": "water_index",
    "sanitation": "sanitation_index",
    "child_mortality": "under5_mort",
    "women_edu": "women_edu",
    "women_bmi": "women_bmi",
}


def load_sustainbench(data_dir: str | None = None) -> list[Benchmark]:
    """Load six SustainBench DHS indices with the official trainval/test split.

    ``data_dir`` reads the original label CSVs instead of CoordBench.
    """
    if data_dir:
        tr = pd.read_csv(f"{data_dir}/dhs_trainval_labels.csv")
        te = pd.read_csv(f"{data_dir}/dhs_test_labels.csv")
        tr["_is_test"] = False
        te["_is_test"] = True
        df = pd.concat([tr, te], ignore_index=True)
        is_test = df["_is_test"].to_numpy()
    else:
        df = _load_coordbench("sustainbench")  # merged table; split == "trainval" | "test"
        is_test = (df["split"] == "test").to_numpy()
    out = []
    for key, col in SUSTAINBENCH_TASKS.items():
        m = df[col].notna() & df["lat"].notna() & df["lon"].notna()
        sub = df[m]
        vals = sub[col].to_numpy(np.float64)
        out.append(
            Benchmark(
                name=f"sustainbench-{key}",
                lat=sub["lat"].to_numpy(np.float64),
                lon=sub["lon"].to_numpy(np.float64),
                tasks={col: vals},
                test_mask=is_test[m.to_numpy()],
            )
        )
    return out


# Better Together (van der Plas et al., arXiv:2605.18667; MIT code, CC-BY-SA data).
# Dynamic World, bioclim, and footprint share the random_sample == 1 point pool.
def load_better_together(data_dir: str | None = None) -> list[Benchmark]:
    """``data_dir`` reads the 5 raw CSVs instead of CoordBench. bt_bioclim/bt_human_footprint are
    restricted to the random_sample==1 pool below, matching the original protocol."""

    def _get(config: str, filename: str) -> pd.DataFrame:
        return pd.read_csv(f"{data_dir}/{filename}") if data_dir else _load_coordbench(config)

    out: list[Benchmark] = []

    ch = _get("bt_cropharvest", "cropharvest.csv")
    out.append(
        Benchmark(
            name="bt-cropharvest",
            lat=ch["lat"].to_numpy(np.float64),
            lon=ch["lon"].to_numpy(np.float64),
            tasks={"crop_type": ch["label_name"].to_numpy()},
            task_type="classification",
        )
    )

    bm = _get(
        "bt_biomass", "biomass.csv"
    )  # biomass_mean is clean; biomass_center has -9999 sentinels
    m = bm["biomass_mean"].notna()
    out.append(
        Benchmark(
            name="bt-biomass",
            lat=bm.loc[m, "lat"].to_numpy(np.float64),
            lon=bm.loc[m, "lon"].to_numpy(np.float64),
            tasks={"biomass": bm.loc[m, "biomass_mean"].to_numpy(np.float64)},
        )
    )

    dw = _get("bt_landcover", "dw_locations.csv")
    dw = dw[
        dw["random_sample"] == 1
    ]  # the 10k pool the paper probes (bioclim/footprint share its ids)
    lc_cols = [
        "water",
        "trees",
        "grass",
        "flooded_vegetation",
        "crops",
        "shrub_and_scrub",
        "built",
        "bare",
        "snow_and_ice",
    ]
    out.append(
        Benchmark(
            name="bt-landcover",
            lat=dw["lat"].to_numpy(np.float64),
            lon=dw["lon"].to_numpy(np.float64),
            tasks={
                c: dw[c].to_numpy(np.float64) for c in lc_cols
            },  # fractional cover -> multi-target regression
        )
    )

    pool_ids = set(dw["id"])
    bio = _get("bt_bioclim", "bioclim.csv")
    bio = (
        bio.merge(dw[["id", "lat", "lon"]], on="id") if data_dir else bio[bio["id"].isin(pool_ids)]
    )
    bio_cols = [f"bioclim_{i:02d}" for i in range(1, 20)]
    out.append(
        Benchmark(
            name="bt-bioclim",
            lat=bio["lat"].to_numpy(np.float64),
            lon=bio["lon"].to_numpy(np.float64),
            tasks={c: bio[c].to_numpy(np.float64) for c in bio_cols},
        )
    )

    hf = _get("bt_human_footprint", "human_footprint.csv")
    hf = hf.merge(dw[["id", "lat", "lon"]], on="id") if data_dir else hf[hf["id"].isin(pool_ids)]
    out.append(
        Benchmark(
            name="bt-population",
            lat=hf["lat"].to_numpy(np.float64),
            lon=hf["lon"].to_numpy(np.float64),
            tasks={"pop_density": hf["pop_density"].to_numpy(np.float64)},
        )
    )
    out.append(
        Benchmark(
            name="bt-distroad",
            lat=hf["lat"].to_numpy(np.float64),
            lon=hf["lon"].to_numpy(np.float64),
            tasks={
                "maxdist_road": hf["maxdist_road"].to_numpy(np.float64)
            },  # clipped at 50 km in-source
        )
    )
    return out


# CDC PLACES 2023 BRFSS prevalence estimates at ZCTA centroids.
CDC_PLACES_URL = "https://data.cdc.gov/resource/c7b2-4ecy.csv?$limit=60000"
CDC_PLACES_MEASURES = {  # task name to column prefix; CrudePrev is a percentage
    "phys_health": "PHLTH",
    "diabetes": "DIABETES",
    "copd": "COPD",
    "cancer": "CANCER",
    "chd": "CHD",
    "mental_health": "MHLTH",
    "checkup": "CHECKUP",
    "sleep_lt7": "SLEEP",
    "asthma": "CASTHMA",
    "obesity": "OBESITY",
    "smoking": "CSMOKING",
    "high_chol": "HIGHCHOL",
}


def load_cdc_places(url: str | None = None) -> list[Benchmark]:
    """Load CDC PLACES prevalence measures at ZCTA centroids.

    ``url`` reads the Socrata source instead of CoordBench.
    """
    if url:
        df = pd.read_csv(_download(url, "cdc_places_zcta_2023.csv"))
        df.columns = [c.lower() for c in df.columns]
        # geolocation is "POINT (lon lat)"; parse to floats
        geo = df["geolocation"].str.extract(r"POINT \(([-0-9.]+) ([-0-9.]+)\)").astype(float)
        df["lon"], df["lat"] = geo[0], geo[1]
    else:
        df = _load_coordbench("cdc_places")
    out = []
    for key, pref in CDC_PLACES_MEASURES.items():
        col = f"{pref.lower()}_crudeprev"
        if col not in df.columns:
            raise ValueError(
                f"load_cdc_places: expected column {col!r} for measure {key!r} is missing "
                f"(source schema may have changed); columns present: {list(df.columns)}"
            )
        m = df[col].notna() & df["lat"].notna() & df["lon"].notna()
        sub = df[m]
        out.append(
            Benchmark(
                name=f"places-{key}",
                lat=sub["lat"].to_numpy(np.float64),
                lon=sub["lon"].to_numpy(np.float64),
                tasks={col: sub[col].to_numpy(np.float64)},
            )
        )
    return out


def load_usavars() -> list[Benchmark]:
    """Load one MOSAIKS/USAVars regression benchmark per label.

    Missing coordinates are joined by ID from the other label tables.
    """
    frames = {}
    for label in USAVARS_LABELS:
        df = _load_coordbench(f"usavars_{label}")
        df.columns = [str(c).strip() for c in df.columns]
        frames[label] = df

    id_maps = []
    for df in frames.values():
        cl = {c.lower(): c for c in df.columns}
        if {"id", "lat", "lon"} <= set(cl):
            m = df[[cl["id"], cl["lat"], cl["lon"]]].copy()
            m.columns = ["ID", "lat", "lon"]
            id_maps.append(m)
    id2ll = pd.concat(id_maps).dropna().drop_duplicates("ID").set_index("ID") if id_maps else None

    benches: list[Benchmark] = []
    for label, df in frames.items():
        cl = {c.lower(): c for c in df.columns}
        # Exclude schema metadata from candidate target columns.
        skip = {"id", "lat", "lon"} | _CANONICAL_EXTRA
        cands = [
            c
            for c in df.columns
            if c.lower() not in skip
            and not c.startswith("Unnamed")
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not cands:
            continue
        value_col = next((c for c in cands if label in c.lower()), None)
        value_col = value_col or next((c for c in cands if c.lower() == "price"), None) or cands[0]
        if {"lat", "lon"} <= set(cl):
            lat, lon, val = df[cl["lat"]], df[cl["lon"]], df[value_col]
        elif id2ll is not None and "id" in cl:
            j = (
                df[[cl["id"], value_col]]
                .rename(columns={cl["id"]: "ID"})
                .merge(id2ll, left_on="ID", right_index=True)
            )
            lat, lon, val = j["lat"], j["lon"], j[value_col]
        else:
            continue
        lat_a = lat.to_numpy(np.float64)
        lon_a = lon.to_numpy(np.float64)
        val_a = val.to_numpy(np.float64)
        keep = val_a != USAVARS_NODATA
        lat_a, lon_a, val_a = lat_a[keep], lon_a[keep], val_a[keep]
        if label in USAVARS_LOG_LABELS:
            val_a = np.log1p(val_a)
        benches.append(
            Benchmark(name=f"mosaiks-{label}", lat=lat_a, lon=lon_a, tasks={label: val_a})
        )
    return benches


def load_country(n_points: int = 30000, seed: int = 0, source: str = "auto") -> Benchmark:
    """Global country classification: random land points labelled by Natural Earth admin-0.

    The default (n_points, seed) reads CoordBench's frozen ``country`` sample. Any other value,
    or ``source="live"``, re-derives from Natural Earth (needs geopandas/shapely).
    """
    if source != "live" and n_points == 30000 and seed == 0:
        df = _load_coordbench("country")
        return Benchmark(
            name="country",
            lat=df["lat"].to_numpy(np.float64),
            lon=df["lon"].to_numpy(np.float64),
            tasks={"country": df["country"].to_numpy()},
            task_type="classification",
        )

    import geopandas as gpd
    from shapely.geometry import Point

    path = _download(NATURAL_EARTH_COUNTRIES, "ne_110m_admin_0_countries.geojson")
    countries = gpd.read_file(path)[["ADMIN", "geometry"]]
    lat, lon = _random_land_points(np.random.default_rng(seed), n_points)
    pts = gpd.GeoDataFrame(
        {"lat": lat, "lon": lon},
        geometry=[Point(x, y) for x, y in zip(lon, lat, strict=True)],
        crs=4326,
    )
    joined = gpd.sjoin(pts, countries, how="inner", predicate="within")
    keep = joined[joined.groupby("ADMIN")["ADMIN"].transform("count") >= 10]  # drop tiny classes
    return Benchmark(
        name="country",
        lat=keep["lat"].to_numpy(np.float64),
        lon=keep["lon"].to_numpy(np.float64),
        tasks={"country": keep["ADMIN"].to_numpy()},
        task_type="classification",
    )


def load_deepmind_eval(root: str | None = None, names: set[str] | None = None) -> list[Benchmark]:
    """DeepMind/AlphaEarth geospatial eval suite: one Benchmark per eval CSV.

    Standard format columns: ``label``, lon/lat, a valid-time-start (-> year). Classification vs
    regression is inferred (integer label + few classes => classification). ``names`` (eval stems,
    e.g. {"openet_ensemble"}) restricts which evals load. ``root`` reads raw eval CSVs from a local
    directory instead of CoordBench's DEEPMIND_EVAL_CONFIGS.
    """
    if root is not None:
        stems_and_frames = [
            (csv.stem, pd.read_csv(csv))
            for csv in sorted(Path(root).rglob("*.csv"))
            if "trial_groups" not in csv.name and (names is None or csv.stem in names)
        ]
    else:
        stems_and_frames = [
            (stem, _load_coordbench(f"dm_{stem}"))
            for stem in DEEPMIND_EVAL_CONFIGS
            if names is None or stem in names
        ]

    if names is not None:
        found = {stem for stem, _ in stems_and_frames}
        missing_names = names - found
        if missing_names:
            raise ValueError(
                f"load_deepmind_eval: requested names not found (check for a typo): {sorted(missing_names)}"
            )

    benches: list[Benchmark] = []
    for stem, df in stems_and_frames:
        lon_col = _first_col(df, "lon", "longitude", "Lon", "x")
        lat_col = _first_col(df, "lat", "latitude", "Lat", "y")
        if "label" not in df.columns or lon_col is None or lat_col is None:
            raise ValueError(
                f"load_deepmind_eval: {stem!r} is missing an expected column "
                f"(label/lon/lat); columns present: {list(df.columns)}"
            )
        ts_col = _first_col(df, "timestamp", "valid_time_start_ms")
        label = df["label"].to_numpy()
        integral = np.all(np.isfinite(label)) and np.allclose(label, np.round(label))
        is_clf = bool(integral and np.unique(label).size <= 100)
        # All-null timestamps mean no year information.
        year = None
        if ts_col is not None:
            yr = pd.to_datetime(df[ts_col], unit="ms").dt.year
            if yr.notna().any():
                year = yr.to_numpy()
        benches.append(
            Benchmark(
                name=f"dm-{stem}",
                lat=df[lat_col].to_numpy(np.float64),
                lon=df[lon_col].to_numpy(np.float64),
                tasks={stem: label.astype(np.int64) if is_clf else label.astype(np.float64)},
                task_type="classification" if is_clf else "regression",
                year=year,
            )
        )
    return benches


def _sample_raster(src, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:  # noqa: ANN001
    """Sample WGS84 coordinates in the raster CRS, replacing nodata with NaN."""
    from rasterio.warp import transform as warp_transform

    xs, ys = warp_transform("EPSG:4326", src.crs, lon.tolist(), lat.tolist())
    vals = np.array([v[0] for v in src.sample(zip(xs, ys, strict=True))], dtype=np.float64)
    if src.nodata is not None:
        vals = np.where(vals == src.nodata, np.nan, vals)
    return vals


def load_worldclim(
    bio_vars: tuple[int, ...] = (1, 12), n_points: int = 20000, seed: int = 0, source: str = "auto"
) -> list:
    """Load WorldClim v2.1 bioclim at 10 arc-minute resolution.

    Defaults read CoordBench; other parameters or ``source="live"`` sample the
    rasters. bio1 is annual mean temperature; bio12 is annual precipitation.
    """
    if source != "live" and bio_vars == (1, 12) and n_points == 20000 and seed == 0:
        df = _load_coordbench("worldclim_bio")
        return [
            Benchmark(
                name=f"worldclim-bio{v}",
                lat=df["lat"].to_numpy(np.float64),
                lon=df["lon"].to_numpy(np.float64),
                tasks={f"bio{v}": df[f"bio{v}"].to_numpy(np.float64)},
            )
            for v in bio_vars
            if f"bio{v}" in df.columns
        ]

    import rasterio

    cache = CACHE / "worldclim_bio.csv"
    if not cache.exists():
        zip_path = _download(WORLDCLIM_BIO_URL, "wc2.1_10m_bio.zip")
        out_dir = CACHE / "worldclim10m"
        if not out_dir.exists():
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(out_dir)
        rng = np.random.default_rng(seed)
        lat, lon = _random_land_points(rng, n_points * 3)  # oversample; ocean -> nodata, dropped
        cols = {"lat": lat, "lon": lon}
        for v in bio_vars:
            with rasterio.open(out_dir / f"wc2.1_10m_bio_{v}.tif") as src:
                cols[f"bio{v}"] = _sample_raster(src, lat, lon)
        df = pd.DataFrame(cols).dropna()
        if len(df) > n_points:
            df = df.sample(n_points, random_state=seed)
        df.to_csv(cache, index=False)
    df = pd.read_csv(cache)
    return [
        Benchmark(
            name=f"worldclim-bio{v}",
            lat=df["lat"].to_numpy(np.float64),
            lon=df["lon"].to_numpy(np.float64),
            tasks={f"bio{v}": df[f"bio{v}"].to_numpy(np.float64)},
        )
        for v in bio_vars
        if f"bio{v}" in df.columns
    ]


def load_soilgrids(
    props: tuple[str, ...] = ("soc", "phh2o"),
    depth: str = "0-5cm",
    n_points: int = 6000,
    seed: int = 0,
    source: str = "auto",
) -> list:
    """Load SoilGrids v2 soil organic carbon and pH.

    Defaults read CoordBench; other parameters or ``source="live"`` sample
    decimated global COGs after reprojecting coordinates to Goode Homolosine.
    """
    if (
        source != "live"
        and props == ("soc", "phh2o")
        and depth == "0-5cm"
        and n_points == 6000
        and seed == 0
    ):
        df = _load_coordbench("soilgrids")
        return [
            Benchmark(
                name=f"soilgrids-{p}",
                lat=df["lat"].to_numpy(np.float64),
                lon=df["lon"].to_numpy(np.float64),
                tasks={p: df[p].to_numpy(np.float64)},
            )
            for p in props
            if p in df.columns
        ]

    import rasterio
    from rasterio.transform import Affine, rowcol
    from rasterio.warp import transform as warp_transform

    cache = CACHE / "soilgrids.csv"
    if not cache.exists():
        rng = np.random.default_rng(seed)
        wc = CACHE / "worldclim_bio.csv"  # reuse known land points to skip ocean
        if wc.exists():
            d = pd.read_csv(wc)
            d = d.sample(min(n_points, len(d)), random_state=seed)
            lat, lon = d["lat"].to_numpy(), d["lon"].to_numpy()
        else:
            lat, lon = _random_land_points(rng, n_points)
        cols = {"lat": lat, "lon": lon}
        for p in props:
            url = "/vsicurl/" + SOILGRIDS_COG.format(prop=p, depth=depth)
            with rasterio.open(url) as src:
                out_w, out_h = 4000, 2000  # sample a coarse global overview
                arr = src.read(1, out_shape=(out_h, out_w))
                tr = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
                xs, ys = warp_transform("EPSG:4326", src.crs, lon.tolist(), lat.tolist())
                rows, cs = rowcol(tr, xs, ys)
                rows = np.clip(np.asarray(rows), 0, out_h - 1)
                cs = np.clip(np.asarray(cs), 0, out_w - 1)
                v = arr[rows, cs].astype(np.float64)
                if src.nodata is not None:
                    v[v == src.nodata] = np.nan
            v[v <= 0] = np.nan  # SoilGrids encodes no-soil/ocean as 0
            cols[p] = v
        df = pd.DataFrame(cols).dropna()
        df.to_csv(cache, index=False)
    df = pd.read_csv(cache)
    return [
        Benchmark(
            name=f"soilgrids-{p}",
            lat=df["lat"].to_numpy(np.float64),
            lon=df["lon"].to_numpy(np.float64),
            tasks={p: df[p].to_numpy(np.float64)},
        )
        for p in props
        if p in df.columns
    ]


def load_ecoregions(n_points: int = 30000, seed: int = 0, source: str = "auto") -> Benchmark:
    """Global biome + ecoregion classification via RESOLVE Ecoregions2017.

    The default (n_points, seed) reads CoordBench's frozen ``ecoregions`` sample. Any other value,
    or ``source="live"``, re-derives from the Ecoregions2017 shapefile (needs geopandas/shapely).
    """
    if source != "live" and n_points == 30000 and seed == 0:
        df = _load_coordbench("ecoregions")
        return Benchmark(
            name="ecoregions",
            lat=df["lat"].to_numpy(np.float64),
            lon=df["lon"].to_numpy(np.float64),
            tasks={"biome": df["biome"].to_numpy(), "ecoregion": df["ecoregion"].to_numpy()},
            task_type="classification",
        )

    import geopandas as gpd
    from shapely.geometry import Point

    CACHE.mkdir(parents=True, exist_ok=True)
    shp_dir = CACHE / "ecoregions2017"
    if not shp_dir.exists():
        with (
            urllib.request.urlopen(ECOREGIONS_ZIP) as resp,
            zipfile.ZipFile(io.BytesIO(resp.read())) as zf,
        ):
            zf.extractall(shp_dir)
    shp = next(shp_dir.rglob("*.shp"))
    eco = gpd.read_file(shp)[["BIOME_NAME", "ECO_NAME", "geometry"]]
    lat, lon = _random_land_points(np.random.default_rng(seed), n_points)
    pts = gpd.GeoDataFrame(
        {"lat": lat, "lon": lon},
        geometry=[Point(x, y) for x, y in zip(lon, lat, strict=True)],
        crs=4326,
    )
    joined = gpd.sjoin(pts, eco, how="inner", predicate="within")
    return Benchmark(
        name="ecoregions",
        lat=joined["lat"].to_numpy(np.float64),
        lon=joined["lon"].to_numpy(np.float64),
        tasks={
            "biome": joined["BIOME_NAME"].to_numpy(),
            "ecoregion": joined["ECO_NAME"].to_numpy(),
        },
        task_type="classification",
    )
