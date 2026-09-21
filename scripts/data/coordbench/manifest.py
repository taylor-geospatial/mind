"""CoordBench source tables, file locations, and licenses.

``license_status``:
  clear: no redistribution caveat identified.
  needs_verification: terms not fully confirmed.
  unresolved_risk: redistribution restrictions or unresolved access terms.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigSpec:
    key: str  # data/<key>/data.parquet
    family: str  # raw/<family>/
    raw_files: tuple[str, ...]  # relative paths under raw/<family>/
    source_url: str
    license: str
    license_status: str  # clear | needs_verification | unresolved_risk
    citation: str
    task_type: str  # regression | classification
    notes: str = ""
    snapshot: bool = False  # Sampled from rasters (WorldClim/SoilGrids).


CONFIGS: tuple[ConfigSpec, ...] = (
    ConfigSpec(
        key="pdfm_conus27",
        family="pdfm_conus27",
        raw_files=("conus27.csv",),
        source_url=(
            "https://raw.githubusercontent.com/google-research/population-dynamics/master/"
            "data/benchmarks/conus27.csv"
        ),
        license="Apache-2.0 (repo); data file itself not explicitly cleared",
        license_status="needs_verification",
        citation="Google Research, Population Dynamics Foundation Model benchmarks (conus27)",
        task_type="regression",
        notes="27 US place-level health/socio/env tasks + 3 official split-scheme columns "
        "(imputation_split, superresolution_split, extrapolation_split) kept as-is, not collapsed "
        "into one canonical split (they're independent regimes, not one train/test partition).",
    ),
    ConfigSpec(
        key="air_temp",
        family="air_temp",
        raw_files=("stationDataAll.csv",),
        source_url="https://api.figshare.com/v2/file/download/12609182",
        license="figshare-hosted (Hooker et al. 2018); not explicitly confirmed",
        license_status="needs_verification",
        citation="Hooker et al. 2018, global high-resolution mean air temperature stations",
        task_type="regression",
    ),
    ConfigSpec(
        key="california_housing",
        family="california_housing",
        raw_files=("california_housing.csv",),
        source_url="sklearn.datasets.fetch_california_housing (StatLib, Pace & Barry 1997)",
        license="public domain (StatLib)",
        license_status="clear",
        citation="Pace, R. Kelley and Ronald Barry, 1997",
        task_type="regression",
    ),
    ConfigSpec(
        key="satclip_country",
        family="satclip_official",
        raw_files=("lat_lon_country.csv",),
        source_url="https://drive.google.com/drive/folders/1tI2qo6iioRrv3P1OxSXwHKObpLCrinad",
        license="unspecified (Google Drive share, github.com/microsoft/satclip issue #6)",
        license_status="needs_verification",
        citation="Klemmer et al., SatCLIP",
        task_type="classification",
        notes="country==0 is the ocean/nodata sentinel; NOT filtered here (consumer's choice).",
    ),
    ConfigSpec(
        key="satclip_ecoregion",
        family="satclip_official",
        raw_files=("lat_lon_ecoregion.csv",),
        source_url="https://drive.google.com/drive/folders/1tI2qo6iioRrv3P1OxSXwHKObpLCrinad",
        license="unspecified (Google Drive share, github.com/microsoft/satclip issue #6)",
        license_status="needs_verification",
        citation="Klemmer et al., SatCLIP",
        task_type="classification",
        notes="ecoregion_class_index/biome_class_index/realmbiome_class_index == -1 is the "
        "ocean/nodata sentinel; NOT filtered here. realmbiome_class_index wasn't previously "
        "exposed by benchmarks.py's loader -- present in the raw file, kept here.",
    ),
    ConfigSpec(
        key="satclip_population",
        family="satclip_official",
        raw_files=("labels_population.csv",),
        source_url="https://drive.google.com/drive/folders/1tI2qo6iioRrv3P1OxSXwHKObpLCrinad",
        license="unspecified (Google Drive share, github.com/microsoft/satclip issue #6)",
        license_status="needs_verification",
        citation="Klemmer et al., SatCLIP",
        task_type="regression",
    ),
    ConfigSpec(
        key="satclip_elevation",
        family="satclip_official",
        raw_files=("labels_elevation.csv",),
        source_url="https://drive.google.com/drive/folders/1tI2qo6iioRrv3P1OxSXwHKObpLCrinad",
        license="unspecified (Google Drive share, github.com/microsoft/satclip issue #6)",
        license_status="needs_verification",
        citation="Klemmer et al., SatCLIP",
        task_type="regression",
    ),
    *(
        ConfigSpec(
            key=f"usavars_{label}",
            family="usavars",
            raw_files=(f"usavars_{label}.csv",),
            source_url=(
                "https://hf.co/datasets/torchgeo/usavars/resolve/"
                f"01377abfaf50c0cc8548aaafb79533666bbf288f/{label}.csv"
            ),
            license="CC-BY-4.0",
            license_status="clear",
            citation="torchgeo/usavars (MOSAIKS)",
            task_type="regression",
            notes="income/housing carry no native lat/lon; joined from the coord-bearing USAVars "
            "files by shared ID at normalize time (a join, not a value change). -999 nodata "
            "sentinel kept as-is, not dropped.",
        )
        for label in (
            "treecover",
            "elevation",
            "population",
            "nightlights",
            "income",
            "roads",
            "housing",
        )
    ),
    ConfigSpec(
        key="worldclim_bio",
        family="worldclim",
        raw_files=("wc2.1_10m_bio.zip",),
        source_url="https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip",
        license="non-commercial/academic use only; redistribution requires WorldClim's permission",
        license_status="unresolved_risk",
        citation="Fick & Hijmans 2017, WorldClim 2",
        task_type="regression",
        snapshot=True,
        notes="Point sample procedurally generated (benchmarks.py:load_worldclim, seed=0, "
        "n=20000) from these global rasters. Redistribution requires source permission.",
    ),
    ConfigSpec(
        key="soilgrids",
        family="soilgrids",
        raw_files=("soilgrids_overview_soc.tif", "soilgrids_overview_phh2o.tif"),
        source_url="https://files.isric.org/soilgrids/latest/data/{prop}/{prop}_0-5cm_mean.vrt",
        license="CC-BY-4.0",
        license_status="clear",
        citation="Poggio et al. 2021, SoilGrids 2.0",
        task_type="regression",
        snapshot=True,
        notes="'Raw' here is the coarse 4000x2000 overview grid benchmarks.py:load_soilgrids "
        "actually reads (a single decimated fetch), not the full-resolution global COG -- "
        "mirroring the multi-GB original would be impractical and isn't what the benchmark uses.",
    ),
    ConfigSpec(
        key="sustainbench",
        family="sustainbench",
        raw_files=("dhs_trainval_labels.csv", "dhs_test_labels.csv"),
        source_url="https://api.figshare.com/v2/articles/26026798 (TorchSpatial/LocBench pack)",
        license="DHS Program-derived indices; redistribution terms not confirmed (dhsprogram.com "
        "access-gated)",
        license_status="unresolved_risk",
        citation="Yeh et al., SustainBench; DHS Program surveys",
        task_type="regression",
        notes="One config for the full merged trainval+test table (32 cols incl. all 6 SDG "
        "indices + normalized variants + urban/survey/adm1 metadata), not split per index. "
        "split = 'trainval' | 'test' by source file. id = DHSID_EA. timestamp from 'year' "
        "(survey year, year-granularity -> Jan-1-UTC ms).",
    ),
    ConfigSpec(
        key="cdc_places",
        family="cdc_places",
        raw_files=("cdc_places_zcta_2023.csv",),
        source_url="https://data.cdc.gov/resource/c7b2-4ecy.csv?$limit=60000",
        license="US federal government work / CDC Open Data (Socrata) - public",
        license_status="clear",
        citation="CDC PLACES 2023 (model-based ZCTA estimates)",
        task_type="regression",
        notes="One config for all 12 CrudePrev health measures (one shared ZCTA-level source "
        "table), not split per measure. geolocation POINT parsed to lon/lat (parse, not a value "
        "change); original geolocation column kept as-is too.",
    ),
    # DeepMind configs and licenses are defined in benchmarks.py and dm_eval.py.
)

DEEPMIND_EVAL_SOURCE_URL = "https://zenodo.org/records/16585402"
DEEPMIND_EVAL_LICENSE_NOTE = (
    "Apache-2.0 (code); CC-BY-4.0 (most data); us_trees is CC-BY-NC-4.0 (iNaturalist-derived); "
    "canada_crops_* is Open Government Licence - Canada. Per-config license is set per-file in "
    "dm_eval.py, NOT blanket CC-BY-4.0 -- see the deepmind_eval README.md shipped in the zip."
)


def by_family(family: str) -> tuple[ConfigSpec, ...]:
    return tuple(c for c in CONFIGS if c.family == family)


def by_key(key: str) -> ConfigSpec:
    (spec,) = (c for c in CONFIGS if c.key == key)
    return spec


FAMILIES: tuple[str, ...] = tuple(dict.fromkeys(c.family for c in CONFIGS)) + ("deepmind_eval",)
