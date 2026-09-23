# Build CoordBench

The scripts construct 33 source tables for 52 datasets and 78 prediction targets.
They preserve source labels and columns; evaluation applies filtering, logarithms, and folds in `src/mind/eval/benchmarks.py` and `probe.py`.
The build directory defaults to `data/coordbench`, with sources in `raw/<family>/` and output tables in `data/<config>/data.parquet`.

```bash
export COORDBENCH_BUILD_ROOT=/path/to/coordbench
uv run python -m scripts.data.coordbench.fetch_raw --satclip-dir /path/to/satclip-csvs
uv run python -m scripts.data.coordbench.normalize
uv run python -m scripts.data.coordbench.validate
export COORDBENCH_ROOT="$COORDBENCH_BUILD_ROOT"
```

Each command accepts `--only FAMILY [FAMILY ...]`:

```bash
uv run python -m scripts.data.coordbench.fetch_raw --only california_housing
uv run python -m scripts.data.coordbench.normalize --only california_housing
uv run python -m scripts.data.coordbench.validate --only california_housing
```

Families are `pdfm_conus27`, `air_temp`, `california_housing`, `satclip_official`, `usavars`, `worldclim`, `soilgrids`, `sustainbench`, `cdc_places`, and `deepmind_eval`.
The default run builds all families.
Fetch and normalize WorldClim before SoilGrids, which uses the same seed-0 land sample.

## Sources

[`manifest.py`](manifest.py) lists source URLs and transformations.
USAVars uses a pinned revision; SustainBench uses the 11 MB DHS label archive from TorchSpatial/LocBench.
DeepMind uses 15 tables from the 22 MB evaluation archive in Zenodo record 16585402.
No satellite imagery downloads are needed.

Download SatCLIP's CSVs from the [upstream Google Drive folder](https://drive.google.com/drive/folders/1tI2qo6iioRrv3P1OxSXwHKObpLCrinad) into the `--satclip-dir` directory:

- `lat_lon_country.csv`
- `lat_lon_ecoregion.csv`
- `labels_population.csv`
- `labels_elevation.csv`

Files already in `raw/satclip_official/` are reused.
WorldClim and SoilGrids use fixed-seed raster samples; SoilGrids stores 4000-by-2000 source overviews.
`validate` compares tables with raw sources; raster checks may access the network.
Use the released CoordBench tables for the paper's exact evaluation rows.

## Table format

Tables begin with `lon`, `lat`, `timestamp`, `timestamp_end`, `split`, and `id`, followed by source columns.
Coordinates are WGS84 degrees; timestamps are nullable Unix milliseconds, with year-only observations dated January 1 UTC.
`split` preserves official source membership.

One table can supply several datasets or targets, including SustainBench's six DHS indices and CDC PLACES' twelve health measures.
USAVars income and housing acquire coordinates through ID joins to other USAVars files.
Raw nodata sentinels and labels are preserved.

## Licenses

[`manifest.py`](manifest.py) records license status, and [`dm_eval.py`](dm_eval.py) lists DeepMind's per-file licenses.
PDFM, air temperature, and SatCLIP have unresolved source-license status; WorldClim requires redistribution permission, and DHS-derived SustainBench redistribution terms are unconfirmed.
Most DeepMind tables are CC-BY-4.0; `us_trees` is CC-BY-NC-4.0, and `canada_crops_*` use the Open Government Licence - Canada.
Retain upstream terms when redistributing tables.
