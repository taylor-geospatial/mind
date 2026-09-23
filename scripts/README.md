# Experiments

Run modules from the repository root after `make install`.
Set `MIND_CHECKPOINT` to the released weights and `COORDBENCH_ROOT` to the downloaded CoordBench directory.
Outputs default to `results/`.

Ridge and Chunked Penalty probes use training-split centering and an unpenalized intercept, matching the updated paper results. Recompute evaluations into fresh output files when comparing with results produced before this correction; existing CSVs do not identify the intercept convention.

## Main comparison

Run the three commands in the root README for fold seeds 0 through 4.
Keep the cell ladder `0 0.25 0.5 1 2 5 10 20 40`; 5 degrees is needed to detect saturated 10-degree assignments.
Cell 0 uses random folds; positive values use regional folds.
The summarizer reports cells 0, 2, 10, 20, and 40.

Evaluation uses the same five outer folds for each representation, replacing official source splits.
Predictor selection uses each outer training pool, with regional inner folds for regional evaluation.
`--only` accepts `pdfm`, `air_temp`, `calhousing`, `sustainbench`, `mosaiks`, `satclip_official`, `worldclim`, `soilgrids`, `cdc_places`, and `deepmind`.

- `blocksize_sweep` compares frozen encoders. Use `--baseline-names` to select baselines, `--no-baselines` for MIND alone, or `--ckpts path=tag` for another checkpoint.
- `tikhonov_sweep` evaluates Chunked Penalty and training-fold prefix selection.
- `coord_interp_baseline` evaluates coordinate IDW. The paper uses `--ks 50`; the neighbor-count sweep uses `--ks 1 2 5 10 20 35 50 100 200 500`.

SatCLIP requires the original L40 weights, either the full upstream checkpoint or its extracted location encoder.
The recorded configuration does not identify the image-backbone variant.

## Aggregate CSVs

Create `results/inputs.json` with the input files and fold seeds.
Penalty files also need an encoder name in `source`:

```json
[
  {"path": "encoders-0.csv", "seed": 0},
  {"path": "penalty-0.csv", "seed": 0, "source": "mind"},
  {"path": "idw-0.csv", "seed": 0}
]
```

Add entries for seeds 1 through 4.
For sharded runs, use a glob and an `expected_files` count to check for missing shards.

```bash
uv run python -m scripts.paper.summarize --manifest results/inputs.json \
  --methods mind_64 mind_3072 mind/cv_chunk coord_idw/k50 \
  --categories --out results/summary.csv
```

The output includes per-seed scores, mean, sample standard deviation, and the dataset cohort.
Regression scores are floored at -1 per target, averaged within each dataset, then averaged across datasets with equal weight.
Classification uses the same dataset weighting.
All methods and seeds share the eligible dataset set at each block size.
Unchanged nearest-training-distance statistics between successive positive block sizes identify saturated assignments.

Method names are the encoder name, `source/config` for penalty runs, or `coord_idw/k50` for IDW.
Omit `--methods` to include all methods in the inputs.
Do not mix unrelated training runs in one fold-seed entry.
Use `--seeds 0` for runs with one fold assignment.

## Training controls

Add these arguments to the root README's training command:

| Control                  | Arguments                         |
| ------------------------ | --------------------------------- |
| Non-nested               | `--matryoshka ""`                 |
| Fixed 64 dimensions      | `--embed-dim 64 --matryoshka ""`  |
| Fixed 256 dimensions     | `--embed-dim 256 --matryoshka ""` |
| Additional training seed | `--seed 1`                        |

Evaluate fixed-width models with `blocksize_sweep --arch-preset resiren64` or `resiren256`, `--no-baselines`, and `--ckpts path=tag`.
Keep training-seed and fold-seed comparisons separate.
Evaluate the non-nested checkpoint with the prefix and Chunked Penalty sweeps.
Run-specific `--vicreg-var`, `--vicreg-cov`, and training seeds were not retained for the `vicreg_covariance` and `vicreg_variance_covariance` controls.

For order controls, run `tikhonov_sweep` with `--permute` and separately with `--encoder geoclip`, using fold seed 0 and the full cell ladder.
Give each run a distinct `source` in its input manifest.
`summarize --seeds 0 --cohort results/summary.csv` uses the main comparison's dataset cohort.

## Spatial analyses

```bash
uv run python -m scripts.paper.distance_error --out results/distance-error.csv
uv run python -m scripts.paper.variogram \
  --n-anchors 20000 --pairs-per-lag 4096 \
  --ckpts weights/mind.safetensors=mind --out results/variogram.csv
uv run python -m scripts.paper.buffered_exclusion \
  --radii 0 10 25 50 100 --fold-seed 0 --out results/buffered-0.csv
```

Repeat buffered exclusion for fold seeds 0 through 4.
It holds test points fixed and compares radius-based training exclusions with random deletions of the same size.
Aggregate these files in a separate manifest with `summarize --buffered --task-type regression`.

`variogram` estimates half-sill distances using geodesic offsets and land masks.
Multiple `--ckpts path=tag` values compare objectives at the same locations.

```bash
uv run python -m scripts.paper.width_pca_maps
uv run python -m scripts.paper.width_band_maps
```

These scripts compute covariance participation ratios, prefix PCA maps, and selected chunk maps using the WorldClim land mask.
Set `MIND_DEVICE=cpu` for CPU computation and `MIND_LAND_RASTER` to a local WorldClim bio1 raster, or let the script download it.
Maps use `results/map-cache/`; change the cache directory when changing models.
`plot_spatial_splits --evaluation-root data/coordbench/data` renders the geographic-fold example.
Use `--temperature` and `--africa-crops` to supply those source tables explicitly.

## Construct MINDSET

Use the released MINDSET tables to train on the paper's coordinates and targets.
Pixel sampling uses the AlphaEarth mosaic configured in `src/mind/data/zarr_aef.py`.
The sampler uses nine annual mosaics, GeoNames cities with at least 15,000 people, 16,000 blocks of 256 × 256 pixels, and up to 768 valid points per block.
The builder computes three coordinate teachers and writes Hilbert-sorted GeoParquet tables.

```bash
uv run python -m scripts.data.build_aef_urban --out data/aef-pixels --workers 32 --seed 0
uv run python -m scripts.data.build_mindset --data data/aef-pixels --out-dir data/mindset
```

See the [CoordBench guide](data/coordbench/README.md) for its data pipeline.
