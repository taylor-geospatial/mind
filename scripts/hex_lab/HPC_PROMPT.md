# HPC handoff prompt

Copy the following prompt into an agent session on the HPC. Replace host and scratch paths only after inspecting the actual cluster environment.

```text
Continue work on the MIND hex explorer and generate the additional global H3 Parquet embeddings.

Repository: https://github.com/taylor-geospatial/mind
Branch: feat/hex-embedding-explorer

If a checkout does not already exist, run:
  gh repo clone taylor-geospatial/mind mind
  cd mind
  git switch feat/hex-embedding-explorer
  uv sync

If the repository already exists, inspect git status and remotes first. Preserve local changes, fetch origin, and check out the existing branch without resetting anything. Read AGENTS.md, scripts/hex_lab/README.md, scripts/hex_lab/prepare.py, and scripts/hex_lab/analysis.py before editing.

The explorer already supports arbitrary emb_<integer> columns, --dims START:STOP, PCA fitting, and configurable cache paths. The published https://data.source.coop/tge-labs/mind/mind_hex_r5.parquet contains only dimensions 0–63. Do not treat those 64 columns as the full embedding or fabricate additional dimensions from them.

Build the remaining 47 contiguous 64-dimensional storage blocks and a full 3,072-dimensional H3 r5 Parquet, with reproducible generation code and provenance. Storage block j covers [64*j, 64*(j+1)). These 48 storage blocks are different from the seven training chunks (0:64, 64:128, 128:256, 256:512, 512:1024, 1024:2048, 2048:3072).

First recover and inspect the existing hex-generation pipeline and full-width source on the HPC. A candidate public source is https://data.source.coop/tge-labs/mind/mind.zarr. docs/index.html has a small AOI example: decode embedding * emb_scale + emb_offset and apply mask == 1. Verify current metadata, dimensions, pyramid level, coordinate orientation, nodata, and quantization parameters before using it. Prefer an existing local copy or original producer when available.

Match the published Parquet's actual source grid/pyramid level, land mask, pixel selection, and per-hex aggregation. Do not assume the base Zarr level reproduces its counts. Do not replace the mean of sampled pixel embeddings with model inference at the hex centroid; that is a different dataset. Reproduce the first 64 dimensions on a representative subset and compare H3 IDs, counts, and means to the published file before scaling. If the original method cannot be recovered, report the discrepancy and define a separately named, documented product rather than claiming an exact extension.

Implement a resumable tiled/sharded producer with bounded memory, deterministic ordering, and restart checkpoints. Reuse the same valid spatial samples and counts for every dimension block. Keep emb_<integer> column suffixes as global dimension indices. Write unique h3 strings, positive count, embedding means, and WGS84 centroid geometry with GeoParquet metadata. Fit/store pca1–pca3 independently for each representation and retain each basis, dimension range, sampling seed, model/data provenance, and checksums. Assemble the full-width output by checked one-to-one H3 joins, never by assuming shard row order. Use scratch storage, not Git, for generated data.

Inspect available RAM, scratch capacity, scheduler, and source I/O before choosing job sizes. Run a small regional smoke build first; estimate global time and disk use from it. Then use the cluster's scheduler for the global run with checkpoints and logs, following the available cluster instructions. Do not load the full global pixel × 3,072 array into RAM.

Validate every output: unique r5 IDs, matching spatial coverage and counts across blocks, expected global dimension indices, finite values, coordinate/geometry consistency, correct masks/decoding, and agreement with direct source samples. Include antimeridian and high-latitude samples. Verify the full-width vectors against their component block files.

Prepare explorer caches for representative storage blocks 0, 8, 24, and 47, plus the full embedding. Use python -m scripts.hex_lab.prepare with new output paths and explicit labels; refit PCA for each selected representation. Serve them with python -m scripts.hex_lab.serve --data-dir PATH --port PORT. Provide the exact SSH port-forwarding commands for my actual login/compute hosts and inspect PCA, similarity, and contrast in the browser. If full-width similarity is slow, profile it before changing the algorithm.

Add useful regression tests and generation documentation. Run make check, commit source changes on this branch, and push them to update its PR. Do not merge the PR, publish datasets, modify the GitHub Pages site, or commit generated Parquets/caches. Finish with the job status, output paths and sizes, dimension coverage, validation results, and the commands needed to resume or open the explorer.
```
