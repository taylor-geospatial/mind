# MIND hex explorer

Local PCA colors, distant embedding matches, and neighboring-cell contrast with optional 3D columns. This is a separate research tool; the GitHub Pages site is unchanged.

## Run the published 64-dimensional example

From the repository root:

```bash
uv sync
mkdir -p data/hex
curl -fL --retry 3 \
  https://data.source.coop/tge-labs/mind/mind_hex_r5.parquet \
  -o data/hex/mind_hex_r5.parquet
uv run python -m scripts.hex_lab.prepare \
  --input data/hex/mind_hex_r5.parquet \
  --output results/hex-lab/first-64 --pca stored --label 'First 64 dimensions'
uv run python -m scripts.hex_lab.serve --data-dir results/hex-lab/first-64
```

Open <http://127.0.0.1:8776/>. Try **Contrast → Himalaya → Raise contrast into 3D columns**, or click a location in **Similarity**. The distance control excludes matches within 1,000, 3,000, or 6,000 km; displayed matches are also separated by 500 km.

The source Parquet is about 209 MiB and contains 574,638 H3 r5 cells, 64 embedding columns, three PCA columns, `count`, and centroid geometry. It cannot provide the remaining dimensions; those require additional upstream data. See the [HPC handoff prompt](HPC_PROMPT.md).

## Other dimensions and files

The preparer discovers every column matching `emb_<integer>` and sorts by its numeric suffix. The source must contain unique resolution-5 H3 IDs as hexadecimal strings, positive integer `count`, and finite embeddings with nonzero norms. `geometry` is optional because polygons are reconstructed from H3 IDs.

Each invocation builds one representation in a new output directory. It never overwrites an existing cache. For example, given a **new** file containing all 3,072 dimensions:

```bash
# A later 64-dimensional storage block; stop is exclusive.
uv run python -m scripts.hex_lab.prepare \
  --input /scratch/hex/mind_hex_r5_full.parquet \
  --dims 512:576 --output /scratch/hex-lab/block-08 --label 'Storage block 8 · dimensions 512–575'

# Full embedding: all emb_* columns, automatically discovered.
uv run python -m scripts.hex_lab.prepare \
  --input /scratch/hex/mind_hex_r5_full.parquet \
  --output /scratch/hex-lab/full --label 'Full embedding · 3072 dimensions'

uv run python -m scripts.hex_lab.serve --data-dir /scratch/hex-lab/full --port 8776
```

A Parquet containing just one block also works without `--dims`; retain its global dimension indices in the column names. Prefixes such as `--dims 0:256` and training chunks such as `--dims 128:256` work the same way. MIND's 48 equal 64-dimensional **storage blocks** differ from its seven **training chunks**: `0:64`, `64:128`, `128:256`, `256:512`, `512:1024`, `1024:2048`, and `2048:3072`.

PCA is fitted to the selected raw embedding columns by default, using a reproducible sample of up to 20,000 cells (`--pca-samples`, `--seed`). It is not standardized per dimension. The fitted basis, sampled row indices, explained variance ratios, selected columns, and source checksum are saved with the cache. PCA colors from separate fits do not identify common classes across representations. Use `--pca stored` only when the input's `pca1`–`pca3` already describe **all** its embedding columns; this option is rejected with `--dims`.

Preparation reads Parquet batches into a float32 memory-mapped array, aggregates overviews in 64-channel blocks, and batches neighbor calculations. `--batch-size 512` lowers working memory for full-width runs. At 574,638 × 3,072, each dense float32 source array is approximately 6.6 GiB. Preparation also needs a normalized copy, overview arrays, PCA workspace, and temporary disk space; allow at least 25 GiB of scratch space and provision RAM above the mapped working set. The serving process memory-maps vectors; a global similarity query still scans every dimension and can take longer at full width. Measure a regional run before scheduling the global build.

## HPC access

Run preparation and serving on a compute node using the cluster's normal scheduler. The server binds to `127.0.0.1` only. From your laptop, forward through the login host to the allocated compute node:

```bash
ssh -N -J USER@LOGIN_HOST -L 8776:127.0.0.1:8776 USER@COMPUTE_NODE
```

Then open <http://127.0.0.1:8776/> locally. Substitute the real account/hosts and follow your cluster's SSH policy. If serving directly on an allowed host, omit `-J`. Run caches on separate ports to compare them side by side.

The embedding data remains on the machine running the server. The browser requests MapLibre GL JS 6.10.0 from unpkg, basemap tiles/labels from CARTO/OpenStreetMap, and fonts from Google Fonts. No Node build or GPU is needed for the explorer. Network access is required for that browser styling/basemap.

## Quantities and validation

- **Color atlas:** three PCA scores mapped to RGB with a fixed per-channel 2nd–98th percentile stretch.
- **Similarity:** cosine similarity of L2-normalized selected embedding dimensions. Queries snap to the nearest source cell within 100 km. Rankings always use original r5 cells, including when the map shows an overview. Embedding similarity is not a validated ecological analogue label.
- **Contrast:** mean cosine distance to available immediate r5 H3 neighbors. Gray means no neighbors; it is not zero contrast. The source r5 99th percentile caps display color and height, while hover reports uncapped values. Columns represent embedding contrast, not terrain elevation.
- **Overviews:** r3/r4 means weighted by `count`, calculated before L2 normalization. PCA scores are averaged with the same weights. Contrast overviews average the r5 contrast among cells with neighbors, so the measure and color scale remain consistent across zoom levels. Empty regions are absent from the source file. `count` is displayed as source samples; verify the upstream sampling method before comparing different producers.

Generated data belongs under ignored `data/`, `results/`, or external scratch storage; do not commit Parquets, caches, or screenshots. Run `make check` for repository checks. The hex tests cover additional dimensions, numeric column ordering, PCA selection, weighted overviews, cosine scores, missing neighbors, and antimeridian geometry. Browser verification covers the three views, distance filtering, source-resolution zoom, and mobile layout.
