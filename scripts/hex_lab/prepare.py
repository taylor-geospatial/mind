"""Build a local explorer cache from H3 r5 Parquet embedding columns."""

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import h3.api.basic_int as h3
import numpy as np
import pyarrow.parquet as pq
from numpy.lib.format import open_memmap
from sklearn.decomposition import PCA

from scripts.hex_lab.metrics import (
    aggregate_contrast,
    aggregate_vectors,
    neighbor_contrast,
    normalize_to,
)


def embedding_columns(names, dims=None):
    indexed = sorted((int(name[4:]), name) for name in names if re.fullmatch(r"emb_\d+", name))
    ids = [i for i, _ in indexed]
    if len(set(ids)) != len(ids):
        raise ValueError("Embedding column suffixes must identify unique dimensions.")
    if dims:
        parts = dims.split(":")
        if len(parts) != 2:
            raise ValueError("--dims must be a half-open range such as 64:128.")
        start, stop = map(int, parts)
        if start < 0 or stop <= start or not set(range(start, stop)).issubset(ids):
            raise ValueError(f"The file does not contain every dimension in {dims}.")
        indexed = [(i, name) for i, name in indexed if start <= i < stop]
    if len(indexed) < 3:
        raise ValueError("At least three emb_<integer> columns are required.")
    return [name for _, name in indexed]


def source_arrays(source, columns, path, batch_size):
    file = pq.ParquetFile(source)
    metadata = file.read(columns=["h3", "count"])
    try:
        cells = np.array([int(cell, 16) for cell in metadata["h3"].to_pylist()], np.uint64)
    except (TypeError, ValueError) as error:
        raise ValueError("h3 must contain hexadecimal cell ID strings.") from error
    counts = metadata["count"].to_numpy().astype(np.float64)
    if not len(cells) or len(np.unique(cells)) != len(cells):
        raise ValueError("The source must have one row per unique H3 cell.")
    if any(not h3.is_valid_cell(int(c)) or h3.get_resolution(int(c)) != 5 for c in cells):
        raise ValueError("The explorer currently expects H3 resolution-5 cells.")
    if not np.isfinite(counts).all() or (counts <= 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("count must be a positive integer for every cell.")
    values = open_memmap(path, mode="w+", dtype=np.float32, shape=(len(cells), len(columns)))
    start = 0
    for batch in file.iter_batches(batch_size=batch_size, columns=columns):
        block = np.column_stack([batch[name].to_numpy() for name in columns]).astype(np.float32)
        if not np.isfinite(block).all():
            raise ValueError("Embedding columns must contain finite numeric values.")
        values[start : start + len(block)] = block
        start += len(block)
    values.flush()
    return cells, counts, values


def pca_scores(source, values, mode, samples, seed, batch_size, output):
    if mode == "stored":
        table = pq.read_table(source, columns=["pca1", "pca2", "pca3"])
        scores = np.column_stack([table[f"pca{i}"] for i in (1, 2, 3)]).astype(np.float32)
        if not np.isfinite(scores).all():
            raise ValueError("Stored PCA scores must be finite.")
        return scores, {"method": "stored pca1/pca2/pca3", "fit_rows": None}
    sample_count = min(len(values), samples)
    if sample_count < 3:
        raise ValueError("PCA fitting requires at least three source rows.")
    indices = np.sort(np.random.default_rng(seed).choice(len(values), sample_count, replace=False))
    pca = PCA(n_components=3, svd_solver="randomized", random_state=seed).fit(values[indices])
    scores = np.empty((len(values), 3), np.float32)
    for start in range(0, len(values), batch_size):
        scores[start : start + batch_size] = pca.transform(values[start : start + batch_size])
    np.savez(
        output / "pca-basis.npz",
        mean=pca.mean_,
        components=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        sample_indices=indices,
    )
    return scores, {
        "method": "PCA of raw selected embeddings; no per-channel standardization",
        "fit_rows": sample_count,
        "seed": seed,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def save_level(output, level, cells, counts, values, scores, contrast, stretch, batch_size):
    unit = open_memmap(
        output / f"r{level}-vectors.npy", mode="w+", dtype=np.float32, shape=values.shape
    )
    normalize_to(values, unit, batch_size)
    unit.flush()
    if contrast is None:
        contrast = neighbor_contrast(cells, unit, batch_size)
    positions = np.array([h3.cell_to_latlng(int(cell)) for cell in cells], np.float32)
    rgb = (
        np.clip((scores - stretch[0]) / np.maximum(stretch[1] - stretch[0], 1e-12), 0, 1) * 255
    ).astype(np.uint8)
    np.savez(
        output / f"r{level}-meta.npz",
        cells=cells,
        count=counts,
        positions=positions,
        rgb=rgb,
        contrast=contrast.astype(np.float32),
    )
    finite = contrast[np.isfinite(contrast)]
    quantiles = np.quantile(finite, [0, 0.5, 0.95, 0.99, 1]).tolist() if len(finite) else [0] * 5
    print(f"r{level}: {len(cells):,} cells, {values.shape[1]} dimensions", flush=True)
    return {
        "cells": len(cells),
        "contrast_quantiles": quantiles,
        "isolated_cells": int((~np.isfinite(contrast)).sum()),
    }, contrast


def build(
    source, output, dims=None, pca_mode="fit", samples=20000, seed=0, batch_size=2048, label=None
):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new cache directory.")
    if batch_size < 1 or samples < 3:
        raise ValueError("Use a positive batch size and at least three PCA samples.")
    columns = embedding_columns(pq.ParquetFile(source).schema_arrow.names, dims)
    if dims and pca_mode == "stored":
        raise ValueError(
            "Use --pca fit when selecting dimensions; stored PCA may describe other channels."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hex-build-", dir=output.parent) as temporary:
        work = Path(temporary)
        cache = work / "cache"
        cache.mkdir()
        cells, counts, values = source_arrays(source, columns, work / "raw.npy", batch_size)
        scores, pca_info = pca_scores(source, values, pca_mode, samples, seed, batch_size, cache)
        stretch = np.quantile(scores, [0.02, 0.98], axis=0)
        info, contrast = save_level(
            cache, 5, cells, counts, values, scores, None, stretch, batch_size
        )
        levels = {5: info}
        for level in (4, 3):
            parents = np.array([h3.cell_to_parent(int(cell), level) for cell in cells], np.uint64)
            unique, inverse = np.unique(parents, return_inverse=True)
            average = open_memmap(
                work / f"r{level}-raw.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(unique), len(columns)),
            )
            weights = aggregate_vectors(values, counts, inverse, average)
            parent_scores = np.empty((len(unique), 3), np.float32)
            aggregate_vectors(scores, counts, inverse, parent_scores)
            local_change = aggregate_contrast(contrast, counts, inverse)
            levels[level], _ = save_level(
                cache,
                level,
                unique,
                weights,
                average,
                parent_scores,
                local_change,
                stretch,
                batch_size,
            )
        with source.open("rb") as handle:
            checksum = hashlib.file_digest(handle, "sha256").hexdigest()
        metadata = {
            "format_version": 1,
            "source": source.name,
            "sha256": checksum,
            "label": label or (f"Dimensions {dims}" if dims else source.stem),
            "dimensions": len(columns),
            "embedding_columns": columns,
            "source_cells": len(cells),
            "source_resolution": 5,
            "pca": pca_info,
            "pca_stretch": stretch.tolist(),
            "levels": levels,
            "overview_method": "Source-count-weighted mean of raw embedding vectors",
            "contrast_method": "Mean cosine distance to available r5 H3 neighbors; count-weighted in overviews",
            "similarity_method": "Cosine similarity of L2-normalized selected embeddings",
        }
        (cache / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        cache.rename(output)
    print(f"Cache ready: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Local H3 r5 Parquet")
    parser.add_argument("--output", type=Path, required=True, help="New cache directory")
    parser.add_argument(
        "--dims", help="Half-open dimension range, e.g. 64:128; default: all emb_* columns"
    )
    parser.add_argument("--pca", choices=("fit", "stored"), default="fit")
    parser.add_argument("--pca-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--label", help="Representation label displayed in the explorer")
    args = parser.parse_args()
    build(
        args.input,
        args.output,
        args.dims,
        args.pca,
        args.pca_samples,
        args.seed,
        args.batch_size,
        args.label,
    )


if __name__ == "__main__":
    main()
