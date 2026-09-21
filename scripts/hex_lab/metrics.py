"""Numerical helpers for the hex explorer."""

import h3.api.basic_int as h3
import numpy as np
from scipy.sparse import csr_array


def normalize_to(values, output, batch_size=2048):
    for start in range(0, len(values), batch_size):
        block = values[start : start + batch_size]
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        if not np.isfinite(block).all() or np.any(norms <= 0):
            raise ValueError("Embeddings must be finite and have nonzero norms.")
        output[start : start + batch_size] = block / norms


def neighbor_contrast(cells, vectors, batch_size=2048):
    """Mean cosine distance to present neighbors; NaN for isolated cells."""
    index = {int(cell): i for i, cell in enumerate(cells)}
    result = np.full(len(cells), np.nan, np.float32)
    for start in range(0, len(cells), batch_size):
        chunk = cells[start : start + batch_size]
        neighbors = np.full((len(chunk), 6), -1, np.int32)
        for row, cell in enumerate(chunk):
            adjacent = [index.get(c, -1) for c in h3.grid_disk(int(cell), 1) if c != cell]
            neighbors[row, : len(adjacent)] = adjacent
        valid = neighbors >= 0
        dots = np.einsum(
            "ik,ijk->ij", vectors[start : start + len(chunk)], vectors[np.maximum(neighbors, 0)]
        )
        distance = np.clip(1 - dots, 0, 2) * valid
        np.divide(
            distance.sum(axis=1),
            valid.sum(axis=1),
            out=result[start : start + len(chunk)],
            where=valid.sum(axis=1) > 0,
        )
    return result


def aggregate_vectors(values, counts, inverse, output):
    """Count-weighted mean, processing 64 channels at a time."""
    weights = np.bincount(inverse, weights=counts)
    aggregation = csr_array(
        (counts / weights[inverse], (inverse, np.arange(len(counts)))),
        shape=(len(weights), len(counts)),
    )
    for start in range(0, values.shape[1], 64):
        output[:, start : start + 64] = aggregation @ values[:, start : start + 64]
    return weights


def aggregate_contrast(values, counts, inverse):
    valid = np.isfinite(values)
    totals = np.bincount(inverse, weights=np.where(valid, values * counts, 0))
    weights = np.bincount(inverse, weights=np.where(valid, counts, 0))
    return np.divide(totals, weights, out=np.full_like(totals, np.nan), where=weights > 0)


def sphere(positions):
    lat, lon = np.radians(np.asarray(positions, dtype=np.float64)).T
    return np.column_stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)))


def distances(xyz, query):
    chord = np.linalg.norm(xyz - query, axis=1)
    return 2 * 6371.0088 * np.arcsin(np.minimum(chord / 2, 1))
