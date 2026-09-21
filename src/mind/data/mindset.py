"""Load the released MINDSET GeoParquet tables for coordinate distillation."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import Tensor

from mind.data.quantization import AEF_NODATA, dequantize

_TEACHERS = {"climplicit": "clim", "geoclip": "geo", "sinr": "sinr"}


def _batches(files: list[pq.ParquetFile], columns: list[str]) -> Iterator[pa.RecordBatch]:
    for file in files:
        for batch in file.iter_batches(batch_size=65536, columns=columns):
            if len(batch):
                yield batch


def _vectors(batch: pa.RecordBatch, name: str) -> Tensor:
    column = batch.column(name)
    flat = column.flatten()
    if column.null_count or flat.null_count:
        raise ValueError(f"MINDSET column {name!r} contains null targets")
    values = flat.to_numpy(zero_copy_only=False).copy()
    return torch.from_numpy(values).reshape(len(batch), -1)


def load_mindset(
    directory: str | Path,
) -> tuple[Tensor, Tensor, tuple[int, ...], dict[str, Tensor]]:
    """Join MINDSET teachers and annual AEF vectors by ``point_id``.

    Teachers remain on CPU. AEF vectors are decoded and normalized per year,
    then averaged. Each point must have one vector per recorded year.
    """
    teacher_files, aef_files = [], []
    for path in sorted(Path(directory).rglob("*.parquet")):
        file = pq.ParquetFile(path)
        columns = set(file.schema_arrow.names)
        if {"point_id", "bbox", *_TEACHERS} <= columns:
            teacher_files.append(file)
        elif {"point_id", "year", "aef"} <= columns:
            aef_files.append(file)
    if not teacher_files or not aef_files:
        raise ValueError("Expected MINDSET teacher and annual AEF GeoParquet tables or shards")

    n = sum(file.metadata.num_rows for file in teacher_files)
    if not n:
        raise ValueError("MINDSET contains no training points")
    point_ids = np.empty(n, dtype=np.int64)
    coords = torch.empty((n, 2), dtype=torch.float32)
    targets: dict[str, Tensor] = {}
    start = 0
    for batch in _batches(teacher_files, ["point_id", "bbox", *_TEACHERS]):
        stop = start + len(batch)
        ids = batch.column("point_id")
        if ids.null_count or not pa.types.is_integer(ids.type):
            raise ValueError("MINDSET point_id must contain non-null integers")
        point_ids[start:stop] = ids.to_numpy(zero_copy_only=False)
        bbox = batch.column("bbox")
        latlon = np.column_stack([bbox.field("ymin").to_numpy(), bbox.field("xmin").to_numpy()])
        if not np.isfinite(latlon).all():
            raise ValueError("MINDSET coordinates must be finite")
        coords[start:stop] = torch.from_numpy(latlon)
        for column, key in _TEACHERS.items():
            values = _vectors(batch, column)
            if not values.is_floating_point() or not torch.isfinite(values).all():
                raise ValueError(f"MINDSET {column} targets must be finite floating-point values")
            if key not in targets:
                targets[key] = torch.empty((n, values.shape[1]), dtype=torch.float16)
            if values.shape[1] != targets[key].shape[1]:
                raise ValueError(f"MINDSET {column} dimensions differ between shards")
            targets[key][start:stop] = values
        start = stop

    order = np.argsort(point_ids)
    sorted_ids = point_ids[order]
    if np.any(sorted_ids[1:] == sorted_ids[:-1]):
        raise ValueError("MINDSET teacher point_id values must be unique")
    aef_sum = None
    seen: dict[int, np.ndarray] = {}
    for batch in _batches(aef_files, ["point_id", "year", "aef"]):
        id_column = batch.column("point_id")
        if id_column.null_count or not pa.types.is_integer(id_column.type):
            raise ValueError("MINDSET AEF point_id must contain non-null integers")
        ids = id_column.to_numpy(zero_copy_only=False)
        positions = np.searchsorted(sorted_ids, ids)
        if np.any(positions == n) or not np.array_equal(sorted_ids[positions], ids):
            raise ValueError("MINDSET AEF point_id is missing from the teacher table")
        rows = order[positions]
        years = batch.column("year").to_numpy(zero_copy_only=False)
        if not np.issubdtype(years.dtype, np.integer):
            raise ValueError("MINDSET year must contain non-null integers")
        for year in np.unique(years):
            year_rows = rows[years == year]
            if int(year) not in seen:
                seen[int(year)] = np.zeros(n, dtype=bool)
            present = seen[int(year)]
            if present[year_rows].any() or len(np.unique(year_rows)) != len(year_rows):
                raise ValueError("MINDSET contains a duplicate point_id/year AEF target")
            present[year_rows] = True
        raw = _vectors(batch, "aef")
        if raw.dtype != torch.int8 or torch.any(raw == AEF_NODATA):
            raise ValueError("MINDSET AEF targets must be native int8 vectors without nodata")
        # Retain the pixel loader's float16 rounding before averaging years.
        annual = F.normalize(dequantize(raw), dim=-1).half().float()
        if aef_sum is None:
            aef_sum = torch.zeros((n, annual.shape[1]), dtype=torch.float32)
        aef_sum.index_add_(0, torch.from_numpy(rows), annual)
    if aef_sum is None or any(not present.all() for present in seen.values()):
        raise ValueError("MINDSET is missing AEF years for one or more training points")
    aef_sum /= len(seen)
    return coords, aef_sum, tuple(sorted(seen)), targets
