"""Build Hilbert-sorted MINDSET GeoParquet tables for teacher and AEF embeddings."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

from mind.eval.embedders import embed_climplicit, embed_geoclip, embed_sinr

TEACHERS = [
    ("climplicit", embed_climplicit, 1024),
    ("geoclip", embed_geoclip, 512),
    ("sinr", embed_sinr, 256),
]
GEO_META = {
    "version": "1.1.0",
    "primary_column": "geometry",
    "columns": {
        "geometry": {
            "encoding": "WKB",
            "geometry_types": ["Point"],
            "crs": None,
            "covering": {
                "bbox": {
                    "xmin": ["bbox", "xmin"],
                    "ymin": ["bbox", "ymin"],
                    "xmax": ["bbox", "xmax"],
                    "ymax": ["bbox", "ymax"],
                }
            },
        }
    },
}


def load_points(data, limit=None):
    """Load float32 lat/lon coordinates, int8 targets [N, years, 64], and decoding metadata."""
    d = Path(data)
    man = json.loads((d / "manifest.json").read_text())
    coords = np.concatenate([np.load(d / f"{s}.coords.npy") for s in man["shards"]]).astype(
        np.float32
    )
    targets = np.concatenate([np.load(d / f"{s}.targets.npy") for s in man["shards"]])
    if limit:
        coords, targets = coords[:limit], targets[:limit]
    return coords, targets, man["years"], man.get("dequant", "signed_square")


def hilbert_order(lat, lon, bits=16):
    """Sort coordinates along a Hilbert curve on a 2^bits grid."""
    n = (1 << bits) - 1
    x = np.clip(((lon + 180.0) / 360.0 * n), 0, n).astype(np.int64)
    y = np.clip(((lat + 90.0) / 180.0 * n), 0, n).astype(np.int64)
    d = np.zeros(len(x), np.int64)
    s = 1 << (bits - 1)
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64)
        ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        swap = ry == 0
        flip = swap & (rx == 1)
        x[flip] = s - 1 - x[flip]
        y[flip] = s - 1 - y[flip]
        xs = x[swap].copy()
        x[swap] = y[swap]
        y[swap] = xs
        s >>= 1
    return np.argsort(d, kind="stable")


def geom_cols(lat, lon):
    """Create WKB points and bounding boxes for spatial filtering."""
    wkb = shapely.to_wkb(shapely.points(lon, lat))
    bbox = pa.StructArray.from_arrays(
        [
            pa.array(lon, pa.float64()),
            pa.array(lat, pa.float64()),
            pa.array(lon, pa.float64()),
            pa.array(lat, pa.float64()),
        ],
        names=["xmin", "ymin", "xmax", "ymax"],
    )
    return pa.array(wkb, pa.binary()), bbox


def fsl(arr, np_dt, pa_t):
    a = np.ascontiguousarray(arr, np_dt)
    return pa.FixedSizeListArray.from_arrays(pa.array(a.reshape(-1), pa_t), a.shape[1])


def writer(path, fields, extra=None):
    md = {b"geo": json.dumps(GEO_META).encode()}
    if extra:
        md.update(extra)
    schema = pa.schema(fields).with_metadata(md)
    return pq.ParquetWriter(path, schema, compression="zstd", compression_level=5), schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/aef-pixels")
    ap.add_argument("--out-dir", default="data/mindset")
    ap.add_argument("--chunk", type=int, default=262144)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    coords, targets, years, dq = load_points(args.data, args.limit or None)
    N, Y = len(coords), targets.shape[1]
    order = hilbert_order(coords[:, 0], coords[:, 1])
    print(f"mindset points: N={N:,}, Y={Y} years {years}, dequant={dq}; Hilbert-sorted", flush=True)

    tfields = [
        ("point_id", pa.int64()),
        ("geometry", pa.binary()),
        (
            "bbox",
            pa.struct(
                [
                    ("xmin", pa.float64()),
                    ("ymin", pa.float64()),
                    ("xmax", pa.float64()),
                    ("ymax", pa.float64()),
                ]
            ),
        ),
    ]
    tfields += [(name, pa.list_(pa.float16(), dim)) for name, _, dim in TEACHERS]
    tw, _ = writer(
        out / "mindset_teachers.parquet",
        tfields,
        {
            b"teacher_dtype": b"float16; raw embedder output -- L2-normalize (and optionally PHI-S standardize) at use, as training does"
        },
    )
    for c0 in range(0, N, args.chunk):
        idx = order[c0 : c0 + args.chunk]
        la, lo = coords[idx, 0].astype(np.float64), coords[idx, 1].astype(np.float64)
        wkb, bbox = geom_cols(la, lo)
        cols = [pa.array(idx.astype(np.int64)), wkb, bbox]
        cols += [fsl(fn(la, lo), np.float16, pa.float16()) for _, fn, _ in TEACHERS]
        tw.write_table(pa.table(cols, schema=tw.schema))
        print(f"  teachers rows {c0}-{c0 + len(idx)}/{N}", flush=True)
    tw.close()

    afields = [
        ("point_id", pa.int64()),
        ("year", pa.int16()),
        ("geometry", pa.binary()),
        (
            "bbox",
            pa.struct(
                [
                    ("xmin", pa.float64()),
                    ("ymin", pa.float64()),
                    ("xmax", pa.float64()),
                    ("ymax", pa.float64()),
                ]
            ),
        ),
        ("aef", pa.list_(pa.int8(), 64)),
    ]
    aw, _ = writer(
        out / "mindset_aef.parquet",
        afields,
        {
            b"aef_dtype": b"int8 (native, "
            + dq.encode()
            + b"); dequant f = sign(x)*(|x|/127.5)**2, nodata=-128; then L2-normalize at use"
        },
    )
    yrs = np.asarray(years, np.int16)
    for c0 in range(0, N, args.chunk):
        idx = order[c0 : c0 + args.chunk]
        la, lo = coords[idx, 0].astype(np.float64), coords[idx, 1].astype(np.float64)
        tgt = targets[idx]
        wkb, bbox = geom_cols(la, lo)
        for yi in range(Y):
            cols = [
                pa.array(idx.astype(np.int64)),
                pa.array(np.full(len(idx), yrs[yi], np.int16)),
                wkb,
                bbox,
                fsl(tgt[:, yi, :], np.int8, pa.int8()),
            ]
            aw.write_table(pa.table(cols, schema=aw.schema))
        print(f"  aef rows {c0}-{c0 + len(idx)}/{N} x{Y}yr", flush=True)
    aw.close()

    sz = subprocess.run(["du", "-sh", str(out)], capture_output=True, text=True).stdout.split()[0]
    print(f"\nwrote {out} ({sz}): teachers {N:,} rows, aef {N * Y:,} rows", flush=True)


if __name__ == "__main__":
    main()
