"""Aggregate held-out prediction errors by distance to the nearest training point.

Regression skill is 1 - bin MSE / variance of all held-out targets.
"""

import argparse
import csv
from itertools import pairwise
from pathlib import Path

import numpy as np

from mind.eval.probe import ridge_fold_predictions, spatial_fold_ids

OUT = "results/distance_error.csv"
BIN_EDGES_KM = (0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf)
CELLS = (0.0, 2.0, 10.0)
EARTH_R_KM = 6371.0


def nn_train_km(lat, lon, fold: np.ndarray) -> np.ndarray:
    """Return each held-out point's nearest-training great-circle distance in km."""
    from sklearn.neighbors import BallTree

    rad = np.radians(np.stack([lat, lon], axis=1))
    out = np.full(len(lat), np.nan)
    for f in np.unique(fold):
        te = fold == f
        if te.all() or te.sum() == 0:
            continue
        tree = BallTree(rad[~te], metric="haversine")
        dist, _ = tree.query(rad[te], k=1)
        out[te] = dist[:, 0] * EARTH_R_KM
    return out


def binned(dist: np.ndarray, err2: np.ndarray, ok: np.ndarray, denom: float) -> list[tuple]:
    rows = []
    for lo, hi in pairwise(BIN_EDGES_KM):
        m = ok & (dist >= lo) & (dist < hi)
        n = int(m.sum())
        rows.append((lo, hi, n, float(1.0 - err2[m].mean() / denom) if n >= 30 else None))
    return rows


def run_benchmark(bench, device: str) -> list[dict]:
    from scripts.paper.blocksize_sweep import feature_sets
    from scripts.paper.white_compare import family_of

    feats = feature_sets(bench, device, force=False)
    rows = []
    for cell in CELLS:
        fold_all = (
            np.arange(len(bench.lat)) % 5
            if cell == 0.0
            else spatial_fold_ids(bench.lat, bench.lon, 5, cell, 0)
        )
        for task, y in bench.tasks.items():
            for label, (x, width) in feats.items():
                res = ridge_fold_predictions(
                    x,
                    y,
                    bench.task_type,
                    folds=5,
                    seed=0,
                    device=device,
                    fold_assign=fold_all,
                )
                sub = res["row"]
                dist = nn_train_km(bench.lat[sub], bench.lon[sub], fold_all[sub])
                if bench.task_type == "regression":
                    err2 = (res["truth"] - res["pred"]) ** 2
                    denom = float(np.var(res["truth"]))
                else:
                    err2 = (res["truth"] != res["pred"]).astype(np.float64)
                    denom = 1.0  # Classification skill is accuracy.
                ok = np.isfinite(err2) & np.isfinite(dist)
                if denom <= 0:
                    continue
                for lo, hi, n, skill in binned(dist, err2, ok, denom):
                    rows.append(
                        {
                            "benchmark": bench.name,
                            "family": family_of(bench.name),
                            "task": task,
                            "task_type": bench.task_type,
                            "encoder": label,
                            "width": width,
                            "cell_deg": cell,
                            "dist_lo_km": lo,
                            "dist_hi_km": None if np.isinf(hi) else hi,
                            "n": n,
                            "skill": None if skill is None else round(skill, 5),
                        }
                    )
            narrow = [r for r in rows if r["encoder"] == "mind_64" and r["cell_deg"] == cell]
            wide = [r for r in rows if r["encoder"] == "mind_3072" and r["cell_deg"] == cell]
            cross = next(
                (
                    f"{n['dist_lo_km']:g}-{n['dist_hi_km']}"
                    for n, w in zip(narrow[-9:], wide[-9:], strict=False)
                    if n["skill"] is not None and w["skill"] is not None and n["skill"] > w["skill"]
                ),
                "none",
            )
            print(
                f"{bench.name:<24} {task:<20} cell={cell:5.2f} narrow-beats-wide from {cross} km",
                flush=True,
            )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--out", default=OUT)
    a = p.parse_args()

    from scripts.paper.run_eval import gather_benchmarks

    rows = []
    for bench in gather_benchmarks(
        only=set(a.only) if a.only else None, pdfm_split="extrapolation"
    ):
        rows.extend(run_benchmark(bench, a.device))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "benchmark",
        "family",
        "task",
        "task_type",
        "encoder",
        "width",
        "cell_deg",
        "dist_lo_km",
        "dist_hi_km",
        "n",
        "skill",
    ]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
