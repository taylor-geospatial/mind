"""Evaluate inverse-distance-weighted coordinate k-NN across spatial block sizes."""

import argparse
import csv
from pathlib import Path

import numpy as np

from mind.eval.probe import spatial_fold_ids

OUT = "results/coord_interp.csv"
CELL_LADDER = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
KS = (1, 5, 20)
EARTH_R_KM = 6371.0


def knn_predict_multi(
    lat_tr, lon_tr, y_tr, lat_te, lon_te, ks, task_type: str, n_classes: int, n_jobs: int = 1
) -> dict:
    """Predict with inverse great-circle distance weights for each k, sharing one query."""
    from sklearn.neighbors import BallTree

    tree = BallTree(np.radians(np.stack([lat_tr, lon_tr], axis=1)), metric="haversine")
    kmax = min(max(ks), len(lat_tr))
    q = np.radians(np.stack([lat_te, lon_te], axis=1))
    if n_jobs != 1 and len(q) > 5000:
        from joblib import Parallel, delayed

        chunks = [c for c in np.array_split(np.arange(len(q)), max(n_jobs, 1)) if len(c)]
        res = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(tree.query)(q[c], k=kmax) for c in chunks
        )
        dist = np.concatenate([r[0] for r in res])
        idx = np.concatenate([r[1] for r in res])
    else:
        dist, idx = tree.query(q, k=kmax)
    dist_km = dist * EARTH_R_KM
    # A 10 m floor bounds weights for coincident coordinates.
    inv = 1.0 / np.maximum(dist_km, 0.01)
    neigh = y_tr[idx]

    out = {}
    for k in ks:
        k_eff = min(k, kmax)
        w = inv[:, :k_eff]
        w = w / w.sum(axis=1, keepdims=True)
        nb = neigh[:, :k_eff]
        if task_type == "regression":
            out[k] = (w * nb).sum(axis=1)
            continue
        votes = np.zeros((len(lat_te), n_classes))
        for j in range(k_eff):
            np.add.at(votes, (np.arange(len(lat_te)), nb[:, j]), w[:, j])
        out[k] = votes.argmax(axis=1)
    return out


def random_fold_ids(n: int, folds: int, fold_seed: int) -> np.ndarray:
    """Assign rows to folds using the requested seed."""
    if fold_seed == 0:
        return np.arange(n) % folds
    import torch

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(fold_seed)).numpy()
    fold = np.empty(n, dtype=np.int64)
    fold[perm] = np.arange(n) % folds
    return fold


def score_one(
    bench,
    task: str,
    y_raw: np.ndarray,
    cell: float,
    folds: int = 5,
    fold_seed: int = 0,
    ks: tuple = KS,
    n_jobs: int = 1,
) -> dict:
    lat, lon = bench.lat, bench.lon
    if bench.task_type == "regression":
        valid = np.isfinite(y_raw)
        y = y_raw.astype(np.float64)
        n_classes = 0
    else:
        valid = np.array([v is not None and (isinstance(v, str) or np.isfinite(v)) for v in y_raw])
        _, inverse = np.unique(y_raw[valid], return_inverse=True)
        y = np.zeros(len(y_raw), dtype=np.int64)
        y[valid] = inverse
        n_classes = int(inverse.max()) + 1
    la, lo, yv = lat[valid], lon[valid], y[valid]
    fold = (
        random_fold_ids(len(la), folds, fold_seed)
        if cell == 0.0
        else spatial_fold_ids(la, lo, folds, cell, fold_seed)
    )
    scores = {k: [] for k in ks}
    for f in np.unique(fold):
        te = fold == f
        if te.all() or te.sum() == 0:
            continue
        preds = knn_predict_multi(
            la[~te], lo[~te], yv[~te], la[te], lo[te], ks, bench.task_type, n_classes, n_jobs
        )
        yt = yv[te]
        for k, pred in preds.items():
            if bench.task_type == "regression":
                ss_tot = ((yt - yt.mean()) ** 2).sum()
                scores[k].append(
                    1.0 - ((yt - pred) ** 2).sum() / max(ss_tot, 1e-12) if ss_tot > 0 else np.nan
                )
            else:
                scores[k].append(float((pred == yt).mean()))
    return {k: (float(np.nanmean(v)) if v else float("nan")) for k, v in scores.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--cells", type=float, nargs="*", default=None)
    p.add_argument("--out", default=OUT)
    p.add_argument("--fold-seed", type=int, default=0)
    p.add_argument(
        "--ks",
        type=int,
        nargs="*",
        default=list(KS),
        help="neighbor counts to sweep; the appendix ablation varies this while everything else is "
        "held fixed",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="threads for the BallTree query (the only hot loop); 1 reproduces the committed CSVs",
    )
    a = p.parse_args()

    from scripts.paper.run_eval import gather_benchmarks
    from scripts.paper.white_compare import family_of

    ladder = tuple(a.cells) if a.cells else CELL_LADDER
    rows = []
    for bench in gather_benchmarks(
        only=set(a.only) if a.only else None, pdfm_split="extrapolation"
    ):
        for task, y in bench.tasks.items():
            per_cell = {}
            for cell in ladder:
                per_cell[cell] = score_one(
                    bench, task, y, cell, fold_seed=a.fold_seed, ks=tuple(a.ks), n_jobs=a.n_jobs
                )
                for k, s in per_cell[cell].items():
                    rows.append(
                        {
                            "benchmark": bench.name,
                            "family": family_of(bench.name),
                            "task": task,
                            "task_type": bench.task_type,
                            "cell_deg": cell,
                            "k": k,
                            "score": None if np.isnan(s) else round(s, 5),
                        }
                    )
            kref = 5 if 5 in a.ks else a.ks[0]
            print(
                f"{bench.name:<26} {task:<22} coord-IDW k={kref} random "
                f"{per_cell[ladder[0]][kref]:+.3f} -> cell{ladder[-1]:g} "
                f"{per_cell[ladder[-1]][kref]:+.3f}",
                flush=True,
            )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cols = ["benchmark", "family", "task", "task_type", "cell_deg", "k", "score"]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
