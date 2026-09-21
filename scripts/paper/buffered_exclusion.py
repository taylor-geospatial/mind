"""Evaluate fixed held-out folds with increasing training-point exclusion radii.

Compare each buffer with random removal of the same number of training points.
"""

import argparse
import time
from collections import defaultdict

import numpy as np
import torch

from mind.eval.probe import RIDGE_ALPHAS, _random_folds
from scripts.paper.blocksize_sweep import EARTH_R_KM, MIND_WIDTHS, StreamCSV
from scripts.paper.coord_interp_baseline import knn_predict_multi, random_fold_ids
from scripts.paper.tikhonov_probe import CHUNK_BETAS, _family_choice, build_profiles, path_scores
from scripts.paper.tikhonov_sweep import build_targets, load_trunk, task_groups

RADII_KM = (0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0)
FOLDS = 5
INNER_FOLDS = 5
MIN_TRAIN = 30  # Leave scores empty below this training count.
IDW_K = 50
OUT = "results/buffered_exclusion.csv"
COLS = [
    "benchmark",
    "family",
    "task",
    "task_type",
    "n",
    "n_valid",
    "encoder",
    "config",
    "radius_km",
    "mode",
    "fold_seed",
    "n_folds",
    "n_train_kept",
    "frac_kept",
    "median_nn_km",
    "score",
]


def unit_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    rlat, rlon = np.radians(lat), np.radians(lon)
    return np.stack(
        [np.cos(rlat) * np.cos(rlon), np.cos(rlat) * np.sin(rlon), np.sin(rlat)], axis=1
    )


def chord_to_km(d: np.ndarray) -> np.ndarray:
    return 2.0 * EARTH_R_KM * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))


def scored_configs(
    feats: torch.Tensor,
    targets: torch.Tensor,
    tinfo: list,
    kept: torch.Tensor,
    te: torch.Tensor,
    profiles: list,
    families: dict[str, list[int]],
    dev: torch.device,
) -> dict[str, np.ndarray]:
    """Score each profile and family after inner CV on the retained training points."""
    inner = _random_folds(kept, INNER_FOLDS, 0, dev)
    inner_grid = np.mean(
        [
            path_scores(
                feats,
                targets,
                tinfo,
                torch.sort(torch.cat([inner[j] for j in range(INNER_FOLDS) if j != i])).values,
                inner[i],
                profiles,
                RIDGE_ALPHAS,
                dev,
            )
            for i in range(INNER_FOLDS)
        ],
        axis=0,
    )
    outer = path_scores(feats, targets, tinfo, kept, te, profiles, RIDGE_ALPHAS, dev)
    n_t = outer.shape[2]
    out = {}
    for pi, (name, _) in enumerate(profiles):
        a_star = inner_grid[pi].argmax(axis=0)  # [T]
        out[name] = outer[pi, a_star, np.arange(n_t)]
    for fam, prof_idx in families.items():
        chosen = _family_choice(inner_grid, prof_idx, len(RIDGE_ALPHAS))  # [T, 2]
        out[fam] = outer[chosen[:, 0], chosen[:, 1], np.arange(n_t)]
    return out


def idw_scores(
    lat_g,
    lon_g,
    y_by_task: dict,
    tinfo: list,
    kept_pos: np.ndarray,
    te_pos: np.ndarray,
    n_jobs: int,
) -> dict[str, float]:
    """Score Coordinate IDW on one split using float targets or integer class codes."""
    out = {}
    for task, task_type, _, _ in tinfo:
        y = y_by_task[task]
        n_classes = int(y.max()) + 1 if task_type == "classification" else 0
        pred = knn_predict_multi(
            lat_g[kept_pos],
            lon_g[kept_pos],
            y[kept_pos],
            lat_g[te_pos],
            lon_g[te_pos],
            (IDW_K,),
            task_type,
            n_classes,
            n_jobs=n_jobs,
        )[IDW_K]
        y_te = y[te_pos]
        if task_type == "regression":
            ss_tot = max(float(((y_te - y_te.mean()) ** 2).sum()), 1e-12)
            out[task] = 1.0 - float(((y_te - pred) ** 2).sum()) / ss_tot
        else:
            out[task] = float((pred == y_te).mean())
    return out


def feature_sets(bench, device: str, force: bool) -> dict:
    """Return {encoder: (features, profiles, families)} for MIND and baselines."""
    from pathlib import Path

    from mind.eval.embedders import (
        embed_climplicit,
        embed_raw,
        embed_satclip,
        embed_sinr,
    )
    from scripts.paper.blocksize_sweep import CKPT, embed_cached

    lat, lon = bench.lat, bench.lon
    sc_weights = Path(__file__).resolve().parents[2] / "weights" / "satclip-location-encoder.pt"
    mind = load_trunk(bench, "mind", CKPT, device, force)
    p_mind = build_profiles(mind.shape[1], widths=MIND_WIDTHS, betas=CHUNK_BETAS, pows=())
    names = [n for n, _ in p_mind]
    ridge_full = names.index(f"ridge_w{mind.shape[1]}")
    fams = {
        "cv_width": [i for i, n in enumerate(names) if n.startswith("ridge_w")],
        "cv_chunk": [ridge_full] + [i for i, n in enumerate(names) if n.startswith("tik_chunk_")],
    }
    baselines = {
        "geoclip": lambda: load_trunk(bench, "geoclip", CKPT, device, force),
        "satclip_l40": lambda: embed_cached(
            "satclip_l40",
            bench.name,
            lat,
            lon,
            lambda: embed_satclip(sc_weights, lat, lon, device=device),
            force,
        ),
        "climplicit": lambda: embed_cached(
            "climplicit",
            bench.name,
            lat,
            lon,
            lambda: embed_climplicit(lat, lon, device=device),
            force,
        ),
        "sinr": lambda: embed_cached(
            "sinr", bench.name, lat, lon, lambda: embed_sinr(lat, lon, device=device), force
        ),
        "raw_coords": lambda: embed_raw(lat, lon),
    }
    out = {"mind": (mind, p_mind, fams)}
    for label, fn in baselines.items():
        emb = fn()
        out[label] = (
            emb,
            build_profiles(emb.shape[1], widths=(emb.shape[1],), betas=(), pows=()),
            {},
        )
    return out


def sweep_benchmark(
    bench, device: str, fold_seed: int, radii, sink, force: bool, idw_jobs: int
) -> None:
    from scipy.spatial import KDTree

    from scripts.paper.white_compare import family_of

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    lat, lon = bench.lat, bench.lon
    encoders = feature_sets(bench, device, force)
    fold_full = random_fold_ids(len(lat), FOLDS, fold_seed)
    xyz = unit_xyz(lat, lon)
    for valid, tasks in task_groups(bench, encoders["mind"][0]):
        t0 = time.monotonic()
        targets, tinfo = build_targets(bench, tasks, valid, dev)
        feats_t = {
            name: torch.as_tensor(x[valid], dtype=torch.float32, device=dev)
            for name, (x, _, _) in encoders.items()
        }
        fold_g, xyz_g = fold_full[valid], xyz[valid]
        lat_g, lon_g = lat[valid], lon[valid]
        y_by_task = {}
        for task, task_type, _, _ in tinfo:
            y = bench.tasks[task][valid]
            if task_type == "regression":
                y_by_task[task] = y.astype(np.float64)
            else:
                y_by_task[task] = np.unique(y, return_inverse=True)[1].astype(np.int64)
        # Reuse each fold's nearest-held-out distances across radii.
        folds_info = []
        for f in range(FOLDS):
            te_pos, tr_pos = np.flatnonzero(fold_g == f), np.flatnonzero(fold_g != f)
            if len(te_pos) == 0 or len(tr_pos) == 0:
                continue
            d_km = chord_to_km(KDTree(xyz_g[te_pos]).query(xyz_g[tr_pos])[0])
            folds_info.append((te_pos, tr_pos, d_km))
        rows = []
        for r in radii:
            for mode in ("buffer",) if r == 0.0 else ("buffer", "random_drop"):
                kept_list = []
                for fi, (_te_pos, tr_pos, d_km) in enumerate(folds_info):
                    # Radius zero keeps all points; positive radii exclude coincident points.
                    excl = d_km < r
                    if mode == "buffer":
                        kept = tr_pos[~excl]
                    else:
                        rng = np.random.default_rng([fold_seed, fi, int(r)])
                        kept = rng.choice(tr_pos, len(tr_pos) - int(excl.sum()), replace=False)
                        kept.sort()
                    kept_list.append(kept)
                n_kept = [len(k) for k in kept_list]
                frac = float(
                    np.mean(
                        [
                            len(k) / len(tr)
                            for k, (_, tr, _) in zip(kept_list, folds_info, strict=True)
                        ]
                    )
                )
                meds = [
                    float(np.median(chord_to_km(KDTree(xyz_g[k]).query(xyz_g[te])[0])))
                    for k, (te, _, _) in zip(kept_list, folds_info, strict=True)
                    if len(k) > 0
                ]
                med_nn = float(np.mean(meds)) if meds else float("nan")
                ok = bool(folds_info) and min(n_kept) >= MIN_TRAIN
                acc: dict[tuple[str, str], list] = defaultdict(list)
                if ok:
                    for (te_pos, _, _), kept in zip(folds_info, kept_list, strict=True):
                        te_t = torch.as_tensor(te_pos, device=dev)
                        kept_t = torch.as_tensor(kept, device=dev)
                        for enc, (_, profiles, families) in encoders.items():
                            for config, sc in scored_configs(
                                feats_t[enc], targets, tinfo, kept_t, te_t, profiles, families, dev
                            ).items():
                                acc[(enc, config)].append(sc)
                        idw = idw_scores(lat_g, lon_g, y_by_task, tinfo, kept, te_pos, idw_jobs)
                        acc[("coord_idw", f"idw_k{IDW_K}")].append(
                            np.array([idw[t[0]] for t in tinfo])
                        )
                base = {
                    "benchmark": bench.name,
                    "family": family_of(bench.name),
                    "task_type": bench.task_type,
                    "n": len(lat),
                    "n_valid": int(valid.sum()),
                    "radius_km": r,
                    "mode": mode,
                    "fold_seed": fold_seed,
                    "n_folds": len(folds_info),
                    "n_train_kept": int(np.mean(n_kept)) if n_kept else 0,
                    "frac_kept": round(frac, 4) if n_kept else 0.0,
                    "median_nn_km": round(med_nn, 2) if np.isfinite(med_nn) else "",
                }
                configs = [
                    (enc, config)
                    for enc, (_, profiles, families) in encoders.items()
                    for config in [n for n, _ in profiles] + list(families)
                ] + [("coord_idw", f"idw_k{IDW_K}")]
                for enc, config in configs:
                    per_fold = acc.get((enc, config))
                    for t, (task, _, _, _) in enumerate(tinfo):
                        score = (
                            round(float(np.nanmean([sc[t] for sc in per_fold])), 5)
                            if ok and per_fold
                            else ""
                        )
                        rows.append(
                            {**base, "task": task, "encoder": enc, "config": config, "score": score}
                        )
        sink.write(rows)
        print(
            f"{bench.name:<30} group[{len(tasks)} task(s)] n_valid={int(valid.sum()):>7d} "
            f"{time.monotonic() - t0:7.1f}s",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", nargs="*", default=None, help="benchmark loader keys to restrict to")
    p.add_argument("--radii", type=float, nargs="*", default=None, help="override RADII_KM")
    p.add_argument(
        "--fold-seed", type=int, default=0, help="seed of the fixed test-fold assignment"
    )
    p.add_argument("--force-embed", action="store_true")
    p.add_argument("--idw-jobs", type=int, default=4)
    p.add_argument("--out", default=OUT)
    a = p.parse_args()

    from scripts.paper.run_eval import gather_benchmarks

    radii = tuple(a.radii) if a.radii else RADII_KM
    sink = StreamCSV(a.out, COLS)
    for bench in gather_benchmarks(
        only=set(a.only) if a.only else None, pdfm_split="extrapolation"
    ):
        sweep_benchmark(bench, a.device, a.fold_seed, radii, sink, a.force_embed, a.idw_jobs)
    sink.close()


if __name__ == "__main__":
    main()
