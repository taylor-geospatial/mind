"""Evaluate prefix widths and Tikhonov penalty profiles on CoordBench.

Tasks with the same valid rows share feature matrices and fold assignments.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from mind.eval.probe import _random_folds, spatial_fold_ids
from scripts.paper.blocksize_sweep import ARCH, CKPT, StreamCSV, embed_cached, median_nn_train_km
from scripts.paper.tikhonov_probe import build_families, build_profiles, run_group

CELLS = (0.0, 2.0, 10.0, 40.0)
OUT = "results/tikhonov_sweep.csv"
CHOICES_OUT = "results/tikhonov_choices.csv"
EMBED_TAG = "mind_resiren3072"  # blocksize_sweep's cache tag for the released trunk
COLS = [
    "benchmark",
    "family",
    "task",
    "task_type",
    "n",
    "n_valid",
    "config",
    "rule",
    "cell_deg",
    "median_nn_train_km",
    "n_folds_nonempty",
    "score",
]
CHOICE_COLS = ["benchmark", "task", "cell_deg", "config", "rule", "fold", "profile", "alpha"]


def task_groups(bench, trunk: np.ndarray) -> list[tuple[np.ndarray, list[str]]]:
    """Group tasks by finite-label and finite-feature masks."""
    finite_rows = np.isfinite(trunk).all(axis=1)
    by_mask: dict[bytes, tuple[np.ndarray, list[str]]] = {}
    for task, y in bench.tasks.items():
        if bench.task_type == "regression":
            valid = np.isfinite(y.astype(np.float64))
        else:
            valid = np.array([v is not None and (isinstance(v, str) or np.isfinite(v)) for v in y])
        valid = valid & finite_rows
        key = np.packbits(valid).tobytes()
        if key in by_mask:
            by_mask[key][1].append(task)
        else:
            by_mask[key] = (valid, [task])
    return list(by_mask.values())


def build_targets(bench, tasks: list[str], valid: np.ndarray, dev: torch.device):
    """Return float32 targets [n_valid, K] and task metadata.

    Use one column per regression target and a one-hot block per classification task.
    """
    cols, tinfo, start = [], [], 0
    for task in tasks:
        y = bench.tasks[task][valid]
        if bench.task_type == "regression":
            t = torch.as_tensor(y.astype(np.float64), dtype=torch.float32, device=dev)[:, None]
            cidx = None
        else:
            _, inverse = np.unique(y, return_inverse=True)
            cidx = torch.as_tensor(inverse, device=dev)
            t = torch.nn.functional.one_hot(cidx).float()
        cols.append(t)
        tinfo.append((task, bench.task_type, slice(start, start + t.shape[1]), cidx))
        start += t.shape[1]
    return torch.cat(cols, dim=1), tinfo


def embed_tag_for(ckpt: str) -> str:
    """Include checkpoint identity in the embedding cache tag."""
    if ckpt == CKPT:
        return EMBED_TAG  # keep the released trunk's existing cache entries valid
    return "mind_resiren3072_" + Path(ckpt).stem


def load_trunk(bench, encoder: str, ckpt: str, device: str, force: bool) -> np.ndarray:
    """Load cached MIND or GeoCLIP features."""
    lat, lon = bench.lat, bench.lon
    if encoder == "mind":
        from mind.eval.embedders import embed_ours

        return embed_cached(
            embed_tag_for(ckpt),
            bench.name,
            lat,
            lon,
            lambda: embed_ours(ckpt, lat, lon, feature="pooled", device=device, **ARCH)[:, :3072],
            force,
        )
    if encoder == "geoclip":
        from mind.eval.embedders import embed_geoclip

        return embed_cached(
            "geoclip", bench.name, lat, lon, lambda: embed_geoclip(lat, lon, device=device), force
        )
    raise ValueError(f"unknown encoder {encoder!r}")


def sweep_benchmark(
    bench,
    device: str,
    cells,
    force: bool,
    sink,
    choice_sink,
    permute: bool = False,
    ckpt: str = CKPT,
    fold_seed: int = 0,
    encoder: str = "mind",
) -> None:
    from scripts.paper.white_compare import family_of

    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    lat, lon = bench.lat, bench.lon
    trunk = load_trunk(bench, encoder, ckpt, device, force)
    if permute:
        # Apply one column permutation across all tasks and folds.
        trunk = trunk[:, np.random.default_rng(0).permutation(trunk.shape[1])]
    profiles = build_profiles(trunk.shape[1])
    families = build_families(profiles, trunk.shape[1])
    groups = task_groups(bench, trunk)
    for cell in cells:
        t0 = time.monotonic()
        fold_full = None if cell == 0.0 else spatial_fold_ids(lat, lon, 5, cell, fold_seed)
        nn_km = median_nn_train_km(lat, lon, fold_full, 5)
        rows, crows = [], []
        for valid, tasks in groups:
            feats = torch.as_tensor(trunk[valid], dtype=torch.float32, device=dev)
            targets, tinfo = build_targets(bench, tasks, valid, dev)
            idx = torch.arange(feats.shape[0], device=dev)
            if fold_full is None:
                outer = _random_folds(idx, 5, fold_seed, dev)
            else:
                fa = torch.as_tensor(fold_full[valid], device=dev)
                outer = [idx[fa == f] for f in torch.unique(fa)]
            outer = [f for f in outer if f.numel() > 0]
            # leave-one-train-block-group-out needs >= 2 groups in every training pool
            spatial_inner = fold_full is not None and len(outer) >= 3
            results, choices = run_group(
                feats,
                targets,
                tinfo,
                outer,
                profiles,
                families,
                dev,
                spatial_inner=spatial_inner,
                verbose=feats.shape[0] > 50_000,
            )
            for (config, rule), scores in results.items():
                for t, (task, task_type, _, _) in enumerate(tinfo):
                    rows.append(
                        {
                            "benchmark": bench.name,
                            "family": family_of(bench.name),
                            "task": task,
                            "task_type": task_type,
                            "n": len(lat),
                            "n_valid": int(feats.shape[0]),
                            "config": config,
                            "rule": rule,
                            "cell_deg": cell,
                            "median_nn_train_km": round(nn_km, 2),
                            "n_folds_nonempty": 0 if fold_full is None else len(outer),
                            "score": round(float(scores[t]), 5),
                        }
                    )
            crows.extend({"benchmark": bench.name, "cell_deg": cell, **c} for c in choices)
        sink.write(rows)
        choice_sink.write(crows)
        by = {(r["config"], r["rule"]): r["score"] for r in rows if r["task"] == rows[0]["task"]}
        d = trunk.shape[1]
        print(
            f"{bench.name:<30} cell={cell:5.2f} nn={nn_km:8.1f}km {time.monotonic() - t0:7.1f}s "
            f"ridge{d}[nested]={by[(f'ridge_w{d}', 'nested')]:+.3f} "
            f"cv_width={by[('cv_width', 'nested')]:+.3f} "
            f"cv_chunk={by[('cv_chunk', 'nested')]:+.3f}",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", nargs="*", default=None, help="benchmark loader keys to restrict to")
    p.add_argument("--cells", type=float, nargs="*", default=None)
    p.add_argument("--force-embed", action="store_true")
    p.add_argument(
        "--permute",
        action="store_true",
        help="fixed seed-0 column permutation of the trunk: the order control (see sweep_benchmark)",
    )
    p.add_argument(
        "--ckpt",
        default=CKPT,
        help="checkpoint to read; the embedding cache tag follows it so a different checkpoint "
        "cannot reuse the released trunk's cached embeddings",
    )
    p.add_argument(
        "--encoder",
        default="mind",
        choices=("mind", "geoclip"),
        help="feature source: the nested checkpoint (default) or the GeoCLIP baseline, the "
        "cross-encoder order control (see load_trunk); --ckpt/--permute only apply to mind",
    )
    p.add_argument(
        "--fold-seed",
        type=int,
        default=0,
        help="seed of the outer fold assignment (block-to-fold permutation for spatial CV, shuffle "
        "for random folds); inner selection folds keep their own fixed seed",
    )
    p.add_argument("--out", default=OUT)
    p.add_argument("--choices-out", default=CHOICES_OUT)
    a = p.parse_args()
    if a.encoder != "mind" and (a.permute or a.ckpt != CKPT):
        p.error("--permute/--ckpt only make sense with --encoder mind")

    from scripts.paper.run_eval import gather_benchmarks

    cells = tuple(a.cells) if a.cells else CELLS
    sink = StreamCSV(a.out, COLS)
    choice_sink = StreamCSV(a.choices_out, CHOICE_COLS)
    for bench in gather_benchmarks(
        only=set(a.only) if a.only else None, pdfm_split="extrapolation"
    ):
        sweep_benchmark(
            bench,
            a.device,
            cells,
            a.force_embed,
            sink,
            choice_sink,
            a.permute,
            a.ckpt,
            a.fold_seed,
            a.encoder,
        )
    sink.close()
    choice_sink.close()


if __name__ == "__main__":
    main()
