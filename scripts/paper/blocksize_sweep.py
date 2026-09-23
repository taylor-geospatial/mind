"""Evaluate MIND prefixes and baseline encoders across spatial block sizes."""

import argparse
import csv
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from mind.eval.probe import evaluate_tasks, spatial_fold_ids

CKPT = os.environ.get("MIND_CHECKPOINT", "weights/mind.safetensors")
ARCH = {
    "encoder_type": "resiren",
}
MIND_WIDTHS = (64, 128, 256, 512, 1024, 2048, 3072)
ARCH_PRESETS = {
    "resiren3072": (ARCH, MIND_WIDTHS),
    # Fixed-width training baselines.
    "resiren64": (ARCH, (64,)),
    "resiren256": (ARCH, (256,)),
}
CELL_LADDER = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)
CACHE = Path(os.environ.get("MIND_EMBED_CACHE", Path.home() / ".cache/mind/embeddings"))
OUT = "results/blocksize_sweep.csv"
EARTH_R_KM = 6371.0


class StreamCSV:
    """Append CSV rows and flush after each batch."""

    def __init__(self, path: str, fieldnames: list[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("w", newline="")
        self.writer = csv.DictWriter(self.fh, fieldnames=fieldnames)
        self.writer.writeheader()
        self.fh.flush()
        self.n = 0

    def write(self, rows: list[dict]) -> None:
        self.writer.writerows(rows)
        self.fh.flush()
        self.n += len(rows)

    def close(self) -> None:
        self.fh.close()
        print(f"\nwrote {self.path} ({self.n} rows)")


ENCODER_SOURCES = (
    Path(__file__).resolve().parents[2] / "src/mind/eval/extra_encoders.py",
    Path(__file__).resolve().parents[2] / "src/mind/eval/embedders.py",
    Path(__file__).resolve().parents[2] / "src/mind/inference.py",
    Path(__file__).resolve().parents[2] / "src/mind/encoders/resiren.py",
    Path(__file__).resolve().parents[2] / "src/mind/encoders/rff.py",
)


@lru_cache(maxsize=1)
def encoder_code_digest() -> str:
    """Hash encoder source files for the embedding cache key."""
    h = hashlib.sha1()
    for src in ENCODER_SOURCES:
        h.update(src.read_bytes() if src.exists() else b"")
    return h.hexdigest()[:8]


def cache_path(tag: str, name: str, lat: np.ndarray, lon: np.ndarray) -> Path:
    """Key cached embeddings by encoder tag, coordinates, and encoder source."""
    digest = hashlib.sha1(
        np.ascontiguousarray(np.stack([lat, lon]).astype(np.float64)).tobytes()
    ).hexdigest()[:16]
    return CACHE / f"{tag}__{name}__{digest}__{encoder_code_digest()}.npy"


def legacy_cache_path(tag: str, name: str, lat: np.ndarray, lon: np.ndarray) -> Path:
    """Locate an older cache entry for the cache-miss message."""
    digest = hashlib.sha1(
        np.ascontiguousarray(np.stack([lat, lon]).astype(np.float64)).tobytes()
    ).hexdigest()[:16]
    return CACHE / f"{tag}__{name}__{digest}.npy"


def embed_cached(tag: str, name: str, lat, lon, fn, force: bool = False) -> np.ndarray:
    path = cache_path(tag, name, lat, lon)
    if path.exists() and not force:
        return np.load(path)
    legacy = legacy_cache_path(tag, name, lat, lon)
    if legacy.exists() and not force:
        print(
            f"  [cache] recomputing {tag}/{name}: legacy entry predates the code digest", flush=True
        )
    emb = np.ascontiguousarray(fn().astype(np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb


def _standardize_teacher(t: np.ndarray) -> np.ndarray:
    """Z-score teacher features across the benchmark, zeroing near-constant channels."""
    sd = t.std(0)
    keep = sd > 1e-4
    return np.where(keep, (t - t.mean(0)) / np.where(keep, sd, 1.0), 0.0)


def feature_sets(
    bench,
    device: str,
    force: bool,
    ckpt: str = CKPT,
    tag: str = "mind",
    with_baselines: bool = True,
    preset: str = "resiren3072",
    baselines_only: bool = False,
    baseline_names: tuple | None = None,
) -> dict:
    """Return {label: (features, width)}, slicing MIND prefixes from one embedding."""
    from mind.eval.embedders import (
        embed_climplicit,
        embed_geoclip,
        embed_ours,
        embed_raw,
        embed_satclip,
        embed_sinr,
        embed_xyz,
    )

    lat, lon = bench.lat, bench.lon
    sc_weights = Path(__file__).resolve().parents[2] / "weights" / "satclip-location-encoder.pt"
    out = {}

    arch, widths = ARCH_PRESETS[preset]
    if not baselines_only:
        trunk = embed_cached(
            f"{tag}_{preset}",
            bench.name,
            lat,
            lon,
            lambda: embed_ours(ckpt, lat, lon, feature="pooled", device=device, **arch)[
                :, : max(widths)
            ],
            force,
        )
        for w in widths:
            out[f"{tag}_{w}"] = (trunk[:, :w], w)

    if not with_baselines:
        return out

    from mind.eval.extra_encoders import (
        embed_csp,
        embed_gair,
        embed_sled,
        embed_taxabind,
        embed_tte,
    )

    baselines = {
        "raw_coords": lambda: embed_raw(lat, lon),
        "xyz_coords": lambda: embed_xyz(lat, lon),
        "satclip_l40": lambda: embed_satclip(sc_weights, lat, lon, device=device),
        "geoclip": lambda: embed_geoclip(lat, lon, device=device),
        "climplicit": lambda: embed_climplicit(lat, lon, device=device),
        "sinr": lambda: embed_sinr(lat, lon, device=device),
        "gair": lambda: embed_gair(lat, lon, device=device),
        "csp_inat": lambda: embed_csp(lat, lon, device=device, variant="csp_inat"),
        "csp_fmow": lambda: embed_csp(lat, lon, device=device, variant="csp_fmow"),
        "tte": lambda: embed_tte(lat, lon, device=device),
        "taxabind": lambda: embed_taxabind(lat, lon, device=device),
        # The paper reports the S1+S2+Landsat SLED checkpoint.
        "sled_s2ls": lambda: embed_sled(lat, lon, device=device, variant="sled_s2ls"),
        "sled_s1s2ls": lambda: embed_sled(lat, lon, device=device, variant="sled_s1s2ls"),
    }
    if baseline_names is not None:
        unknown = set(baseline_names) - baselines.keys()
        if unknown:
            raise ValueError(f"Unknown baseline names: {sorted(unknown)}")
        baselines = {k: v for k, v in baselines.items() if k in baseline_names}
    else:
        baselines = {k: v for k, v in baselines.items() if k != "sled_s2ls"}
    for label, fn in baselines.items():
        emb = embed_cached(label, bench.name, lat, lon, fn, force)
        out[label] = (emb, emb.shape[1])

    # Standardize each teacher before concatenation.
    if {"geoclip", "climplicit", "sinr"} <= baselines.keys():
        teachers = [out["geoclip"][0], out["climplicit"][0], out["sinr"][0]]
        concat = np.concatenate([_standardize_teacher(t) for t in teachers], axis=1).astype(
            np.float32
        )
        out["teacher_concat"] = (concat, concat.shape[1])
    return out


def median_nn_train_km(
    lat, lon, fold: np.ndarray | None, folds: int, max_pairs: int = 4000
) -> float:
    """Average the fold-wise median nearest-training distance in km.

    Sample at most ``max_pairs`` training and test points per fold.
    """
    rng = np.random.default_rng(0)
    rlat, rlon = np.radians(lat), np.radians(lon)
    xyz = EARTH_R_KM * np.stack(
        [np.cos(rlat) * np.cos(rlon), np.cos(rlat) * np.sin(rlon), np.sin(rlat)], axis=1
    )
    n = len(xyz)
    meds = []
    for f in range(folds):
        te = np.arange(n) % folds == f if fold is None else fold == f
        tr_i, te_i = np.flatnonzero(~te), np.flatnonzero(te)
        if len(tr_i) == 0 or len(te_i) == 0:
            continue
        if len(tr_i) > max_pairs:
            tr_i = rng.choice(tr_i, max_pairs, replace=False)
        if len(te_i) > max_pairs:
            te_i = rng.choice(te_i, max_pairs, replace=False)
        a, b = xyz[te_i], xyz[tr_i]
        best = np.empty(len(a))
        for i in range(0, len(a), 512):
            d2 = ((a[i : i + 512, None, :] - b[None, :, :]) ** 2).sum(axis=2)
            best[i : i + 512] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
        meds.append(
            np.median(2.0 * EARTH_R_KM * np.arcsin(np.clip(best / (2.0 * EARTH_R_KM), 0, 1)))
        )
    return float(np.mean(meds)) if meds else float("nan")


def sweep_benchmark(
    bench,
    device: str,
    respect_official: bool,
    force: bool,
    ladder,
    ckpt: str = CKPT,
    tag: str = "mind",
    with_baselines: bool = True,
    preset: str = "resiren3072",
    fold_seed: int = 0,
    baselines_only: bool = False,
    baseline_names: tuple | None = None,
) -> list[dict]:
    from scripts.paper.white_compare import family_of

    feats = feature_sets(
        bench, device, force, ckpt, tag, with_baselines, preset, baselines_only, baseline_names
    )
    test_mask = bench.test_mask if respect_official else None
    rows = []
    for cell in ladder:
        fold = None if cell == 0.0 else spatial_fold_ids(bench.lat, bench.lon, 5, cell, fold_seed)
        nn_km = median_nn_train_km(bench.lat, bench.lon, fold, 5)
        n_blocks = 0 if fold is None else len(np.unique(fold))
        for label, (x, width) in feats.items():
            per_task = evaluate_tasks(
                x,
                bench.tasks,
                task_type=bench.task_type,
                device=device,
                clf_probe="ridge",
                lat=bench.lat,
                lon=bench.lon,
                spatial_cv=cell != 0.0,
                cell_deg=cell if cell != 0.0 else 10.0,
                test_mask=test_mask,
                fast_alpha_path=True,
                seed=fold_seed,
            )
            for task, score in per_task.items():
                if task == "MEAN":
                    continue
                rows.append(
                    {
                        "benchmark": bench.name,
                        "family": family_of(bench.name),
                        "task": task,
                        "task_type": bench.task_type,
                        "n": len(bench.lat),
                        "encoder": label,
                        "width": width,
                        "cell_deg": cell,
                        "median_nn_train_km": round(nn_km, 2),
                        "n_folds_nonempty": n_blocks,
                        "score": round(float(score), 5),
                    }
                )
        means = {
            lab: float(
                np.mean([r["score"] for r in rows if r["cell_deg"] == cell and r["encoder"] == lab])
            )
            for lab in feats
        }
        widths = ARCH_PRESETS[preset][1]
        if baselines_only:
            print(
                f"[baselines] {bench.name:<26} cell={cell:5.2f} nn={nn_km:7.1f}km "
                + " ".join(f"{lab}:{means[lab]:+.3f}" for lab in feats),
                flush=True,
            )
            continue
        best_w = max(widths, key=lambda w: means[f"{tag}_{w}"])
        print(
            f"[{tag}] {bench.name:<26} cell={cell:5.2f} nn={nn_km:7.1f}km "
            f"argmax w={best_w:<5d} "
            f"(narrow:{means[f'{tag}_{min(widths)}']:+.3f} wide:{means[f'{tag}_{max(widths)}']:+.3f})"
            + (
                f" concat:{means['teacher_concat']:+.3f} geoclip:{means['geoclip']:+.3f}"
                if with_baselines
                else ""
            ),
            flush=True,
        )
    return rows


def main() -> None:
    global CACHE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", nargs="*", default=None, help="benchmark loader keys to restrict to")
    p.add_argument("--cells", type=float, nargs="*", default=None, help="override the cell ladder")
    p.add_argument("--respect-official-splits", action="store_true")
    p.add_argument(
        "--ckpts",
        nargs="+",
        default=None,
        help="ckpt_path=tag pairs; defaults to the released MRL trunk. Passing several compares training "
        "objectives on one axis, which is how the width knob is shown to come from nesting.",
    )
    p.add_argument(
        "--no-baselines",
        action="store_true",
        help="skip the pretrained baselines and the teacher concatenation (they do not depend on the "
        "checkpoint, so re-scoring them once per objective is wasted work)",
    )
    p.add_argument(
        "--baselines-only",
        action="store_true",
        help="score only the pretrained baselines and the teacher concatenation, skipping the MIND "
        "trunk; used to replicate the baselines under reseeded fold assignments",
    )
    p.add_argument(
        "--baseline-names",
        nargs="*",
        default=None,
        help="restrict the baseline set to these labels (e.g. gair tte); default runs all "
        "coordinate encoders reported in the paper",
    )
    p.add_argument("--force-embed", action="store_true")
    p.add_argument(
        "--arch-preset",
        default="resiren3072",
        choices=tuple(ARCH_PRESETS),
        help="MIND or the separately trained fixed-width ReSIREN controls",
    )
    p.add_argument(
        "--fold-seed",
        type=int,
        default=0,
        help="seed of the fold assignment (block-to-fold permutation for spatial CV, shuffle for "
        "random folds); sweeping it measures split-assignment variance at fixed weights",
    )
    p.add_argument("--out", default=OUT)
    p.add_argument("--cache-dir", type=Path, default=CACHE)
    a = p.parse_args()
    CACHE = a.cache_dir

    from scripts.paper.run_eval import gather_benchmarks

    ladder = tuple(a.cells) if a.cells else CELL_LADDER
    specs = [spec.partition("=") for spec in (a.ckpts or [f"{CKPT}=mind"])]
    cols = [
        "benchmark",
        "family",
        "task",
        "task_type",
        "n",
        "encoder",
        "width",
        "cell_deg",
        "median_nn_train_km",
        "n_folds_nonempty",
        "score",
    ]
    sink = StreamCSV(a.out, cols)
    for bench in gather_benchmarks(
        only=set(a.only) if a.only else None, pdfm_split="extrapolation"
    ):
        sub = bench
        for ckpt, _, tag in specs:
            sink.write(
                sweep_benchmark(
                    sub,
                    a.device,
                    a.respect_official_splits,
                    a.force_embed,
                    ladder,
                    ckpt,
                    tag or "mind",
                    not a.no_baselines,
                    a.arch_preset,
                    a.fold_seed,
                    a.baselines_only,
                    tuple(a.baseline_names) if a.baseline_names else None,
                )
            )
    sink.close()


if __name__ == "__main__":
    main()
