"""Generalized Tikhonov probes for MIND embeddings.

Solve ||Xw - y||^2 + alpha * w.T @ diag(g) @ w by rescaling features
with g**(-1/2). The bias has g=1. Profiles use prefix truncation,
chunk-geometric penalties, or power penalties.

``selected`` maximizes the outer-fold mean score over profiles and alphas.
``nested`` selects them by random inner CV; ``nested_spatial`` uses spatial
block groups within the training pool. Ties select the first grid entry.
"""

import numpy as np
import torch
from torch import Tensor

from mind.eval.probe import RIDGE_ALPHAS, _random_folds

MRL_BOUNDARIES = (64, 128, 256, 512, 1024, 2048)  # trained nested widths of the released trunk
TRUNC_WIDTHS = (16, 32, 64, 128, 256, 512, 1024, 2048, 3072)
CHUNK_BETAS = (3.0, 10.0, 100.0, 1000.0)
POW_EXPONENTS = (0.5, 1.0, 2.0, 4.0)


def build_profiles(
    d: int,
    widths: tuple[int, ...] = TRUNC_WIDTHS,
    betas: tuple[float, ...] = CHUNK_BETAS,
    pows: tuple[float, ...] = POW_EXPONENTS,
    boundaries: tuple[int, ...] = MRL_BOUNDARIES,
) -> list[tuple[str, int | np.ndarray]]:
    """Return (name, truncation width or penalty vector) entries in tie-break order."""
    profiles: list[tuple[str, int | np.ndarray]] = [
        (f"ridge_w{w}", int(w)) for w in widths if w <= d
    ]
    chunk_of = np.searchsorted(np.asarray(boundaries), np.arange(d), side="right")
    profiles.extend((f"tik_chunk_b{beta:g}", beta ** chunk_of.astype(np.float64)) for beta in betas)
    profiles.extend((f"tik_pow_p{p:g}", (np.arange(d, dtype=np.float64) + 1.0) ** p) for p in pows)
    return profiles


def build_families(profiles: list[tuple[str, int | np.ndarray]], d: int) -> dict[str, list[int]]:
    """Group profile indices for CV selection, including full-width ridge in each family."""
    names = [n for n, _ in profiles]
    ridge_full = names.index(f"ridge_w{d}")
    trunc = [i for i, n in enumerate(names) if n.startswith("ridge_w")]
    chunk = [ridge_full] + [i for i, n in enumerate(names) if n.startswith("tik_chunk_")]
    pow_ = [ridge_full] + [i for i, n in enumerate(names) if n.startswith("tik_pow_")]
    return {
        "cv_width": trunc,
        "cv_chunk": chunk,
        "cv_pow": pow_,
        "cv_all": list(range(len(profiles))),
    }


def path_scores(
    feats: Tensor,
    targets: Tensor,
    tinfo: list[tuple[str, str, slice, Tensor | None]],
    train_idx: Tensor,
    test_idx: Tensor,
    profiles: list[tuple[str, int | np.ndarray]],
    alphas: tuple[float, ...],
    dev: torch.device,
) -> np.ndarray:
    """Return scores [profiles, alphas, tasks] for one train/test split.

    Share one standardized float64 Gram matrix across profiles.
    """
    x_tr, x_te = feats[train_idx], feats[test_idx]
    mean, std = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
    x_tr, x_te = (x_tr - mean) / std, (x_te - mean) / std
    x_tr = torch.cat([x_tr, torch.ones(x_tr.shape[0], 1, device=dev)], dim=1).double()
    x_te = torch.cat([x_te, torch.ones(x_te.shape[0], 1, device=dev)], dim=1).double()
    y_tr = targets[train_idx].double()
    gram = x_tr.T @ x_tr  # [D+1, D+1]
    rhs = x_tr.T @ y_tr  # [D+1, K]
    d = feats.shape[1]
    out = np.empty((len(profiles), len(alphas), len(tinfo)), dtype=np.float64)
    for pi, (_, spec) in enumerate(profiles):
        if isinstance(spec, int):  # truncation: sub-block of the shared Gram (+ bias column)
            idx = torch.cat([torch.arange(spec, device=dev), torch.tensor([d], device=dev)])
            g_p, rhs_p, xte_p = gram[idx][:, idx], rhs[idx], x_te[:, idx]
        else:  # graduated: diagonal rescale by s = g^{-1/2} (bias unscaled)
            s = torch.ones(d + 1, dtype=torch.float64, device=dev)
            s[:d] = torch.as_tensor(spec, device=dev) ** -0.5
            g_p, rhs_p, xte_p = s[:, None] * gram * s[None, :], s[:, None] * rhs, x_te * s
        evals, evecs = torch.linalg.eigh(g_p)
        proj = evecs.T @ rhs_p
        xte_v = xte_p @ evecs
        for ai, a in enumerate(alphas):
            pred = xte_v @ (proj / (evals[:, None] + a))
            for ti, (_, task_type, sl, cidx) in enumerate(tinfo):
                if task_type == "regression":
                    y_te = targets[test_idx, sl.start]
                    ss_res = ((y_te - pred[:, sl.start]) ** 2).sum()
                    ss_tot = ((y_te - y_te.mean()) ** 2).sum().clamp_min(1e-12)
                    out[pi, ai, ti] = float(1.0 - ss_res / ss_tot)
                else:
                    assert cidx is not None  # classification tinfo always carries class indices
                    out[pi, ai, ti] = float(
                        (pred[:, sl].argmax(1) == cidx[test_idx]).float().mean()
                    )
    return out


def _family_choice(grid: np.ndarray, prof_idx: list[int], n_alphas: int) -> np.ndarray:
    """Return each task's (profile index, alpha index) as a [T, 2] array. First max wins."""
    sub = grid[prof_idx]  # [Pf, A, T]
    flat = sub.reshape(-1, sub.shape[2]).argmax(axis=0)
    p_local, a_i = np.divmod(flat, n_alphas)
    return np.stack([np.asarray(prof_idx)[p_local], a_i], axis=1)


def run_group(
    feats: Tensor,
    targets: Tensor,
    tinfo: list[tuple[str, str, slice, Tensor | None]],
    outer_folds: list[Tensor],
    profiles: list[tuple[str, int | np.ndarray]],
    families: dict[str, list[int]],
    dev: torch.device,
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
    seed: int = 0,
    folds: int = 5,
    spatial_inner: bool = False,
    verbose: bool = False,
) -> tuple[dict[tuple[str, str], np.ndarray], list[dict]]:
    """Return {(config, rule): per-task scores} and per-fold profile/alpha choices."""
    outer = [f for f in outer_folds if f.numel() > 0]
    nf, n_a, n_t = len(outer), len(alphas), len(tinfo)
    all_fams = {name: [i] for i, (name, _) in enumerate(profiles)} | families
    outer_path = np.empty((nf, len(profiles), n_a, n_t))
    inner_rules: dict[str, np.ndarray] = {"nested": np.empty_like(outer_path)}
    if spatial_inner:
        inner_rules["nested_spatial"] = np.empty_like(outer_path)
    for f, te in enumerate(outer):
        tr = torch.sort(torch.cat([outer[j] for j in range(nf) if j != f])).values
        outer_path[f] = path_scores(feats, targets, tinfo, tr, te, profiles, alphas, dev)
        inner = _random_folds(tr, folds, seed, dev)
        inner_rules["nested"][f] = np.mean(
            [
                path_scores(
                    feats,
                    targets,
                    tinfo,
                    torch.cat([inner[j] for j in range(folds) if j != i]),
                    inner[i],
                    profiles,
                    alphas,
                    dev,
                )
                for i in range(folds)
            ],
            axis=0,
        )
        if spatial_inner:
            groups = [outer[j] for j in range(nf) if j != f]
            inner_rules["nested_spatial"][f] = np.mean(
                [
                    path_scores(
                        feats,
                        targets,
                        tinfo,
                        torch.sort(
                            torch.cat([groups[j] for j in range(len(groups)) if j != k])
                        ).values,
                        groups[k],
                        profiles,
                        alphas,
                        dev,
                    )
                    for k in range(len(groups))
                ],
                axis=0,
            )
        if verbose:
            print(f"    outer fold {f + 1}/{nf} done", flush=True)

    results: dict[tuple[str, str], np.ndarray] = {}
    choices: list[dict] = []
    fold_mean = outer_path.mean(axis=0)  # [P, A, T]
    for fam_name, prof_idx in all_fams.items():
        results[(fam_name, "selected")] = fold_mean[prof_idx].reshape(-1, n_t).max(axis=0)
        for rule, sel_grid in inner_rules.items():
            per_fold = np.empty((nf, n_t))
            for f in range(nf):
                chosen = _family_choice(sel_grid[f], prof_idx, n_a)  # [T, 2]
                for t in range(n_t):
                    p_i, a_i = int(chosen[t, 0]), int(chosen[t, 1])
                    per_fold[f, t] = outer_path[f, p_i, a_i, t]
                    choices.append(
                        {
                            "task": tinfo[t][0],
                            "config": fam_name,
                            "rule": rule,
                            "fold": f,
                            "profile": profiles[p_i][0],
                            "alpha": alphas[a_i],
                        }
                    )
            results[(fam_name, rule)] = per_fold.mean(axis=0)
    return results, choices
