"""Frozen-encoder probes with standardized features."""

import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

# Ridge penalties at half-decade intervals, selected within each training fold.
RIDGE_ALPHAS = tuple(float(10.0**e) for e in np.arange(-4.0, 6.5, 0.5))


def _standardize_train_test(x_tr: Tensor, x_te: Tensor, thresh: float = 1e-4) -> tuple:
    """Standardize with training statistics and zero near-constant channels."""
    mean = x_tr.mean(0, keepdim=True)
    std = x_tr.std(0, keepdim=True)
    keep = std > thresh
    sd = torch.where(keep, std, torch.ones_like(std))
    return ((x_tr - mean) / sd) * keep, ((x_te - mean) / sd) * keep


def _centered_ridge_predict(
    x_tr: Tensor, y_tr: Tensor, x_te: Tensor, alpha: float, dev: torch.device
) -> Tensor:
    """Ridge with an unpenalized intercept: center on the training split, solve, restore the mean."""
    x_tr, x_te, y_tr = x_tr.double(), x_te.double(), y_tr.double()
    x_mean, y_mean = x_tr.mean(0, keepdim=True), y_tr.mean(0, keepdim=True)
    xc = x_tr - x_mean
    eye = torch.eye(xc.shape[1], device=dev, dtype=torch.float64)
    weight = torch.linalg.solve(xc.T @ xc + alpha * eye, xc.T @ (y_tr - y_mean))
    return (x_te - x_mean) @ weight + y_mean


def _ridge_eval(
    feats: Tensor,
    targets: Tensor,
    class_idx: Tensor | None,
    train_idx: Tensor,
    test_idx: Tensor,
    alpha: float,
    task_type: str,
    dev: torch.device,
    standardize: bool = True,
) -> float:
    """Fit ridge on ``train_idx`` and score ``test_idx`` with R^2 or accuracy.

    The intercept is unpenalized; centering uses only the training split.
    """
    x_tr, x_te = feats[train_idx], feats[test_idx]
    if standardize:
        x_tr, x_te = _standardize_train_test(x_tr, x_te)
    pred = _centered_ridge_predict(x_tr, targets[train_idx], x_te, alpha, dev)
    if task_type == "regression":
        y_te = targets[test_idx]
        ss_res = ((y_te - pred) ** 2).sum()
        ss_tot = ((y_te - y_te.mean()) ** 2).sum().clamp_min(1e-12)
        return float(1.0 - ss_res / ss_tot)
    assert class_idx is not None
    return float((pred.argmax(1) == class_idx[test_idx]).float().mean())


def _ridge_eval_alpha_path(
    feats: Tensor,
    targets: Tensor,
    class_idx: Tensor | None,
    train_idx: Tensor,
    test_idx: Tensor,
    alphas: tuple[float, ...],
    task_type: str,
    dev: torch.device,
    standardize: bool = True,
) -> list[float]:
    """Score all ridge penalties using one eigendecomposition of the Gram matrix."""
    x_tr, x_te = feats[train_idx], feats[test_idx]
    if standardize:
        x_tr, x_te = _standardize_train_test(x_tr, x_te)
    y_tr = targets[train_idx].double()
    x_tr, x_te = x_tr.double(), x_te.double()
    x_mean, y_mean = x_tr.mean(0, keepdim=True), y_tr.mean(0, keepdim=True)
    x_tr, x_te = x_tr - x_mean, x_te - x_mean
    gram = x_tr.T @ x_tr
    rhs = x_tr.T @ (y_tr - y_mean)
    evals, evecs = torch.linalg.eigh(gram)
    proj = evecs.T @ rhs
    x_te_v = x_te @ evecs
    out = []
    for a in alphas:
        pred = x_te_v @ (proj / (evals[:, None] + a)) + y_mean
        if task_type == "regression":
            y_te = targets[test_idx]
            ss_res = ((y_te - pred) ** 2).sum()
            ss_tot = ((y_te - y_te.mean()) ** 2).sum().clamp_min(1e-12)
            out.append(float(1.0 - ss_res / ss_tot))
        else:
            assert class_idx is not None
            out.append(float((pred.argmax(1) == class_idx[test_idx]).float().mean()))
    return out


def spatial_fold_ids(
    lat: np.ndarray, lon: np.ndarray, folds: int = 5, cell_deg: float = 10.0, seed: int = 0
) -> np.ndarray:
    """Assign whole latitude/longitude grid cells to folds; ``cell_deg`` is in degrees."""
    cell = np.floor(np.asarray(lat) / cell_deg).astype(np.int64) * 100003 + np.floor(
        np.asarray(lon) / cell_deg
    ).astype(np.int64)
    uniq = np.unique(cell)
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold_of = {int(c): int(order[i] % folds) for i, c in enumerate(uniq)}
    return np.array([fold_of[int(c)] for c in cell], dtype=np.int64)


def _random_folds(pool: Tensor, folds: int, seed: int, dev: torch.device) -> list[Tensor]:
    """Partition ``pool`` into deterministic random folds."""
    g = torch.Generator().manual_seed(seed)
    perm = pool[torch.randperm(pool.shape[0], generator=g).to(dev)]
    return [perm[i::folds] for i in range(folds)]


def _cv_best_alpha(
    feats: Tensor,
    targets: Tensor,
    class_idx: Tensor | None,
    pool: Tensor,
    folds: int,
    alphas: tuple[float, ...],
    task_type: str,
    dev: torch.device,
    seed: int,
    standardize: bool = True,
    fold_index_list: list | None = None,
    fast_alpha_path: bool = False,
) -> tuple[float, float]:
    """Return the penalty with the best mean CV score over ``pool``.

    Use ``fold_index_list`` when supplied; otherwise draw random folds.
    ``fast_alpha_path`` reuses one eigendecomposition per fold.
    """
    if fold_index_list is None:
        fold_ids = _random_folds(pool, folds, seed, dev)
    else:
        fold_ids = [f for f in fold_index_list if f.numel() > 0]
    nf = len(fold_ids)
    train_of = [torch.cat([fold_ids[j] for j in range(nf) if j != f]) for f in range(nf)]
    if fast_alpha_path:
        per_fold = [
            _ridge_eval_alpha_path(
                feats,
                targets,
                class_idx,
                train_of[f],
                fold_ids[f],
                alphas,
                task_type,
                dev,
                standardize,
            )
            for f in range(nf)
        ]
    else:
        per_fold = [
            [
                _ridge_eval(
                    feats,
                    targets,
                    class_idx,
                    train_of[f],
                    fold_ids[f],
                    a,
                    task_type,
                    dev,
                    standardize,
                )
                for a in alphas
            ]
            for f in range(nf)
        ]
    means = np.mean(np.asarray(per_fold, dtype=np.float64), axis=0)
    best_i = int(np.argmax(means))  # ties select the first alpha
    best_alpha, best_score = alphas[best_i], float(means[best_i])
    if len(alphas) > 1 and best_alpha in (alphas[0], alphas[-1]):
        warnings.warn(
            f"ridge alpha selected at grid edge ({best_alpha:g}); widen RIDGE_ALPHAS -- the probe may "
            "be under/over-regularized (esp. high-dim features)",
            stacklevel=2,
        )
    return best_alpha, best_score


def torch_probe_score(
    features: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    folds: int = 5,
    seed: int = 0,
    device: str = "cuda",
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
    test_mask: np.ndarray | None = None,
    standardize: bool = True,
    fold_assign: np.ndarray | None = None,
    fast_alpha_path: bool = False,
) -> float:
    """Mean ridge R^2 or one-hot ridge accuracy with nested penalty selection.

    Split precedence: ``test_mask``, ``fold_assign``, then random folds. Penalties
    are selected within each training split. Spatial inner folds require at least
    three nonempty outer groups; otherwise inner folds are random.
    """
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    if task_type == "regression":
        valid = np.isfinite(labels)
    else:
        valid = np.array([v is not None and (isinstance(v, str) or np.isfinite(v)) for v in labels])
    valid = valid & np.isfinite(features).all(axis=1)
    feats = torch.as_tensor(features[valid], dtype=torch.float32, device=dev)
    class_idx = None
    if task_type == "regression":
        targets = torch.as_tensor(
            labels[valid].astype(np.float64), dtype=torch.float32, device=dev
        )[:, None]
    else:
        _, inverse = np.unique(labels[valid], return_inverse=True)
        class_idx = torch.as_tensor(inverse, device=dev)
        targets = torch.nn.functional.one_hot(class_idx).float()

    all_idx = torch.arange(feats.shape[0], device=dev)
    if test_mask is not None:
        is_test = torch.as_tensor(np.asarray(test_mask)[valid], device=dev, dtype=torch.bool)
        train_pool, test_idx = all_idx[~is_test], all_idx[is_test]
        best_alpha, _ = _cv_best_alpha(
            feats,
            targets,
            class_idx,
            train_pool,
            folds,
            alphas,
            task_type,
            dev,
            seed,
            standardize,
            fast_alpha_path=fast_alpha_path,
        )
        return _ridge_eval(
            feats, targets, class_idx, train_pool, test_idx, best_alpha, task_type, dev, standardize
        )
    fil = None
    if fold_assign is not None:
        fa = torch.as_tensor(np.asarray(fold_assign)[valid], device=dev)
        fil = [all_idx[fa == f] for f in torch.unique(fa)]
    outer = fil if fil is not None else _random_folds(all_idx, folds, seed, dev)
    outer = [f for f in outer if f.numel() > 0]
    scores = []
    for f, test_idx in enumerate(outer):
        # Spatial inner CV needs at least two remaining groups.
        inner_folds = (
            [outer[j] for j in range(len(outer)) if j != f]
            if fil is not None and len(outer) >= 3
            else None
        )
        # Preserve row order before drawing random inner folds.
        train_pool = torch.sort(torch.cat([outer[j] for j in range(len(outer)) if j != f])).values
        best_alpha, _ = _cv_best_alpha(
            feats,
            targets,
            class_idx,
            train_pool,
            folds,
            alphas,
            task_type,
            dev,
            seed,
            standardize,
            fold_index_list=inner_folds,
            fast_alpha_path=fast_alpha_path,
        )
        scores.append(
            _ridge_eval(
                feats,
                targets,
                class_idx,
                train_pool,
                test_idx,
                best_alpha,
                task_type,
                dev,
                standardize,
            )
        )
    return float(np.mean(scores))


def ridge_fold_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    folds: int = 5,
    seed: int = 0,
    device: str = "cuda",
    alphas: tuple[float, ...] = RIDGE_ALPHAS,
    standardize: bool = True,
    fold_assign: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return out-of-fold ``pred``, ``truth``, ``fold``, and original ``row`` indices.

    Rows with nonfinite labels or features are dropped. Each penalty is selected
    by random-fold CV within the corresponding training pool.
    """
    dev = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    if task_type == "regression":
        valid = np.isfinite(labels)
    else:
        valid = np.array([v is not None and (isinstance(v, str) or np.isfinite(v)) for v in labels])
    valid = valid & np.isfinite(features).all(axis=1)
    rows = np.flatnonzero(valid)
    feats = torch.as_tensor(features[valid], dtype=torch.float32, device=dev)
    if task_type == "regression":
        targets = torch.as_tensor(
            labels[valid].astype(np.float64), dtype=torch.float32, device=dev
        )[:, None]
        class_idx = None
        truth = labels[valid].astype(np.float64)
    else:
        _, inverse = np.unique(labels[valid], return_inverse=True)
        class_idx = torch.as_tensor(inverse, device=dev)
        targets = torch.nn.functional.one_hot(class_idx).float()
        truth = inverse.astype(np.int64)
    all_idx = torch.arange(feats.shape[0], device=dev)
    if fold_assign is None:
        g = torch.Generator().manual_seed(seed)
        perm = all_idx[torch.randperm(all_idx.shape[0], generator=g).to(dev)]
        fold_ids = [perm[i::folds] for i in range(folds)]
    else:
        fa = torch.as_tensor(np.asarray(fold_assign)[valid], device=dev)
        fold_ids = [all_idx[fa == f] for f in torch.unique(fa)]
    fold_ids = [f for f in fold_ids if f.numel() > 0]
    pred = np.full(len(rows), np.nan)
    fold_of = np.full(len(rows), -1, dtype=np.int64)
    for f, te in enumerate(fold_ids):
        tr = torch.cat([fold_ids[j] for j in range(len(fold_ids)) if j != f])
        best_alpha, _ = _cv_best_alpha(
            feats,
            targets,
            class_idx,
            tr,
            folds,
            alphas,
            task_type,
            dev,
            seed,
            standardize,
            fast_alpha_path=True,
        )
        x_tr, x_te = feats[tr], feats[te]
        if standardize:
            x_tr, x_te = _standardize_train_test(x_tr, x_te)
        out = _centered_ridge_predict(x_tr, targets[tr], x_te, best_alpha, dev)
        te_np = te.cpu().numpy()
        pred[te_np] = (
            out.squeeze(1).cpu().numpy()
            if task_type == "regression"
            else out.argmax(1).cpu().numpy()
        )
        fold_of[te_np] = f
    return {"pred": pred, "truth": truth, "fold": fold_of, "row": rows}


def ridge_cv_r2(features: np.ndarray, labels: np.ndarray, folds: int = 5, seed: int = 0) -> float:
    """Mean held-out R^2 of a ridge probe (standardized features, RidgeCV alpha per fold)."""
    valid = np.isfinite(labels)
    features, labels = features[valid], labels[valid]
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: list[float] = []
    for train_idx, test_idx in kf.split(features):
        scaler = StandardScaler().fit(features[train_idx])
        model = RidgeCV(alphas=RIDGE_ALPHAS)
        model.fit(scaler.transform(features[train_idx]), labels[train_idx])
        pred = model.predict(scaler.transform(features[test_idx]))
        scores.append(r2_score(labels[test_idx], pred))
    return float(np.mean(scores))


def logistic_cv_acc(
    features: np.ndarray, labels: np.ndarray, folds: int = 5, seed: int = 0
) -> float:
    """Mean held-out accuracy of a logistic-regression probe (standardized features)."""
    valid = np.array([isinstance(v, str) or np.isfinite(v) for v in labels])
    features, labels = features[valid], labels[valid]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: list[float] = []
    for train_idx, test_idx in skf.split(features, labels):
        scaler = StandardScaler().fit(features[train_idx])
        model = LogisticRegression(max_iter=200, C=1.0)
        model.fit(scaler.transform(features[train_idx]), labels[train_idx])
        pred = model.predict(scaler.transform(features[test_idx]))
        scores.append(accuracy_score(labels[test_idx], pred))
    return float(np.mean(scores))


def logistic_csweep_acc(
    features: np.ndarray,
    labels: np.ndarray,
    folds: int = 5,
    seed: int = 0,
    device: str = "cuda",
    c_range: tuple[float, float, int] = (-4.0, 4.0, 9),
) -> float:
    """Mean logistic accuracy, selecting C by test accuracy within each fold."""
    from mind.eval.logreg import LBFGSLogReg

    valid = np.array([isinstance(v, str) or np.isfinite(v) for v in labels])
    features, labels = features[valid], labels[valid]
    _, y = np.unique(labels, return_inverse=True)
    c_values = 10.0 ** np.linspace(*c_range)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: list[float] = []
    for train_idx, test_idx in skf.split(features, y):
        scaler = StandardScaler().fit(features[train_idx])
        x_tr = torch.as_tensor(scaler.transform(features[train_idx]), dtype=torch.float32)
        x_te = torch.as_tensor(scaler.transform(features[test_idx]), dtype=torch.float32)
        y_tr = torch.as_tensor(y[train_idx], dtype=torch.long)
        y_te = y[test_idx]
        best = -1.0
        for c in c_values:
            model = LBFGSLogReg(c=float(c), max_iter=500, device=device, seed=seed).fit(x_tr, y_tr)
            acc = float((model.predict(x_te) == y_te).mean())
            best = max(best, acc)
        scores.append(best)
    return float(np.mean(scores))


def knn_cv_acc(
    features: np.ndarray,
    labels: np.ndarray,
    folds: int = 5,
    seed: int = 0,
    k: int = 10,
    device: str = "cuda",
) -> float:
    """Mean held-out k-NN accuracy on standardized features, using faissknn."""
    from faissknn import FaissKNNClassifier

    valid = np.array([isinstance(v, str) or np.isfinite(v) for v in labels])
    features, labels = features[valid], labels[valid]
    classes, y = np.unique(labels, return_inverse=True)
    dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: list[float] = []
    for train_idx, test_idx in skf.split(features, y):
        scaler = StandardScaler().fit(features[train_idx])
        x_tr = scaler.transform(features[train_idx]).astype(np.float32)
        x_te = scaler.transform(features[test_idx]).astype(np.float32)
        model = FaissKNNClassifier(n_neighbors=k, n_classes=len(classes), device=dev, metric="l2")
        model.fit(x_tr, y[train_idx])
        pred = model.predict(x_te)
        scores.append(float((np.asarray(pred).ravel() == y[test_idx]).mean()))
    return float(np.mean(scores))


def mlp_probe_score(
    features: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    folds: int = 5,
    seed: int = 0,
    device: str = "cuda",
    test_mask: np.ndarray | None = None,
    hidden: int = 128,
    dropout: float = 0.5,
    epochs: int = 100,
    batch: int = 1024,
    lr: float = 1e-3,
    wd: float = 1e-5,
    fold_assign: np.ndarray | None = None,
) -> float:
    """Mean R^2 or accuracy with a SatCLIP/OSMGraphCLIP-style two-layer MLP.

    Split precedence: ``test_mask``, ``fold_assign``, then random folds.
    """
    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    valid = np.array([isinstance(v, str) or np.isfinite(v) for v in labels])
    feats, lab = features[valid], labels[valid]
    is_reg = task_type == "regression"
    if is_reg:
        y_raw = lab.astype(np.float32)
        n_out = 1
    else:
        _, y_all = np.unique(lab, return_inverse=True)
        n_out = int(y_all.max()) + 1
    if test_mask is not None:
        tm = test_mask[valid]
        splits = [(np.where(~tm)[0], np.where(tm)[0])]
    elif fold_assign is not None:
        fa = np.asarray(fold_assign)[valid]
        splits = [(np.where(fa != f)[0], np.where(fa == f)[0]) for f in np.unique(fa)]
    else:
        g = np.random.default_rng(seed)
        perm = g.permutation(len(feats))
        splits = [(np.setdiff1d(perm, perm[i::folds]), perm[i::folds]) for i in range(folds)]

    def run(tr_i: np.ndarray, te_i: np.ndarray) -> float:
        torch.manual_seed(seed)
        mean, std = feats[tr_i].mean(0), feats[tr_i].std(0)
        # Zero near-constant channels before standardizing.
        keep = std > 1e-4
        sd = np.where(keep, std, 1.0)

        def stdz(a: np.ndarray) -> torch.Tensor:
            z = (a - mean) / sd
            z[:, ~keep] = 0.0
            return torch.as_tensor(np.clip(z, -10.0, 10.0), dtype=torch.float32, device=dev)

        xtr, xte = stdz(feats[tr_i]), stdz(feats[te_i])
        if is_reg:
            ymu, ysd = (
                float(y_raw[tr_i].mean()),
                float(y_raw[tr_i].std() + 1e-8),
            )
            y_use = (y_raw - ymu) / ysd
        else:
            y_use = y_all
        # Reserve 10% of the training rows for early stopping.
        ntr = len(tr_i)
        nval = max(1, ntr // 10)
        ytr_all = torch.as_tensor(y_use[tr_i], device=dev)
        yt_all = ytr_all.float().unsqueeze(1) if is_reg else ytr_all.long()
        # Shuffle before splitting, since input rows may be spatially ordered.
        gtr = torch.Generator(device="cpu").manual_seed(seed)
        rp = torch.randperm(ntr, generator=gtr).to(dev)
        xtr_s, yt_all_s = xtr[rp], yt_all[rp]
        xv, yv = xtr_s[:nval], yt_all_s[:nval]
        xt, yt = xtr_s[nval:], yt_all_s[nval:]
        net = nn.Sequential(
            nn.Linear(feats.shape[1], hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        ).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        lossf = nn.MSELoss() if is_reg else nn.CrossEntropyLoss()
        best_val, best_state, patience, since = float("inf"), None, 12, 0
        for _ in range(epochs):
            net.train()
            for b in range(0, len(xt), batch):
                idx = slice(b, b + batch)
                opt.zero_grad(set_to_none=True)
                lossf(net(xt[idx]), yt[idx]).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(lossf(net(xv), yv))
            if vl < best_val - 1e-5:
                best_val, since = vl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                since += 1
                if since >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            pred = net(xte)
            if is_reg:
                p = pred.squeeze(1).cpu().numpy()
                yt_te = y_use[te_i]
                ss = ((yt_te - p) ** 2).sum()
                tot = ((yt_te - yt_te.mean()) ** 2).sum() + 1e-12
                return float(1.0 - ss / tot)
            return float((pred.argmax(1).cpu().numpy() == y_use[te_i]).mean())

    return float(np.mean([run(tr, te) for tr, te in splits]))


def evaluate_tasks(
    features: np.ndarray,
    labels: dict[str, np.ndarray],
    task_type: str = "regression",
    folds: int = 5,
    seed: int = 0,
    backend: str = "torch",
    device: str = "cuda",
    test_mask: np.ndarray | None = None,
    standardize: bool = True,
    clf_probe: str = "ridge",
    lat: np.ndarray | None = None,
    lon: np.ndarray | None = None,
    spatial_cv: bool = False,
    cell_deg: float = 10.0,
    fast_alpha_path: bool = False,
) -> dict[str, float]:
    """Return per-task R^2 or accuracy and their mean.

    ``clf_probe`` selects ridge, logistic, k-NN, or MLP. Logistic and k-NN fall
    back to ridge outside random classification folds.
    ``fast_alpha_path`` reuses the ridge eigendecomposition across penalties.
    """
    fold_assign = None
    if spatial_cv and test_mask is None and lat is not None and lon is not None:
        fold_assign = spatial_fold_ids(lat, lon, folds, cell_deg, seed)
    if clf_probe == "mlp":
        per_task = {
            t: mlp_probe_score(
                features,
                y,
                task_type,
                folds,
                seed,
                device=device,
                test_mask=test_mask,
                fold_assign=fold_assign,
            )
            for t, y in labels.items()
        }
        per_task["MEAN"] = float(np.mean(list(per_task.values())))
        return per_task
    cv_clf = (
        task_type == "classification"
        and clf_probe in ("logistic", "knn")
        and test_mask is None
        and fold_assign is None
    )
    if cv_clf:
        if clf_probe == "knn":
            per_task = {
                t: knn_cv_acc(features, y, folds, seed, device=device) for t, y in labels.items()
            }
        else:
            per_task = {
                t: logistic_csweep_acc(features, y, folds, seed, device=device)
                for t, y in labels.items()
            }
    else:
        per_task = {
            task: torch_probe_score(
                features,
                y,
                task_type,
                folds,
                seed,
                device=device,
                test_mask=test_mask,
                standardize=standardize,
                fold_assign=fold_assign,
                fast_alpha_path=fast_alpha_path,
            )
            for task, y in labels.items()
        }
    per_task["MEAN"] = float(np.mean(list(per_task.values())))
    return per_task
