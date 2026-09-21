"""Measure empirical variograms and characteristic distances of MIND prefixes."""

import argparse
import csv
from pathlib import Path

import numpy as np

from scripts.paper.blocksize_sweep import ARCH, CKPT, MIND_WIDTHS

OUT = "results/variogram.csv"
EARTH_R_KM = 6371.0
# Limit lags to 8,000 km to avoid near-antipodal pairs.
LAGS_KM = np.geomspace(0.01, 8000.0, 48)
SHORT_LAGS_KM = (0.1, 1.0, 10.0, 100.0)


def land_tree():
    """STRtree over the Natural Earth 110m land union, for point-on-land tests."""
    import shapely
    from cartopy.io import shapereader

    reader = shapereader.Reader(
        shapereader.natural_earth(resolution="110m", category="physical", name="land")
    )
    return shapely.STRtree([shapely.union_all(list(reader.geometries()))])


def land_masked_offset(
    lat: np.ndarray,
    lon: np.ndarray,
    dist_km: float,
    rng: np.random.Generator,
    tree,
    max_tries: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Redraw bearings until offsets reach land, up to ``max_tries`` per anchor.

    Return (lat2, lon2, keep), where ``keep`` marks successful offsets.
    """
    import shapely

    la, lo = np.empty_like(lat), np.empty_like(lon)
    keep = np.zeros(len(lat), dtype=bool)
    pending = np.arange(len(lat))
    for _ in range(max_tries):
        pl, pn = geodesic_offset(lat[pending], lon[pending], dist_km, rng)
        pts = shapely.points(pn, pl)
        ok = np.zeros(len(pending), dtype=bool)
        ok[tree.query(pts, predicate="within")[0]] = True
        idx = pending[ok]
        la[idx], lo[idx] = pl[ok], pn[ok]
        keep[idx] = True
        pending = pending[~ok]
        if len(pending) == 0:
            break
    return la, lo, keep


def land_anchors(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample area-weighted land anchors between 56 S and 70 N.

    Latitude is sampled uniformly in sin(lat) before the Natural Earth land filter.
    """
    import shapely
    from cartopy.io import shapereader

    reader = shapereader.Reader(
        shapereader.natural_earth(resolution="110m", category="physical", name="land")
    )
    land = shapely.union_all(list(reader.geometries()))
    tree = shapely.STRtree([land])

    rng = np.random.default_rng(seed)
    s_lo, s_hi = np.sin(np.radians(-56.0)), np.sin(np.radians(70.0))
    lats: list[np.ndarray] = []
    lons: list[np.ndarray] = []
    got = 0
    while got < n:
        cand_lat = np.degrees(np.arcsin(rng.uniform(s_lo, s_hi, n)))
        cand_lon = rng.uniform(-180.0, 180.0, n)
        pts = shapely.points(cand_lon, cand_lat)
        hit = tree.query(pts, predicate="within")[0]  # indices of candidates inside the land union
        keep = np.zeros(n, dtype=bool)
        keep[hit] = True
        lats.append(cand_lat[keep])
        lons.append(cand_lon[keep])
        got += int(keep.sum())
    return np.concatenate(lats)[:n], np.concatenate(lons)[:n]


def geodesic_offset(
    lat: np.ndarray, lon: np.ndarray, dist_km: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Move each anchor ``dist_km`` along a random bearing on a sphere."""
    brg = rng.uniform(0.0, 2.0 * np.pi, size=len(lat))
    d = dist_km / EARTH_R_KM
    p1, l1 = np.radians(lat), np.radians(lon)
    p2 = np.arcsin(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(brg))
    l2 = l1 + np.arctan2(np.sin(brg) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), (np.degrees(l2) + 180.0) % 360.0 - 180.0


def embed(lat, lon, device: str, ckpt: str) -> np.ndarray:
    from mind.eval.embedders import embed_ours

    return embed_ours(ckpt, lat, lon, feature="pooled", device=device, **ARCH)[
        :, : max(MIND_WIDTHS)
    ]


def sill(gamma: np.ndarray) -> float:
    """Estimate the sill as the median semivariance over the largest quartile of lags."""
    tail = gamma[max(1, int(0.75 * len(gamma))) :]
    return float(np.median(tail))


def half_sill_km(lags: np.ndarray, gamma: np.ndarray, frac: float = 0.5) -> float:
    """Log-interpolate the first lag reaching ``frac`` of the sill.

    Return the largest measured lag as a lower bound if the threshold is not reached.
    """
    target = frac * sill(gamma)
    hit = np.flatnonzero(gamma >= target)
    if len(hit) == 0:
        return float(lags[-1])
    i = int(hit[0])
    if i == 0:
        return float(lags[0])
    g0, g1 = gamma[i - 1], gamma[i]
    if g1 == g0:
        return float(lags[i])
    frac = (target - g0) / (g1 - g0)
    return float(np.exp(np.log(lags[i - 1]) + frac * (np.log(lags[i]) - np.log(lags[i - 1]))))


def run(
    n_anchors: int,
    pairs_per_lag: int,
    device: str,
    seed: int,
    ckpt: str,
    tag: str,
    land_mask: bool = False,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    lat0, lon0 = land_anchors(n_anchors, seed)
    base = embed(lat0, lon0, device, ckpt)
    tree = land_tree() if land_mask else None

    # gamma[width][lag]
    gammas = {w: [] for w in MIND_WIDTHS}
    rel_change = {w: {} for w in MIND_WIDTHS}
    for lag in LAGS_KM:
        sel = rng.choice(len(lat0), min(pairs_per_lag, len(lat0)), replace=False)
        if tree is not None:
            la, lo, keep = land_masked_offset(lat0[sel], lon0[sel], float(lag), rng, tree)
            sel, la, lo = sel[keep], la[keep], lo[keep]
            if keep.mean() < 1.0:
                print(f"  land mask kept {keep.mean():.1%} of pairs at {lag:.0f} km", flush=True)
        else:
            la, lo = geodesic_offset(lat0[sel], lon0[sel], float(lag), rng)
        moved = embed(la, lo, device, ckpt)
        for w in MIND_WIDTHS:
            a, b = base[sel, :w].astype(np.float64), moved[:, :w].astype(np.float64)
            diff = np.linalg.norm(a - b, axis=1)
            gammas[w].append(0.5 * float(np.mean(diff**2)))
            for s in SHORT_LAGS_KM:
                if abs(lag - s) / s < 0.15:
                    rel_change[w][s] = float(np.mean(diff / np.linalg.norm(a, axis=1).clip(1e-9)))
        print(
            f"lag {lag:10.3f} km  gamma(64)={gammas[64][-1]:.5g}  gamma(3072)={gammas[3072][-1]:.5g}",
            flush=True,
        )

    rows = []
    for w in MIND_WIDTHS:
        g = np.asarray(gammas[w])
        hs = half_sill_km(LAGS_KM, g)
        # Sensitivity to the sill threshold.
        hs_lo, hs_hi = half_sill_km(LAGS_KM, g, 0.45), half_sill_km(LAGS_KM, g, 0.55)
        for lag, gv in zip(LAGS_KM, g, strict=True):
            rows.append(
                {
                    "ckpt": tag,
                    "encoder": f"{tag}_{w}",
                    "width": w,
                    "lag_km": round(float(lag), 4),
                    "gamma": float(f"{gv:.6g}"),
                    "half_sill_km": round(hs, 2),
                    "half_sill_lo_km": round(hs_lo, 2),
                    "half_sill_hi_km": round(hs_hi, 2),
                    "sill": float(f"{sill(g):.6g}"),
                }
            )
        short = "  ".join(
            f"{s:g}km:{rel_change[w].get(s, float('nan')):.3g}" for s in SHORT_LAGS_KM
        )
        print(
            f"[{tag}] w={w:<5d} half-sill {hs:9.1f} km  (45-55% sill: {hs_lo:7.1f}-{hs_hi:7.1f})  "
            f"rel L2 change  {short}",
            flush=True,
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-anchors", type=int, default=20000)
    p.add_argument("--pairs-per-lag", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--ckpts",
        nargs="+",
        default=[f"{CKPT}=mrl_v4long"],
        help="ckpt_path=tag pairs. Several tags let one run compare training objectives, which is how "
        "the front-loading claim is tested: only a nesting objective should order prefixes by scale.",
    )
    p.add_argument("--out", default=OUT)
    p.add_argument(
        "--land-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="redraw bearings until each offset lands on land, so long lags measure the land field "
        "(the committed CSVs use this; --no-land-mask reproduces the ocean-pair estimator)",
    )
    a = p.parse_args()

    rows = []
    for spec in a.ckpts:
        ckpt, _, tag = spec.partition("=")
        rows.extend(
            run(a.n_anchors, a.pairs_per_lag, a.device, a.seed, ckpt, tag or ckpt, a.land_mask)
        )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "ckpt",
        "encoder",
        "width",
        "lag_km",
        "gamma",
        "half_sill_km",
        "half_sill_lo_km",
        "half_sill_hi_km",
        "sill",
    ]
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
