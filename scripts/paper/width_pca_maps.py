"""Compute MIND prefix/chunk PCA maps and covariance participation ratios.

Use 0.16-degree global and 0.04-degree regional grids with a WorldClim land mask.
Set MIND_CHECKPOINT and MIND_DEVICE to select the model and device.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Times New Roman"

import cartopy.crs as ccrs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from cartopy.io import shapereader  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from scripts.paper.blocksize_sweep import CKPT  # noqa: E402
from scripts.paper.map_grid import map_grid  # noqa: E402

LEVEL = 4  # 0.16 degrees
ARCH = {
    "encoder_type": "resiren",
}
WIDTHS = (64, 128, 256, 512, 1024, 3072)  # nested prefixes the checkpoint was trained at
BANDS = ((0, 64), (64, 128), (128, 256), (256, 512), (512, 1024), (1024, 2048), (2048, 3072))
ZOOM = {"lon": (-125.0, -95.0), "lat": (28.0, 50.0), "level": "2"}  # western North America
LAT_MIN = -60.0  # Exclude Antarctica from the color stretch.
FLAT_LAT = (-58.0, 84.0)
FLAT_ASPECT = 360.0 / (FLAT_LAT[1] - FLAT_LAT[0])
OUT = Path("results/figures")
CACHE = Path("results/map-cache")

INK = "#241a18"
OCEAN = "#eef1f4"
LAB_LMIN, LAB_LSPAN, LAB_CHROMA = 32.0, 60.0, 58.0


def land_grid():
    land, lats, lons = map_grid(LEVEL)
    land[lats < LAT_MIN, :] = False
    return land, lats, lons


def zoom_grid():
    return map_grid(2, (ZOOM["lon"][0], ZOOM["lat"][0], ZOOM["lon"][1], ZOOM["lat"][1]))


def _lab_to_srgb(L, a, b) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fy = (L + 16.0) / 116.0
    fx, fz = fy + a / 500.0, fy - b / 200.0

    def finv(t) -> np.ndarray:
        c = t**3
        return np.where(c > 0.008856, c, (t - 16.0 / 116.0) / 7.787)

    X, Y, Z = 0.95047 * finv(fx), finv(fy), 1.08883 * finv(fz)
    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    bl = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

    def gam(c) -> np.ndarray:
        c = np.clip(c, 0.0, 1.0)
        return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)

    return gam(r), gam(g), gam(bl)


def lab_stretch(proj):
    """Compute percentile stretch parameters for a shared color scale."""
    l1, h1 = np.percentile(proj[:, 0], [2, 98])
    m2 = np.percentile(np.abs(proj[:, 1]), 98)
    m3 = np.percentile(np.abs(proj[:, 2]), 98)
    return float(l1), float(h1), float(m2), float(m3)


def rank_uniform(proj):
    """Rank-transform each column to a uniform spread on [-1, 1] (histogram equalisation)."""
    out = np.empty_like(proj, dtype=np.float32)
    n = proj.shape[0]
    for k in range(proj.shape[1]):
        ranks = np.empty(n, dtype=np.float32)
        ranks[np.argsort(proj[:, k], kind="stable")] = np.arange(n, dtype=np.float32)
        out[:, k] = (ranks + 0.5) / n * 2.0 - 1.0
    return out


def lab_rgb(proj, stretch=None, equalize=False):
    """Map three PCA scores to sRGB: PC1 lightness, PC2/3 chroma.

    ``equalize`` rank-transforms each component before color mapping.
    """
    if equalize:
        u = rank_uniform(proj)
        L = LAB_LMIN + (u[:, 0] + 1.0) / 2.0 * LAB_LSPAN
        a, b = u[:, 1] * LAB_CHROMA, u[:, 2] * LAB_CHROMA
        return np.stack(_lab_to_srgb(L, a, b), axis=-1).astype(np.float32)
    l1, h1, m2, m3 = stretch if stretch is not None else lab_stretch(proj)
    L = LAB_LMIN + np.clip((proj[:, 0] - l1) / (h1 - l1 + 1e-9), 0, 1) * LAB_LSPAN
    a = np.clip(proj[:, 1] / (m2 + 1e-9), -1, 1) * LAB_CHROMA
    b = np.clip(proj[:, 2] / (m3 + 1e-9), -1, 1) * LAB_CHROMA
    return np.stack(_lab_to_srgb(L, a, b), axis=-1).astype(np.float32)


def align_to(proj, ref):
    """Align three PCA columns to a reference by signed permutation."""
    c = np.corrcoef(proj.T, ref.T)[:3, 3:]  # rows: proj comps, cols: ref comps
    out = np.empty_like(proj)
    used: set[int] = set()
    for j in range(3):
        i = max((i for i in range(3) if i not in used), key=lambda i: abs(c[i, j]))
        used.add(i)
        s = np.sign(c[i, j])
        out[:, j] = proj[:, i] * (s if s != 0 else 1.0)
    return out


def _heavy_tail_positive(proj) -> np.ndarray:
    for k in range(proj.shape[1]):
        if np.percentile(proj[:, k], 98) < -np.percentile(proj[:, k], 2):
            proj[:, k] *= -1
    return proj


def effdim(evals):
    """Return the covariance participation ratio, (sum l)^2 / sum l^2."""
    evals = np.clip(evals, 0, None)
    return float(evals.sum() ** 2 / ((evals**2).sum() + 1e-12))


def compute_scores(tag, land, lats, lons):
    """Cache prefix/chunk PCA scores and covariance spectra for the land grid.

    ``own`` fits each prefix or chunk; ``shared`` uses the full embedding basis.
    """
    cache = CACHE / f"pca_scores_{tag}.npz"
    if cache.exists():
        print(f"{tag}: loaded {cache}", flush=True)
        return dict(np.load(cache).items())

    from mind.eval.embedders import embed_ours

    ii, jj = np.nonzero(land)
    print(f"{tag}: embedding {len(ii)} land pixels", flush=True)
    emb = embed_ours(
        CKPT,
        lats[ii],
        lons[jj],
        feature="pooled",
        device=os.environ.get("MIND_DEVICE", "cuda"),
        **ARCH,
    )
    emb -= emb.mean(axis=0)
    emb /= emb.std(axis=0) + 1e-8
    emb = emb.astype(np.float32)
    n, _d = emb.shape

    out: dict[str, np.ndarray | np.floating] = {}

    # full-width basis: top-3 PCs of the standardised 3072-d field
    pca_full = PCA(n_components=3, random_state=0).fit(emb)
    V = pca_full.components_.astype(np.float32)  # [3, 3072]
    full = emb @ V.T
    for k in range(3):  # heavier tail positive fixes the basis sign
        if np.percentile(full[:, k], 98) < -np.percentile(full[:, k], 2):
            full[:, k] *= -1
            V[k] *= -1
    out["full"] = full

    # Covariance spectrum for participation ratios.
    cov = (emb.T @ emb) / n
    out["evals"] = np.linalg.eigvalsh(cov.astype(np.float64))[::-1].copy()
    for w in WIDTHS:
        out[f"effdim_pref_{w}"] = np.float64(effdim(np.linalg.eigvalsh(cov[:w, :w])))
    for lo, hi in BANDS:
        out[f"effdim_band_{lo}_{hi}"] = np.float64(effdim(np.linalg.eigvalsh(cov[lo:hi, lo:hi])))
    del cov

    ref = None
    for w in WIDTHS:
        own = PCA(n_components=3, random_state=0).fit_transform(emb[:, :w])
        own = _heavy_tail_positive(own) if ref is None else align_to(own, ref)
        if ref is None:
            ref = own
        out[f"pref_own_{w}"] = own.astype(np.float32)
        out[f"pref_shared_{w}"] = emb[:, :w] @ V[:, :w].T
        print(f"{tag}: prefix {w} done", flush=True)

    for lo, hi in BANDS:
        own = PCA(n_components=3, random_state=0).fit_transform(emb[:, lo:hi])
        out[f"band_own_{lo}_{hi}"] = align_to(own, full).astype(np.float32)
        out[f"band_shared_{lo}_{hi}"] = emb[:, lo:hi] @ V[:, lo:hi].T
        print(f"{tag}: band {lo}:{hi} done", flush=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    print(f"{tag}: wrote {cache}", flush=True)
    return out


def draw_panel(ax, rgb_land, land, lats, lons, globe, title=None):
    H, W = land.shape
    rgba = np.zeros((H, W, 4), np.float32)
    rgba[land, :3] = rgb_land
    rgba[..., 3] = land.astype(np.float32)
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    if globe:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.spines["geo"].set_visible(False)
    ax.patch.set_facecolor(OCEAN)
    ax.imshow(
        rgba,
        origin="upper",
        extent=extent,
        transform=ccrs.PlateCarree(),
        interpolation="bilinear",
        regrid_shape=1600,
    )
    scale = "110m" if globe else "50m"
    coast = shapereader.Reader(
        shapereader.natural_earth(resolution=scale, category="physical", name="coastline")
    ).geometries()
    ax.add_geometries(  # Match the coastline extent to the land mask.
        [g for g in coast if g.bounds[3] > LAT_MIN],
        ccrs.PlateCarree(),
        facecolor="none",
        edgecolor=INK,
        linewidth=0.3,
        alpha=0.55,
    )
    if title:
        ax.set_title(title, fontsize=9, pad=3)


def render(rgb_land, land, lats, lons, path, globe):
    fig = plt.figure(figsize=(11, 5.6) if globe else (7, 5.6))
    proj = ccrs.EqualEarth() if globe else ccrs.PlateCarree()
    ax = fig.add_axes([0.005, 0.005, 0.99, 0.99], projection=proj)
    draw_panel(ax, rgb_land, land, lats, lons, globe)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path, flush=True)


def render_flat(rgb_land, land, lats, lons, path, width_in=7.5, label=None):
    """Render a lon/lat map cropped to FLAT_LAT, with an optional corner label."""
    fig = plt.figure(figsize=(width_in, width_in / FLAT_ASPECT))
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    draw_panel(ax, rgb_land, land, lats, lons, globe=False)
    ax.set_extent([-180.0, 180.0, FLAT_LAT[0], FLAT_LAT[1]], crs=ccrs.PlateCarree())
    if label is not None:
        ax.text(
            0.015,
            0.025,
            label,
            transform=ax.transAxes,
            fontsize=26,
            fontfamily="Nimbus Roman",
            ha="left",
            va="bottom",
            zorder=10,
            bbox={
                "boxstyle": "square,pad=0.35",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 1.8,
            },
        )
    fig.savefig(path, dpi=300, facecolor="white", pad_inches=0)
    plt.close(fig)
    print("wrote", path, flush=True)


def render_preview(panels, land, lats, lons, path, globe, ncols=2):
    """Render a panel grid from (rgb_land, title) pairs."""
    nrows = -(-len(panels) // ncols)
    w, h = (5.5, 3.0) if globe else (3.5, 3.0)
    proj = ccrs.EqualEarth() if globe else ccrs.PlateCarree()
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(w * ncols, h * nrows), subplot_kw={"projection": proj}
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(panels) :]:
        ax.set_visible(False)
    for ax, (rgb_land, title) in zip(axes, panels, strict=False):
        draw_panel(ax, rgb_land, land, lats, lons, globe, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", path, flush=True)


PREVIEW_WIDTHS = (64, 256, 1024, 3072)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, (land, lats, lons), globe in (
        ("globe", land_grid(), True),
        ("zoom", zoom_grid(), False),
    ):
        s = compute_scores(tag, land, lats, lons)
        panels = []
        for w in WIDTHS:
            rgb = lab_rgb(s[f"pref_own_{w}"])
            render(rgb, land, lats, lons, OUT / f"prefix-own-{tag}-{w}.png", globe)
            if w in PREVIEW_WIDTHS:
                ed = float(s[f"effdim_pref_{w}"])
                panels.append((rgb, f"First {w} Dims (Participation Ratio {ed:.0f})"))
        render_preview(panels, land, lats, lons, OUT / f"prefix-own-{tag}-panels.png", globe)
        for w in WIDTHS:
            print(f"{tag}: effdim(prefix {w}) = {float(s[f'effdim_pref_{w}']):.1f}", flush=True)


if __name__ == "__main__":
    main()
