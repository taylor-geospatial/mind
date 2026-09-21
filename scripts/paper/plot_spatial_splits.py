"""Plot held-out and training points for two CoordBench spatial block sizes."""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from mind.eval.probe import spatial_fold_ids
from scripts.paper.figstyle import INK, IVORY, RED, apply_rc

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/figures/spatial-splits"
SEED = 0
HELD_OUT_FOLD = 0


def _read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    lon_col = next(c for c in ("lon", "longitude", "Lon", "x") if c in frame)
    lat_col = next(c for c in ("lat", "latitude", "Lat", "y") if c in frame)
    lon = frame[lon_col].to_numpy(float)
    lat = frame[lat_col].to_numpy(float)
    keep = np.isfinite(lon) & np.isfinite(lat)
    return lat[keep], lon[keep]


def _cell_pairs(lat: np.ndarray, lon: np.ndarray, cell_deg: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.floor(lat / cell_deg).astype(np.int64),
        np.floor(lon / cell_deg).astype(np.int64),
    )


def _grid_ticks(lo: float, hi: float, cell_deg: float) -> np.ndarray:
    """Return grid lines at the same zero-origin boundaries as the fold cells."""
    first = int(np.ceil(lo / cell_deg))
    last = int(np.floor(hi / cell_deg))
    return np.arange(first, last + 1, dtype=float) * cell_deg


def _extent(
    lat: np.ndarray, lon: np.ndarray, global_map: bool
) -> tuple[float, float, float, float]:
    if global_map:
        return -180, 180, -90, 90
    pad = 1.5
    return (
        float(lon.min() - pad),
        float(lon.max() + pad),
        float(lat.min() - pad),
        float(lat.max() + pad),
    )


def _draw_panel(
    ax: plt.Axes,
    lat: np.ndarray,
    lon: np.ndarray,
    cell_deg: float,
    extent: tuple[float, float, float, float],
    label: str,
    panel: str,
) -> None:
    folds = spatial_fold_ids(lat, lon, folds=5, cell_deg=cell_deg, seed=SEED)
    lat_cells, lon_cells = _cell_pairs(lat, lon, cell_deg)
    held = folds == HELD_OUT_FOLD
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_facecolor(IVORY)
    ax.add_feature(
        cfeature.LAND.with_scale("110m"), facecolor="#ece9df", edgecolor="none", zorder=0
    )
    ax.add_feature(
        cfeature.COASTLINE.with_scale("110m"), edgecolor="#8e817a", linewidth=0.35, zorder=1
    )
    ax.gridlines(
        draw_labels=False,
        linewidth=0.35,
        color="#b7afa5",
        alpha=0.65,
        linestyle=":",
        xlocs=_grid_ticks(extent[0], extent[1], cell_deg),
        ylocs=_grid_ticks(extent[2], extent[3], cell_deg),
    )
    held_cells = np.unique(np.column_stack((lat_cells[held], lon_cells[held])), axis=0)
    for lat_cell, lon_cell in held_cells:
        ax.add_patch(
            Rectangle(
                (int(lon_cell) * cell_deg, int(lat_cell) * cell_deg),
                cell_deg,
                cell_deg,
                transform=ccrs.PlateCarree(),
                facecolor=RED,
                edgecolor=RED,
                linewidth=0.65,
                alpha=0.18,
                zorder=2,
            )
        )
    ax.scatter(
        lon[~held],
        lat[~held],
        s=2.1 if len(lon) > 2000 else 4.2,
        color="#7f8994",
        alpha=0.36,
        linewidths=0,
        transform=ccrs.PlateCarree(),
        zorder=3,
        rasterized=True,
    )
    ax.scatter(
        lon[held],
        lat[held],
        s=3.3 if len(lon) > 2000 else 6.0,
        color=RED,
        alpha=0.82,
        linewidths=0,
        transform=ccrs.PlateCarree(),
        zorder=4,
        rasterized=True,
    )
    print(f"{label}: {held.sum()} held-out of {len(lat)} points")
    ax.text(
        0.02,
        0.97,
        f"{panel}  {label}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.4,
        fontweight="bold",
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2},
        zorder=6,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--temperature", type=Path, default=None)
    parser.add_argument("--africa-crops", type=Path, default=None)
    args = parser.parse_args()
    root = args.evaluation_root
    temperature = args.temperature or root / "worldclim_bio" / "data.parquet"
    africa_crops = args.africa_crops or root / "dm_africa_crop_mask" / "data.parquet"
    if temperature.suffix == ".parquet":
        temp_frame = pd.read_parquet(temperature)
        valid = temp_frame["bio1"].notna().to_numpy()
        temp_lat = temp_frame.loc[valid, "lat"].to_numpy(float)
        temp_lon = temp_frame.loc[valid, "lon"].to_numpy(float)
    else:
        temp_lat, temp_lon = _read(temperature)
    crop_lat, crop_lon = _read(africa_crops)
    crop_extent = _extent(crop_lat, crop_lon, False)
    crop_aspect = (crop_extent[3] - crop_extent[2]) / (crop_extent[1] - crop_extent[0])
    apply_rc()
    projection = ccrs.PlateCarree()
    # Match the map aspect ratios, allowing for the margins and subplot gaps below.
    panel_width = 5.5 * 0.98 / 2.03
    figure_height = panel_width * (0.5 + crop_aspect) * 1.02 / 0.985
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(5.5, figure_height),
        gridspec_kw={"height_ratios": (0.5, crop_aspect)},
        subplot_kw={"projection": projection},
        facecolor="white",
    )
    _draw_panel(
        axes[0, 0],
        temp_lat,
        temp_lon,
        20,
        _extent(temp_lat, temp_lon, True),
        "Temperature · 20° cells",
        "a",
    )
    _draw_panel(
        axes[0, 1],
        temp_lat,
        temp_lon,
        40,
        _extent(temp_lat, temp_lon, True),
        "Temperature · 40° cells",
        "b",
    )
    _draw_panel(
        axes[1, 0],
        crop_lat,
        crop_lon,
        20,
        crop_extent,
        "Africa crops · 20° cells",
        "c",
    )
    _draw_panel(
        axes[1, 1],
        crop_lat,
        crop_lon,
        40,
        crop_extent,
        "Africa crops · 40° cells",
        "d",
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.995, wspace=0.03, hspace=0.04)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{OUTPUT}.pdf", bbox_inches="tight", facecolor="white", dpi=300)
    fig.savefig(f"{OUTPUT}.svg", bbox_inches="tight", facecolor="white", dpi=180)
    svg_path = OUTPUT.with_suffix(".svg")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n"
    )
    plt.close(fig)
    print(f"WorldClim bio1 points={len(temp_lon)}, Africa crop points={len(crop_lon)}")
    print(f"wrote {OUTPUT}.pdf and {OUTPUT}.svg")


if __name__ == "__main__":
    main()
