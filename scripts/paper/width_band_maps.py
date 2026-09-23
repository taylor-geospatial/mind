"""Render MIND chunk PCA maps. Use --full for all chunks and the regional zoom."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from scripts.paper.width_pca_maps import (
    BANDS,
    OUT,
    compute_scores,
    lab_rgb,
    lab_stretch,
    land_grid,
    render,
    render_flat,
    render_preview,
    zoom_grid,
)

PAPER_PANELS = {  # (variant, band) -> (paper figure file, corner label); None = full embedding
    ("own", BANDS[0]): ("band-chunk-first.png", "Chunk 0"),
    ("own", BANDS[3]): ("band-chunk-mid.png", "Chunk 3"),
    ("own", BANDS[-1]): ("band-chunk-last.png", "Chunk 6"),
    ("own", None): ("band-full.png", "All Chunks"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also render the zoom grid and shared variant, plus every per-band diagnostic "
        "PNG. Slow; only the default (paper panels + band-own-globe-panels.png) is needed "
        "to rebuild the paper.",
    )
    args = parser.parse_args()

    Path(OUT).mkdir(parents=True, exist_ok=True)
    tags = (
        (("globe", land_grid(), True), ("zoom", zoom_grid(), False))
        if args.full
        else (("globe", land_grid(), True),)
    )
    variants = ("shared", "own") if args.full else ("own",)
    for tag, (land, lats, lons), globe in tags:
        s = compute_scores(tag, land, lats, lons)
        shared_stretch = lab_stretch(s["full"])
        for variant in variants:
            stretch = shared_stretch if variant == "shared" else None
            equalize = variant == "own"
            panels = []
            for chunk, (lo, hi) in enumerate(BANDS):
                rgb = lab_rgb(s[f"band_{variant}_{lo}_{hi}"], stretch, equalize=equalize)
                if args.full:
                    render(
                        rgb, land, lats, lons, OUT / f"band-{variant}-{tag}-{lo}-{hi}.png", globe
                    )
                panels.append((rgb, f"Chunk {chunk} (Dim {lo}–{hi - 1})"))  # noqa: RUF001
                if globe and (variant, (lo, hi)) in PAPER_PANELS:
                    fname, label = PAPER_PANELS[variant, (lo, hi)]
                    render_flat(rgb, land, lats, lons, OUT / fname, label=label)
            rgb_full = lab_rgb(s["full"], shared_stretch, equalize=equalize)
            if args.full:
                render(rgb_full, land, lats, lons, OUT / f"band-{variant}-{tag}-full.png", globe)
            panels.append((rgb_full, "All Chunks (Dim 0–3071)"))  # noqa: RUF001
            if globe and (variant, None) in PAPER_PANELS:
                fname, label = PAPER_PANELS[variant, None]
                render_flat(rgb_full, land, lats, lons, OUT / fname, label=label)
            if args.full or (variant == "own" and tag == "globe"):
                render_preview(
                    panels, land, lats, lons, OUT / f"band-{variant}-{tag}-panels.png", globe
                )
        for lo, hi in BANDS:
            print(
                f"{tag}: Participation Ratio (Band {lo}:{hi}) = {float(s[f'effdim_band_{lo}_{hi}']):.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
