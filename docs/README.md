# MIND project page

GitHub Pages serves this static site from `main:/docs` at <https://research.taylorgeospatial.org/mind/>. No build step is needed.

```bash
uv run python -m http.server 8765 --bind 127.0.0.1 --directory docs
```

The page introduces MIND, reports CoordBench results, and displays precomputed embedding maps. `map.js` selects three 64-dimensional storage chunks or the full-field image. `store.js` displays the 48 storage chunks using metadata in `assets/store.json`. Image bounds must agree with that metadata. The supervised widths and seven training chunks are distinct from the storage chunks.

The introductory globe cycles through storage chunks 0, 8, 24, and 47, then all chunks. Its embedding strip highlights the corresponding positions in the 48-chunk vector, using the same crossfade as the PCA overlay. Pausing or reducing motion also pauses the strip; flat previews retain the matching highlight when WebGL is unavailable.

[The globe video](assets/mind-linkedin-2160x2700.mp4) is a silent, looping 30-second export with the official Taylor Geospatial logo, at 2160 × 2700 pixels (4:5), encoded as H.264 at 30 fps for sharing.

The checked-in site assets are sufficient to serve the page. The research repository maintains their generation and provenance records separately.

The results table and paper figures are synchronized with `mind-paper` commit `377c226`. The table follows `paper/tables/coordbench.tex`, including ranking marks; SVG panels are exported from the manuscript's TikZ definitions and committed data. `assets/paper-figures.json` records the full source commit and checksums. The method diagram retains a white background for the repository README. The results use ridge probes with unpenalized intercepts.

The header uses the official brown one-line Taylor Geospatial logo in `assets/taylor-geospatial-oneline.svg`, copied unchanged from the TG brand suite. Preserve its proportions and at least one brandmark's height of clear space around it.

The logo links to the Taylor Geospatial Innovation Program. Resource links retain the compact text layout and use the page's accent color to distinguish them from the authors.

Verify the page at 320, 390, 768, and 1280 px. Check table scrolling, chunk selection, keyboard navigation, pan/zoom, and animation playback. Preview switching should continue when the Leaflet CDN is unavailable. After changing assets, update their URL versions in `index.html` and run `node --check docs/map.js`, `node --check docs/store.js`, and `make check`.
