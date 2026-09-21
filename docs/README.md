# MIND project page

GitHub Pages serves this static site from `main:/docs` at <https://research.taylorgeospatial.org/mind/>. No build step is needed.

```bash
uv run python -m http.server 8765 --bind 127.0.0.1 --directory docs
```

The page introduces MIND, reports CoordBench results, and displays precomputed embedding maps. `map.js` selects three 64-dimensional storage chunks or the full-field image. `store.js` displays the 48 storage chunks using metadata in `assets/store.json`. Image bounds must agree with that metadata. The supervised widths and seven training chunks are distinct from the storage chunks.

The checked-in site assets are sufficient to serve the page. The research repository maintains their generation and provenance records separately.

Verify the page at 320, 390, 768, and 1280 px. Check table scrolling, chunk selection, keyboard navigation, pan/zoom, and animation playback. Preview switching should continue when the Leaflet CDN is unavailable. After changing assets, update their URL versions in `index.html` and run `node --check docs/map.js`, `node --check docs/store.js`, and `make check`.
