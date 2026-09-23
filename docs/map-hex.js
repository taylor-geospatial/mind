/* Hexbin layer for the MIND landing map: H3 r4/r5 overviews read live from GeoParquet on
 * Source Coop with hyparquet (only the h3, count, and pca1..3 columns are fetched, by HTTP range
 * request), colored either PCA(3)->CIELab of the 64-d cell mean or by cell density (count of
 * 1 km land pixels per hexagon). Every cell ring is projected once at zoom 0 and drawn per frame
 * with a linear transform on one canvas.
 *
 * window.MINDHEX(ctx) -> { ensure(res), setColor(mode), layer(), stats(res), disable() } */
(function () {
  "use strict";

  var HEX_BASE = "https://data.source.coop/tge-labs/mind/mind_hex_";
  var HYPARQUET_CDNS = ["https://cdn.jsdelivr.net/npm/hyparquet/+esm", "https://esm.sh/hyparquet"];
  var H3_CDNS = ["https://cdn.jsdelivr.net/npm/h3-js/+esm", "https://esm.sh/h3-js"];

  function importFirst(cdns) {
    var i = 0;
    function next(lastErr) {
      if (i >= cdns.length) return Promise.reject(lastErr || new Error("no CDN reachable"));
      return import(/* webpackIgnore: true */ cdns[i++]).then(function (m) {
        return m && m.default && !m.parquetReadObjects && !m.cellToBoundary ? m.default : m;
      }, next);
    }
    return next(null);
  }

  window.MINDHEX = function (ctx) {
    var L = window.L, C = window.MINDCOLOR;
    var map = ctx.map, pane = ctx.pane;
    var libs = null; // Promise -> {hp, h3}
    var sets = {}; // res -> Promise -> {cells:[...], bytes}
    var current = { res: "r5", color: "pca" };
    var layer = null;

    function ensureLibs() {
      if (libs) return libs;
      libs = Promise.all([importFirst(HYPARQUET_CDNS), importFirst(H3_CDNS)]).then(function (m) {
        if (!m[0].parquetReadObjects || !m[0].asyncBufferFromUrl) throw new Error("hyparquet API missing");
        if (typeof m[1].cellToBoundary !== "function") throw new Error("h3-js API missing");
        return { hp: m[0], h3: m[1] };
      });
      return libs;
    }

    function load(res) {
      if (sets[res]) return sets[res];
      sets[res] = ensureLibs().then(function (m) {
        return m.hp.asyncBufferFromUrl({ url: HEX_BASE + res + ".parquet" }).then(function (file) {
          return m.hp
            .parquetReadObjects({ file: file, columns: ["h3", "count", "pca1", "pca2", "pca3"] })
            .then(function (rows) { return build(m.h3, rows); });
        });
      });
      return sets[res];
    }

    function build(h3, rows) {
      if (!rows || !rows.length) throw new Error("no hex rows");
      var p1 = [], p2 = [], p3 = [], cnt = [];
      for (var i = 0; i < rows.length; i++) {
        p1.push(rows[i].pca1); p2.push(rows[i].pca2); p3.push(rows[i].pca3); cnt.push(Number(rows[i].count));
      }
      var color = C.pcaColorizer(p1, p2, p3);
      var sc = cnt.slice().sort(function (a, b) { return a - b; });
      var cmax = C.pct(sc, 0.98) || 1;
      var cells = [];
      for (var j = 0; j < rows.length; j++) {
        var ring = h3.cellToBoundary(rows[j].h3);
        if (!ring || ring.length < 3) continue;
        var minLng = 180, maxLng = -180, minLat = 90, maxLat = -90;
        for (var k = 0; k < ring.length; k++) {
          var la = ring[k][0], lo = ring[k][1];
          if (lo < minLng) minLng = lo;
          if (lo > maxLng) maxLng = lo;
          if (la < minLat) minLat = la;
          if (la > maxLat) maxLat = la;
        }
        if (maxLng - minLng > 180) continue; // antimeridian wrap
        var xy = new Float64Array(ring.length * 2);
        for (var q = 0; q < ring.length; q++) {
          var p0 = map.project([ring[q][0], ring[q][1]], 0);
          xy[q * 2] = p0.x;
          xy[q * 2 + 1] = p0.y;
        }
        var rgb = color(p1[j], p2[j], p3[j]);
        var t = Math.min(1, Math.sqrt(cnt[j] / cmax));
        var ci = ((t * 255) | 0) * 3;
        cells.push({
          xy: xy,
          pca: "rgb(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ")",
          dens: "rgb(" + C.MAGMA[ci] + "," + C.MAGMA[ci + 1] + "," + C.MAGMA[ci + 2] + ")",
          n: [minLat, maxLat, minLng, maxLng],
        });
      }
      return { cells: cells, n: rows.length };
    }

    var HexLayer = L.Layer.extend({
      onAdd: function (m) {
        this._map = m;
        var p = m.getPane(pane) || m.getPanes().overlayPane;
        this._canvas = L.DomUtil.create("canvas", "mind-hex-canvas");
        this._canvas.style.position = "absolute";
        p.appendChild(this._canvas);
        m.on("moveend zoomend resize viewreset", this._redraw, this);
        this._redraw();
      },
      onRemove: function (m) {
        m.off("moveend zoomend resize viewreset", this._redraw, this);
        if (this._canvas && this._canvas.parentNode) this._canvas.parentNode.removeChild(this._canvas);
        this._canvas = null;
      },
      _redraw: function () {
        var m = this._map;
        if (!m || !this._canvas) return;
        var set = sets[current.res] && sets[current.res]._value;
        var size = m.getSize();
        var tl = m.containerPointToLayerPoint([0, 0]);
        L.DomUtil.setPosition(this._canvas, tl);
        this._canvas.width = size.x;
        this._canvas.height = size.y;
        var cx = this._canvas.getContext("2d");
        cx.clearRect(0, 0, size.x, size.y);
        if (!set) return;
        var s = Math.pow(2, m.getZoom());
        var origin = m.getPixelOrigin();
        var ax = origin.x + tl.x, ay = origin.y + tl.y;
        var b = m.getBounds();
        var north = b.getNorth() + 3, south = b.getSouth() - 3, west = b.getWest() - 3, east = b.getEast() + 3;
        var key = current.color === "density" ? "dens" : "pca";
        var cells = set.cells;
        for (var i = 0; i < cells.length; i++) {
          var nb = cells[i].n;
          if (nb[0] > north || nb[1] < south || nb[2] > east || nb[3] < west) continue;
          var xy = cells[i].xy;
          cx.beginPath();
          cx.moveTo(xy[0] * s - ax, xy[1] * s - ay);
          for (var k = 2; k < xy.length; k += 2) cx.lineTo(xy[k] * s - ax, xy[k + 1] * s - ay);
          cx.closePath();
          cx.fillStyle = cells[i][key];
          cx.fill();
        }
      },
    });

    function redraw() { if (layer && layer._map) layer._redraw(); }

    return {
      // Resolve once the requested resolution is loaded and drawn.
      ensure: function (res) {
        current.res = res || current.res;
        var p = load(current.res);
        return p.then(function (set) {
          p._value = set;
          if (!layer) layer = new HexLayer();
          redraw();
          return set;
        });
      },
      setColor: function (mode) { current.color = mode; redraw(); },
      layer: function () { return layer; },
      res: function () { return current.res; },
      color: function () { return current.color; },
    };
  };
})();
