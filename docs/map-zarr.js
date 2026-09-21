/* Zarr-backed views for the MIND landing map, read live from the 3072-d multiscale store on
 * Source Coop with zarrita:
 *   - a band cache: one 64-d Matryoshka chunk of level 5 (562 x 1125, ~14 MB compressed) per
 *     fetch, shared by Chunk and Dimension modes, with a running byte counter;
 *   - Chunk: PCA(3)->CIELab of one cached band, computed in the browser;
 *   - Dimension: one channel of a cached band, viridis, robust 2-98% stretch;
 *   - Compare: two points' embeddings from level 6 bands (~4 MB each), cosine similarity across
 *     nested prefixes.
 *
 * window.MINDZARR(ctx) -> { ensure(), band(b), bandBytes(b), fetched(), renderChunk(b),
 *                           renderDim(d), vectors(pts, width), grid() } */
(function () {
  "use strict";

  var ZARR_URL = "https://data.source.coop/tge-labs/mind/mind.zarr";
  var ZARRITA_CDNS = ["https://cdn.jsdelivr.net/npm/zarrita@0.4.0/+esm", "https://esm.sh/zarrita"];
  var BAND = 64, NBANDS = 48;
  var LEVEL = "5"; // Chunk + Dimension grid
  var CMP_LEVEL = "6"; // Compare grid
  var LAT_MIN = -60; // Antarctica: outlier embeddings hijack every stretch (matches the paper figs)

  function importZarrita() {
    var idx = 0;
    function tryNext(lastErr) {
      if (idx >= ZARRITA_CDNS.length) return Promise.reject(lastErr || new Error("no zarrita CDN reachable"));
      return import(/* webpackIgnore: true */ ZARRITA_CDNS[idx++]).then(
        function (mod) { return mod && mod.FetchStore ? mod : mod && mod.default ? mod.default : mod; },
        tryNext
      );
    }
    return tryNext(null);
  }

  window.MINDZARR = function (ctx) {
    var C = window.MINDCOLOR;
    var zarr = null, ready = null;
    var Z = { emb: null, cmpEmb: null, H: 0, W: 0, mask: null, lat: null, lon: null, scale: null, offset: null,
              cmp: { H: 0, W: 0, lat: null, lon: null, mask: null } };
    var bands = {}; // b -> Promise<Uint8Array (64*H*W)>
    var cmpBands = {}; // b -> Promise<Uint8Array (64*h*w)>
    var bandSizes = ctx.store && ctx.store.levels ? ctx.store.levels[+LEVEL].bytes_per_band : null;
    var cmpSizes = ctx.store && ctx.store.levels ? ctx.store.levels[+CMP_LEVEL].bytes_per_band : null;
    var fetched = 0;

    function openLevel(root, lvl) {
      return Promise.all([
        zarr.open(root.resolve(lvl + "/embedding"), { kind: "array" }),
        zarr.open(root.resolve(lvl + "/mask"), { kind: "array" }),
        zarr.open(root.resolve(lvl + "/lat"), { kind: "array" }),
        zarr.open(root.resolve(lvl + "/lon"), { kind: "array" }),
      ]).then(function (a) {
        return Promise.all([a[0], zarr.get(a[1]), zarr.get(a[2]), zarr.get(a[3])]);
      }).then(function (g) {
        var shp = g[0].shape;
        var mask = g[1].data, lat = g[2].data;
        var H = shp[1], W = shp[2];
        for (var r = 0; r < H; r++) if (Number(lat[r]) < LAT_MIN) for (var c = 0; c < W; c++) mask[r * W + c] = 0;
        return { emb: g[0], H: H, W: W, mask: mask, lat: lat, lon: g[3].data };
      });
    }

    function ensure() {
      if (ready) return ready;
      ready = importZarrita().then(function (mod) {
        zarr = mod;
        if (!zarr || typeof zarr.open !== "function" || !zarr.FetchStore) throw new Error("zarrita module missing expected API");
        var root = zarr.root(new zarr.FetchStore(ZARR_URL));
        return Promise.all([
          openLevel(root, LEVEL),
          openLevel(root, CMP_LEVEL),
          zarr.open(root.resolve("emb_scale"), { kind: "array" }).then(function (a) { return zarr.get(a); }),
          zarr.open(root.resolve("emb_offset"), { kind: "array" }).then(function (a) { return zarr.get(a); }),
        ]);
      }).then(function (r) {
        Z.emb = r[0].emb; Z.H = r[0].H; Z.W = r[0].W; Z.mask = r[0].mask; Z.lat = r[0].lat; Z.lon = r[0].lon;
        Z.cmp = r[1]; Z.cmpEmb = r[1].emb;
        Z.scale = r[2].data; Z.offset = r[3].data;
      });
      return ready;
    }

    function gridBounds(g) {
      var la0 = Number(g.lat[0]), la1 = Number(g.lat[g.H - 1]);
      var lo0 = Number(g.lon[0]), lo1 = Number(g.lon[g.W - 1]);
      var dlat = Math.abs(la1 - la0) / (g.H - 1), dlon = Math.abs(lo1 - lo0) / (g.W - 1);
      return {
        latMin: Math.min(la0, la1) - dlat / 2, latMax: Math.max(la0, la1) + dlat / 2,
        lonMin: Math.min(lo0, lo1) - dlon / 2, lonMax: Math.max(lo0, lo1) + dlon / 2,
        latDescending: la0 > la1,
      };
    }

    function band(b) {
      if (bands[b]) return bands[b];
      bands[b] = ensure().then(function () {
        return zarr.get(Z.emb, [zarr.slice(b * BAND, (b + 1) * BAND), null, null]);
      }).then(function (res) {
        fetched += bandSizes ? bandSizes[b] : 0;
        return res.data;
      });
      bands[b].catch(function () { delete bands[b]; });
      return bands[b];
    }

    function cmpBand(b) {
      if (cmpBands[b]) return cmpBands[b];
      cmpBands[b] = ensure().then(function () {
        return zarr.get(Z.cmpEmb, [zarr.slice(b * BAND, (b + 1) * BAND), null, null]);
      }).then(function (res) {
        fetched += cmpSizes ? cmpSizes[b] : 0;
        return res.data;
      });
      cmpBands[b].catch(function () { delete cmpBands[b]; });
      return cmpBands[b];
    }

    // Paint an RGBA (H x W) image north-up and return a data URL plus bounds.
    function paint(fill) {
      var H = Z.H, W = Z.W;
      var canvas = document.createElement("canvas");
      canvas.width = W; canvas.height = H;
      var cx = canvas.getContext("2d");
      var img = cx.createImageData(W, H);
      var px = img.data;
      var gb = gridBounds(Z);
      var flip = !gb.latDescending;
      for (var y = 0; y < H; y++) {
        var src = (flip ? H - 1 - y : y) * W;
        var dst = y * W;
        for (var x = 0; x < W; x++) fill(src + x, (dst + x) * 4, px);
      }
      cx.putImageData(img, 0, 0);
      return { url: canvas.toDataURL(), bounds: [[gb.latMin, gb.lonMin], [gb.latMax, gb.lonMax]] };
    }

    // Chunk b -> PCA(3) on standardized land pixels -> CIELab color.
    function renderChunk(b) {
      return band(b).then(function (data) {
        var H = Z.H, W = Z.W, n = H * W, mask = Z.mask;
        var land = [];
        for (var i = 0; i < n; i++) if (mask[i]) land.push(i);
        var nl = land.length;
        // dequantize + standardize each dim over land (paper recipe), row-major (nl x 64)
        var x = new Float32Array(nl * BAND);
        for (var d = 0; d < BAND; d++) {
          var sc = Number(Z.scale[b * BAND + d]), of = Number(Z.offset[b * BAND + d]);
          var base = d * n, s = 0, s2 = 0, k;
          for (k = 0; k < nl; k++) { var v = data[base + land[k]] * sc + of; s += v; s2 += v * v; }
          var mu = s / nl, sd = Math.sqrt(Math.max(s2 / nl - mu * mu, 1e-12));
          for (k = 0; k < nl; k++) x[k * BAND + d] = (data[base + land[k]] * sc + of - mu) / sd;
        }
        var p = C.pca3(x, nl, BAND);
        var sc1 = new Float32Array(nl), sc2 = new Float32Array(nl), sc3 = new Float32Array(nl);
        for (k = 0; k < nl; k++) {
          var a = 0, bb = 0, c = 0, row = k * BAND;
          for (d = 0; d < BAND; d++) {
            var xv = x[row + d] - p.mean[d];
            a += xv * p.comps[0][d]; bb += xv * p.comps[1][d]; c += xv * p.comps[2][d];
          }
          sc1[k] = a; sc2[k] = bb; sc3[k] = c;
        }
        var color = C.pcaColorizer(sc1, sc2, sc3);
        var rgb = new Uint8Array(n * 3);
        for (k = 0; k < nl; k++) {
          var q = color(sc1[k], sc2[k], sc3[k]);
          rgb[land[k] * 3] = q[0]; rgb[land[k] * 3 + 1] = q[1]; rgb[land[k] * 3 + 2] = q[2];
        }
        return paint(function (si, di, px) {
          if (!mask[si]) { px[di + 3] = 0; return; }
          px[di] = rgb[si * 3]; px[di + 1] = rgb[si * 3 + 1]; px[di + 2] = rgb[si * 3 + 2]; px[di + 3] = 255;
        });
      });
    }

    // Dimension d -> viridis with a robust 2-98% stretch over land.
    function renderDim(d) {
      var b = Math.floor(d / BAND), dd = d % BAND;
      return band(b).then(function (data) {
        var n = Z.H * Z.W, mask = Z.mask, base = dd * n;
        var sc = Number(Z.scale[d]), of = Number(Z.offset[d]);
        var vals = [];
        for (var i = 0; i < n; i++) if (mask[i]) vals.push(data[base + i] * sc + of);
        vals.sort(function (a, b) { return a - b; });
        var lo = C.pct(vals, 0.02), hi = C.pct(vals, 0.98), span = hi - lo > 0 ? hi - lo : 1;
        var V = C.VIRIDIS;
        return paint(function (si, di, px) {
          if (!mask[si]) { px[di + 3] = 0; return; }
          var t = (data[base + si] * sc + of - lo) / span;
          t = t < 0 ? 0 : t > 1 ? 1 : t;
          var ci = ((t * 255 + 0.5) | 0) * 3;
          px[di] = V[ci]; px[di + 1] = V[ci + 1]; px[di + 2] = V[ci + 2]; px[di + 3] = 255;
        });
      });
    }

    function rowCol(g, lat, lon) {
      var gb = gridBounds(g);
      var rowT = gb.latDescending ? (gb.latMax - lat) / (gb.latMax - gb.latMin) : (lat - gb.latMin) / (gb.latMax - gb.latMin);
      var colT = (lon - gb.lonMin) / (gb.lonMax - gb.lonMin);
      return [Math.min(g.H - 1, Math.max(0, Math.floor(rowT * g.H))), Math.min(g.W - 1, Math.max(0, Math.floor(colT * g.W)))];
    }

    // Embeddings (Float32Array of `width` dims) at each {lat, lon}, from the compare grid.
    function vectors(pts, width) {
      return ensure().then(function () {
        var g = Z.cmp;
        var idx = pts.map(function (p) { var rc = rowCol(g, p.lat, p.lon); return rc[0] * g.W + rc[1]; });
        for (var i = 0; i < idx.length; i++) if (!g.mask[idx[i]]) return Promise.reject(new Error("one of those points is open ocean (not stored in this layer) — click two points on land"));
        var nb = Math.ceil(width / BAND);
        var reads = [];
        for (var b = 0; b < nb; b++) reads.push(cmpBand(b));
        return Promise.all(reads).then(function (datas) {
          var n = g.H * g.W;
          return pts.map(function (_, pi) {
            var out = new Float32Array(width);
            for (var d = 0; d < width; d++) {
              var bb = Math.floor(d / BAND), dd = d % BAND;
              out[d] = datas[bb][dd * n + idx[pi]] * Number(Z.scale[d]) + Number(Z.offset[d]);
            }
            return out;
          });
        });
      });
    }

    return {
      ensure: ensure,
      band: band,
      bandBytes: function (b) { return bandSizes ? bandSizes[b] : null; },
      cmpBandBytes: function (b) { return cmpSizes ? cmpSizes[b] : null; },
      hasBand: function (b) { return !!bands[b]; },
      hasCmpBand: function (b) { return !!cmpBands[b]; },
      fetched: function () { return fetched; },
      renderChunk: renderChunk,
      renderDim: renderDim,
      vectors: vectors,
      BAND: BAND,
      NBANDS: NBANDS,
      LEVEL: LEVEL,
      CMP_LEVEL: CMP_LEVEL,
    };
  };
})();
