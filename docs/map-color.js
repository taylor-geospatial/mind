/* Shared color helpers for the MIND landing page (map + store explorer).
 * Exposed on window.MINDCOLOR; no dependencies. */
(function () {
  "use strict";

  // viridis, 17 control points sampled from matplotlib, linearly interpolated to 256 stops,
  // packed as a flat [r0,g0,b0, r1,g1,b1, ...] Uint8Array.
  function buildLut(ctrl) {
    var n = ctrl.length;
    var lut = new Uint8Array(256 * 3);
    for (var i = 0; i < 256; i++) {
      var t = (i / 255) * (n - 1);
      var lo = Math.floor(t);
      var hi = Math.min(lo + 1, n - 1);
      var f = t - lo;
      for (var c = 0; c < 3; c++) {
        lut[i * 3 + c] = Math.round(ctrl[lo][c] + (ctrl[hi][c] - ctrl[lo][c]) * f);
      }
    }
    return lut;
  }

  var VIRIDIS = buildLut([
    [68, 1, 84], [71, 19, 101], [72, 40, 120], [69, 55, 129], [64, 71, 136], [57, 86, 140],
    [51, 99, 141], [45, 113, 142], [40, 125, 142], [35, 138, 141], [31, 151, 139], [33, 164, 133],
    [47, 177, 121], [70, 190, 105], [102, 202, 84], [141, 211, 59], [184, 219, 38], [253, 231, 37],
  ]);

  var MAGMA = buildLut([
    [0, 0, 4], [29, 17, 71], [81, 18, 124], [132, 38, 129], [183, 55, 121], [229, 81, 99],
    [252, 135, 97], [254, 197, 134], [252, 253, 191],
  ]);

  // CIELab (D65) -> sRGB 0..255; mirrors scripts/data/build_mind_cog_hex.py and
  // scripts/data/web_store_assets.py so every PCA rendering on the page shares one palette.
  function labToSrgb(L, a, b) {
    var fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
    function finv(t) {
      var c = t * t * t;
      return c > 0.008856 ? c : (t - 16 / 116) / 7.787;
    }
    var X = 0.95047 * finv(fx), Y = 1.0 * finv(fy), Z = 1.08883 * finv(fz);
    var r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z;
    var g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
    var bl = 0.0557 * X - 0.204 * Y + 1.057 * Z;
    function gam(c) {
      c = c < 0 ? 0 : c > 1 ? 1 : c;
      return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
    }
    return [Math.round(gam(r) * 255), Math.round(gam(g) * 255), Math.round(gam(bl) * 255)];
  }
  var LAB = { LMIN: 32, LSPAN: 60, CHROMA: 58 };

  function pct(sorted, q) {
    var i = Math.max(0, Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1))));
    return sorted[i];
  }

  // Robust stretch for three PCA score columns: sign fix on PC1 (positive tail up) and 2-98%
  // percentile scaling, returning a function (p1,p2,p3) -> [r,g,b].
  function pcaColorizer(p1, p2, p3) {
    var s1 = Array.prototype.slice.call(p1).sort(function (a, b) { return a - b; });
    var s2 = Array.prototype.slice.call(p2).sort(function (a, b) { return a - b; });
    var s3 = Array.prototype.slice.call(p3).sort(function (a, b) { return a - b; });
    var sign1 = pct(s1, 0.98) < -pct(s1, 0.02) ? -1 : 1;
    var sign2 = pct(s2, 0.98) < -pct(s2, 0.02) ? -1 : 1;
    var sign3 = pct(s3, 0.98) < -pct(s3, 0.02) ? -1 : 1;
    var l1 = sign1 * pct(s1, sign1 > 0 ? 0.02 : 0.98);
    var h1 = sign1 * pct(s1, sign1 > 0 ? 0.98 : 0.02);
    var m2 = Math.max(Math.abs(pct(s2, 0.02)), Math.abs(pct(s2, 0.98)));
    var m3 = Math.max(Math.abs(pct(s3, 0.02)), Math.abs(pct(s3, 0.98)));
    return function (a, b, c) {
      var t = (sign1 * a - l1) / (h1 - l1 + 1e-9);
      t = t < 0 ? 0 : t > 1 ? 1 : t;
      var La = LAB.LMIN + t * LAB.LSPAN;
      var aa = Math.max(-1, Math.min(1, (sign2 * b) / (m2 + 1e-9))) * LAB.CHROMA;
      var bb = Math.max(-1, Math.min(1, (sign3 * c) / (m3 + 1e-9))) * LAB.CHROMA;
      return labToSrgb(La, aa, bb);
    };
  }

  // Top-3 principal directions of an (n x d) row-major Float32Array via a few power
  // iterations with deflation on the d x d covariance. d <= 64 here, so this is cheap.
  function pca3(x, n, d) {
    var mean = new Float64Array(d);
    var i, j, k;
    for (i = 0; i < n; i++) for (j = 0; j < d; j++) mean[j] += x[i * d + j];
    for (j = 0; j < d; j++) mean[j] /= n;
    var cov = new Float64Array(d * d);
    var row = new Float64Array(d);
    for (i = 0; i < n; i++) {
      for (j = 0; j < d; j++) row[j] = x[i * d + j] - mean[j];
      for (j = 0; j < d; j++) {
        var rj = row[j];
        if (rj === 0) continue;
        for (k = j; k < d; k++) cov[j * d + k] += rj * row[k];
      }
    }
    for (j = 0; j < d; j++) for (k = 0; k < j; k++) cov[j * d + k] = cov[k * d + j];
    var comps = [];
    for (var c = 0; c < 3; c++) {
      var v = new Float64Array(d);
      for (j = 0; j < d; j++) v[j] = Math.sin(j * 1.7 + c);
      for (var it = 0; it < 60; it++) {
        var w = new Float64Array(d);
        for (j = 0; j < d; j++) {
          var s = 0;
          for (k = 0; k < d; k++) s += cov[j * d + k] * v[k];
          w[j] = s;
        }
        var nrm = 0;
        for (j = 0; j < d; j++) nrm += w[j] * w[j];
        nrm = Math.sqrt(nrm) || 1;
        for (j = 0; j < d; j++) v[j] = w[j] / nrm;
      }
      var lam = 0;
      for (j = 0; j < d; j++) {
        var s2 = 0;
        for (k = 0; k < d; k++) s2 += cov[j * d + k] * v[k];
        lam += v[j] * s2;
      }
      for (j = 0; j < d; j++) for (k = 0; k < d; k++) cov[j * d + k] -= lam * v[j] * v[k];
      comps.push(v);
    }
    return { mean: mean, comps: comps };
  }

  window.MINDCOLOR = {
    VIRIDIS: VIRIDIS,
    MAGMA: MAGMA,
    labToSrgb: labToSrgb,
    pcaColorizer: pcaColorizer,
    pca3: pca3,
    pct: pct,
  };
})();
