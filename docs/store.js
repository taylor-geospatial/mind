/* "Inside the store" section: three views of the released Zarr, all driven by assets/store.json
 * (written by scripts/data/web_store_assets.py from the live bucket).
 *   1. Explorer  an SVG stack of the 48 storage chunks; dimension + level sliders highlight what
 *                a read touches and total its measured bytes.
 *   2. Density   level-0 shard map of chunk 0 (36 x 71 shards, bytes on disk), ocean shards absent.
 *   3. Strip     48 per-chunk PCA thumbnails with roughness bars, a focus panel, an autoplay, and
 *                a roughness-by-chunk / characteristic-distance chart. */
(function () {
  "use strict";

  var C = window.MINDCOLOR;
  var OURS = "#ff4f2c", INK = "#f2ede2", MUTED = "#a39a8e", RULE = "#3a302a";
  var BOUNDS = [64, 128, 256, 512, 1024, 2048, 3072];

  function $(id) { return document.getElementById(id); }
  function fmtBytes(b) {
    if (b == null) return "—";
    if (b >= 1e9) return (b / 1e9).toFixed(b < 1e10 ? 2 : 1) + " GB";
    if (b >= 1e6) return (b / 1e6).toFixed(b < 1e7 ? 1 : 0) + " MB";
    return (b / 1e3).toFixed(0) + " kB";
  }
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function svgEl(tag, attrs) {
    var e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    return e;
  }

  // ---------------- explorer ----------------
  function explorer(S) {
    var svg = $("ex-svg"), pre = $("ex-prefix"), lev = $("ex-level");
    if (!svg || !pre || !lev) return;
    var N = S.n_bands, levels = S.levels;
    var state = { nb: 1, level: 0 };

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      var L = levels[state.level];
      var rows = L.shard_grid[0], cols = L.shard_grid[1];
      // isometric slab stack: front face = lat x lon shard grid, depth = emb chunks
      var x0 = 36, y0 = 292, fw = 290, fh = 145; // front face
      var dx = 3.6, dy = -2.3; // per-chunk depth offset (48 chunks -> ~173 px right, 110 px up)
      var cw = fw / cols, ch = fh / rows;
      var step = Math.max(1, Math.round(cols / 18));
      for (var b = N - 1; b >= 0; b--) {
        var ox = x0 + b * dx, oy = y0 + b * dy;
        var hot = b < state.nb;
        var face = svgEl("rect", { x: ox, y: oy - fh, width: fw, height: fh, rx: 1.5,
          fill: hot ? OURS : "#221b17", "fill-opacity": hot ? 0.9 : 0.92,
          stroke: hot ? "#ffd9cf" : "#4a3f37", "stroke-width": hot ? 0.7 : 0.5 });
        svg.appendChild(face);
        if (b === 0 || b === state.nb - 1) {
          // shard grid on the front-most face and on the selected boundary face
          var g = svgEl("g", { stroke: hot ? "#fff" : MUTED, "stroke-opacity": 0.5, "stroke-width": 0.5 });
          for (var c = step; c < cols; c += step) g.appendChild(svgEl("line", { x1: ox + c * cw, y1: oy - fh, x2: ox + c * cw, y2: oy }));
          for (var r = step; r < rows; r += step) g.appendChild(svgEl("line", { x1: ox, y1: oy - r * ch, x2: ox + fw, y2: oy - r * ch }));
          svg.appendChild(g);
        }
      }
      // bracket along the depth axis for the selected truncation
      var bx0 = x0 + fw, by0 = y0 - fh, bx1 = x0 + fw + state.nb * dx, by1 = y0 - fh + state.nb * dy;
      svg.appendChild(svgEl("line", { x1: bx0, y1: by0 - 6, x2: bx1, y2: by1 - 6, stroke: OURS, "stroke-width": 2 }));
      // Matryoshka boundary labels: evenly spaced at the right, each with a leader to its edge
      var tick = svgEl("g", { "font-family": "JetBrains Mono, monospace", "font-size": "10" });
      var lx = x0 + fw + N * dx + 26, lyBase = y0 - 4, lyStep = 15;
      BOUNDS.forEach(function (w, i) {
        var b = w / S.band_dims; // boundary after chunk b-1
        var ex = x0 + fw + b * dx, ey = y0 + b * dy;
        var ly = lyBase - i * lyStep;
        var on = w <= state.nb * S.band_dims;
        tick.appendChild(svgEl("line", { x1: ex + 2, y1: ey - 1, x2: lx - 4, y2: ly - 3, stroke: on ? OURS : RULE, "stroke-width": on ? 1 : 0.6 }));
        var t = svgEl("text", { x: lx, y: ly, fill: on ? INK : MUTED, "font-weight": on ? "500" : "400" });
        t.textContent = String(w);
        tick.appendChild(t);
      });
      svg.appendChild(tick);
      var lab = svgEl("g", { "font-family": "JetBrains Mono, monospace", "font-size": "10", fill: MUTED });
      var t1 = svgEl("text", { x: x0, y: y0 + 14 }); t1.textContent = "lon · " + L.shape[2].toLocaleString() + " px · " + cols + " shards";
      var t2 = svgEl("text", { x: x0 - 6, y: y0 - fh / 2, transform: "rotate(-90 " + (x0 - 6) + " " + (y0 - fh / 2) + ")", "text-anchor": "middle" }); t2.textContent = "lat · " + L.shape[1].toLocaleString() + " px · " + rows + " shards";
      var t3 = svgEl("text", { x: x0 + fw + 12, y: y0 + 14, fill: OURS }); t3.textContent = "emb → 48 chunks × 64 dims";
      lab.appendChild(t1); lab.appendChild(t2); lab.appendChild(t3);
      svg.appendChild(lab);
      readout();
    }

    function readout() {
      var L = levels[state.level];
      var nb = state.nb, dims = nb * S.band_dims;
      var bytes = null;
      if (L.bytes_per_band) { bytes = 0; for (var b = 0; b < nb; b++) bytes += L.bytes_per_band[b]; }
      var shards = L.shard_grid[0] * L.shard_grid[1];
      var present = S.shard_density_L0_band0 ? S.shard_density_L0_band0.present : null;
      $("ex-prefix-lab").textContent = String(dims);
      $("ex-level-lab").textContent = String(state.level);
      $("ex-res-lab").textContent = L.resolution_deg + "° · " + (L.resolution_deg * 111).toFixed(L.resolution_deg < 0.1 ? 1 : 0) + " km";
      $("ex-readout").textContent = "embedding[:" + dims + "] at level " + state.level;
      var hs = S.halfsill ? S.halfsill.find(function (h) { return h.width === dims; }) : null;
      $("ex-stats").innerHTML =
        '<div class="st">bytes read<b class="ours">' + fmtBytes(bytes) + "</b>measured, zstd-5</div>" +
        '<div class="st">shard files<b>' + (nb * (state.level === 0 && present ? present : shards)).toLocaleString() + "</b>" + nb + " chunk" + (nb > 1 ? "s" : "") + " × " + (state.level === 0 && present ? present + " land shards" : shards + " shards") + "</div>" +
        '<div class="st">characteristic distance<b>' + (hs ? Math.round(hs.km) + " km" : "—") + "</b>for these leading chunks</div>";
    }

    pre.addEventListener("input", function () { state.nb = BOUNDS[parseInt(pre.value, 10) || 0] / S.band_dims; draw(); });
    lev.addEventListener("input", function () { state.level = parseInt(lev.value, 10) || 0; draw(); });
    draw();
  }

  // ---------------- density ----------------
  function density(S) {
    var cv = $("dn-canvas"), D = S.shard_density_L0_band0;
    if (!cv || !D) { if (cv) cv.parentNode.style.display = "none"; return; }
    var rows = D.rows, cols = D.cols, bytes = D.bytes;
    var max = 0; for (var i = 0; i < bytes.length; i++) if (bytes[i] > max) max = bytes[i];
    var cx = cv.getContext("2d");
    var cw = cv.width / cols, ch = cv.height / rows;
    function paint(hi) {
      cx.fillStyle = "#0e0b09"; cx.fillRect(0, 0, cv.width, cv.height);
      for (var r = 0; r < rows; r++) for (var c = 0; c < cols; c++) {
        var v = bytes[r * cols + c];
        if (!v) continue;
        var t = Math.sqrt(v / max);
        var ci = ((t * 255) | 0) * 3;
        cx.fillStyle = "rgb(" + C.MAGMA[ci] + "," + C.MAGMA[ci + 1] + "," + C.MAGMA[ci + 2] + ")";
        cx.fillRect(c * cw + 0.5, r * ch + 0.5, cw - 1, ch - 1);
      }
      if (hi) { cx.strokeStyle = "#fff"; cx.lineWidth = 1.5; cx.strokeRect(hi[1] * cw + 0.5, hi[0] * ch + 0.5, cw - 1, ch - 1); }
    }
    paint(null);
    var total = 0, present = 0; for (i = 0; i < bytes.length; i++) if (bytes[i]) { total += bytes[i]; present++; }
    var ro = $("dn-readout");
    ro.textContent = present + " of " + rows * cols + " shards written · " + fmtBytes(total);
    cv.addEventListener("mousemove", function (e) {
      var rect = cv.getBoundingClientRect();
      var c = Math.floor(((e.clientX - rect.left) / rect.width) * cols), r = Math.floor(((e.clientY - rect.top) / rect.height) * rows);
      if (c < 0 || r < 0 || c >= cols || r >= rows) return;
      var v = bytes[r * cols + c];
      var lat0 = 90 - r * 5.12, lon0 = -180 + c * 5.12;
      ro.textContent = "shard " + r + "/" + c + " · " + lat0.toFixed(0) + "°–" + (lat0 - 5.12).toFixed(0) + "° lat, " + lon0.toFixed(0) + "°–" + (lon0 + 5.12).toFixed(0) + "° lon · " + (v ? fmtBytes(v) + " (64 × 512 × 512 uint8 = 16.8 MB raw)" : "not written (ocean)");
      paint([r, c]);
    });
    cv.addEventListener("mouseleave", function () { paint(null); ro.textContent = present + " of " + rows * cols + " shards written · " + fmtBytes(total); });
  }

  // ---------------- strip ----------------
  function strip(S) {
    var host = $("strip"), img = $("strip-img"), meta = $("strip-meta"), play = $("strip-play"), ro = $("strip-readout");
    if (!host || !S.chunks) return;
    var chunks = S.chunks, N = chunks.length;
    var rmax = 0; chunks.forEach(function (c) { if (c.roughness > rmax) rmax = c.roughness; });
    var sel = 0, chips = [], timer = null;

    chunks.forEach(function (c, i) {
      var chip = el("button", "chip" + (BOUNDS.indexOf(i * S.band_dims) >= 0 && i > 0 ? " bound" : ""));
      chip.type = "button";
      chip.setAttribute("aria-label", "Storage chunk " + i);
      var im = el("img"); im.loading = "lazy"; im.src = "assets/chunks/chunk_" + (i < 10 ? "0" + i : i) + ".png"; im.alt = "storage chunk " + i;
      chip.appendChild(im);
      chip.appendChild(el("span", "n", String(i)));
      var bar = el("div", "bar"); bar.appendChild(el("i")); bar.firstChild.style.width = Math.round((c.roughness / rmax) * 100) + "%";
      chip.appendChild(bar);
      chip.addEventListener("click", function () { stop(); select(i); });
      host.appendChild(chip);
      chips.push(chip);
    });

    function select(i) {
      sel = i;
      chips.forEach(function (ch, j) {
        ch.classList.toggle("sel", j === i);
        ch.setAttribute("aria-pressed", String(j === i));
      });
      var c = chunks[i];
      img.src = "assets/chunks/chunk_" + (i < 10 ? "0" + i : i) + ".png";
      var lo = i * S.band_dims, hi = lo + S.band_dims - 1;
      var pfx = BOUNDS.filter(function (w) { return w > hi; })[0] || 3072;
      var hs = S.halfsill ? S.halfsill.filter(function (h) { return h.width === pfx; })[0] : null;
      var l0 = S.levels && S.levels[0].bytes_per_band ? S.levels[0].bytes_per_band[i] : null;
      meta.innerHTML =
        '<span class="big">Chunk ' + i + ' <span class="ours">·</span> dims ' + lo + "–" + hi + "</span>" +
        "roughness <b>" + c.roughness.toFixed(3) + "</b> (" + (c.roughness / chunks[0].roughness).toFixed(1) + "× chunk 0)<br>" +
        "participation ratio <b>" + c.participation_ratio.toFixed(0) + "</b> of 64<br>" +
        "share of raw variance <b>" + (100 * c.raw_variance_share).toFixed(1) + "%</b><br>" +
        "on disk at 0.01° <b>" + fmtBytes(l0) + "</b><br>" +
        "within leading dimensions <b>[:" + pfx + "]</b>" + (hs ? ", characteristic distance <b>" + Math.round(hs.km) + " km</b>" : "");
      ro.textContent = "chunk " + i + " / " + (N - 1);
      drawChart(S, i);
    }
    function stop() { if (timer) { clearInterval(timer); timer = null; play.textContent = "Play"; } }
    if ("IntersectionObserver" in window) {
      var playbackObserver = new IntersectionObserver(function (entries) {
        if (!entries[0].isIntersecting) stop();
      });
      playbackObserver.observe(host);
    }
    document.addEventListener("visibilitychange", function () { if (document.hidden) stop(); });
    play.addEventListener("click", function () {
      if (timer) return stop();
      play.textContent = "Stop";
      timer = setInterval(function () { select((sel + 1) % N); }, 420);
    });

    // staggered reveal on scroll
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (!en.isIntersecting) return;
          chips.forEach(function (ch, i) { setTimeout(function () { ch.classList.add("in"); }, i * 28); });
          io.disconnect();
        });
      }, { threshold: 0.15 });
      io.observe(host);
    } else chips.forEach(function (ch) { ch.classList.add("in"); });
    select(0);
  }

  // ---------------- roughness / half-sill chart ----------------
  function drawChart(S, sel) {
    var cv = $("scale-chart");
    if (!cv || !S.chunks) return;
    var cx = cv.getContext("2d"), W = cv.width, H = cv.height;
    var padL = 54, padR = 88, padT = 18, padB = 34, pw = W - padL - padR, ph = H - padT - padB;
    var N = S.chunks.length, rmax = 0;
    S.chunks.forEach(function (c) { if (c.roughness > rmax) rmax = c.roughness; });
    cx.clearRect(0, 0, W, H);
    cx.font = "10px 'JetBrains Mono', monospace";
    // bars: roughness per chunk
    var bw = pw / N;
    for (var i = 0; i < N; i++) {
      var h = (S.chunks[i].roughness / rmax) * ph;
      cx.fillStyle = i === sel ? "#fff" : OURS;
      cx.globalAlpha = i === sel ? 1 : 0.75;
      cx.fillRect(padL + i * bw + 1, padT + ph - h, bw - 2, h);
    }
    cx.globalAlpha = 1;
    // boundaries + half-sill line (log-ish y on the right axis)
    var hs = S.halfsill || [];
    var kmMax = 800, kmMin = 150;
    function yk(km) { return padT + (1 - (Math.log(km) - Math.log(kmMin)) / (Math.log(kmMax) - Math.log(kmMin))) * ph; }
    cx.strokeStyle = RULE; cx.lineWidth = 1;
    BOUNDS.slice(0, -1).forEach(function (w) {
      var x = padL + (w / S.band_dims) * bw;
      cx.beginPath(); cx.moveTo(x, padT); cx.lineTo(x, padT + ph); cx.stroke();
    });
    cx.strokeStyle = INK; cx.lineWidth = 1.4; cx.beginPath();
    hs.forEach(function (h, j) { var x = padL + (h.width / S.band_dims) * bw; var y = yk(h.km); if (j === 0) cx.moveTo(x, y); else cx.lineTo(x, y); });
    cx.stroke();
    hs.forEach(function (h) {
      var x = padL + (h.width / S.band_dims) * bw, y = yk(h.km);
      cx.beginPath(); cx.arc(x, y, 3, 0, Math.PI * 2); cx.fillStyle = INK; cx.fill();
      cx.fillStyle = INK; cx.textAlign = "center"; cx.fillText(Math.round(h.km) + " km", x, y - 8);
    });
    // axes + labels
    cx.strokeStyle = MUTED; cx.lineWidth = 0.8;
    cx.beginPath(); cx.moveTo(padL, padT + ph + 0.5); cx.lineTo(padL + pw, padT + ph + 0.5); cx.stroke();
    cx.fillStyle = MUTED; cx.textAlign = "center";
    [0, 8, 16, 24, 32, 40, 47].forEach(function (i) { cx.fillText("chunk " + i, padL + (i + 0.5) * bw, padT + ph + 14); });
    BOUNDS.forEach(function (w) { cx.fillText(String(w), padL + (w / S.band_dims) * bw, padT + ph + 27); });
    cx.textAlign = "right"; cx.fillStyle = OURS; cx.fillText("roughness", padL - 8, padT + 10);
    cx.fillText(rmax.toFixed(2), padL - 8, padT + 22);
    cx.fillText("0", padL - 8, padT + ph);
    cx.textAlign = "left"; cx.fillStyle = INK; cx.fillText("characteristic", padL + pw + 8, padT + 10);
    cx.fillText("distance", padL + pw + 8, padT + 22);
  }

  var initialized = false;
  function boot() {
    if (initialized) return;
    initialized = true;
    $("store-retry").hidden = true;
    fetch("assets/store.json").then(function (r) { return r.ok ? r.json() : null; }).then(function (S) {
      if (!S) throw new Error("Store metadata unavailable");
      try { explorer(S); } catch (e) { if (window.console) console.warn("[MIND store] explorer:", e); }
      try { density(S); } catch (e) { if (window.console) console.warn("[MIND store] density:", e); }
      try { strip(S); } catch (e) { if (window.console) console.warn("[MIND store] strip:", e); }
    }).catch(function (e) {
      initialized = false;
      $("store-retry").hidden = false;
      $("ex-readout").textContent = "Could not load storage data.";
      if (window.console) console.warn("[MIND store] store.json:", e);
    });
  }
  var section = $("store");
  if (!section) return;
  $("store-retry").addEventListener("click", boot);
  if ("IntersectionObserver" in window) {
    var observer = new IntersectionObserver(function (entries) {
      if (entries.some(function (entry) { return entry.isIntersecting; })) {
        observer.disconnect();
        boot();
      }
    }, { rootMargin: "200px" });
    observer.observe(section);
  } else boot();
})();
