/* Autoplay and accessible controls for the introductory chunk globe. */
(function () {
  "use strict";
  var canvas = document.getElementById("chunk-globe");
  if (!canvas) return;
  var fallback = document.getElementById("globe-fallback");
  var play = document.getElementById("globe-play");
  var status = document.getElementById("globe-status");
  var embedding = document.getElementById("globe-embedding");
  var chunks = document.getElementById("globe-chunks");
  var cells = Array.from({ length: 48 }, function () {
    var cell = document.createElement("span");
    cell.className = "globe-chunk";
    chunks.appendChild(cell);
    return cell;
  });
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
  var views = [0, 8, 24, 47].map(function (chunk) {
    return { file: "chunks/chunk_" + String(chunk).padStart(2, "0") + ".png",
      label: "Chunk " + chunk, chunk: chunk };
  });
  views.push({ file: "global_pca_rgb.png", label: "All chunks", full: true });
  var images = new Map(), renderer = null;
  var current = 0, previous = 0, request = 0;
  var playing = !reduced.matches, visible = false, ready = false;
  var frame = 0, last = null, elapsed = 0, fade = 1, loading = false;
  var longitude = 0.25, latitude = 0.25, drag = null;
  var HOLD = 2750, FADE = 550;

  function load(index) {
    if (images.has(index)) return images.get(index);
    var promise = new Promise(function (resolve, reject) {
      var img = new Image();
      var timeout = setTimeout(function () { reject(new Error("Globe image timed out")); }, 12000);
      img.onload = function () { clearTimeout(timeout); resolve(img); };
      img.onerror = function () { clearTimeout(timeout); reject(new Error("Globe image unavailable")); };
      img.src = "assets/" + views[index].file;
    }).catch(function (error) { images.delete(index); throw error; });
    images.set(index, promise);
    return promise;
  }

  function update() {
    var view = views[current];
    var description = "PCA colors of MIND " + (view.full ? "across all chunks." : "storage " + view.label.toLowerCase() + ".");
    canvas.setAttribute("aria-label", "Globe showing " + description);
    fallback.src = "assets/" + view.file;
    fallback.alt = description;
    embedding.hidden = false;
    embedding.setAttribute("aria-label", "MIND embedding: 3,072 dimensions in 48 storage chunks of 64 dimensions, ordered coarse to fine. " +
      (view.full ? "All chunks highlighted." : view.label + " highlighted, dimensions " + (view.chunk*64) + " through " + (view.chunk*64+63) + "."));
    cells.forEach(function (cell, index) {
      cell.classList.toggle("was-active", !!views[previous].full || views[previous].chunk === index);
      cell.classList.toggle("is-active", !!view.full || view.chunk === index);
    });
    embedding.style.setProperty("--blend", renderer ? fade*fade*(3-2*fade) : 1);
    play.setAttribute("aria-label", playing ? "Pause animation" : "Play animation");
    play.title = play.getAttribute("aria-label");
  }

  function draw() {
    var blend = renderer ? fade*fade*(3-2*fade) : 1;
    embedding.style.setProperty("--blend", blend);
    if (renderer && ready) renderer.draw(views[previous], views[current], blend, longitude, latitude);
  }

  async function select(index) {
    var token = ++request;
    loading = true;
    try {
      var img = await load(index);
      if (token !== request) return;
      if (renderer && !views[index].texture) views[index].texture = renderer.texture(img);
      previous = current;
      current = index;
      fade = reduced.matches || !renderer ? 1 : 0;
      elapsed = 0;
      status.textContent = "";
      update();
      draw();
    } catch (error) {
      if (token !== request) return;
      playing = false;
      status.textContent = "An overlay could not load. Press play to retry.";
      update();
    } finally {
      if (token === request) { loading = false; sync(); }
    }
  }

  function tick(time) {
    frame = 0;
    var dt = last === null ? 0 : Math.min(time-last, 100);
    last = time;
    longitude = (longitude + dt*0.00012) % (Math.PI*2);
    fade = Math.min(1, fade + dt/FADE);
    elapsed += dt;
    if (elapsed >= HOLD && !loading) select((current+1)%views.length);
    draw();
    sync();
  }

  function sync() {
    if (ready && playing && visible && !document.hidden && !drag) {
      if (!frame) frame = requestAnimationFrame(tick);
    } else {
      cancelAnimationFrame(frame);
      frame = 0;
      last = null;
    }
  }

  function useFallback() {
    renderer = null;
    canvas.hidden = true;
    fallback.hidden = false;
  }

  async function start() {
    try { renderer = window.createMINDGlobe(canvas); }
    catch (error) { useFallback(); }
    try {
      var img = await load(0);
      if (renderer) {
        try { views[0].texture = renderer.texture(img); }
        catch (error) { useFallback(); }
      }
      if (renderer) { canvas.hidden = false; fallback.hidden = true; }
      ready = true;
      play.hidden = false;
      update();
      draw();
      sync();
      if (renderer) {
        var terrain = new Image();
        terrain.onload = function () {
          if (!renderer) return;
          try { renderer.setTerrain(terrain); draw(); }
          catch (error) { /* Keep the shaded sphere if terrain is unavailable. */ }
        };
        terrain.src = "assets/globe-dem.png";
      }
      // Five local images, fetched once; the rest of the site reuses the same URLs.
      views.forEach(function (_, index) { load(index).catch(function () { /* Retry on selection. */ }); });
    } catch (error) {
      useFallback();
      ready = true;
      playing = false;
      play.hidden = false;
      elapsed = HOLD;
      status.textContent = "The preview could not load. Press play to retry.";
      update();
    }
  }

  play.addEventListener("click", function () {
    playing = !playing;
    update();
    draw();
    sync();
  });
  reduced.addEventListener("change", function () {
    playing = !reduced.matches;
    fade = 1;
    update();
    draw();
    sync();
  });
  canvas.addEventListener("pointerdown", function (event) {
    if (event.button !== 0) return;
    drag = { id: event.pointerId, x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
    sync();
  });
  canvas.addEventListener("pointermove", function (event) {
    if (!drag || drag.id !== event.pointerId) return;
    longitude -= (event.clientX-drag.x)*0.008;
    latitude = Math.max(-1.2, Math.min(1.2, latitude+(event.clientY-drag.y)*0.006));
    drag.x = event.clientX;
    drag.y = event.clientY;
    draw();
  });
  function endDrag() { drag = null; sync(); }
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("lostpointercapture", endDrag);
  canvas.addEventListener("webglcontextlost", function (event) {
    event.preventDefault();
    ++request;
    loading = false;
    useFallback();
    status.textContent = "Showing flat PCA previews while the globe is unavailable.";
    update();
    sync();
  });
  document.addEventListener("visibilitychange", sync);
  window.addEventListener("resize", draw);
  if ("IntersectionObserver" in window) {
    var started = false;
    new IntersectionObserver(function (entries) {
      visible = entries[0].isIntersecting;
      if (visible && !started) { started = true; start(); }
      sync();
    }).observe(canvas.parentElement);
  } else { visible = true; start(); }
})();
