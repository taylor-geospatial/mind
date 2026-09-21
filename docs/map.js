/* Small, precomputed previews. Leaflet loads only when pan and zoom is requested. */
(function () {
  "use strict";

  var preview = document.getElementById("map-preview");
  var controls = document.getElementById("map-controls");
  var status = document.getElementById("map-status");
  var activate = document.getElementById("map-activate");
  var mapEl = document.getElementById("map");
  if (!preview || !controls || !status || !activate || !mapEl) return;

  // Level-6 cell edges recorded in assets/store.json (thumb_bounds).
  var bounds = [[-89.84000000009183, -180.00000000000028], [89.99999999999984, 179.67999999967316]];
  var views = [
    { file: "chunks/chunk_00.png", label: "First chunk", dims: "0–63", description: "broad geographic structure" },
    { file: "chunks/chunk_23.png", label: "Middle chunk", dims: "1472–1535", description: "intermediate spatial granularity" },
    { file: "chunks/chunk_47.png", label: "Last chunk", dims: "3008–3071", description: "fine spatial granularity" },
    { file: "band-full.png?v=8475d3d", label: "All chunks", dims: "0–3071", description: "the full embedding", bounds: [[-58, -180], [84, 180]] },
  ];
  var buttons = Array.from(controls.querySelectorAll("[data-view]"));
  var images = new Map();
  var selected = 0, request = 0, map = null, layer = null;
  controls.hidden = false;

  function url(index) { return "assets/" + views[index].file; }
  function viewBounds() { return views[selected].bounds || bounds; }
  function label(index) { return views[index].label + " · dimensions " + views[index].dims + "."; }
  function setStatus(text, warning) {
    status.textContent = text;
    status.classList.toggle("is-warn", !!warning);
  }

  function loadImage(index) {
    if (images.has(index)) return images.get(index);
    var promise = new Promise(function (resolve, reject) {
      var img = new Image();
      var timeout = setTimeout(function () { reject(new Error("Preview timed out")); }, 10000);
      img.onload = function () { clearTimeout(timeout); resolve(img); };
      img.onerror = function () { clearTimeout(timeout); reject(new Error("Preview unavailable")); };
      img.src = url(index);
    }).catch(function (error) {
      images.delete(index);
      throw error;
    });
    images.set(index, promise);
    return promise;
  }

  async function select(index) {
    var current = ++request;
    setStatus("Loading " + views[index].label.toLowerCase() + "…");
    try {
      await loadImage(index);
      if (current !== request) return;
      selected = index;
      preview.src = url(index);
      preview.alt = views[index].label + ": " + views[index].description + " in PCA false color.";
      if (layer) { layer.setBounds(viewBounds()); layer.setUrl(url(index)); }
      buttons.forEach(function (button, i) { button.setAttribute("aria-pressed", String(i === index)); });
      setStatus(label(index));
    } catch (error) {
      if (current === request) setStatus("That preview could not load. The previous view is still shown; select a chunk to retry.", true);
    }
  }

  buttons.forEach(function (button, index) {
    button.addEventListener("click", function () { select(index); });
  });

  // Warm the three small previews near the viewport. All chunks reuses the paper image.
  function warmPreviews() {
    views.slice(0, 3).forEach(function (_, index) { loadImage(index).catch(function () { /* Retry on selection. */ }); });
  }
  if ("IntersectionObserver" in window) {
    var observer = new IntersectionObserver(function (entries) {
      if (entries.some(function (entry) { return entry.isIntersecting; })) {
        observer.disconnect();
        warmPreviews();
      }
    }, { rootMargin: "200px" });
    observer.observe(preview);
  }

  function loadAsset(tag, href, integrity) {
    return new Promise(function (resolve, reject) {
      var element = document.createElement(tag);
      var timeout = setTimeout(fail, 10000);
      function fail() {
        clearTimeout(timeout);
        element.remove();
        reject(new Error("Map library unavailable"));
      }
      element.integrity = integrity;
      element.crossOrigin = "anonymous";
      element.onload = function () { clearTimeout(timeout); resolve(); };
      element.onerror = fail;
      if (tag === "link") { element.rel = "stylesheet"; element.href = href; }
      else element.src = href;
      document.head.appendChild(element);
    });
  }

  var leaflet = null;
  function loadLeaflet() {
    if (!leaflet) {
      leaflet = Promise.all([
        loadAsset("link", "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css", "sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="),
        loadAsset("script", "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js", "sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="),
      ]).catch(function (error) { leaflet = null; throw error; });
    }
    return leaflet;
  }

  activate.addEventListener("click", async function () {
    if (map) {
      map.fitBounds(viewBounds(), { padding: [8, 8], animate: false });
      return;
    }
    activate.disabled = true;
    activate.textContent = "Loading map…";
    try {
      await loadLeaflet();
      await loadImage(selected);
      mapEl.hidden = false;
      mapEl.parentElement.classList.add("is-interactive");
      // The previews are latitude/longitude grids; preserve that projection when zooming.
      map = L.map(mapEl, {
        crs: L.CRS.EPSG4326,
        scrollWheelZoom: false,
        dragging: true,
        maxBounds: [[-100, -190], [100, 190]],
        maxBoundsViscosity: 1,
        zoomSnap: 0.25,
        zoomControl: true,
        maxZoom: 4,
      });
      map.fitBounds(viewBounds(), { padding: [8, 8], animate: false });
      map.setMinZoom(map.getZoom());
      map.on("resize", function () { map.setMinZoom(map.getBoundsZoom(bounds, false, [16, 16])); });
      layer = L.imageOverlay(url(selected), viewBounds(), { attribution: "MIND · Taylor Geospatial" }).addTo(map);
      preview.hidden = true;
      activate.textContent = "Reset view";
      setStatus(label(selected) + " Drag to pan; pinch or use + / − to zoom.");
    } catch (error) {
      if (map) map.remove();
      map = null;
      layer = null;
      mapEl.hidden = true;
      mapEl.parentElement.classList.remove("is-interactive");
      preview.hidden = false;
      activate.textContent = "Retry pan & zoom";
      setStatus("Pan and zoom could not load. You can still switch between the four previews.", true);
    } finally {
      activate.disabled = false;
    }
  });
})();
