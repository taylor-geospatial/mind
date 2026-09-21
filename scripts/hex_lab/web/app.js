import {Map, NavigationControl, Marker} from 'https://unpkg.com/maplibre-gl@6.10.0/dist/maplibre-gl.mjs';

const $ = id => document.getElementById(id);
const places = {
  amazon: {name: 'Amazon', center: [-63, -4], zoom: 4.3},
  sahara: {name: 'Sahara', center: [20, 25], zoom: 4.3},
  himalaya: {name: 'Himalaya', center: [86.9, 28.7], zoom: 5.2},
  cornbelt: {name: 'Corn Belt', center: [-93, 42], zoom: 5.5},
};
const views = {
  pca: ['01 / COLOR ATLAS', 'A world of structure.', 'Three PCA components become red, green and blue. Follow the broad patterns, then zoom into individual hexes.'],
  similarity: ['02 / EMBEDDING SIMILARITY', 'Familiar, far from home.', 'Choose a hex to compare embeddings. Brighter colors mean a closer cosine match.'],
  contrast: ['03 / LOCAL CONTRAST', 'Where patterns change.', 'See how much each embedding differs from its immediate neighbors. Bright cells mark stronger local transitions.'],
};
const state = {mode: 'pca', query: null, matches: [], minimumKm: 3000, globe: true, relief: false, ready: false,
  request: 0, selection: 0, controller: null, queryMarker: null, resultMarkers: [], resolution: 3};
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const mobile = () => innerWidth <= 700;
const padding = () => mobile() ? {top: 30, bottom: innerHeight * .35, left: 0, right: 0} : {top: 0, bottom: 0, left: 330, right: 0};
const coordinates = (lat, lon) => `${Math.abs(lat).toFixed(1)}°${lat >= 0 ? 'N' : 'S'}, ${Math.abs(lon).toFixed(1)}°${lon >= 0 ? 'E' : 'W'}`;
const number = n => n.toLocaleString('en-US');
let noticeTimer;
function notice(message) {
  $('notice').textContent = message;
  $('notice').hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { $('notice').hidden = true; }, 6000);
}
async function get(url, signal) {
  const response = await fetch(url, {signal});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

const metadata = await get('/api/meta');
$('dataset-label').textContent = metadata.label;
$('dataset-description').textContent = `${number(metadata.source_cells)} source H3 cells · ${metadata.dimensions} embedding dimensions · ${metadata.source}`;
views.similarity[2] = `Choose a hex to find places with similar ${metadata.dimensions}-dimensional embeddings. Brighter colors mean a closer cosine match.`;
const style = await fetch('https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json').then(r => {
  if (!r.ok) throw new Error('Basemap unavailable');
  return r.json();
}).catch(() => ({version: 8, sources: {}, layers: [{id: 'background', type: 'background', paint: {'background-color': '#20221f'}}]}));
style.projection = {type: 'globe'};
style.sky = {'sky-color': '#24211b', 'horizon-color': '#8e826d', 'fog-color': '#262b29', 'atmosphere-blend': .5};
for (const layer of style.layers) {
  if (layer.type === 'background') layer.paint['background-color'] = '#272c29';
  if (layer.id === 'water') layer.paint['fill-color'] = '#182629';
}
const map = new Map({container: 'map', style, center: [15, 12], zoom: mobile() ? .65 : 2.05,
  maxZoom: 10, minZoom: -.3, attributionControl: {compact: true}, canvasContextAttributes: {antialias: true}});
map.setPadding(padding());
map.addControl(new NavigationControl({visualizePitch: true}), 'top-right');
window.hexLab = {map, state};
map.on('error', event => { console.warn('Map:', event.error?.message); });
map.on('load', () => {
  const before = map.getStyle().layers.find(layer => layer.type === 'symbol')?.id;
  map.addSource('hexes', {type: 'geojson', data: {type: 'FeatureCollection', features: []}, tolerance: 0, maxzoom: 10});
  map.addLayer({id: 'hex-fill', type: 'fill', source: 'hexes', paint: {'fill-color': ['get', 'color'], 'fill-opacity': .94}}, before);
  map.addLayer({id: 'hex-line', type: 'line', source: 'hexes', paint: {'line-color': '#151b18', 'line-width': .45, 'line-opacity': ['interpolate', ['linear'], ['zoom'], 1, .12, 6, .42]}}, before);
  map.addLayer({id: 'hex-relief', type: 'fill-extrusion', source: 'hexes', layout: {visibility: 'none'}, paint: {
    'fill-extrusion-color': ['get', 'color'], 'fill-extrusion-opacity': .95,
    'fill-extrusion-height': ['interpolate', ['linear'], ['zoom'], 0, ['*', ['get', 'height'], 2], 4, ['*', ['get', 'height'], .6], 6, ['*', ['get', 'height'], .15], 8, ['*', ['get', 'height'], .035]],
    'fill-extrusion-base': 0,
  }}, before);
  state.ready = true;
  refresh();
});

async function refresh() {
  if (!state.ready || (state.mode === 'similarity' && !state.query)) return;
  state.controller?.abort();
  state.controller = new AbortController();
  const request = ++state.request;
  const zoom = map.getZoom();
  const bounds = map.getBounds();
  const level = zoom < 2.8 ? 3 : zoom < 4.7 ? 4 : 5;
  const bbox = zoom < 2.8 ? [-180, -90, 180, 90] : [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()];
  const params = new URLSearchParams({level, bbox: bbox.join(','), mode: state.mode});
  if (state.query) params.set('query', state.query.h3);
  $('status').textContent = 'Updating hexes…';
  $('status').classList.add('busy');
  try {
    const data = await get(`/api/hexes?${params}`, state.controller.signal);
    if (request !== state.request) return;
    map.getSource('hexes').setData(data);
    state.resolution = data.resolution;
    state.featureCount = data.count;
    $('status').textContent = `${number(data.count)} hexes · H3 r${data.resolution} ${data.resolution === 5 ? 'source cells' : 'overview'} · ${metadata.dimensions} dimensions`;
    $('status').classList.remove('busy');
    if (state.mode === 'contrast') {
      $('legend-labels').innerHTML = `<span>0</span><span>${(data.contrast_max / 2).toFixed(3)}</span><span>≥ ${data.contrast_max.toFixed(3)}</span>`;
    }
  } catch (error) {
    if (error.name !== 'AbortError') {
      notice(error.message);
      $('status').textContent = 'Unable to load hexes. Move the map to retry.';
      $('status').classList.remove('busy');
    }
  }
}
map.on('moveend', refresh);
window.addEventListener('resize', () => { map.setPadding(padding()); });

function layers() {
  if (!state.ready) return;
  map.setLayoutProperty('hex-fill', 'visibility', state.relief ? 'none' : 'visible');
  map.setLayoutProperty('hex-line', 'visibility', state.relief ? 'none' : 'visible');
  map.setLayoutProperty('hex-relief', 'visibility', state.relief ? 'visible' : 'none');
}
function legend() {
  $('legend-bar').className = `legend-bar ${state.mode}`;
  const entries = {
    pca: ['PCA → RGB', '<span>PC1 → R</span><span>PC2 → G</span><span>PC3 → B</span>', 'Same color stretch at every zoom.'],
    similarity: ['Cosine similarity to selected hex', '<span>≤ 0</span><span>0.5</span><span>1.0</span>', 'Ranked matches use source r5 cells.'],
    contrast: ['Mean cosine distance to neighbors', '<span>0</span><span>0.010</span><span>≥ 0.021</span>', 'Color ceiling: source r5 99th percentile.'],
  };
  const entry = entries[state.mode];
  $('legend-title').textContent = entry[0];
  $('legend-labels').innerHTML = entry[1];
  $('legend-note').textContent = entry[2];
}
async function setMode(mode) {
  state.mode = mode;
  state.controller?.abort();
  $('hover').hidden = true;
  document.querySelectorAll('[data-mode]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.mode === mode)));
  ['view-number', 'view-title', 'view-description'].forEach((id, i) => { $(id).textContent = views[mode][i]; });
  $('similarity-controls').hidden = mode !== 'similarity';
  $('contrast-controls').hidden = mode !== 'contrast';
  if (mode !== 'contrast' && state.relief) {
    state.relief = false;
    $('relief').checked = false;
    map.easeTo({pitch: 0, duration: reducedMotion ? 0 : 500});
  }
  state.queryMarker?.getElement().toggleAttribute('hidden', mode !== 'similarity');
  state.resultMarkers.forEach(marker => marker.getElement().toggleAttribute('hidden', mode !== 'similarity'));
  legend();
  layers();
  if (mode === 'similarity' && !state.query) {
    await choose(...places.amazon.center, 'Amazon');
    if (state.mode === 'similarity') fly(places.amazon.center, mobile() ? .65 : 2.05);
  }
  else refresh();
}
document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => setMode(button.dataset.mode)));

function fly(center, zoom, extra = {}) {
  map.flyTo({center, zoom, padding: padding(), speed: 1.15, curve: 1.2,
    pitch: state.relief ? 50 : 0, bearing: state.relief ? -15 : 0, duration: reducedMotion ? 0 : 1300, ...extra});
}
document.querySelectorAll('[data-place]').forEach(button => button.addEventListener('click', async () => {
  const place = places[button.dataset.place];
  if (state.mode === 'similarity') {
    await choose(...place.center, place.name);
    if (state.mode === 'similarity') fly(place.center, mobile() ? .65 : 2.05);
  } else fly(place.center, place.zoom);
}));
$('world').addEventListener('click', () => fly([15, 12], mobile() ? .65 : 2.05, {pitch: 0, bearing: 0}));
$('distance').addEventListener('change', () => {
  state.minimumKm = Number($('distance').value);
  if (state.query) choose(state.query.lon, state.query.lat);
});
$('projection').addEventListener('click', () => {
  state.globe = !state.globe;
  map.setProjection({type: state.globe ? 'globe' : 'mercator'});
  $('projection').textContent = state.globe ? 'Globe' : 'Flat map';
  $('projection').setAttribute('aria-pressed', String(state.globe));
  refresh();
});
$('relief').addEventListener('change', () => {
  state.relief = $('relief').checked;
  layers();
  if (state.relief && map.getZoom() < 3.5) fly(places.himalaya.center, 4.7);
  else map.easeTo({pitch: state.relief ? 50 : 0, duration: reducedMotion ? 0 : 650});
});

async function choose(lon, lat, name = '') {
  const selection = ++state.selection;
  $('query-summary').textContent = `Comparing all ${number(metadata.source_cells)} source cells…`;
  $('matches').replaceChildren();
  $('match-note').hidden = true;
  try {
    const data = await get(`/api/select?lat=${lat}&lon=${((lon + 180) % 360 + 360) % 360 - 180}&minimum_km=${state.minimumKm}`);
    if (selection !== state.selection) return;
    state.query = data.query;
    state.matches = data.matches;
    const q = data.query;
    $('query-summary').innerHTML = `${name || 'Selected hex'} · ${coordinates(q.lat, q.lon)}<small>${q.h3} · ${q.count} source samples</small>`;
    state.queryMarker?.remove();
    state.resultMarkers.forEach(marker => marker.remove());
    const element = document.createElement('div');
    element.className = 'query-marker';
    element.setAttribute('aria-label', 'Selected source hex');
    state.queryMarker = new Marker({element}).setLngLat([q.lon, q.lat]).addTo(map);
    state.resultMarkers = data.matches.map((match, i) => {
      const button = document.createElement('button');
      button.className = 'match';
      button.innerHTML = `<span class="rank">${i + 1}</span><span class="coordinate">${coordinates(match.lat, match.lon)}<small>${number(Math.round(match.distance_km))} km away</small></span><span class="score">${match.score.toFixed(3)}</span>`;
      button.setAttribute('aria-label', `Inspect match ${i + 1}, ${coordinates(match.lat, match.lon)}, cosine similarity ${match.score.toFixed(3)}`);
      button.addEventListener('click', () => fly([match.lon, match.lat], 5.6));
      $('matches').append(button);
      const dot = document.createElement('div');
      dot.className = 'result-marker';
      dot.textContent = i + 1;
      return new Marker({element: dot}).setLngLat([match.lon, match.lat]).addTo(map);
    });
    $('match-note').hidden = false;
    if (!data.matches.length) $('matches').textContent = 'No sampled cells beyond this distance.';
    if (state.mode !== 'similarity') {
      state.queryMarker.getElement().hidden = true;
      state.resultMarkers.forEach(marker => { marker.getElement().hidden = true; });
    }
    refresh();
  } catch (error) {
    if (selection !== state.selection) return;
    notice(error.message);
    $('query-summary').textContent = state.query ? `Selected hex · ${coordinates(state.query.lat, state.query.lon)}` : 'Choose a sampled hex on land.';
  }
}

map.on('click', event => {
  if (state.mode !== 'similarity' || !state.ready) return;
  choose(event.lngLat.lng, event.lngLat.lat);
});
map.on('mousemove', event => {
  if (!state.ready || mobile()) return;
  const features = map.queryRenderedFeatures(event.point, {layers: [state.relief ? 'hex-relief' : 'hex-fill']});
  map.getCanvas().style.cursor = state.mode === 'similarity' && features.length ? 'crosshair' : '';
  if (!features.length) { $('hover').hidden = true; return; }
  const p = features[0].properties;
  const value = p.value == null ? 'No available neighbors' : Number(p.value).toFixed(4);
  const metric = state.mode === 'similarity' ? `Cosine similarity <strong>${value}</strong>` : `Local contrast <strong>${value}</strong>`;
  $('hover').innerHTML = `${coordinates(p.lat, p.lon)}<br>${metric}<small>H3 r${p.level} ${p.level < 5 ? 'overview' : 'source cell'} · ${number(p.count)} source samples</small>`;
  $('hover').hidden = false;
  $('hover').style.left = `${Math.min(event.point.x + 18, innerWidth - 295)}px`;
  $('hover').style.top = `${Math.min(event.point.y + 18, innerHeight - 105)}px`;
});
map.getCanvas().addEventListener('mouseleave', () => { $('hover').hidden = true; });
