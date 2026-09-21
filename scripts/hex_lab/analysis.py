"""Read-only exploration of prepared H3 embeddings."""

import json
import threading
from functools import lru_cache

import h3.api.basic_int as h3
import numpy as np
from scipy.spatial import KDTree

from scripts.hex_lab.metrics import distances, sphere


@lru_cache(maxsize=60000)
def polygon(cell):
    """Keep antimeridian rings continuous rather than spanning the world."""
    center_lon = h3.cell_to_latlng(cell)[1]
    ring = [
        [round(center_lon + (lon - center_lon + 180) % 360 - 180, 5), round(lat, 5)]
        for lat, lon in h3.cell_to_boundary(cell)
    ]
    return {"type": "Polygon", "coordinates": [ring + [ring[0]]]}


def visible_indices(data, level, bbox):
    west, south, east, north = bbox
    lat, lon = data["positions"].T
    pad = {3: 2.5, 4: 1.0, 5: 0.5}[level]
    mask = (lat >= south - pad) & (lat <= north + pad)
    if east - west < 360:
        center = (west + east) / 2
        wrapped = (lon - center + 180) % 360 - 180
        mask &= np.abs(wrapped) <= (east - west) / 2 + pad
    return np.flatnonzero(mask)


def colors(values, mode, maximum):
    if mode == "contrast":
        t = np.clip(np.nan_to_num(values) / maximum, 0, 1)
        stops = np.array(
            [[26, 59, 62], [54, 120, 118], [198, 177, 111], [239, 132, 65], [255, 234, 175]]
        )
    else:
        t = np.clip(values, 0, 1)
        stops = np.array(
            [[37, 41, 54], [64, 89, 111], [68, 137, 136], [150, 191, 130], [255, 231, 157]]
        )
    ramp = t * (len(stops) - 1)
    low = np.minimum(ramp.astype(int), len(stops) - 2)
    rgb = np.rint(stops[low] + (stops[low + 1] - stops[low]) * (ramp - low)[:, None]).astype(
        np.uint8
    )
    rgb[~np.isfinite(values)] = [112, 112, 112]
    return rgb


class Explorer:
    def __init__(self, directory) -> None:
        self.meta = json.loads((directory / "metadata.json").read_text())
        self.levels = {}
        for resolution in (3, 4, 5):
            with np.load(directory / f"r{resolution}-meta.npz") as archive:
                data = dict(archive)
            data["vectors"] = np.load(directory / f"r{resolution}-vectors.npy", mmap_mode="r")
            if data["vectors"].shape != (len(data["cells"]), self.meta["dimensions"]):
                raise ValueError(
                    "Cache metadata does not match the embedding arrays. Rebuild the cache."
                )
            self.levels[resolution] = data
        self.xyz = sphere(self.levels[5]["positions"])
        self.tree = KDTree(self.xyz)
        self.index = {int(cell): i for i, cell in enumerate(self.levels[5]["cells"])}
        self.lock = threading.Lock()
        self.maximum = max(self.meta["levels"]["5"]["contrast_quantiles"][3], 1e-12)
        # Each cache holds only one query; full-width vectors remain memory-mapped.
        self.scores = lru_cache(maxsize=1)(self._scores)

    def detail(self, index, score=None, distance=None):
        data = self.levels[5]
        lat, lon = data["positions"][index]
        contrast = float(data["contrast"][index])
        result = {
            "h3": f"{int(data['cells'][index]):x}",
            "lat": float(lat),
            "lon": float(lon),
            "count": int(data["count"][index]),
            "contrast": contrast if np.isfinite(contrast) else None,
        }
        if score is not None:
            result.update(score=float(score), distance_km=float(distance))
        return result

    def _scores(self, query) -> dict[int, np.ndarray]:
        index = self.index.get(int(query, 16))
        if index is None:
            raise ValueError("That source cell is not in this dataset.")
        vector = self.levels[5]["vectors"][index]
        return {
            level: np.clip(data["vectors"] @ vector, -1, 1) for level, data in self.levels.items()
        }

    def select(self, lat, lon, minimum_km=3000):
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("Choose a valid latitude and longitude.")
        if minimum_km not in (1000, 3000, 6000):
            raise ValueError("Choose a search distance of 1000, 3000, or 6000 km.")
        chord, index = self.tree.query(sphere([[lat, lon]])[0])
        nearest_km = 2 * 6371.0088 * np.arcsin(min(chord / 2, 1))
        if nearest_km > 100:
            raise ValueError("No sampled hex within 100 km. Choose a location on land.")
        query = self.detail(index)
        with self.lock:
            scores = self.scores(query["h3"])[5]
        distance = distances(self.xyz, self.xyz[index])
        candidates = np.flatnonzero(distance >= minimum_km)
        ranked = candidates[np.argsort(scores[candidates])[::-1]]
        selected = []
        for candidate in ranked:
            if selected and np.min(distances(self.xyz[selected], self.xyz[candidate])) < 500:
                continue
            selected.append(int(candidate))
            if len(selected) == 6:
                break
        matches = [self.detail(i, scores[i], distance[i]) for i in selected]
        return {
            "query": query,
            "matches": matches,
            "snap_km": float(nearest_km),
            "minimum_km": minimum_km,
        }

    def hexes(self, level, bbox, mode, query=None):
        if level not in self.levels or mode not in ("pca", "similarity", "contrast"):
            raise ValueError("Unknown resolution or view.")
        if len(bbox) != 4 or not np.isfinite(bbox).all() or bbox[2] < bbox[0] or bbox[3] < bbox[1]:
            raise ValueError("Invalid viewport bounds.")
        indices = visible_indices(self.levels[level], level, bbox)
        while len(indices) > 22000 and level > 3:
            level -= 1
            indices = visible_indices(self.levels[level], level, bbox)
        data = self.levels[level]
        values = data["contrast"][indices]
        if mode == "pca":
            rgb = data["rgb"][indices]
        else:
            if mode == "similarity":
                if query is None:
                    raise ValueError("Choose a source hex first.")
                with self.lock:
                    values = self.scores(query)[level][indices]
            rgb = colors(values, mode, self.maximum)
        features = []
        for offset, i in enumerate(indices):
            cell = int(data["cells"][i])
            lat, lon = data["positions"][i]
            value = float(values[offset])
            height = float(np.nan_to_num(data["contrast"][i])) / self.maximum
            props = {
                "h3": f"{cell:x}",
                "level": level,
                "count": int(data["count"][i]),
                "lat": float(lat),
                "lon": float(lon),
                "value": round(value, 6) if np.isfinite(value) else None,
                "color": "#" + "".join(f"{c:02x}" for c in rgb[offset]),
                "height": round(float(np.clip(height, 0, 1)) * 100000),
            }
            features.append(
                {
                    "type": "Feature",
                    "id": f"{cell:x}",
                    "properties": props,
                    "geometry": polygon(cell),
                }
            )
        return {
            "type": "FeatureCollection",
            "features": features,
            "resolution": level,
            "contrast_max": self.maximum,
            "count": len(features),
        }
