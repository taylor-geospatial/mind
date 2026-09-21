import json
import tempfile
import unittest
from pathlib import Path

import h3.api.basic_int as h3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.hex_lab.analysis import Explorer, polygon
from scripts.hex_lab.metrics import aggregate_contrast, neighbor_contrast
from scripts.hex_lab.prepare import build, embedding_columns


class HexLabTests(unittest.TestCase):
    def test_column_ranges(self) -> None:
        names = [f"emb_{i}" for i in range(128)][::-1]
        assert embedding_columns(names) == list(reversed(names))
        assert embedding_columns(names, "64:128") == [f"emb_{i}" for i in range(64, 128)]
        for selection in ("128:192", "-1:64", "64:64", "64"):
            with self.subTest(selection=selection), np.testing.assert_raises(ValueError):
                embedding_columns(names, selection)
        with np.testing.assert_raises(ValueError):
            embedding_columns(["emb_0", "emb_00", "emb_1", "emb_2"])

    def test_wide_parquet_round_trip(self) -> None:
        cells = sorted(
            {
                c
                for lat, lon in ((0, 0), (40, 80), (-15, -65), (52, 179.8))
                for c in h3.grid_disk(h3.latlng_to_cell(lat, lon, 5), 1)
            }
        )
        rng = np.random.default_rng(42)
        values = rng.normal(size=(len(cells), 128)).astype(np.float32)
        # A silent truncation to 64 dimensions would make every score one.
        values[:, :64] = 1
        values[:, 64:] *= 4
        counts = np.arange(1, len(cells) + 1)
        data = {"h3": [f"{c:x}" for c in cells], "count": counts}
        data.update({f"emb_{i}": values[:, i] for i in reversed(range(128))})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "wide.parquet"
            pq.write_table(pa.table(data), source, row_group_size=5)
            build(source, root / "full", samples=20, batch_size=3)
            explorer = Explorer(root / "full")
            assert explorer.meta["dimensions"] == 128
            assert explorer.meta["pca"]["fit_rows"] == 20
            unit = values / np.linalg.norm(values, axis=1, keepdims=True)
            np.testing.assert_allclose(explorer.levels[5]["vectors"], unit, atol=1e-7)
            query = f"{cells[0]:x}"
            np.testing.assert_allclose(explorer.scores(query)[5], unit @ unit[0], atol=1e-6)
            assert float(explorer.scores(query)[5][-1]) < 0.9
            parent = int(explorer.levels[4]["cells"][0])
            ids = [i for i, cell in enumerate(cells) if h3.cell_to_parent(cell, 4) == parent]
            mean = np.average(values[ids], axis=0, weights=counts[ids])
            np.testing.assert_allclose(
                explorer.levels[4]["vectors"][0], mean / np.linalg.norm(mean), atol=1e-6
            )
            selection = explorer.select(*h3.cell_to_latlng(cells[0]), minimum_km=6000)
            assert selection["matches"]
            for match in selection["matches"]:
                assert match["distance_km"] >= 6000
            result = explorer.hexes(5, [170, 40, 190, 60], "similarity", query)
            assert result["count"] > 0
            json.dumps(result, allow_nan=False)
            build(source, root / "late", dims="64:128", samples=20, batch_size=3)
            later = Explorer(root / "late")
            assert later.meta["embedding_columns"][0] == "emb_64"
            selected = values[:, 64:]
            expected = selected / np.linalg.norm(selected, axis=1, keepdims=True)
            np.testing.assert_allclose(later.levels[5]["vectors"], expected, atol=1e-7)
            with np.testing.assert_raises(ValueError):
                build(source, root / "full")
            with np.testing.assert_raises(ValueError):
                build(source, root / "invalid", dims="64:128", pca_mode="stored")
            assert not (root / "invalid").exists()

    def test_neighbors_and_missing_cells(self) -> None:
        first = h3.latlng_to_cell(20, 20, 5)
        second = next(c for c in h3.grid_disk(first, 1) if c != first)
        isolated = h3.latlng_to_cell(-40, -70, 5)
        values = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        result = neighbor_contrast(np.array([first, second, isolated]), values, batch_size=1)
        np.testing.assert_allclose(result[:2], [1, 1])
        assert np.isnan(result[2])
        overview = aggregate_contrast(result, np.array([1, 2, 100]), np.array([0, 0, 0]))
        np.testing.assert_allclose(overview, [1])

    def test_dateline_polygon(self) -> None:
        cell = h3.latlng_to_cell(52, 179.999, 5)
        ring = np.array(polygon(cell)["coordinates"][0])
        assert float(np.max(np.abs(np.diff(ring[:, 0])))) < 5
        np.testing.assert_array_equal(ring[0], ring[-1])

    def test_failed_build_does_not_leave_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cells = [f"{c:x}" for c in h3.grid_disk(h3.latlng_to_cell(0, 0, 5), 1)]
            source = root / "invalid.parquet"
            pq.write_table(
                pa.table(
                    {
                        "h3": cells,
                        "count": np.zeros(len(cells)),
                        **{f"emb_{i}": np.ones(len(cells)) for i in range(3)},
                    }
                ),
                source,
            )
            with np.testing.assert_raises(ValueError):
                build(source, root / "output")
            assert not (root / "output").exists()
            assert sorted(p.name for p in root.iterdir()) == ["invalid.parquet"]
