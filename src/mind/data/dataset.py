"""Read and write sharded annual AEF pixel targets.

Shards store latitude/longitude coordinates and native int8 vectors for every year.
The training loader dequantizes and normalizes each year before averaging targets.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def write_shard(out_dir: str | Path, name: str, coords: np.ndarray, targets: np.ndarray) -> None:
    """Write one shard: ``coords[N,2] float32`` (lat, lon) and ``targets[N,Y,64] int8``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{name}.coords.npy", coords.astype(np.float32))
    np.save(out / f"{name}.targets.npy", targets)


def write_manifest(
    out_dir: str | Path,
    years: tuple[int, ...],
    shard_names: list[str],
    n_samples: int,
    dequant: str = "signed_square",
) -> None:
    out = Path(out_dir)
    manifest = {
        "years": list(years),
        "n_samples": n_samples,
        "shards": shard_names,
        "dequant": dequant,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


class AEFDataset:
    """Holds coordinates and raw annual targets loaded from pixel shards."""

    def __init__(
        self,
        coords: Tensor,
        targets: Tensor,
        years: tuple[int, ...],
        dequant: str = "signed_square",
    ) -> None:
        if coords.shape[0] != targets.shape[0]:
            raise ValueError("coords and targets disagree on N")
        if targets.shape[1] != len(years):
            raise ValueError("targets year axis disagrees with years")
        self.coords = coords  # [N, 2] float32 (lat, lon)
        self.targets = targets  # [N, Y, 64] int8
        self.years = years
        self.dequant = dequant

    @classmethod
    def from_dir(cls, data_dir: str | Path, dequant: str = "signed_square") -> "AEFDataset":
        d = Path(data_dir)
        manifest = json.loads((d / "manifest.json").read_text())
        coords = np.concatenate(
            [np.load(d / f"{s}.coords.npy") for s in manifest["shards"]], axis=0
        )
        targets = np.concatenate(
            [np.load(d / f"{s}.targets.npy") for s in manifest["shards"]], axis=0
        )
        return cls(
            torch.from_numpy(coords),
            torch.from_numpy(targets),
            tuple(manifest["years"]),
            dequant=manifest.get("dequant", dequant),
        )

    def __len__(self) -> int:
        return self.coords.shape[0]
