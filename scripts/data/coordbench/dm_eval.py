"""Discover DeepMind evaluation CSVs (Zenodo 16585402) and their per-file licenses."""

from pathlib import Path

CC_BY_NC = "CC-BY-NC-4.0 (iNaturalist)"
OGL_CANADA = "Open Government Licence - Canada"
CC_BY = "CC-BY-4.0"

PER_FILE_LICENSE = {
    "us_trees": CC_BY_NC,
    "canada_crops_coarse": OGL_CANADA,
    "canada_crops_fine": OGL_CANADA,
}


def license_for(stem: str) -> str:
    return PER_FILE_LICENSE.get(stem, CC_BY)


def discover(raw_dir: Path) -> list[Path]:
    """Find evaluation CSVs, excluding trial-group files."""
    return sorted(p for p in raw_dir.rglob("*.csv") if "trial_groups" not in p.name)
