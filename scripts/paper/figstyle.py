"""Shared palette and matplotlib defaults for paper figures."""

import matplotlib
from matplotlib.colors import LinearSegmentedColormap

BROWN = "#3b1e1c"
PERIWINKLE = "#80a0d8"
IVORY = "#f4f4eb"
RED = "#ff4f2c"
LIGHTBLUE = "#a7d0dc"
GREEN = "#cff29e"

OURS = RED
TEACH = BROWN
BASE = PERIWINKLE
INK = "#241a18"
MACRO = "#241a18"
PANEL = IVORY
FAM = {"socio": RED, "env": PERIWINKLE, "land": BROWN, "other": "#9c9c9c"}

_TGWARM = LinearSegmentedColormap.from_list("tgwarm", [IVORY, RED, BROWN])
if "tgwarm" not in matplotlib.colormaps:
    matplotlib.colormaps.register(_TGWARM)
SEQ = "tgwarm"


def apply_rc() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
