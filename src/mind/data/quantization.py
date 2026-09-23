"""Dequantization for AlphaEarth int8 embeddings (nodata -128).

Apply signed-square decoding before L2 normalization.
"""

import torch
from torch import Tensor

AEF_NODATA = -128
SIGNED_SQUARE_SCALE = 127.5


def dequantize_signed_square(x: Tensor) -> Tensor:
    """Decode to float32 with ``sign(x) * (abs(x) / 127.5)**2``."""
    f = x.float()
    return torch.sign(f) * (f.abs() / SIGNED_SQUARE_SCALE).pow(2)


def dequantize(x: Tensor, method: str = "signed_square") -> Tensor:
    if method == "signed_square":
        return dequantize_signed_square(x)
    if method == "linear":
        return x.float() / 127.0
    if method == "none":
        return x.float()  # already-float targets (e.g. precomputed patch means)
    raise ValueError(f"unknown dequantization method {method!r}")
