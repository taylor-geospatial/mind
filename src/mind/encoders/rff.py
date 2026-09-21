"""Equal-Earth projection, adapted from UniGeoCLIP under the MIT License.

Source: https://github.com/gastruc/unigeoclip
"""

import math

import torch
from torch import Tensor

# Equal-Earth projection polynomial constants (Savric et al., 2018).
_A1 = 1.340264
_A2 = -0.081106
_A3 = 0.000893
_A4 = 0.003796
# UniGeoCLIP projection scale.
_SF = 66.50336
_SQRT3 = math.sqrt(3.0)


def equal_earth_projection(latlon: Tensor) -> Tensor:
    """Project [..., 2] (latitude, longitude) degrees to Equal-Earth (x, y)."""
    latitude_rad = torch.deg2rad(latlon[..., 0])
    longitude_rad = torch.deg2rad(latlon[..., 1])
    sin_theta = (_SQRT3 / 2.0) * torch.sin(latitude_rad)
    theta = torch.asin(sin_theta)
    denominator = 3.0 * (9.0 * _A4 * theta**8 + 7.0 * _A3 * theta**6 + 3.0 * _A2 * theta**2 + _A1)
    x = (2.0 * _SQRT3 * longitude_rad * torch.cos(theta)) / denominator
    y = _A4 * theta**9 + _A3 * theta**7 + _A2 * theta**3 + _A1 * theta
    return (torch.stack((x, y), dim=-1) * _SF) / 180.0
