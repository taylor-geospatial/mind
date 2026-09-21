"""MIND's Equal-Earth residual SIREN encoder."""

import math

import torch
from torch import Tensor, nn

from mind.encoders.rff import equal_earth_projection


class SIRENLayer(nn.Module):
    """Linear layer with SIREN, FINER, or first-layer H-SIREN activation."""

    def __init__(
        self,
        in_f: int,
        out_f: int,
        w0: float = 1.0,
        is_first: bool = False,
        act: str = "siren",
        finer_k: float = 10.0,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_f, out_f)
        self.w0 = w0
        # H-SIREN applies sinh only in the first layer.
        self.act = act if (is_first or act != "hsiren") else "siren"
        with torch.no_grad():
            bound = 1.0 / in_f if is_first else math.sqrt(6.0 / in_f) / w0
            self.linear.weight.uniform_(-bound, bound)
            if self.act == "finer" and is_first:
                self.linear.bias.uniform_(-finer_k, finer_k)
            else:
                nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        z = self.linear(x)
        if self.act == "finer":
            z = (z.abs() + 1.0) * z
        elif self.act == "hsiren":
            z = torch.sinh(z)
        return torch.sin(self.w0 * z)


class ReSIRENLocationEncoder(nn.Module):
    """Residual SIREN over Equal-Earth coordinates, with optional Fourier year features."""

    def __init__(
        self,
        embed_dim: int = 3072,
        out_dim: int | None = 64,
        depth: int = 12,
        use_year: bool = False,
        w0_first: float = 30.0,
        w0: float = 1.0,
        variant: str = "siren",
        finer_k: float = 10.0,
        year_frequencies: int = 8,
        year_ref: float = 2021.0,
        year_scale: float = 4.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.use_year = use_year
        self.year_ref = year_ref
        self.year_scale = year_scale
        self.register_buffer("year_freqs", 2.0 ** torch.arange(year_frequencies).float())
        in_dim = 2 + (2 * year_frequencies if use_year else 0)
        self.first = SIRENLayer(
            in_dim, embed_dim, w0=w0_first, is_first=True, act=variant, finer_k=finer_k
        )
        self.blocks = nn.ModuleList(
            [
                SIRENLayer(embed_dim, embed_dim, w0=w0, act=variant, finer_k=finer_k)
                for _ in range(depth)
            ]
        )
        self.head: nn.Module = (
            nn.Identity()
            if (out_dim is None or out_dim == embed_dim)
            else nn.Linear(embed_dim, out_dim)
        )

    def _year_feats(self, year: Tensor) -> Tensor:
        yn = ((year.float() - self.year_ref) / self.year_scale).reshape(-1, 1)  # [B, 1]
        ang = yn * self.year_freqs.unsqueeze(0)  # [B, F]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # [B, 2F]

    def forward(
        self,
        latlon: Tensor,
        year: Tensor | None = None,
        return_features: bool = False,
    ) -> Tensor:
        squeeze_k = latlon.dim() == 2
        if squeeze_k:
            latlon = latlon.unsqueeze(1)  # [B, 1, 2] -- single point
        b, k, _ = latlon.shape
        loc = equal_earth_projection(latlon.reshape(b * k, 2))  # [B*K, 2]
        if self.use_year:
            if year is None:
                raise ValueError("year is required when use_year=True")
            yf = self._year_feats(year).unsqueeze(1).expand(b, k, -1).reshape(b * k, -1)
            x = torch.cat([loc, yf], dim=-1)
        else:
            x = loc
        h = self.first(x)
        for blk in self.blocks:
            h = h + blk(h)
        h = h.reshape(b, k, self.embed_dim)
        pooled = h.mean(1)  # K=1 recovers the single point
        return pooled if return_features else self.head(pooled)
