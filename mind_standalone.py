"""Standalone MIND loading and coordinate embedding.

from mind_standalone import load_mind, embed
model = load_mind("mind.safetensors")
features = embed(model, lats, lons, dim=64)
"""

import math

import numpy as np
import torch
from torch import Tensor, nn

# Equal-Earth projection polynomial constants (Savric et al. 2018).
EE_A1, EE_A2, EE_A3, EE_A4 = 1.340264, -0.081106, 0.000893, 0.003796
EE_SCALE = 66.50336
SQRT3 = math.sqrt(3.0)


def equal_earth_projection(latlon: Tensor) -> Tensor:
    """Project (lat, lon) in degrees -> Equal-Earth (x, y). ``latlon`` is [..., 2] (lat, lon)."""
    lat, lon = torch.deg2rad(latlon[..., 0]), torch.deg2rad(latlon[..., 1])
    theta = torch.asin((SQRT3 / 2.0) * torch.sin(lat))
    denom = 3.0 * (9.0 * EE_A4 * theta**8 + 7.0 * EE_A3 * theta**6 + 3.0 * EE_A2 * theta**2 + EE_A1)
    x = (2.0 * SQRT3 * lon * torch.cos(theta)) / denom
    y = EE_A4 * theta**9 + EE_A3 * theta**7 + EE_A2 * theta**3 + EE_A1 * theta
    return (torch.stack((x, y), dim=-1) * EE_SCALE) / 180.0


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
            nn.Identity() if out_dim in (None, embed_dim) else nn.Linear(embed_dim, out_dim)
        )

    def year_feats(self, year: Tensor) -> Tensor:
        ang = ((year.float() - self.year_ref) / self.year_scale).reshape(-1, 1) * self.year_freqs
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)

    def forward(
        self, latlon: Tensor, year: Tensor | None = None, return_features: bool = False
    ) -> Tensor:
        loc = equal_earth_projection(latlon)
        if self.use_year:
            if year is None:
                raise ValueError("year is required when use_year=True")
            loc = torch.cat([loc, self.year_feats(year)], dim=-1)
        h = self.first(loc)
        for blk in self.blocks:
            h = h + blk(h)
        return h if return_features else self.head(h)


def load_mind(ckpt_path: str = "mind.safetensors", device: str = "cpu") -> ReSIRENLocationEncoder:
    """Load a trained MIND checkpoint (.safetensors or .pt), inferring its architecture from the weights."""
    if str(ckpt_path).endswith(".safetensors"):
        from safetensors.torch import load_file

        state = {k: v.float() for k, v in load_file(ckpt_path).items()}
    else:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
    embed_dim, in_dim = state["first.linear.weight"].shape
    depth = sum(1 for k in state if k.startswith("blocks.") and k.endswith(".linear.weight"))
    out_dim = state["head.weight"].shape[0] if "head.weight" in state else embed_dim
    year_freqs = int(state["year_freqs"].shape[0])
    use_year = in_dim == 2 + 2 * year_freqs
    model = ReSIRENLocationEncoder(
        embed_dim, out_dim, depth, use_year=use_year, year_frequencies=year_freqs
    )
    if "head.weight" in state and isinstance(model.head, nn.Identity):
        # Restore an explicit head even when its width matches the trunk.
        model.head = nn.Linear(embed_dim, out_dim)
    model.load_state_dict(state)
    return model.to(device).eval()


def from_pretrained(
    repo: str = "taylor-geospatial/MIND", filename: str = "mind.safetensors", device: str = "cpu"
) -> ReSIRENLocationEncoder:
    """Download the released weights from the Hugging Face Hub and load them (needs huggingface_hub)."""
    from huggingface_hub import hf_hub_download

    return load_mind(hf_hub_download(repo, filename), device=device)


@torch.no_grad()
def embed(
    model: ReSIRENLocationEncoder,
    lat: np.ndarray,
    lon: np.ndarray,
    dim: int = 64,
    year: int | np.ndarray | None = None,
    device: str | None = None,
    batch_size: int = 8192,
    half: bool = False,
    feature: str = "pooled",
) -> np.ndarray:
    """Return [N, dim] embeddings from coordinates in degrees.

    ``feature="pooled"`` selects the trunk prefix; ``feature="head"`` selects the
    reconstruction head. ``device`` defaults to CUDA when available; ``half``
    enables float16 autocasting on CUDA.
    """
    if feature not in {"pooled", "head"}:
        raise ValueError("feature must be 'pooled' or 'head'")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = torch.device(device).type
    latlon = torch.stack(
        [torch.as_tensor(np.asarray(lat)), torch.as_tensor(np.asarray(lon))], dim=1
    ).float()
    yr = (
        None
        if year is None
        else torch.as_tensor(np.broadcast_to(year, (len(lat),)), dtype=torch.float32)
    )
    out = []
    for a in range(0, len(latlon), batch_size):
        chunk = latlon[a : a + batch_size].to(device)
        yc = yr[a : a + batch_size].to(device) if yr is not None else None
        with torch.autocast("cuda", dtype=torch.float16, enabled=half and device_type == "cuda"):
            emb = model(chunk, yc, return_features=(feature == "pooled"))
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out)[:, :dim]


if __name__ == "__main__":
    model = load_mind("mind.safetensors")
    e = embed(model, np.array([37.77, 51.51]), np.array([-122.42, -0.13]))
    print("embedding shape:", e.shape)  # (2, 64)
