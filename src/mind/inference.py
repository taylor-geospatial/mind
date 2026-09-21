"""Load MIND weights and embed latitude/longitude coordinates."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from mind.encoders import ReSIRENLocationEncoder


def load_mind(
    ckpt_path: str | Path, device: str = "cpu", variant: str = "siren", w0_first: float = 30.0
) -> ReSIRENLocationEncoder:
    """Load a state dict, an older metadata-wrapped checkpoint, or safetensors weights.

    Width, depth, head size and year conditioning are inferred from the weights. The released
    checkpoint uses the default SIREN activation and input frequency.
    """
    if Path(ckpt_path).suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(ckpt_path))
    else:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            args = state.get("meta", {}).get("args", {})
            encoder = args.get("encoder", variant)
            variant = "siren" if encoder == "resiren" else encoder
            w0_first = args.get("w0_first", w0_first)
            state = state["state_dict"]
    if "first.linear.weight" not in state:
        raise ValueError("Expected a MIND ReSIREN checkpoint")
    if variant not in {"siren", "finer", "hsiren"}:
        raise ValueError(f"Unsupported ReSIREN activation: {variant}")
    width, inputs = state["first.linear.weight"].shape
    depth = sum(k.startswith("blocks.") and k.endswith(".linear.weight") for k in state)
    out_dim = state["head.weight"].shape[0] if "head.weight" in state else width
    frequencies = len(state["year_freqs"])
    if inputs not in {2, 2 + 2 * frequencies}:
        raise ValueError(f"Unexpected coordinate input dimension: {inputs}")
    model = ReSIRENLocationEncoder(
        embed_dim=width,
        out_dim=out_dim,
        depth=depth,
        use_year=inputs != 2,
        year_frequencies=frequencies,
        variant=variant,
        w0_first=w0_first,
    )
    if "head.weight" in state and isinstance(model.head, nn.Identity):
        model.head = nn.Linear(width, out_dim)
    model.load_state_dict(state)
    return model.to(device).eval()


def from_pretrained(
    repo: str = "taylor-geospatial/MIND", filename: str = "mind.safetensors", device: str = "cpu"
) -> ReSIRENLocationEncoder:
    """Download and load the released MIND checkpoint."""
    from huggingface_hub import hf_hub_download

    return load_mind(hf_hub_download(repo, filename), device=device)


@torch.no_grad()
def embed(
    model: ReSIRENLocationEncoder,
    lat: np.ndarray,
    lon: np.ndarray,
    dim: int | None = 64,
    year: int | np.ndarray | None = None,
    batch_size: int = 8192,
    feature: str = "pooled",
) -> np.ndarray:
    """Return float32 embeddings, keeping the first ``dim`` channels.

    Inference uses the model's device. ``dim=None`` keeps the full width. ``feature='head'``
    selects the reconstruction head, which is the output embedding for MIND-small.
    """
    if feature not in {"pooled", "head"}:
        raise ValueError("feature must be 'pooled' or 'head'")
    width = model.embed_dim if feature == "pooled" else (model.out_dim or model.embed_dim)
    dim = width if dim is None else dim
    if not 0 < dim <= width or batch_size < 1:
        raise ValueError("dim must be within the embedding width and batch_size must be positive")
    lat, lon = np.asarray(lat), np.asarray(lon)
    if lat.ndim != 1 or lat.shape != lon.shape:
        raise ValueError("lat and lon must be one-dimensional arrays of equal length")
    if model.use_year and year is None:
        raise ValueError("year is required by this checkpoint")
    coords = torch.as_tensor(np.stack([lat, lon], axis=1), dtype=torch.float32)
    years = None if year is None else np.broadcast_to(year, lat.shape).copy()
    device = next(model.parameters()).device
    result = np.empty((len(lat), dim), dtype=np.float32)
    for start in range(0, len(lat), batch_size):
        stop = start + batch_size
        yr = None if years is None else torch.as_tensor(years[start:stop], device=device)
        values = model(coords[start:stop].to(device), yr, return_features=feature == "pooled")
        result[start:stop] = values[:, :dim].float().cpu().numpy()
    return result
