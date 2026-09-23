"""torch.hub entrypoints for MIND.

import torch
model = torch.hub.load("taylor-geospatial/mind", "mind")  # downloads weights from the HF Hub
"""

dependencies = ["torch", "numpy", "safetensors", "huggingface_hub"]

from mind_standalone import ReSIRENLocationEncoder, from_pretrained  # noqa: E402


def mind(
    pretrained: bool = True, device: str = "cpu", repo: str = "taylor-geospatial/MIND"
) -> ReSIRENLocationEncoder:
    """Return the MIND encoder; ``pretrained`` downloads weights from the Hugging Face Hub."""
    if pretrained:
        return from_pretrained(repo, device=device)
    return ReSIRENLocationEncoder(embed_dim=3072, out_dim=None, depth=12).to(device).eval()
