"""Embedding metrics logged during training."""

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def effective_rank(emb: Tensor) -> float:
    """Participation ratio ``sum(s)^2 / sum(s^2)`` of centered singular values."""
    s = torch.linalg.svdvals(emb.float() - emb.float().mean(0, keepdim=True))
    return float((s.sum() ** 2) / (s.pow(2).sum().clamp_min(1e-12)))


@torch.no_grad()
def mean_pairwise_cosine(emb: Tensor, max_n: int = 2048) -> float:
    """Mean off-diagonal cosine similarity for the first ``max_n`` embeddings."""
    x = F.normalize(emb[:max_n].float(), dim=-1)
    sim = x @ x.t()
    n = x.shape[0]
    off = (sim.sum() - sim.diagonal().sum()) / (n * (n - 1))
    return float(off)


@torch.no_grad()
def alignment_uniformity(pred: Tensor, target: Tensor, max_n: int = 4096) -> tuple[float, float]:
    """Wang & Isola alignment (mean ||pred-target||^2 on unit sphere) and uniformity of pred."""
    p = F.normalize(pred[:max_n].float(), dim=-1)
    t = F.normalize(target[:max_n].float(), dim=-1)
    alignment = (p - t).pow(2).sum(dim=-1).mean()
    sq_dist = torch.pdist(p).pow(2)
    uniformity = sq_dist.mul(-2).exp().mean().clamp_min(1e-12).log()
    return float(alignment), float(uniformity)


@torch.no_grad()
def embedding_health(pred: Tensor, target: Tensor) -> dict[str, float]:
    """Compute rank, cosine similarity, alignment, and uniformity for a batch."""
    align, uniform = alignment_uniformity(pred, target)
    return {
        "eff_rank": effective_rank(pred),
        "pairwise_cos": mean_pairwise_cosine(pred),
        "alignment": align,
        "uniformity": uniform,
    }
