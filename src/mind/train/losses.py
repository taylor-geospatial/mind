"""Distillation losses and embedding regularizers."""

import inspect
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CosineMSELoss(nn.Module):
    """Weighted cosine and MSE losses on L2-normalized predictions and targets."""

    def __init__(self, mse_weight: float = 1.0, cos_weight: float = 1.0) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.cos_weight = cos_weight

    def forward(self, pred: Tensor, target: Tensor) -> dict[str, Tensor]:
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        cosine = (pred * target).sum(dim=-1)
        cos_loss = (1.0 - cosine).mean()
        mse = F.mse_loss(pred, target)
        loss = self.cos_weight * cos_loss + self.mse_weight * mse
        return {"loss": loss, "cosine": cosine.mean().detach(), "mse": mse.detach()}


class InfoNCELoss(nn.Module):
    """Symmetric in-batch contrastive loss (CLIP/SatCLIP style)."""

    def __init__(self, init_temperature: float = 0.07, learnable: bool = True) -> None:
        super().__init__()
        log_t = torch.log(torch.tensor(1.0 / init_temperature))
        self.logit_scale = nn.Parameter(log_t) if learnable else log_t

    def forward(self, pred: Tensor, target: Tensor) -> dict[str, Tensor]:
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * pred @ target.t()  # [B, B]
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        with torch.no_grad():
            cosine = (pred * target).sum(dim=-1).mean()
        return {"loss": loss, "cosine": cosine, "mse": F.mse_loss(pred, target).detach()}


def vicreg_regularizer(
    z: Tensor, var_weight: float = 1.0, cov_weight: float = 0.04, eps: float = 1e-4
) -> Tensor:
    """VICReg variance and covariance penalties on [B, D] embeddings."""
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = F.relu(1.0 - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / (n - 1)
    cov_loss = (cov - torch.diag(torch.diagonal(cov))).pow(2).sum() / d
    return var_weight * var_loss + cov_weight * cov_loss


def build_loss(name: str = "cosine_mse", **kwargs: Any) -> nn.Module:
    classes = {"cosine_mse": CosineMSELoss, "infonce": InfoNCELoss}
    if name not in classes:
        raise ValueError(f"unknown loss {name!r}")
    cls = classes[name]
    valid = set(inspect.signature(cls).parameters)
    unknown = set(kwargs) - valid
    if unknown:
        raise ValueError(
            f"loss {name!r} does not accept {sorted(unknown)} (accepts {sorted(valid)}); "
            "check for a typo'd config key"
        )
    return cls(**kwargs)
