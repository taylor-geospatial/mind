"""GPU LBFGS logistic regression for single-label probing.

Adapted from torchgeo-bench (src/torchgeo_bench/linear.py). The objective matches
sklearn: mean(CE) + 0.5 / (C * n) * ||W||^2.
"""

import numpy as np
import torch
from torch import Tensor


class LBFGSLogReg:
    """Single-label logistic regression solved with torch LBFGS (sklearn-matched objective)."""

    def __init__(
        self,
        c: float = 1.0,
        max_iter: int = 2000,
        tol: float = 1e-6,
        device: str = "cuda",
        seed: int = 0,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "LBFGSLogReg(device='cuda') requested but no CUDA device is available; "
                "pass device='cpu' explicitly if that is really intended"
            )
        self.c = float(c)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.device = torch.device(device)
        self.seed = seed
        self.model: torch.nn.Linear | None = None

    def fit(self, x: Tensor, y: Tensor) -> "LBFGSLogReg":
        torch.manual_seed(self.seed)
        x = x.to(self.device, torch.float32).contiguous()
        y = y.to(self.device, torch.long).contiguous()
        n, d = x.shape
        n_classes = int(y.max().item()) + 1
        model = torch.nn.Linear(d, n_classes, bias=True).to(self.device)
        torch.nn.init.zeros_(model.weight)
        torch.nn.init.zeros_(model.bias)
        crit = torch.nn.CrossEntropyLoss(reduction="mean")
        reg = 0.5 * (1.0 / self.c) / float(n)  # matches sklearn scaling
        opt = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=self.max_iter,
            history_size=10,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=self.tol * 0.1,
        )

        def closure() -> Tensor:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y) + reg * model.weight.mul(model.weight).sum()
            loss.backward()
            return loss

        opt.step(closure)
        model.eval()
        self.model = model
        return self

    @torch.no_grad()
    def predict(self, x: Tensor) -> np.ndarray:
        assert self.model is not None
        x = x.to(self.device, torch.float32).contiguous()
        return self.model(x).argmax(1).cpu().numpy()
