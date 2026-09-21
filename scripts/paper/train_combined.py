"""Train MIND on AEF pixel year means and three frozen coordinate teachers."""

import argparse
from pathlib import Path

import torch

from mind.encoders import ReSIRENLocationEncoder
from mind.train.multitarget_loop import CombinedDataset, train_combined

# (w_aef, w_clim, w_geo, w_sinr)
WEIGHTS = {
    "aef_only": (1.0, 0.0, 0.0, 0.0),
    "aef_clim": (1.0, 1.0, 0.0, 0.0),
    "clim_only": (0.0, 1.0, 0.0, 0.0),
    "geo_only": (0.0, 0.0, 1.0, 0.0),
    "aef_geo": (1.0, 0.0, 1.0, 0.0),
    "aef_clim_geo": (1.0, 1.0, 1.0, 0.0),
    "aef_sinr": (1.0, 0.0, 0.0, 1.0),
    "aef_clim_geo_sinr": (1.0, 1.0, 1.0, 1.0),
    "socio": (0.0, 1.0, 1.0, 0.0),
    "env": (1.0, 1.0, 0.0, 1.0),
}
# (cos_weight, mse_weight, infonce_weight)
LOSSES = {
    "cosine_mse": (1.0, 1.0, 0.0),
    "cosine": (1.0, 0.0, 0.0),
    "mse": (0.0, 1.0, 0.0),
    "infonce": (0.0, 0.0, 1.0),
    "cosine_infonce": (1.0, 0.0, 1.0),
    "cosine_mse_infonce": (1.0, 1.0, 1.0),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=list(WEIGHTS), default="aef_clim_geo_sinr")
    p.add_argument("--name", required=True)
    p.add_argument("--data", required=True, help="local MINDSET GeoParquet or AEF pixel directory")
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument(
        "--lr", type=float, default=3e-4, help="AdamW peak LR (scale up with batch size)"
    )
    p.add_argument("--embed-dim", type=int, default=3072, help="ReSIREN trunk width")
    p.add_argument(
        "--loss", choices=list(LOSSES), default="cosine_mse", help="distillation loss (ablation)"
    )
    p.add_argument(
        "--w0-first", type=float, default=30.0, help="SIREN first-layer w0 (frequency knob)"
    )
    p.add_argument("--w-clim", type=float, default=None, help="override Climplicit teacher weight")
    p.add_argument("--w-geo", type=float, default=None, help="override GeoCLIP teacher weight")
    p.add_argument("--w-sinr", type=float, default=None, help="override SINR teacher weight")
    p.add_argument(
        "--vicreg-var",
        type=float,
        default=0.0,
        help="VICReg variance (anti-collapse) weight on pooled",
    )
    p.add_argument(
        "--vicreg-cov",
        type=float,
        default=0.0,
        help="VICReg covariance (decorrelation) weight on pooled",
    )
    p.add_argument(
        "--standardize-teachers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="PHI-S global standardization: balance the per-teacher MSE by standardizing each "
        "teacher target (per-channel mean + scalar RMS) so socio teachers aren't starved",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="torch/numpy seed (for mean+-std over seeds)"
    )
    p.add_argument(
        "--mrl-teachers",
        default="",
        help="comma list of teachers (aef,clim,geo,sinr) that supervise the nested prefixes; empty = "
        "all active teachers. Others still supervise the full width.",
    )
    p.add_argument(
        "--matryoshka",
        default="64,128,256,512,1024,2048",
        help="comma list of nested prefix dims (e.g. 128,256,512,1024,2048) trained to be "
        "independently usable, so one model truncates to any of these dims at deploy",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()
    Path(args.name).parent.mkdir(parents=True, exist_ok=True)

    import numpy as np

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = args.device
    ds = CombinedDataset.from_dir(args.data, dev)
    print(f"pixels: N={len(ds)} years={ds.years}", flush=True)
    w_aef, w_clim, w_geo, w_sinr = WEIGHTS[args.variant]
    if args.w_clim is not None:
        w_clim = args.w_clim
    if args.w_geo is not None:
        w_geo = args.w_geo
    if args.w_sinr is not None:
        w_sinr = args.w_sinr
    cos_w, mse_w, infonce_w = LOSSES[args.loss]

    enc = ReSIRENLocationEncoder(
        embed_dim=args.embed_dim,
        out_dim=ds.aef.shape[-1],
        depth=12,
        use_year=False,
        w0_first=args.w0_first,
    )
    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project="mind", name=args.name, config=vars(args))

    def log_fn(step: int, m: dict[str, float]) -> None:
        if step % 200 == 0:
            print(f"step {step}: " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
        if run is not None:
            run.log({f"train/{k}": v for k, v in m.items()}, step=step)

    ckpt = str(Path.cwd() / f"{args.name}.pt")
    train_combined(
        enc,
        ds,
        steps=args.steps,
        batch_size=args.batch_size,
        w_aef=w_aef,
        w_clim=w_clim,
        w_geo=w_geo,
        w_sinr=w_sinr,
        cos_weight=cos_w,
        mse_weight=mse_w,
        infonce_weight=infonce_w,
        log_fn=log_fn,
        ckpt_path=ckpt,
        lr=args.lr,
        standardize_teachers=args.standardize_teachers,
        vicreg_var_weight=args.vicreg_var,
        vicreg_cov_weight=args.vicreg_cov,
        matryoshka_dims=tuple(int(m) for m in args.matryoshka.split(",") if m.strip()),
        mrl_teachers=tuple(t.strip() for t in args.mrl_teachers.split(",") if t.strip()),
    )
    print(f"saved encoder -> {ckpt}", flush=True)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
