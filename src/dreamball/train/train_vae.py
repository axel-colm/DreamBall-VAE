"""Entraînement du VAE.

Usage :
    python -m dreamball.train.train_vae
    python -m dreamball.train.train_vae --scenes box ramp valley stairs --epochs 20

Sorties :
    checkpoints/vae/best.pt, last.pt
    outputs/vae/recon_epoch_XXX.png   # ligne du haut : vraies images ; du bas : reconstructions
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler

from dreamball.data.datasets import FrameDataset, load_split
from dreamball.models.vae import VAE, VAEConfig, vae_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/raw"))
    p.add_argument("--scenes", nargs="+", default=None, help="défaut : toutes les scènes du split")
    p.add_argument("--z-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument(
        "--samples-per-epoch",
        type=int,
        default=200_000,
        help="images tirées par epoch (le train en a 750k, très corrélées entre frames voisines)",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--ball-weight", type=float, default=20.0)
    p.add_argument("--decor-weight", type=float, default=10.0)
    p.add_argument("--free-bits", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--ckpt-dir", type=Path, default=Path("checkpoints/vae"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/vae"))
    p.add_argument("--no-compile", action="store_true", help="désactive torch.compile (démarrage plus rapide)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def save_recon_grid(model: VAE, batch: torch.Tensor, path: Path) -> None:
    """Vraies images en haut, reconstructions (à partir de mu, sans bruit) en bas."""
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        mu, _ = model.encode(batch)
        recon = model.decode(mu).float()
    rows = [torch.cat(list(t[:, 0]), dim=1) for t in (batch, recon)]  # (H, N*W) par ligne
    grid = (torch.cat(rows, dim=0).clamp(0, 1) * 255).byte().cpu().numpy()
    Image.fromarray(grid).save(path)


@torch.no_grad()
def evaluate(model: VAE, loader: DataLoader, device: torch.device, loss_kwargs: dict) -> dict[str, float]:
    model.eval()
    totals, n = {}, 0
    for x in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            mu, logvar = model.encode(x)
            recon = model.decode(mu)  # évaluation déterministe : pas d'échantillonnage
        losses = vae_loss(recon.float(), x, mu.float(), logvar.float(), **loss_kwargs)
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item() * len(x)
        n += len(x)
    return {k: v / n for k, v in totals.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # taille d'entrée fixe : cuDNN choisit l'algo le plus rapide
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ données
    t0 = time.perf_counter()
    train_data = load_split(args.data, "train", args.scenes)
    # Val restreint aux scènes du train : une scène jamais vue (ex. pegs, réservée au test)
    # fausserait la sélection du meilleur checkpoint
    val_data = load_split(args.data, "val", train_data.scene_names)
    print(f"Données : train {train_data.frames.shape}, val {val_data.frames.shape} ({time.perf_counter() - t0:.0f}s)")

    train_ds, val_ds = FrameDataset(train_data), FrameDataset(val_data)
    sampler = RandomSampler(train_ds, num_samples=args.samples_per_epoch)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    # Validation : un sous-ensemble fixe (1 frame sur 10) pour que les epochs soient comparables
    val_subset = torch.utils.data.Subset(val_ds, range(0, len(val_ds), 10))
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, num_workers=args.workers)

    # Batch fixe pour les images de suivi : 2 épisodes par scène, à t = 20 et t = 80
    T = val_data.frames.shape[1]
    vis_idx = [ep * T + t for s in range(len(val_data.scene_names)) for ep in np.flatnonzero(val_data.scene_ids == s)[:2] for t in (20, 80)]
    vis_batch = torch.stack([val_ds[i] for i in vis_idx]).to(device)

    # ------------------------------------------------------------------ modèle
    cfg = VAEConfig(in_channels=vis_batch.shape[1], img_size=vis_batch.shape[-1], z_dim=args.z_dim)
    model = VAE(cfg).to(device)
    # torch.compile fusionne les opérations (norm + activation...) : ~2x plus rapide ici.
    # On garde `model` pour sauvegarder et évaluer, `train_model` pour l'entraînement.
    train_model = model if args.no_compile else torch.compile(model)
    print(f"VAE : {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M paramètres, z_dim={cfg.z_dim}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.05
    )
    loss_kwargs = {"beta": args.beta, "ball_weight": args.ball_weight, "decor_weight": args.decor_weight, "free_bits": args.free_bits}

    # ------------------------------------------------------------------ boucle
    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            # bf16 : ~2x plus rapide sur GPU récent, sans GradScaler (contrairement au fp16)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                recon, mu, logvar = train_model(x)
            losses = vae_loss(recon.float(), x, mu.float(), logvar.float(), **loss_kwargs)

            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            for k in running:
                running[k] += losses[k].item()

        train_metrics = {k: v / steps_per_epoch for k, v in running.items()}
        val_metrics = evaluate(model, val_loader, device, loss_kwargs)
        elapsed = time.perf_counter() - t0
        print(
            f"epoch {epoch:3d}  "
            f"train loss {train_metrics['loss']:8.1f} (recon {train_metrics['recon']:8.1f}, kl {train_metrics['kl']:5.1f})  "
            f"val loss {val_metrics['loss']:8.1f} (recon {val_metrics['recon']:8.1f})  {elapsed:4.0f}s"
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        save_recon_grid(model, vis_batch, args.out_dir / f"recon_epoch_{epoch:03d}.png")
        extra = {"epoch": epoch, "val": val_metrics, "args": {k: str(v) for k, v in vars(args).items()}}
        model.save(args.ckpt_dir / "last.pt", **extra)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            model.save(args.ckpt_dir / "best.pt", **extra)

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Meilleure val loss : {best_val:.1f} -> {args.ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
