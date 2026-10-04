"""VAE convolutionnel : image (C, 128, 128) <-> latent z (z_dim,).

Encodeur : 5 convolutions stride 2 (128 -> 4), puis deux couches linéaires -> mu, logvar.
Décodeur : symétrique, avec upsample + conv (moins d'artefacts en damier que ConvTranspose).

Reparamétrisation : z = mu + sigma * eps, eps ~ N(0, I). Le tirage aléatoire passe dans
eps, ce qui laisse le gradient traverser mu et sigma.
"""

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class VAEConfig:
    in_channels: int = 1
    img_size: int = 128
    z_dim: int = 64
    channels: tuple[int, ...] = (32, 64, 128, 256, 256)  # une entrée = une division par 2


def conv_block(c_in: int, c_out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1),
        nn.GroupNorm(8, c_out),
        nn.SiLU(),
    )


class Encoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        layers, c = [], cfg.in_channels
        for c_out in cfg.channels:
            layers += [conv_block(c, c_out, stride=2), conv_block(c_out, c_out)]
            c = c_out
        self.convs = nn.Sequential(*layers)
        self.feat_size = cfg.img_size // 2 ** len(cfg.channels)
        flat = c * self.feat_size**2
        # Couche linéaire (pas de global pooling) : on garde l'information de POSITION,
        # qui est justement ce qu'on veut encoder (où est la balle, où sont les pentes)
        self.mu = nn.Linear(flat, cfg.z_dim)
        self.logvar = nn.Linear(flat, cfg.z_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.convs(x).flatten(1)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        rev = cfg.channels[::-1]
        self.feat_size = cfg.img_size // 2 ** len(cfg.channels)
        self.c0 = rev[0]
        self.fc = nn.Linear(cfg.z_dim, rev[0] * self.feat_size**2)

        layers, c = [], rev[0]
        for c_out in rev[1:] + (rev[-1],):
            # Une seule conv par étage (l'encodeur en a deux) : les étages à haute résolution
            # coûtent cher, et le décodeur était 4x plus lent que l'encodeur
            layers += [nn.Upsample(scale_factor=2, mode="nearest"), conv_block(c, c_out)]
            c = c_out
        self.convs = nn.Sequential(*layers)
        self.out = nn.Conv2d(c, cfg.in_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(-1, self.c0, self.feat_size, self.feat_size)
        return self.out(self.convs(h))  # logits ; sigmoid -> image dans [0, 1]


class VAE(nn.Module):
    def __init__(self, cfg: VAEConfig = VAEConfig()):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decoder(z))

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # ------------------------------------------------------------------ checkpoints

    def save(self, path, **extra) -> None:
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "VAE":
        ckpt = torch.load(path, map_location=map_location)
        cfg = ckpt["config"]
        cfg["channels"] = tuple(cfg["channels"])
        model = cls(VAEConfig(**cfg))
        model.load_state_dict(ckpt["state_dict"])
        return model


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    ball_weight: float = 20.0,
    decor_weight: float = 10.0,
    ball_threshold: float = 0.75,
    decor_threshold: float = 0.1,
    free_bits: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Perte du VAE, moyennée sur le batch.

    - Reconstruction : erreur quadratique sommée sur les pixels, pondérée par type de pixel.
      En gris : fond = 0, décor = 0.5, balle = 1.0. La balle (~0.5 % des pixels) et les
      lignes du décor (fines) seraient ignorées par une MSE simple, d'où `ball_weight`
      et `decor_weight` (le fond garde un poids de 1).
    - KL : écart entre q(z|x) et N(0, I), avec des « free bits » : les `free_bits` premiers
      nats par dimension ne sont pas pénalisés (cf. World Models). Ça évite que la KL
      écrase le latent et rende les reconstructions floues.
    """
    weight = torch.ones_like(target)
    weight = torch.where(target > decor_threshold, decor_weight, weight)
    weight = torch.where(target > ball_threshold, ball_weight, weight)
    recon_loss = (weight * (recon - target) ** 2).flatten(1).sum(1).mean()

    kl = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(1).mean()
    kl_floor = free_bits * mu.shape[1]
    kl_loss = torch.clamp(kl, min=kl_floor)

    return {"loss": recon_loss + beta * kl_loss, "recon": recon_loss, "kl": kl}
