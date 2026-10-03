"""Chargement du dataset généré par `generate.py`.

    data = load_split("data/raw", "train")       # tout en RAM, en uint8
    vae_ds = FrameDataset(data)                  # 1 image  -> (1, H, W) float [0, 1]
    seq_ds = SequenceDataset(data, seq_len=32)   # 1 fenêtre -> (L, 1, H, W) + états (L, 6)

Pourquoi tout charger en RAM : 5000 épisodes en 128 px = ~12 Go en uint8. Ça tient,
et c'est bien plus rapide que de décompresser un .npz à chaque batch. On garde le uint8
en mémoire (4x moins que du float32) et la conversion en float se fait par échantillon.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class EpisodeData:
    frames: np.ndarray  # (E, T, H, W) ou (E, T, 2, H, W) uint8
    states: np.ndarray  # (E, T, 6) float32
    scene_ids: np.ndarray  # (E,) int, indice dans `scene_names`
    seeds: np.ndarray  # (E,) uint32
    scene_names: list[str]
    metadata: dict
    params: list[dict] = field(default_factory=list)  # paramètres de scène par épisode

    def __len__(self) -> int:
        return len(self.frames)

    def select_scenes(self, names: list[str]) -> "EpisodeData":
        """Sous-ensemble par scène (copie). Utile pour exclure une scène de l'entraînement."""
        ids = [self.scene_names.index(n) for n in names]
        mask = np.isin(self.scene_ids, ids)
        return EpisodeData(
            frames=self.frames[mask],
            states=self.states[mask],
            scene_ids=self.scene_ids[mask],
            seeds=self.seeds[mask],
            scene_names=self.scene_names,
            metadata=self.metadata,
            params=[p for p, m in zip(self.params, mask) if m],
        )


def load_split(root: str | Path, split: str, scenes: list[str] | None = None) -> EpisodeData:
    """Charge tous les shards d'un split dans un seul tableau pré-alloué."""
    split_dir = Path(root) / split
    metadata = json.loads((split_dir / "metadata.json").read_text())
    scene_names = scenes or metadata["scenes"]

    shards = [(i, p) for i, name in enumerate(scene_names) for p in sorted((split_dir / name).glob("shard_*.npz"))]
    if not shards:
        raise FileNotFoundError(f"Aucun shard dans {split_dir} pour {scene_names}")

    # 1re passe : nombre d'épisodes par shard. On ne lit que `seeds` (minuscule) :
    # un .npz est un zip, chaque tableau est décompressé seulement quand on y accède.
    counts = []
    for _, path in shards:
        with np.load(path) as z:
            counts.append(len(z["seeds"]))
    offsets = np.concatenate([[0], np.cumsum(counts)])
    n_total = int(offsets[-1])

    # Forme d'une frame lue sur le premier shard
    with np.load(shards[0][1]) as z:
        frame_shape, n_steps = z["frames"].shape[2:], z["frames"].shape[1]

    # Pré-allocation : évite de concaténer (qui doublerait le pic mémoire)
    frames = np.empty((n_total, n_steps, *frame_shape), dtype=np.uint8)
    states = np.empty((n_total, n_steps, len(metadata["state_keys"])), dtype=np.float32)
    scene_ids = np.empty(n_total, dtype=np.int64)
    seeds = np.empty(n_total, dtype=np.uint32)
    params: list[dict] = [{}] * n_total

    def load_shard(k: int) -> None:
        scene_id, path = shards[k]
        sl = slice(offsets[k], offsets[k + 1])
        with np.load(path) as z:
            frames[sl] = z["frames"]
            states[sl] = z["states"]
            seeds[sl] = z["seeds"]
            params[sl] = [json.loads(p) for p in z["params"]]
        scene_ids[sl] = scene_id

    # La décompression zlib libère le GIL : des threads suffisent pour paralléliser
    with ThreadPoolExecutor() as pool:
        list(pool.map(load_shard, range(len(shards))))

    return EpisodeData(frames, states, scene_ids, seeds, list(scene_names), metadata, params)


class FrameDataset(Dataset):
    """Images individuelles, pour entraîner le VAE. Indice = (épisode, t) aplati."""

    def __init__(self, data: EpisodeData):
        self.data = data
        self.n_steps = data.frames.shape[1]

    def __len__(self) -> int:
        return len(self.data) * self.n_steps

    def __getitem__(self, idx: int) -> torch.Tensor:
        ep, t = divmod(idx, self.n_steps)
        frame = self.data.frames[ep, t]
        x = torch.from_numpy(frame).float().div_(255)
        return x.unsqueeze(0) if x.ndim == 2 else x  # (1, H, W) ou (2, H, W)


class SequenceDataset(Dataset):
    """Fenêtres de `seq_len` frames consécutives, pour entraîner le modèle de dynamique.

    `stride` espace les débuts de fenêtres : stride=1 donne toutes les fenêtres possibles
    (très redondantes), stride=seq_len des fenêtres disjointes.
    """

    def __init__(self, data: EpisodeData, seq_len: int, stride: int = 1):
        n_steps = data.frames.shape[1]
        if seq_len > n_steps:
            raise ValueError(f"seq_len={seq_len} > {n_steps} frames par épisode")
        self.data = data
        self.seq_len = seq_len
        self.starts = np.arange(0, n_steps - seq_len + 1, stride)

    def __len__(self) -> int:
        return len(self.data) * len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep, k = divmod(idx, len(self.starts))
        t = self.starts[k]
        frames = self.data.frames[ep, t : t + self.seq_len]
        x = torch.from_numpy(frames).float().div_(255)
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (L, H, W) -> (L, 1, H, W)
        return {
            "frames": x,
            "states": torch.from_numpy(self.data.states[ep, t : t + self.seq_len]),
            "scene_id": int(self.data.scene_ids[ep]),
        }


if __name__ == "__main__":
    import random
    
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("TkAgg")
        
    data = load_split("data/raw", "train")
    print("Frames shape:", data.frames.shape)
    print("States shape:", data.states.shape)
    
    # Show random frames (5x5 frames)
    plt.figure(figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            ep = random.randint(0, len(data) - 1)
            t = random.randint(0, data.frames.shape[1] - 1)
            frame = data.frames[ep, t]
            plt.subplot(5, 5, i * 5 + j + 1)
            plt.imshow(frame)
            plt.title(f"Ep {ep}, F {t}")
            plt.axis("off")
    plt.show()