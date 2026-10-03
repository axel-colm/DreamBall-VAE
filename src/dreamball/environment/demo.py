"""Démo : un épisode de chaque scène, côte à côte, exporté en gif.

Usage :
    pip install -e .
    python src/dreamball/environment/demo.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

from dreamball.environment import SCENES, BallEnv

IMAGE_SIZE = 128
ZOOM = 4 
N_STEPS = 240  # 8 s à 30 fps


def main() -> None:
    episodes = []
    for i, name in enumerate(SCENES):
        env = BallEnv(name, seed=i, img_size=IMAGE_SIZE, show_rotation=True)
        ep = env.rollout(N_STEPS)
        episodes.append(ep["frames"])
        print(f"{name:7s} {ep['scene_params']}")

    # (n_scenes, T, H, W) -> T images de (H, n_scenes * W), séparées par une bande grise
    sep = np.full((N_STEPS, IMAGE_SIZE, 2), IMAGE_SIZE, dtype=np.uint8)
    tiles = [x for ep in episodes for x in (ep, sep)][:-1]
    video = np.concatenate(tiles, axis=2)

    frames = [
        Image.fromarray(f).resize((f.shape[1] * ZOOM, f.shape[0] * ZOOM), Image.Resampling.NEAREST)
        for f in video
    ]
    out = Path("outputs/demo_scenes.gif")
    out.parent.mkdir(exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=1000 // 30, loop=0)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
