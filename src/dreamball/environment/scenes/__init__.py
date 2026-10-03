"""Registre des scènes. Ajouter une scène = écrire `make(rng) -> Scene` et l'inscrire ici."""

from collections.abc import Callable

import numpy as np

from ..scene import Scene
from . import box, pegs, ramp, stairs, valley

SCENES: dict[str, Callable[[np.random.Generator], Scene]] = {
    "box": box.make,
    "ramp": ramp.make,
    "valley": valley.make,
    "stairs": stairs.make,
    "pegs": pegs.make,
}


def make_scene(name: str, rng: np.random.Generator, allow_mirror: bool = True) -> Scene:
    """Tire une variante aléatoire de la scène `name` (miroir gauche-droite une fois sur deux)."""
    if name not in SCENES:
        raise KeyError(f"Scène inconnue {name!r}. Disponibles : {list(SCENES)}")
    scene = SCENES[name](rng)
    if allow_mirror and rng.random() < 0.5:
        scene = scene.mirrored()
    return scene
