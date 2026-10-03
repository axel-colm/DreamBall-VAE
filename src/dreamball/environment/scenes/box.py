"""Cube vide : la baseline. Seulement des rebonds sur des murs droits."""

import numpy as np

from ..scene import Scene, box_walls
from ._common import random_damping, random_material


def make(rng: np.random.Generator) -> Scene:
    mat = random_material(rng)
    return Scene(
        name="box",
        segments=box_walls(**mat),
        damping=random_damping(rng),
        spawn=(0.1, 0.9, 0.1, 0.9),
        params=mat,
    )
