"""Deux pentes en V : la balle oscille d'un côté à l'autre et finit au fond."""

import numpy as np

from ..scene import Scene, Segment, box_walls
from ._common import random_damping, random_material


def make(rng: np.random.Generator) -> Scene:
    mat = random_material(rng)
    cx = float(rng.uniform(0.35, 0.65))
    y_bottom = float(rng.uniform(0.05, 0.2))
    h_left = float(rng.uniform(0.4, 0.65))
    h_right = float(rng.uniform(0.4, 0.65))

    return Scene(
        name="valley",
        segments=box_walls(**mat)
        + (
            Segment((0.0, h_left), (cx, y_bottom), **mat),
            Segment((cx, y_bottom), (1.0, h_right), **mat),
        ),
        damping=random_damping(rng),
        spawn=(0.1, 0.9, max(h_left, h_right) + 0.08, 0.92),
        params={**mat, "cx": cx, "y_bottom": y_bottom, "h_left": h_left, "h_right": h_right},
    )
