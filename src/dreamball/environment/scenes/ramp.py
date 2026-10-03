"""Une pente qui part du mur gauche : la balle tombe, rebondit puis roule dessus."""

import math

import numpy as np

from ..scene import Scene, Segment, box_walls
from ._common import random_damping, random_material


def make(rng: np.random.Generator) -> Scene:
    mat = random_material(rng)
    angle = float(rng.uniform(15, 40))
    y_top = float(rng.uniform(0.45, 0.7))
    slope = math.tan(math.radians(angle))
    # La pente s'arrête avant de toucher le sol, pour laisser passer la balle dessous
    x_end = min(float(rng.uniform(0.5, 0.75)), (y_top - 0.15) / slope)
    y_end = y_top - x_end * slope

    return Scene(
        name="ramp",
        segments=box_walls(**mat) + (Segment((0.0, y_top), (x_end, y_end), **mat),),
        damping=random_damping(rng),
        spawn=(0.05, x_end * 0.6, y_top + 0.08, 0.92),
        params={**mat, "angle": angle, "y_top": y_top, "x_end": x_end},
    )
