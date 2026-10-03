"""Escalier descendant vers la droite : rebonds en cascade de marche en marche."""

import numpy as np

from ..scene import Scene, Segment, box_walls
from ._common import random_damping, random_material


def make(rng: np.random.Generator) -> Scene:
    mat = random_material(rng)
    n_steps = int(rng.integers(3, 6))
    y_top = float(rng.uniform(0.5, 0.7))
    span = float(rng.uniform(0.6, 0.85))
    step_w = span / n_steps
    step_h = (y_top - 0.05) / n_steps * float(rng.uniform(0.6, 1.0))

    segments = []
    x, y = 0.0, y_top
    for _ in range(n_steps):
        segments.append(Segment((x, y), (x + step_w, y), **mat))  # marche
        segments.append(Segment((x + step_w, y), (x + step_w, y - step_h), **mat))  # contremarche
        x, y = x + step_w, y - step_h

    return Scene(
        name="stairs",
        segments=box_walls(**mat) + tuple(segments),
        damping=random_damping(rng),
        spawn=(0.05, max(0.06, step_w - 0.05), y_top + 0.08, 0.92),
        params={**mat, "n_steps": n_steps, "y_top": y_top, "step_w": step_w, "step_h": step_h},
    )
