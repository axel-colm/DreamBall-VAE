"""Grille de plots en quinconce (Plinko) : trajectoires très sensibles aux conditions initiales."""

import numpy as np

from ..scene import Peg, Scene, box_walls
from ._common import random_damping, random_material


def make(rng: np.random.Generator) -> Scene:
    mat = random_material(rng)
    n_rows = int(rng.integers(3, 6))
    n_cols = int(rng.integers(3, 6))  # max 5 colonnes : l'écart entre plots reste > diamètre de la balle
    radius = float(rng.uniform(0.02, 0.035))

    pegs = []
    for row, y in enumerate(np.linspace(0.65, 0.2, n_rows)):
        # Rangées impaires décalées d'une demi-case
        offset = 0.5 if row % 2 else 0.0
        for col in range(n_cols):
            x = (col + 0.5 + offset) / (n_cols + 0.5)
            jitter = rng.uniform(-0.01, 0.01, size=2)
            pegs.append(Peg((float(x + jitter[0]), float(y + jitter[1])), radius, **mat))

    return Scene(
        name="pegs",
        segments=box_walls(**mat),
        pegs=tuple(pegs),
        damping=random_damping(rng),
        spawn=(0.1, 0.9, 0.78, 0.92),
        params={**mat, "n_rows": n_rows, "n_cols": n_cols, "peg_radius": radius},
    )
