"""Tirages aléatoires partagés par toutes les scènes."""

import numpy as np


def random_material(rng: np.random.Generator) -> dict:
    """Rebond et friction des surfaces d'une scène."""
    return {
        "elasticity": float(rng.uniform(0.5, 0.9)),
        "friction": float(rng.uniform(0.3, 0.9)),
    }


def random_damping(rng: np.random.Generator) -> float:
    """Fraction de vitesse conservée par seconde."""
    return float(rng.uniform(0.85, 0.98))
