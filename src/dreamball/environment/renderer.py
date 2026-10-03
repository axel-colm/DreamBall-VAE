"""Dessine une Scene + l'état de la balle en image numpy.

Le rendu se fait en sur-échantillonnage (`supersample` x plus grand) puis on réduit :
ça donne un anti-crénelage. À 64x64, la balle ne fait que ~5 px de diamètre : sans
anti-crénelage, sa position serait quantifiée au pixel près et le modèle verrait des
sauts au lieu d'un mouvement continu.
"""

import math
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw

from .engine import BallState
from .scene import Scene

DECOR_VALUE = 128
BALL_VALUE = 255


class Renderer:
    def __init__(
        self,
        img_size: int = 64,
        supersample: int = 4,
        mode: Literal["gray", "layers"] = "gray",
        show_rotation: bool = False,
    ):
        """
        mode="gray"   -> (H, W) uint8 : décor gris, balle blanche.
        mode="layers" -> (2, H, W) uint8 : canal 0 = décor, canal 1 = balle.
        show_rotation : trace un rayon sur la balle pour rendre le roulement visible.
        """
        self.img_size = img_size
        self.ss = supersample
        self.mode = mode
        self.show_rotation = show_rotation
        # Le décor est statique pendant un épisode : on le dessine une seule fois
        self._bg_scene: Scene | None = None
        self._bg: Image.Image | None = None

    def _to_px(self, scene: Scene, p) -> tuple[float, float]:
        """Monde (m, y vers le haut) -> image sur-échantillonnée (px, y vers le bas)."""
        w, h = scene.world_size
        size = self.img_size * self.ss
        return (p[0] / w * size, size - p[1] / h * size)

    def _scale(self, scene: Scene, length: float) -> float:
        return length / scene.world_size[0] * self.img_size * self.ss

    def _background(self, scene: Scene) -> Image.Image:
        if self._bg_scene is scene:
            return self._bg
        size = self.img_size * self.ss
        img = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(img)
        for seg in scene.segments:
            # Au moins 1 px final d'épaisseur, sinon les murs disparaissent à la réduction
            width = max(self.ss, round(self._scale(scene, 2 * seg.radius)))
            draw.line([self._to_px(scene, seg.a), self._to_px(scene, seg.b)], fill=255, width=width)
        for peg in scene.pegs:
            cx, cy = self._to_px(scene, peg.center)
            r = self._scale(scene, peg.radius)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        self._bg_scene, self._bg = scene, img
        return img

    def _ball(self, scene: Scene, state: BallState, radius: float) -> Image.Image:
        size = self.img_size * self.ss
        img = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(img)
        cx, cy = self._to_px(scene, (state.x, state.y))
        r = self._scale(scene, radius)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
        if self.show_rotation:
            end = (cx + r * math.cos(state.angle), cy - r * math.sin(state.angle))
            draw.line([(cx, cy), end], fill=0, width=self.ss)
        return img

    def _downsample(self, img: Image.Image) -> np.ndarray:
        return np.asarray(img.resize((self.img_size, self.img_size), Image.Resampling.BOX))

    def render(self, scene: Scene, state: BallState, radius: float) -> np.ndarray:
        decor = self._downsample(self._background(scene))
        ball = self._downsample(self._ball(scene, state, radius))
        if self.mode == "layers":
            return np.stack([decor, ball])
        # Balle par-dessus le décor
        gray = decor.astype(np.float32) / 255 * DECOR_VALUE
        alpha = ball.astype(np.float32) / 255
        return (gray * (1 - alpha) + BALL_VALUE * alpha).round().astype(np.uint8)
