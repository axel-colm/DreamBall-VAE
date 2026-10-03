"""BallEnv : point d'entrée unique de l'environnement.

    env = BallEnv("ramp", seed=0)
    frame = env.reset()          # nouvelle variante de scène + balle aléatoire
    frame = env.step()           # avance d'une frame
    episode = env.rollout(100)   # {"frames": (T, H, W), "states": (T, 6)}

Chaque `reset()` tire une nouvelle variante de la scène (angle de pente, matériaux...)
et une nouvelle position/vitesse de balle.
"""

from dataclasses import astuple
from typing import Literal

import numpy as np

from .engine import BallState, Engine
from .renderer import Renderer
from .scene import Scene
from .scenes import make_scene

STATE_KEYS = ("x", "y", "vx", "vy", "angle", "angular_velocity")


class BallEnv:
    def __init__(
        self,
        scene: str,
        seed: int | None = None,
        img_size: int = 128,
        render_mode: Literal["gray", "layers"] = "gray",
        show_rotation: bool = False,
        dt: float = 1 / 30,
        substeps: int = 10,
        ball_radius: float = 0.04,
        max_init_speed: float = 1.5,
    ):
        self.scene_name = scene
        self.dt = dt
        self.substeps = substeps
        self.ball_radius = ball_radius
        self.max_init_speed = max_init_speed
        self.rng = np.random.default_rng(seed)
        self.renderer = Renderer(img_size, mode=render_mode, show_rotation=show_rotation)

        self.scene: Scene | None = None
        self.engine: Engine | None = None

    # ------------------------------------------------------------------ construction

    def _build_engine(self, scene: Scene) -> Engine:
        """Traduit la Scene (données) en monde pymunk."""
        engine = Engine(
            world_size=scene.world_size,
            gravity=scene.gravity,
            dt=self.dt,
            substeps=self.substeps,
            damping=scene.damping,
        )
        for s in scene.segments:
            engine.add_segment(s.a, s.b, s.radius, s.elasticity, s.friction)
        for p in scene.pegs:
            engine.add_circle(p.center, p.radius, p.elasticity, p.friction)
        return engine

    def _spawn_ball(self, engine: Engine, scene: Scene) -> None:
        """Place la balle au hasard dans la zone de spawn, sans chevaucher le décor."""
        x0, x1, y0, y1 = scene.spawn
        margin = self.ball_radius + 0.01
        for _ in range(100):
            pos = (float(self.rng.uniform(x0, x1)), float(self.rng.uniform(y0, y1)))
            if engine.is_free(pos, margin):
                break
        else:
            raise RuntimeError(f"Impossible de placer la balle dans la scène {scene.name!r}")

        speed = self.rng.uniform(0, self.max_init_speed)
        direction = self.rng.uniform(0, 2 * np.pi)
        vel = (float(speed * np.cos(direction)), float(speed * np.sin(direction)))
        # La balle reprend le matériau des murs : un seul couple (elasticity, friction) par scène
        mat = scene.segments[0]
        engine.add_ball(pos, vel, self.ball_radius, elasticity=mat.elasticity, friction=mat.friction)

    # ------------------------------------------------------------------ API

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.scene = make_scene(self.scene_name, self.rng)
        self.engine = self._build_engine(self.scene)
        self._spawn_ball(self.engine, self.scene)
        return self.render()

    def step(self) -> np.ndarray:
        self.engine.step()
        return self.render()

    def state(self) -> BallState:
        return self.engine.ball_state()

    def render(self) -> np.ndarray:
        return self.renderer.render(self.scene, self.state(), self.ball_radius)

    def rollout(self, n_steps: int, seed: int | None = None) -> dict:
        """Un épisode complet. `states` suit l'ordre de STATE_KEYS."""
        frames = [self.reset(seed)]
        states = [astuple(self.state())]
        for _ in range(n_steps - 1):
            frames.append(self.step())
            states.append(astuple(self.state()))
        return {
            "frames": np.stack(frames),
            "states": np.asarray(states, dtype=np.float32),
            "scene": self.scene.name,
            "scene_params": self.scene.params,
        }
