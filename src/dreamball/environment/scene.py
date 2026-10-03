"""Description d'une scène : pures données, aucune dépendance à pymunk.

Une scène = la géométrie statique (segments, pegs) + les paramètres physiques + la zone
où la balle peut apparaître. `env.py` la transforme en `Engine`, `renderer.py` la dessine.
Garder ça séparé permet de sauvegarder une scène (params dans le dataset) et de la
redessiner sans simuler.
"""

from dataclasses import dataclass, field, replace

Point = tuple[float, float]


@dataclass(frozen=True)
class Segment:
    a: Point
    b: Point
    radius: float = 0.005
    elasticity: float = 0.8
    friction: float = 0.6


@dataclass(frozen=True)
class Peg:
    center: Point
    radius: float
    elasticity: float = 0.8
    friction: float = 0.6


@dataclass(frozen=True)
class Scene:
    name: str
    segments: tuple[Segment, ...] = ()
    pegs: tuple[Peg, ...] = ()
    world_size: tuple[float, float] = (1.0, 1.0)
    gravity: Point = (0.0, -9.81)
    damping: float = 0.9
    # Zone d'apparition de la balle : (x_min, x_max, y_min, y_max)
    spawn: tuple[float, float, float, float] = (0.1, 0.9, 0.6, 0.9)
    # Paramètres tirés au hasard par la fabrique, sauvegardés avec le dataset
    params: dict = field(default_factory=dict)

    def mirrored(self) -> "Scene":
        """Symétrie gauche-droite : double la diversité des scènes gratuitement."""
        w = self.world_size[0]

        def fx(p: Point) -> Point:
            return (w - p[0], p[1])

        x0, x1, y0, y1 = self.spawn
        return replace(
            self,
            segments=tuple(replace(s, a=fx(s.a), b=fx(s.b)) for s in self.segments),
            pegs=tuple(replace(p, center=fx(p.center)) for p in self.pegs),
            spawn=(w - x1, w - x0, y0, y1),
            params={**self.params, "mirrored": True},
        )


def box_walls(
    world_size: tuple[float, float] = (1.0, 1.0),
    elasticity: float = 0.8,
    friction: float = 0.6,
) -> tuple[Segment, ...]:
    """Les 4 murs du monde, utilisés par toutes les scènes."""
    w, h = world_size
    corners = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    return tuple(
        Segment(a, b, elasticity=elasticity, friction=friction)
        for a, b in zip(corners, corners[1:] + corners[:1])
    )
