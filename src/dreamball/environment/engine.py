"""Moteur physique 2D de DreamBall, basé sur pymunk.

Les 4 concepts de pymunk
------------------------
1. Space  : le monde. Contient la gravité et fait avancer la simulation (`space.step(dt)`).
2. Body   : un objet physique *invisible* : masse, position, vitesse, rotation.
            - DYNAMIC : bougé par les forces (notre balle).
            - STATIC  : ne bouge jamais (murs, pentes). `space.static_body` est fourni d'office.
3. Shape  : la forme de collision attachée à un Body : Circle, Segment, Poly.
            C'est la shape qui porte `elasticity` (rebond) et `friction`.
4. step   : `space.step(dt)` avance le temps de dt secondes : intègre les forces,
            détecte les collisions, résout les contacts.

Un Body sans Shape ne touche rien ; une Shape sans Body n'existe pas.
Il faut ajouter les deux au Space : `space.add(body, shape)`.

Repère : pymunk n'impose rien. Ici, on choisit des mètres, origine en bas à gauche,
y vers le haut. Le monde fait `world_size` (1 x 1 m par défaut).
"""

from dataclasses import dataclass

import pymunk


@dataclass
class BallState:
    """État complet de la balle à un instant : ce qu'on sauvegarde comme vérité terrain."""

    x: float
    y: float
    vx: float
    vy: float
    angle: float  # rad, utile pour voir le roulement sur une pente
    angular_velocity: float


class Engine:
    def __init__(
        self,
        world_size: tuple[float, float] = (1.0, 1.0),
        gravity: tuple[float, float] = (0.0, -9.81),
        dt: float = 1 / 30,
        substeps: int = 10,
        damping: float = 1.0,
    ):
        self.world_size = world_size
        self.dt = dt
        self.substeps = substeps

        self.space = pymunk.Space()
        self.space.gravity = gravity
        # Fraction de vitesse conservée par seconde (1.0 = aucune perte). pymunk n'a pas de
        # résistance au roulement : sans damping < 1, une balle qui roule ne s'arrête jamais.
        self.space.damping = damping

        # Piège classique : les tolérances par défaut de pymunk sont pensées pour des
        # unités en pixels (slop = 0.1 px). Avec un monde de 1 m, 0.1 = 10 % de la scène :
        # la balle s'enfoncerait visiblement dans le sol. On les ramène à l'échelle.
        self.space.collision_slop = 1e-4

        self.ball_body: pymunk.Body | None = None
        self.ball_shape: pymunk.Circle | None = None
        self.static_shapes: list[pymunk.Shape] = []

    # ------------------------------------------------------------------ décor

    def add_segment(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        radius: float = 0.005,
        elasticity: float = 0.8,
        friction: float = 0.6,
    ) -> pymunk.Segment:
        """Ajoute un segment statique (mur, sol, pente). `radius` = demi-épaisseur."""
        # Attaché à space.static_body : pas besoin de créer un Body, il ne bougera jamais.
        seg = pymunk.Segment(self.space.static_body, a, b, radius)
        seg.elasticity = elasticity
        seg.friction = friction
        self.space.add(seg)  # le static_body est déjà dans le space, on n'ajoute que la shape
        self.static_shapes.append(seg)
        return seg

    def add_circle(
        self,
        center: tuple[float, float],
        radius: float,
        elasticity: float = 0.8,
        friction: float = 0.6,
    ) -> pymunk.Circle:
        """Ajoute un obstacle rond statique (peg)."""
        # offset = position du cercle relative au body ; static_body est à l'origine.
        circle = pymunk.Circle(self.space.static_body, radius, offset=center)
        circle.elasticity = elasticity
        circle.friction = friction
        self.space.add(circle)
        self.static_shapes.append(circle)
        return circle

    def is_free(self, pos: tuple[float, float], radius: float) -> bool:
        """True si un disque de rayon `radius` en `pos` ne touche aucune shape."""
        # point_query_nearest renvoie la shape la plus proche à moins de `radius`, sinon None.
        return self.space.point_query_nearest(pos, radius, pymunk.ShapeFilter()) is None

    def add_box(self, **kwargs) -> None:
        """Les 4 murs du monde."""
        w, h = self.world_size
        corners = [(0, 0), (w, 0), (w, h), (0, h)]
        for a, b in zip(corners, corners[1:] + corners[:1]):
            self.add_segment(a, b, **kwargs)

    # ------------------------------------------------------------------ balle

    def add_ball(
        self,
        pos: tuple[float, float],
        vel: tuple[float, float] = (0.0, 0.0),
        radius: float = 0.04,
        mass: float = 1.0,
        elasticity: float = 0.8,
        friction: float = 0.6,
    ) -> None:
        # Le moment d'inertie dit à quel point l'objet résiste à la rotation.
        # Sans lui (ou avec float("inf")), la balle glisserait sur une pente au lieu de rouler.
        moment = pymunk.moment_for_circle(mass, 0, radius)

        body = pymunk.Body(mass, moment)  # DYNAMIC par défaut
        body.position = pos
        body.velocity = vel

        shape = pymunk.Circle(body, radius)
        # Lors d'un contact, pymunk MULTIPLIE les coefficients des deux shapes :
        # rebond effectif = ball.elasticity * mur.elasticity (0.8 * 0.8 = 0.64).
        shape.elasticity = elasticity
        shape.friction = friction

        self.space.add(body, shape)
        self.ball_body, self.ball_shape = body, shape

    # ------------------------------------------------------------------ simulation

    def step(self) -> None:
        """Avance d'une frame. Sous-pas = plus stable (moins de traversée de murs)."""
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            self.space.step(h)  # toujours un dt FIXE : pymunk est instable avec un dt variable

    def ball_state(self) -> BallState:
        b = self.ball_body
        return BallState(
            x=b.position.x,
            y=b.position.y,
            vx=b.velocity.x,
            vy=b.velocity.y,
            angle=b.angle,
            angular_velocity=b.angular_velocity,
        )
