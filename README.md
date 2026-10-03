# DreamBall-VAE

Apprendre à un modèle à **rêver** le mouvement d'une balle : rebonds, roulement sur des pentes, chocs contre des obstacles, prédits entièrement dans un espace latent appris.

Le projet suit l'approche *World Models* (Ha & Schmidhuber, 2018) :

1. **V (VAE)** compresse chaque image en un vecteur latent `z`.
2. **M (modèle de dynamique)** prédit `z_{t+1}` à partir de l'historique des `z`.
3. **Le rêve** : à partir de quelques images réelles, M tourne en boucle sur ses propres prédictions, et le décodeur du VAE transforme les latents en vidéo.

## État d'avancement

- [x] Environnement physique 2D procédural (5 types de scènes)
- [ ] Génération du dataset
- [ ] VAE
- [ ] Modèle de dynamique
- [ ] Rêve et évaluation (dérive de la position selon l'horizon, généralisation à des scènes jamais vues)

## Installation

Python 3.12 ou plus récent.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Environnement

La physique est simulée avec [pymunk](https://www.pymunk.org) (Chipmunk2D) et le rendu est fait en niveaux de gris avec anti-crénelage.

```python
from dreamball.environment import BallEnv

env = BallEnv("ramp", seed=0)
frame = env.reset()          # nouvelle variante de scène + balle aléatoire
frame = env.step()           # avance d'une frame (1/30 s)

episode = env.rollout(150)
episode["frames"]            # (150, H, W) uint8
episode["states"]            # (150, 6) : x, y, vx, vy, angle, angular_velocity
episode["scene_params"]      # paramètres tirés pour cette scène
```

Chaque `reset()` tire une nouvelle variante de la scène (géométrie, élasticité, friction, amortissement), avec une symétrie gauche-droite une fois sur deux, ainsi qu'une position et une vitesse initiales aléatoires pour la balle.

### Scènes

| Nom      | Description                                   | Paramètres aléatoires                  |
|----------|-----------------------------------------------|----------------------------------------|
| `box`    | Boîte vide, rebonds purs (baseline)           | matériaux                              |
| `ramp`   | Pente partant d'un mur : chute puis roulement | angle, hauteur, longueur               |
| `valley` | Deux pentes en V : oscillations               | position et profondeur du creux        |
| `stairs` | Escalier : rebonds en cascade                 | nombre, largeur et hauteur des marches |
| `pegs`   | Plots en quinconce façon Plinko : chaotique   | nombre de rangées et colonnes, rayon   |

Pour ajouter une scène, il suffit d'écrire une fonction `make(rng) -> Scene` dans `scenes/` et de l'inscrire dans `SCENES`.

### Options utiles

- `img_size` : résolution des images.
- `render_mode="layers"` : renvoie `(2, H, W)`, avec le décor et la balle sur deux canaux séparés.
- `show_rotation=True` : trace un rayon sur la balle pour voir le roulement (débogage uniquement).

### Démo

```bash
python src/dreamball/environment/demo.py
```

Ce script génère `outputs/demo_scenes.gif`, qui montre un épisode de chaque scène côte à côte.

## Structure

```
src/dreamball/
└── environment/
    ├── engine.py      # wrapper pymunk : Space, balle, segments, pegs
    ├── scene.py       # description d'une scène (données pures)
    ├── scenes/        # fabriques de scènes procédurales + registre
    ├── renderer.py    # Scene + état de la balle -> image numpy
    ├── env.py         # BallEnv : reset / step / rollout
    └── demo.py
```

## Références

- D. Ha, J. Schmidhuber. [*World Models*](https://arxiv.org/abs/1803.10122), 2018.
- D. Hafner et al. [*Mastering Diverse Domains through World Models* (DreamerV3)](https://arxiv.org/abs/2301.04104), 2023.
