"""Génère le dataset d'épisodes de balle.

Usage :
    python -m dreamball.data.generate --split train --episodes 1000
    python -m dreamball.data.generate --split val --episodes 100
    python -m dreamball.data.generate --split test --episodes 100 --scenes pegs

Organisation sur disque :
    data/raw/
    └── <split>/
        ├── metadata.json            # config de génération
        └── <scene>/
            ├── shard_000.npz        # `shard_size` épisodes
            └── ...

Contenu d'un shard :
    frames  (N, T, H, W) ou (N, T, 2, H, W)  uint8
    states  (N, T, 6)                         float32, ordre de STATE_KEYS
    seeds   (N,)                              uint32, pour rejouer un épisode exact
    params  (N,)                              str (JSON), paramètres de la scène

Reproductibilité : la graine de chaque épisode dépend uniquement de
(seed, split, scène, numéro d'épisode). Résultat identique quel que soit le nombre de
workers, et les splits ne partagent jamais d'épisode.
"""

import argparse
import json
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dreamball.environment import SCENES, STATE_KEYS, BallEnv


@dataclass(frozen=True)
class ShardJob:
    split: str
    scene: str
    shard_idx: int
    first_episode: int
    n_episodes: int
    n_steps: int
    img_size: int
    render_mode: str
    seed: int
    path: Path


def episode_seed(seed: int, split: str, scene: str, episode: int) -> int:
    # crc32 plutôt que hash() : hash() de str change à chaque lancement de Python
    entropy = [seed, zlib.crc32(split.encode()), zlib.crc32(scene.encode()), episode]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def generate_shard(job: ShardJob) -> tuple[ShardJob, float]:
    start = time.perf_counter()
    env = BallEnv(job.scene, img_size=job.img_size, render_mode=job.render_mode)

    frames, states, seeds, params = [], [], [], []
    for ep in range(job.first_episode, job.first_episode + job.n_episodes):
        s = episode_seed(job.seed, job.split, job.scene, ep)
        episode = env.rollout(job.n_steps, seed=s)
        frames.append(episode["frames"])
        states.append(episode["states"])
        seeds.append(s)
        params.append(json.dumps(episode["scene_params"]))

    # Écriture dans un fichier temporaire puis renommage : un shard interrompu
    # (Ctrl+C, crash) ne laisse jamais de .npz corrompu que --resume prendrait pour bon
    tmp = job.path.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        frames=np.stack(frames),
        states=np.stack(states),
        seeds=np.asarray(seeds, dtype=np.uint32),
        params=np.asarray(params),
    )
    tmp.replace(job.path)
    return job, time.perf_counter() - start


def make_jobs(args: argparse.Namespace) -> list[ShardJob]:
    jobs = []
    for scene in args.scenes:
        scene_dir = args.out / args.split / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        n_shards = -(-args.episodes // args.shard_size)  # division arrondie au supérieur
        for i in range(n_shards):
            first = i * args.shard_size
            jobs.append(
                ShardJob(
                    split=args.split,
                    scene=scene,
                    shard_idx=i,
                    first_episode=first,
                    n_episodes=min(args.shard_size, args.episodes - first),
                    n_steps=args.steps,
                    img_size=args.img_size,
                    render_mode=args.render_mode,
                    seed=args.seed,
                    path=scene_dir / f"shard_{i:03d}.npz",
                )
            )
    return jobs


def write_metadata(args: argparse.Namespace) -> None:
    env = BallEnv(args.scenes[0])
    metadata = {
        "split": args.split,
        "scenes": args.scenes,
        "episodes_per_scene": args.episodes,
        "shard_size": args.shard_size,
        "steps": args.steps,
        "img_size": args.img_size,
        "render_mode": args.render_mode,
        "seed": args.seed,
        "dt": env.dt,
        "substeps": env.substeps,
        "ball_radius": env.ball_radius,
        "state_keys": list(STATE_KEYS),
    }
    path = args.out / args.split / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="train", help="nom du split (train, val, test...)")
    p.add_argument("--scenes", nargs="+", default=list(SCENES), choices=list(SCENES))
    p.add_argument("--episodes", type=int, default=500, help="épisodes par scène")
    p.add_argument("--steps", type=int, default=150, help="frames par épisode (30 fps)")
    p.add_argument("--img-size", type=int, default=128)
    p.add_argument("--render-mode", choices=["gray", "layers"], default="gray")
    p.add_argument("--shard-size", type=int, default=100, help="épisodes par fichier .npz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("data/raw"))
    p.add_argument("--workers", type=int, default=None, help="défaut : nombre de CPU")
    p.add_argument("--resume", action="store_true", help="saute les shards déjà présents")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    jobs = make_jobs(args)
    if args.resume:
        jobs = [j for j in jobs if not j.path.exists()]
    write_metadata(args)

    total_eps = sum(j.n_episodes for j in jobs)
    print(f"[{args.split}] {len(jobs)} shards, {total_eps} épisodes -> {args.out / args.split}")

    start = time.perf_counter()
    done_eps = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(generate_shard, job) for job in jobs]
        for fut in as_completed(futures):
            job, elapsed = fut.result()
            done_eps += job.n_episodes
            size_mb = job.path.stat().st_size / 1e6
            print(
                f"  {done_eps:6d}/{total_eps}  {job.scene:7s} shard {job.shard_idx:03d}"
                f"  {elapsed:5.1f}s  {size_mb:6.1f} Mo"
            )

    print(f"Terminé en {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
