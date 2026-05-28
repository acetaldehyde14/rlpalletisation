"""
Train MaskablePPO on the Item-Ordering Environment.

The RL agent decides in what ORDER to place items. The EP heuristic
handles all placement decisions. This avoids the spatial perception
problem entirely.

Usage
    python train_order.py --timesteps 2000000 --n_envs 16 --subproc
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, List

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from order_env import ItemOrderEnv, MAX_ITEMS
from pallet_env import expand_items


def _mask_fn(env):
    return env.action_masks()


def _load_items(data_path: str):
    with open(data_path) as f:
        data = json.load(f)
    return expand_items(data)


def _make_env(data_path: str, seed: int = 0,
              pallet_length: float = 1200.0,
              pallet_width: float = 1100.0,
              pallet_height: float = 1150.0,
              max_weight: float = 1500.0,
              num_rotations: int = 6):
    def _init():
        items = _load_items(data_path)
        env = ItemOrderEnv(
            items,
            pallet_length=pallet_length,
            pallet_width=pallet_width,
            pallet_height=pallet_height,
            max_pallet_weight=max_weight,
            num_rotations=num_rotations,
        )
        env = ActionMasker(env, _mask_fn)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def linear_schedule(initial_lr: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return max(progress_remaining * initial_lr, initial_lr * 0.1)
    return func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=None)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--save_dir", default="./checkpoints/order_ppo")
    parser.add_argument("--log_dir", default="./logs/order_ppo")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--rotations", type=int, default=6)
    parser.add_argument("--pallet_length", type=float, default=1200.0)
    parser.add_argument("--pallet_width", type=float, default=1100.0)
    parser.add_argument("--pallet_height", type=float, default=1150.0)
    parser.add_argument("--max_weight", type=float, default=1500.0)
    parser.add_argument("--subproc", action="store_true")
    parser.add_argument("--no_tensorboard", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    data_paths = args.data
    if data_paths is None:
        data_paths = ["data/orders"]
    expanded = []
    for p in data_paths:
        if os.path.isdir(p):
            for fn in sorted(os.listdir(p)):
                if fn.endswith(".json") and fn != "catalog.json":
                    expanded.append(os.path.join(p, fn))
        else:
            expanded.append(p)
    data_paths = expanded
    n_datasets = len(data_paths)

    train_fns = [
        _make_env(data_path=data_paths[i % n_datasets], seed=i,
                  pallet_length=args.pallet_length, pallet_width=args.pallet_width,
                  pallet_height=args.pallet_height, max_weight=args.max_weight,
                  num_rotations=args.rotations)
        for i in range(args.n_envs)
    ]

    eval_fns = [_make_env(data_path=data_paths[0], seed=999,
                           pallet_length=args.pallet_length,
                           pallet_width=args.pallet_width,
                           pallet_height=args.pallet_height,
                           max_weight=args.max_weight,
                           num_rotations=args.rotations)]

    vec_cls = SubprocVecEnv if (args.subproc and args.n_envs > 1) else DummyVecEnv
    env = vec_cls(train_fns)
    eval_env = DummyVecEnv(eval_fns)

    print(f"Item-Ordering Env: action_space=Discrete({env.action_space.n})")
    print(f"EP handles all placement, RL picks ordering only")
    print(f"Datasets: {n_datasets} files")

    tb_log = None if args.no_tensorboard else str(log_dir)

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=linear_schedule(args.lr),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=tb_log,
    )

    eval_freq = max(10_000 // args.n_envs, 1)
    save_freq = max(50_000 // args.n_envs, 1)

    callbacks = [
        CheckpointCallback(save_freq=save_freq, save_path=str(save_dir),
                           name_prefix="order_ppo"),
        MaskableEvalCallback(eval_env, best_model_save_path=str(save_dir),
                             log_path=str(log_dir), eval_freq=eval_freq,
                             deterministic=True, n_eval_episodes=3, verbose=1),
    ]

    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                tb_log_name="order_ppo")

    final_path = save_dir / "final_model"
    model.save(final_path)
    print(f"\nTraining done.")
    print(f"  Final model : {final_path}.zip")
    print(f"  Best model  : {save_dir / 'best_model.zip'}")


if __name__ == "__main__":
    main()
