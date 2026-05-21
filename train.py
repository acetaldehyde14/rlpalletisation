"""
Train a MaskablePPO agent on the pallet packing environment.

Speed improvements over v1:
  1. Action space is now K_EP * NUM_ROTATIONS = 300 vs old grid_l * grid_w * 6 ~ 20 000
     --> ~66x fewer actions to explore, much denser reward signal per step.
  2. Mask computation is O(K_EP * 6 * N_placed) instead of O(grid * 6 * N_placed).
  3. Tuned hyperparameters: larger n_steps, batch_size; linear LR schedule.
  4. SubprocVecEnv spawns real OS processes for true CPU parallelism.

Usage
    Quick test  (< 1 min):
        python train.py --timesteps 20000 --n_envs 2 --no_tensorboard

    Standard run  (15-30 min on 4-core CPU):
        python train.py --timesteps 500000 --n_envs 4 --no_tensorboard

    Thorough run  (best results, 1-2 h):
        python train.py --timesteps 2000000 --n_envs 8 --subproc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from pallet_env import PalletPackingEnv, expand_items


def _mask_fn(env: PalletPackingEnv):
    return env.action_masks()


def _make_env(data_path: str, seed: int = 0,
              pallet_length: float = 1200.0,
              pallet_width: float  = 1100.0,
              pallet_height: float = 1150.0,
              max_weight: float    = 1500.0):
    def _init():
        with open(data_path) as f:
            data = json.load(f)
        items = expand_items(data)
        env = PalletPackingEnv(
            items,
            pallet_length=pallet_length,
            pallet_width=pallet_width,
            pallet_height=pallet_height,
            max_pallet_weight=max_weight,
        )
        env = ActionMasker(env, _mask_fn)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def linear_schedule(initial_lr: float) -> Callable[[float], float]:
    """Linear decay from initial_lr to 10% of initial_lr over training."""
    def func(progress_remaining: float) -> float:
        return max(progress_remaining * initial_lr, initial_lr * 0.1)
    return func


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         default="data.json")
    parser.add_argument("--timesteps",    type=int,   default=500_000)
    parser.add_argument("--n_envs",       type=int,   default=4)
    parser.add_argument("--save_dir",     default="./checkpoints")
    parser.add_argument("--log_dir",      default="./logs")
    parser.add_argument("--lr",           type=float, default=3e-4,
                        help="Initial learning rate (linearly decayed to 10%)")
    parser.add_argument("--n_steps",      type=int,   default=1024,
                        help="Steps per env per update (larger = more data, less overhead)")
    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--n_epochs",     type=int,   default=5,
                        help="PPO epochs per update (5 is enough with small action space)")
    parser.add_argument("--ent_coef",     type=float, default=0.005,
                        help="Entropy bonus (lower than v1 because action space is smaller)")
    parser.add_argument("--pallet_length",type=float, default=1200.0)
    parser.add_argument("--pallet_width", type=float, default=1100.0)
    parser.add_argument("--pallet_height",type=float, default=1150.0)
    parser.add_argument("--max_weight",   type=float, default=1500.0)
    parser.add_argument("--subproc",      action="store_true",
                        help="SubprocVecEnv for real CPU parallelism (needs __main__ guard)")
    parser.add_argument("--no_tensorboard", action="store_true",
                        help="Skip tensorboard (use if not installed)")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        data_path=args.data,
        pallet_length=args.pallet_length,
        pallet_width=args.pallet_width,
        pallet_height=args.pallet_height,
        max_weight=args.max_weight,
    )

    env_fns  = [_make_env(**env_kwargs, seed=i) for i in range(args.n_envs)]
    eval_fns = [_make_env(**env_kwargs, seed=999)]

    vec_cls = SubprocVecEnv if (args.subproc and args.n_envs > 1) else DummyVecEnv
    env      = vec_cls(env_fns)
    eval_env = DummyVecEnv(eval_fns)

    print(f"Action space size: {env.action_space.n}  "
          f"(K_EP={50} x {6} rotations)")
    print(f"Observation keys: heightmap, ep_obs, item, progress")

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=linear_schedule(args.lr),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.995,         # higher gamma for longer episodes
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=None if args.no_tensorboard else args.log_dir,
    )

    eval_freq = max(10_000 // args.n_envs, 1)
    save_freq = max(50_000 // args.n_envs, 1)

    callbacks = [
        CheckpointCallback(
            save_freq=save_freq,
            save_path=args.save_dir,
            name_prefix="ppo",
        ),
        MaskableEvalCallback(
            eval_env,
            best_model_save_path=args.save_dir,
            log_path=args.log_dir,
            eval_freq=eval_freq,
            deterministic=True,
            n_eval_episodes=3,
            verbose=1,
        ),
    ]

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name="maskable_ppo_ep",
    )
    final_path = Path(args.save_dir) / "final_model"
    model.save(final_path)
    print(f"\nTraining done.")
    print(f"  Final model : {final_path}.zip")
    print(f"  Best model  : {Path(args.save_dir) / 'best_model.zip'}")


if __name__ == "__main__":
    main()
