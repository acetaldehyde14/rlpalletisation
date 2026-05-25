"""
Train RL agents on the pallet packing environment.

Supported algorithms: maskable_ppo (default), dqn, a2c.

Usage
    MaskablePPO  (action-masked, recommended):
        python train.py --algo maskable_ppo --timesteps 5000000 --n_envs 8 --subproc

    DQN:
        python train.py --algo dqn --timesteps 5000000 --n_envs 8 --subproc

    A2C:
        python train.py --algo a2c --timesteps 5000000 --n_envs 8 --subproc

    Quick smoke test:
        python train.py --timesteps 20000 --n_envs 2 --no_tensorboard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import DQN, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from pallet_env import PalletPackingEnv, expand_items


def _mask_fn(env):
    return env.action_masks()


def _make_env(data_path: str, seed: int = 0,
              pallet_length: float = 1200.0,
              pallet_width: float  = 1100.0,
              pallet_height: float = 1150.0,
              max_weight: float    = 1500.0,
              num_rotations: int   = 6,
              use_masking: bool    = True):
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
            num_rotations=num_rotations,
        )
        if use_masking:
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
    parser.add_argument("--algo", choices=["maskable_ppo", "dqn", "a2c"],
                        default="maskable_ppo")
    parser.add_argument("--data",         nargs="+",
                        default=["data.json", "data_old.json"])
    parser.add_argument("--timesteps",    type=int,   default=5_000_000)
    parser.add_argument("--n_envs",       type=int,   default=4)
    parser.add_argument("--save_dir",     default="./checkpoints")
    parser.add_argument("--log_dir",      default="./logs")
    parser.add_argument("--lr",           type=float, default=3e-4,
                        help="Initial learning rate (linearly decayed to 10%%)")
    parser.add_argument("--n_steps",      type=int,   default=1024)
    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--n_epochs",     type=int,   default=5)
    parser.add_argument("--ent_coef",     type=float, default=0.005)
    parser.add_argument("--rotations",    type=int,   default=6,
                        help="Number of box rotations (2 or 6)")
    parser.add_argument("--pallet_length",type=float, default=1200.0)
    parser.add_argument("--pallet_width", type=float, default=1100.0)
    parser.add_argument("--pallet_height",type=float, default=1150.0)
    parser.add_argument("--max_weight",   type=float, default=1500.0)
    parser.add_argument("--subproc",      action="store_true")
    parser.add_argument("--no_tensorboard", action="store_true")
    args = parser.parse_args()

    use_masking = (args.algo == "maskable_ppo")
    save_dir = Path(args.save_dir) / args.algo
    log_dir  = Path(args.log_dir)  / args.algo
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    data_paths = args.data
    n_datasets = len(data_paths)
    print(f"Algo: {args.algo}  |  Rotations: {args.rotations}  |  "
          f"Datasets: {data_paths}  |  Masking: {use_masking}")

    base_kwargs = dict(
        pallet_length=args.pallet_length,
        pallet_width=args.pallet_width,
        pallet_height=args.pallet_height,
        max_weight=args.max_weight,
        num_rotations=args.rotations,
        use_masking=use_masking,
    )

    env_fns = [
        _make_env(data_path=data_paths[i % n_datasets], seed=i, **base_kwargs)
        for i in range(args.n_envs)
    ]
    eval_fns = [_make_env(data_path=data_paths[0], seed=999, **base_kwargs)]

    vec_cls = SubprocVecEnv if (args.subproc and args.n_envs > 1) else DummyVecEnv
    env      = vec_cls(env_fns)
    eval_env = DummyVecEnv(eval_fns)

    action_n = env.action_space.n
    print(f"Action space: {action_n}  (K_EP=50 x {args.rotations} rotations)")

    tb_log = None if args.no_tensorboard else str(log_dir)

    if args.algo == "maskable_ppo":
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
    elif args.algo == "dqn":
        model = DQN(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            learning_rate=linear_schedule(args.lr),
            batch_size=args.batch_size,
            gamma=0.995,
            buffer_size=100_000,
            learning_starts=10_000,
            target_update_interval=1000,
            tensorboard_log=tb_log,
        )
    elif args.algo == "a2c":
        model = A2C(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            learning_rate=linear_schedule(args.lr),
            n_steps=args.n_steps,
            gamma=0.995,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            tensorboard_log=tb_log,
        )

    eval_freq = max(10_000 // args.n_envs, 1)
    save_freq = max(50_000 // args.n_envs, 1)

    callbacks = [
        CheckpointCallback(
            save_freq=save_freq,
            save_path=str(save_dir),
            name_prefix="ppo" if args.algo == "maskable_ppo" else args.algo,
        ),
    ]

    if use_masking:
        callbacks.append(
            MaskableEvalCallback(
                eval_env,
                best_model_save_path=str(save_dir),
                log_path=str(log_dir),
                eval_freq=eval_freq,
                deterministic=True,
                n_eval_episodes=3,
                verbose=1,
            )
        )
    else:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(save_dir),
                log_path=str(log_dir),
                eval_freq=eval_freq,
                deterministic=True,
                n_eval_episodes=3,
                verbose=1,
            )
        )

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        tb_log_name=f"{args.algo}_r{args.rotations}",
    )
    final_path = save_dir / "final_model"
    model.save(final_path)
    print(f"\nTraining done ({args.algo}).")
    print(f"  Final model : {final_path}.zip")
    print(f"  Best model  : {save_dir / 'best_model.zip'}")


if __name__ == "__main__":
    main()
