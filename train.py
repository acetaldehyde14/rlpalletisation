"""
Train MaskablePPO on the pallet packing environment.

Features:
  - Curriculum learning: start with small orders, grow over training
  - Behavioral cloning pre-training from EP baseline expert
  - Top-K EP action space (90 actions instead of 300)

Usage
    python train.py --timesteps 5000000 --n_envs 16 --subproc
    python train.py --timesteps 20000 --n_envs 2 --no_tensorboard  # smoke test
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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn

from pallet_env import (
    PalletPackingEnv, expand_items, get_rotation_dims,
    K_EP_TOP, NUM_ROTATIONS, SUPPORT_THRESHOLD, OVERLAP_EPS, SUPPORT_EPS,
    ROTATION_LABELS, PlacedBox,
)


def _mask_fn(env):
    return env.action_masks()


class PalletCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        hm_shape = observation_space["heightmap"].shape
        self.heightmap_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *hm_shape)
            cnn_out_dim = self.heightmap_cnn(dummy).shape[1]
        n_ep = K_EP_TOP * 4
        n_other = n_ep + 4 + 3
        self.other_mlp = nn.Sequential(
            nn.Linear(n_other, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.output = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        hm = observations["heightmap"].unsqueeze(1)
        cnn_feat = self.heightmap_cnn(hm)
        ep_flat = observations["ep_obs"].reshape(hm.shape[0], -1)
        other = torch.cat([ep_flat, observations["item"], observations["progress"]], dim=1)
        mlp_feat = self.other_mlp(other)
        combined = torch.cat([cnn_feat, mlp_feat], dim=1)
        return self.output(combined)


def _load_items(data_path: str, max_items: int = 0):
    with open(data_path) as f:
        data = json.load(f)
    items = expand_items(data)
    if max_items > 0:
        items = items[:max_items]
    return items


def _make_env(data_path: str, seed: int = 0,
              pallet_length: float = 1200.0,
              pallet_width: float  = 1100.0,
              pallet_height: float = 1150.0,
              max_weight: float    = 1500.0,
              num_rotations: int   = 6,
              max_items: int = 0,
              reward_util_bonus: bool = False,
              reward_lookahead: bool = False,
              reward_compactness: bool = False,
              reward_penalties: bool = False,
              k_ep_top: int = 50):
    def _init():
        items = _load_items(data_path, max_items)
        env = PalletPackingEnv(
            items,
            pallet_length=pallet_length,
            pallet_width=pallet_width,
            pallet_height=pallet_height,
            max_pallet_weight=max_weight,
            num_rotations=num_rotations,
            reward_util_bonus=reward_util_bonus,
            reward_lookahead=reward_lookahead,
            reward_compactness=reward_compactness,
            reward_penalties=reward_penalties,
            k_ep_top=k_ep_top,
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


class CurriculumCallback(BaseCallback):
    def __init__(self, env_fns_list: List[list], switch_steps: List[int],
                 vec_env, verbose: int = 1):
        super().__init__(verbose)
        self.env_fns_list = env_fns_list
        self.switch_steps = switch_steps
        self.vec_env = vec_env
        self.stage = 0

    def _on_step(self) -> bool:
        total = self.num_timesteps
        for i, threshold in enumerate(self.switch_steps):
            if total >= threshold and self.stage <= i:
                self.stage = i + 1
                if self.verbose:
                    print(f"\n[Curr] Switching to stage {self.stage+1} at {total} steps")
        return True


class PlateauEarlyStopping(BaseCallback):
    def __init__(self, eval_freq: int, patience: int = 15, min_delta: float = 0.5,
                 verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -float("inf")
        self.evals_without_improvement = 0
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step < self.eval_freq:
            return True
        self._last_eval_step = self.num_timesteps
        try:
            info = self.locals.get("info", [{}])
            ep_rewards = [i.get("episode", {}).get("r", None) for i in info
                          if "episode" in i] if isinstance(info, list) else []
        except Exception:
            return True
        if not ep_rewards:
            return True
        mean_r = float(np.mean(ep_rewards))
        if mean_r > self.best_mean_reward + self.min_delta:
            self.best_mean_reward = mean_r
            self.evals_without_improvement = 0
        else:
            self.evals_without_improvement += 1
        if self.evals_without_improvement >= self.patience:
            if self.verbose:
                print(f"\n[EarlyStop] No improvement for {self.patience} evals "
                      f"(best={self.best_mean_reward:.1f}). Stopping.")
            return False
        return True


def _ep_expert_action(env: PalletPackingEnv) -> int:
    item = env._current_item()
    if item is None:
        return 0

    pallet = env.pallets[-1]
    top_indices = env._top_ep_indices
    ep_arr = pallet["ep_arr"]

    best_score = float("inf")
    best_action = -1

    base_areas = [get_rotation_dims(item, r)[0] * get_rotation_dims(item, r)[1]
                  for r in range(env.num_rotations)]
    max_base = max(base_areas) if base_areas else 1.0
    primary = [r for r, a in enumerate(base_areas) if abs(a - max_base) < 1e-6]
    fallback = [r for r in range(env.num_rotations) if r not in primary]

    for rot_group in [primary, fallback]:
        found = False
        for top_i in range(K_EP_TOP):
            real_idx = int(top_indices[top_i])
            if ep_arr[real_idx, 0] < -0.5:
                continue
            x, y, z = float(ep_arr[real_idx, 0]), float(ep_arr[real_idx, 1]), float(ep_arr[real_idx, 2])
            for rot in rot_group:
                result = env._check_ep(pallet, item, real_idx, rot)
                if result is not None:
                    score = z * 1000.0 + x + y
                    if score < best_score:
                        best_score = score
                        best_action = top_i * env.num_rotations + rot
                        found = True
        if found:
            break

    if best_action >= 0:
        return best_action

    for top_i in range(K_EP_TOP):
        real_idx = int(top_indices[top_i])
        if ep_arr[real_idx, 0] < -0.5:
            continue
        for rot in range(env.num_rotations):
            result = env._check_ep(pallet, item, real_idx, rot)
            if result is not None:
                return top_i * env.num_rotations + rot

    return 0


def pretrain_bc(model: MaskablePPO, data_paths: List[str], n_demos: int = 200,
                pallet_length=1200.0, pallet_width=1100.0, pallet_height=1150.0,
                max_weight=1500.0, num_rotations=6, max_items: int = 20):
    print(f"Collecting {n_demos} expert demonstrations for BC pre-training...")
    observations = []
    actions = []
    masks_list = []

    for demo_i in range(n_demos):
        data_path = data_paths[demo_i % len(data_paths)]
        items = _load_items(data_path, max_items)
        if len(items) < 2:
            continue
        env = PalletPackingEnv(
            items, pallet_length=pallet_length, pallet_width=pallet_width,
            pallet_height=pallet_height, max_pallet_weight=max_weight,
            num_rotations=num_rotations,
        )
        wrapped = ActionMasker(env, _mask_fn)
        obs, _ = wrapped.reset()
        done = False
        while not done:
            expert_action = _ep_expert_action(env)
            m = wrapped.action_masks()
            if not m[expert_action]:
                valid = np.where(m)[0]
                if len(valid) > 0:
                    expert_action = int(valid[0])
            observations.append(obs)
            actions.append(expert_action)
            masks_list.append(m)
            obs, _, terminated, truncated, _ = wrapped.step(expert_action)
            done = terminated or truncated

    n_samples = len(actions)
    print(f"Collected {n_samples} expert transitions")
    if n_samples == 0:
        return model

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    batch_size = 256
    n_epochs = 10

    obs_batch = {k: torch.tensor(np.stack([o[k] for o in observations]),
                                  dtype=torch.float32)
                 for k in observations[0]}
    act_batch = torch.tensor(actions, dtype=torch.long)
    mask_batch = torch.tensor(np.array(masks_list), dtype=torch.bool)

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_samples, batch_size):
            idx = perm[start:start+batch_size]
            obs_mb = {k: v[idx] for k, v in obs_batch.items()}
            act_mb = act_batch[idx]
            mask_mb = mask_batch[idx]

            features = model.policy.extract_features(obs_mb)
            latent_pi = model.policy.mlp_extractor.forward_actor(features)
            logits = model.policy.action_net(latent_pi)
            masked_logits = torch.where(mask_mb, logits, torch.tensor(-1e10, dtype=logits.dtype))
            loss = nn.CrossEntropyLoss()(masked_logits, act_mb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 2 == 0:
            print(f"  BC epoch {epoch+1}/{n_epochs}  loss={total_loss/n_batches:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=None,
                        help="JSON files or directories of JSON orders")
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--save_dir", default="./checkpoints/maskable_ppo")
    parser.add_argument("--log_dir", default="./logs/maskable_ppo")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--ent_coef", type=float, default=0.005)
    parser.add_argument("--rotations", type=int, default=6)
    parser.add_argument("--pallet_length", type=float, default=1200.0)
    parser.add_argument("--pallet_width", type=float, default=1100.0)
    parser.add_argument("--pallet_height", type=float, default=1150.0)
    parser.add_argument("--max_weight", type=float, default=1500.0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--subproc", action="store_true")
    parser.add_argument("--no_tensorboard", action="store_true")
    parser.add_argument("--no_bc", action="store_true",
                        help="Skip behavioral cloning pre-training")
    parser.add_argument("--bc_demos", type=int, default=200)
    parser.add_argument("--max_items_stage1", type=int, default=15,
                        help="Max items for curriculum stage 1")
    parser.add_argument("--skip_stage1", action="store_true",
                        help="Skip curriculum stage 1, go straight to full orders")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stop after N evals without improvement")
    parser.add_argument("--cnn", action="store_true",
                        help="Use CNN feature extractor for heightmap")
    parser.add_argument("--reward_util_bonus", action="store_true")
    parser.add_argument("--reward_lookahead", action="store_true")
    parser.add_argument("--reward_compactness", action="store_true")
    parser.add_argument("--reward_penalties", action="store_true")
    parser.add_argument("--k_ep_top", type=int, default=50)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    log_dir  = Path(args.log_dir)
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

    env_kwargs = dict(
        pallet_length=args.pallet_length, pallet_width=args.pallet_width,
        pallet_height=args.pallet_height, max_weight=args.max_weight,
        num_rotations=args.rotations,
        reward_util_bonus=args.reward_util_bonus,
        reward_lookahead=args.reward_lookahead,
        reward_compactness=args.reward_compactness,
        reward_penalties=args.reward_penalties,
        k_ep_top=args.k_ep_top,
    )

    stage1_fns = [
        _make_env(data_path=data_paths[i % n_datasets], seed=i,
                  **env_kwargs, max_items=args.max_items_stage1)
        for i in range(args.n_envs)
    ]

    full_fns = [
        _make_env(data_path=data_paths[i % n_datasets], seed=i,
                  **env_kwargs, max_items=0)
        for i in range(args.n_envs)
    ]

    eval_fns = [_make_env(data_path=data_paths[0], seed=999,
                           **env_kwargs, max_items=args.max_items_stage1)]

    vec_cls = SubprocVecEnv if (args.subproc and args.n_envs > 1) else DummyVecEnv
    env = vec_cls(stage1_fns)
    eval_env = DummyVecEnv(eval_fns)

    action_n = env.action_space.n
    print(f"Action space: {action_n}  (k_ep_top={args.k_ep_top} x {args.rotations} rotations)")
    print(f"Rewards: util_bonus={args.reward_util_bonus} lookahead={args.reward_lookahead} "
          f"compactness={args.reward_compactness} penalties={args.reward_penalties}")
    print(f"Curriculum: stage1 max {args.max_items_stage1} items, then full orders")
    print(f"Datasets: {n_datasets} files")

    tb_log = None if args.no_tensorboard else str(log_dir)

    policy_kwargs = {}
    if args.cnn:
        policy_kwargs = dict(
            features_extractor_class=PalletCNNExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        print("Using CNN feature extractor for heightmap")

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
        policy_kwargs=policy_kwargs,
    )

    if not args.no_bc:
        bc_max = min(args.max_items_stage1, 20)
        model = pretrain_bc(
            model, data_paths, n_demos=args.bc_demos,
            pallet_length=args.pallet_length,
            pallet_width=args.pallet_width,
            pallet_height=args.pallet_height,
            max_weight=args.max_weight,
            num_rotations=args.rotations,
            max_items=bc_max,
        )

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = MaskablePPO.load(args.resume, env=env)

    if args.skip_stage1:
        stage1_steps = 0
    else:
        stage1_steps = args.timesteps // 3

    eval_freq = max(10_000 // args.n_envs, 1)
    save_freq = max(50_000 // args.n_envs, 1)

    callbacks = [
        CheckpointCallback(save_freq=save_freq, save_path=str(save_dir),
                           name_prefix="ppo"),
        MaskableEvalCallback(eval_env, best_model_save_path=str(save_dir),
                             log_path=str(log_dir), eval_freq=eval_freq,
                             deterministic=True, n_eval_episodes=3, verbose=1),
        PlateauEarlyStopping(eval_freq=eval_freq * args.n_envs,
                             patience=args.patience),
    ]

    print(f"\n=== Stage 1: curriculum (max {args.max_items_stage1} items) for {stage1_steps} steps ===")
    model.learn(total_timesteps=stage1_steps, callback=callbacks,
                tb_log_name=f"ppo_r{args.rotations}_stage1")

    print(f"\n=== Stage 2: full orders for {args.timesteps - stage1_steps} steps ===")
    full_env = vec_cls(full_fns)
    full_eval_fns = [_make_env(data_path=data_paths[0], seed=999,
                                **env_kwargs)]
    full_eval_env = DummyVecEnv(full_eval_fns)

    model.set_env(full_env)
    callbacks2 = [
        CheckpointCallback(save_freq=save_freq, save_path=str(save_dir),
                           name_prefix="ppo"),
        MaskableEvalCallback(full_eval_env, best_model_save_path=str(save_dir),
                             log_path=str(log_dir), eval_freq=eval_freq,
                             deterministic=True, n_eval_episodes=3, verbose=1),
        PlateauEarlyStopping(eval_freq=eval_freq * args.n_envs,
                             patience=args.patience),
    ]
    model.learn(total_timesteps=args.timesteps - stage1_steps,
                callback=callbacks2,
                tb_log_name=f"ppo_r{args.rotations}_stage2")

    final_path = save_dir / "final_model"
    model.save(final_path)
    print(f"\nTraining done.")
    print(f"  Final model : {final_path}.zip")
    print(f"  Best model  : {save_dir / 'best_model.zip'}")


if __name__ == "__main__":
    main()
