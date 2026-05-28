"""
Masked DQN Agent for pallet packing.

Architecture
    Dueling Double DQN with action masking.

    Observation compression
        Instead of flattening the full 60x55 heightmap (3300 features), the env's
        heightmap is compressed to two 1D profiles:
          col_max  (55,) -- max normalised height in each X column
          row_max  (60,) -- max normalised height in each Y row
        Combined with ep_obs (200), item (4), progress (2) this gives a
        321-dimensional input vector that still captures pallet state but runs
        ~10x faster than the full-heightmap version on CPU.

    Network
        Shared trunk  : Linear(321, 256)  ReLU
                        Linear(256, 128)  ReLU
        Value stream  : Linear(128, 64)   ReLU  → Linear(64, 1)
        Advantage stream: Linear(128, 128) ReLU → Linear(128, 300)
        Q(s,a) = V(s) + A(s,a) - mean(A(s,a))

    Action masking
        Before argmax, invalid actions are set to -1e9 so they are never chosen.
        The Bellman target also masks the next-state argmax over valid actions.

Usage
    python dqn_train.py --orders_dir data/orders --timesteps 300000
    python dqn_train.py --orders_dir data/orders --timesteps 50000 --quick
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pallet_env import PalletPackingEnv, Item, expand_items, K_EP, NUM_ROTATIONS

# ── Observation helper ────────────────────────────────────────────────────────
OBS_DIM = 55 + 60 + K_EP * 4 + 4 + 2   # 321
ACTION_DIM = K_EP * NUM_ROTATIONS        # 300


def flatten_obs(obs: dict) -> np.ndarray:
    """
    Compress dict observation to a 321-dim float32 array.
    col_max (55) + row_max (60) + ep_obs (200) + item (4) + progress (2)
    """
    hm = obs["heightmap"]          # (60, 55) normalised 0-1
    col_max = hm.max(axis=0)       # (55,)
    row_max = hm.max(axis=1)       # (60,)
    return np.concatenate([
        col_max,
        row_max,
        obs["ep_obs"].flatten(),   # (200,)
        obs["item"],               # (4,)
        obs["progress"],           # (2,)
    ]).astype(np.float32)


# ── Replay buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    """Circular replay buffer storing (obs, action, reward, next_obs, done, next_mask)."""

    def __init__(self, capacity: int = 80_000):
        self.buf: deque = deque(maxlen=capacity)

    def push(
        self,
        obs:       np.ndarray,
        action:    int,
        reward:    float,
        next_obs:  np.ndarray,
        done:      bool,
        next_mask: np.ndarray,
    ) -> None:
        self.buf.append((obs, action, reward, next_obs, done, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, acts, rews, nobs, dones, nmasks = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)),
            torch.LongTensor(acts),
            torch.FloatTensor(rews),
            torch.FloatTensor(np.array(nobs)),
            torch.FloatTensor(dones),
            torch.BoolTensor(np.array(nmasks)),
        )

    def __len__(self) -> int:
        return len(self.buf)


# ── Network ───────────────────────────────────────────────────────────────────
class DuelingQNet(nn.Module):
    """Dueling DQN with shared trunk and separate value / advantage streams."""

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.trunk(x)
        v = self.value(shared)                   # (B, 1)
        a = self.advantage(shared)               # (B, action_dim)
        return v + (a - a.mean(dim=1, keepdim=True))


# ── Agent ─────────────────────────────────────────────────────────────────────
class MaskedDQNAgent:
    """
    Masked Double Dueling DQN.

    Key behaviours
      - Invalid actions are set to -1e9 before argmax (both selection and target).
      - Double DQN: online net selects the next action; target net evaluates it.
      - Target network synced with hard copy every target_update_freq steps.
      - Epsilon-greedy over valid actions only during exploration.
    """

    def __init__(
        self,
        obs_dim:           int   = OBS_DIM,
        action_dim:        int   = ACTION_DIM,
        lr:                float = 3e-4,
        gamma:             float = 0.995,
        epsilon_start:     float = 1.0,
        epsilon_end:       float = 0.05,
        epsilon_decay_steps: int = 200_000,
        buffer_capacity:   int   = 80_000,
        batch_size:        int   = 128,
        target_update_freq: int  = 500,
        warmup_steps:      int   = 2_000,
        device:            str   = "cpu",
    ):
        self.gamma             = gamma
        self.epsilon           = epsilon_start
        self.epsilon_end       = epsilon_end
        self.epsilon_decay     = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.batch_size        = batch_size
        self.target_update_freq = target_update_freq
        self.warmup_steps      = warmup_steps
        self.device            = torch.device(device)

        self.online_net = DuelingQNet(obs_dim, action_dim).to(self.device)
        self.target_net = DuelingQNet(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_capacity)

        self.step_count  = 0
        self.update_count = 0
        self.losses: List[float] = []

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, obs_flat: np.ndarray, mask: np.ndarray,
                      greedy: bool = False) -> int:
        """Epsilon-greedy action selection. Only valid (masked) actions chosen."""
        valid = np.where(mask)[0]
        if not greedy and random.random() < self.epsilon:
            return int(random.choice(valid))

        with torch.no_grad():
            q = self.online_net(
                torch.FloatTensor(obs_flat).unsqueeze(0).to(self.device)
            ).squeeze(0).cpu().numpy()
        q[~mask] = -1e9
        return int(q.argmax())

    # ── Training ──────────────────────────────────────────────────────────────

    def push(self, obs, action, reward, next_obs, done, next_mask):
        self.buffer.push(obs, action, float(reward), next_obs, bool(done), next_mask)
        self.step_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def update(self) -> Optional[float]:
        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return None

        obs, acts, rews, nobs, dones, nmasks = self.buffer.sample(self.batch_size)
        obs   = obs.to(self.device)
        acts  = acts.to(self.device)
        rews  = rews.to(self.device)
        nobs  = nobs.to(self.device)
        dones = dones.to(self.device)
        nmasks= nmasks.to(self.device)

        # Current Q-values
        q_curr = self.online_net(obs).gather(1, acts.unsqueeze(1)).squeeze(1)

        # Double DQN target: online net picks next action, target net evaluates
        with torch.no_grad():
            q_next_online = self.online_net(nobs)
            q_next_online[~nmasks] = -1e9
            next_actions = q_next_online.argmax(dim=1)

            q_next_target = self.target_net(nobs)
            q_next_vals   = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target = rews + self.gamma * q_next_vals * (1.0 - dones)

        loss = F.smooth_l1_loss(q_curr, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        loss_val = float(loss.item())
        self.losses.append(loss_val)
        return loss_val

    def save(self, path: str) -> None:
        torch.save({
            "online_state":  self.online_net.state_dict(),
            "target_state":  self.target_net.state_dict(),
            "optimizer":     self.optimizer.state_dict(),
            "step_count":    self.step_count,
            "epsilon":       self.epsilon,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_state"])
        self.target_net.load_state_dict(ckpt["target_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step_count = ckpt["step_count"]
        self.epsilon    = ckpt["epsilon"]


# ── Order loader ──────────────────────────────────────────────────────────────
def load_orders(orders_dir: str) -> List[List[Item]]:
    """Load all JSON orders from a directory into lists of Items."""
    order_files = sorted(
        f for f in os.listdir(orders_dir)
        if f.endswith(".json") and f != "catalog.json"
    )
    orders = []
    for fname in order_files:
        with open(os.path.join(orders_dir, fname)) as f:
            data = json.load(f)
        items = expand_items(data)
        if items:
            orders.append(items)
    return orders


def make_env(items: List[Item], sort: bool = True, **kwargs) -> PalletPackingEnv:
    return PalletPackingEnv(items, sort_items=sort, **kwargs)


# ── Training loop ──────────────────────────────────────────────────────────────
def train(
    orders_dir:    str   = "data/orders",
    save_dir:      str   = "checkpoints_dqn",
    timesteps:     int   = 300_000,
    eval_freq:     int   = 10_000,
    n_eval_eps:    int   = 5,
    lr:            float = 3e-4,
    gamma:         float = 0.995,
    epsilon_decay_steps: int = 180_000,
    batch_size:    int   = 128,
    buffer_cap:    int   = 80_000,
    target_update: int   = 500,
    warmup:        int   = 2_000,
    rand_sort_prob: float = 0.3,
    verbose:       bool  = True,
) -> MaskedDQNAgent:
    """
    Main training loop.

    Each episode:
      1. Randomly sample an order from the pool.
      2. Run full episode (all items placed or skipped).
      3. After each step, push transition to buffer and call agent.update().

    rand_sort_prob
      With this probability the item list is shuffled randomly instead of
      sorted by volume-descending. This exposes the agent to harder item
      orderings and improves generalisation.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print("Loading orders...")
    orders = load_orders(orders_dir)
    print(f"  {len(orders)} orders loaded, "
          f"sizes {min(len(o) for o in orders)}-{max(len(o) for o in orders)} items")

    agent = MaskedDQNAgent(
        lr=lr, gamma=gamma,
        epsilon_decay_steps=epsilon_decay_steps,
        buffer_capacity=buffer_cap,
        batch_size=batch_size,
        target_update_freq=target_update,
        warmup_steps=warmup,
    )

    env_kwargs = dict(
        pallet_length=1200.0, pallet_width=1100.0,
        pallet_height=1150.0, max_pallet_weight=1500.0,
    )

    total_steps  = 0
    episode      = 0
    best_eval    = -float("inf")
    ep_rewards   = []
    ep_utils     = []
    ep_pallets   = []
    t0           = time.perf_counter()

    while total_steps < timesteps:
        # Pick a random order; sometimes shuffle instead of volume-sort
        items     = random.choice(orders)
        use_sort  = random.random() > rand_sort_prob
        env       = make_env(items, sort=use_sort, **env_kwargs)
        obs, _    = env.reset()
        obs_flat  = flatten_obs(obs)
        ep_reward = 0.0
        ep_done   = False

        while not ep_done:
            mask   = env.action_masks()
            action = agent.select_action(obs_flat, mask)
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_done = terminated or truncated

            next_obs_flat = flatten_obs(next_obs)
            next_mask     = env.action_masks() if not ep_done else np.zeros(ACTION_DIM, dtype=bool)

            agent.push(obs_flat, action, reward, next_obs_flat, ep_done, next_mask)
            agent.update()

            obs_flat   = next_obs_flat
            ep_reward += reward
            total_steps += 1

            if total_steps >= timesteps:
                ep_done = True

        ep_rewards.append(ep_reward)
        if "utilization" in info:
            ep_utils.append(info["utilization"] * 100)
            ep_pallets.append(info["num_pallets"])

        episode += 1

        # ── Periodic evaluation ────────────────────────────────────────────
        if total_steps % eval_freq == 0 or total_steps >= timesteps:
            eval_utils, eval_pallets = _evaluate(agent, orders, n_eval_eps, env_kwargs)
            avg_util = np.mean(eval_utils)

            if avg_util > best_eval:
                best_eval = avg_util
                agent.save(os.path.join(save_dir, "best_model_dqn.pt"))

            agent.save(os.path.join(save_dir, f"dqn_step{total_steps}.pt"))

            if verbose:
                elapsed   = time.perf_counter() - t0
                sps       = total_steps / elapsed
                avg_loss  = np.mean(agent.losses[-200:]) if agent.losses else 0.0
                print(
                    f"  step {total_steps:>7,}  eps {agent.epsilon:.3f}  "
                    f"eval_util {avg_util:5.1f}%  eval_pallets {np.mean(eval_pallets):.1f}  "
                    f"loss {avg_loss:.4f}  {sps:.0f} steps/s"
                )

    agent.save(os.path.join(save_dir, "final_model_dqn.pt"))
    print(f"\nTraining done.  Best eval util: {best_eval:.1f}%")
    print(f"  Models saved in {save_dir}/")
    return agent


def _evaluate(
    agent: MaskedDQNAgent,
    orders: List[List[Item]],
    n_eps:  int,
    env_kwargs: dict,
) -> tuple:
    """Run n_eps greedy episodes on randomly chosen orders. Returns (utils, pallets)."""
    utils, pallets = [], []
    sample = random.sample(orders, min(n_eps, len(orders)))
    for items in sample:
        env      = make_env(items, sort=True, **env_kwargs)
        obs, _   = env.reset()
        obs_flat = flatten_obs(obs)
        done     = False
        while not done:
            mask   = env.action_masks()
            action = agent.select_action(obs_flat, mask, greedy=True)
            obs, _, terminated, truncated, info = env.step(action)
            obs_flat = flatten_obs(obs)
            done = terminated or truncated
        if "utilization" in info:
            utils.append(info["utilization"] * 100)
            pallets.append(info["num_pallets"])
    return utils, pallets


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orders_dir",  default="data/orders")
    parser.add_argument("--save_dir",    default="checkpoints_dqn")
    parser.add_argument("--timesteps",   type=int,   default=300_000)
    parser.add_argument("--eval_freq",   type=int,   default=10_000)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--gamma",       type=float, default=0.995)
    parser.add_argument("--buffer",      type=int,   default=80_000,
                        help="Replay buffer capacity")
    parser.add_argument("--target_update", type=int, default=500,
                        help="Hard target network sync every N updates")
    parser.add_argument("--warmup",      type=int,   default=2_000,
                        help="Steps before training starts")
    parser.add_argument("--rand_sort",   type=float, default=0.3,
                        help="Fraction of episodes using random item order (0-1)")
    parser.add_argument("--quick",       action="store_true",
                        help="Quick smoke test: 20K steps only")
    args = parser.parse_args()

    if args.quick:
        args.timesteps = 20_000
        args.warmup    = 500
        args.eval_freq = 5_000

    print(f"Action dim : {ACTION_DIM}")
    print(f"Obs dim    : {OBS_DIM}")
    print(f"Timesteps  : {args.timesteps:,}")

    train(
        orders_dir  = args.orders_dir,
        save_dir    = args.save_dir,
        timesteps   = args.timesteps,
        eval_freq   = args.eval_freq,
        lr          = args.lr,
        gamma       = args.gamma,
        batch_size  = args.batch_size,
        buffer_cap  = args.buffer,
        target_update = args.target_update,
        warmup      = args.warmup,
        rand_sort_prob = args.rand_sort,
    )


if __name__ == "__main__":
    main()
