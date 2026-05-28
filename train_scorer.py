"""
Supervised Placement Scorer.

Train a CNN to score candidate placements by imitating the EP heuristic.
At inference, score all valid (EP, rotation) pairs and pick the highest.

This is pure imitation learning with a CNN — no RL needed.

Usage
    python train_scorer.py --epochs 50
    python evaluate.py --model checkpoints/scorer/best_scorer.pt --scorer
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pallet_env import (
    PalletPackingEnv, expand_items, get_rotation_dims,
    K_EP_TOP, NUM_ROTATIONS, ROTATION_LABELS, PlacedBox,
    SUPPORT_THRESHOLD, OVERLAP_EPS, SUPPORT_EPS,
)
from sb3_contrib.common.wrappers import ActionMasker


class PlacementDataset(Dataset):
    def __init__(self, data_paths: list, n_demos: int = 500,
                 pallet_length=1200.0, pallet_width=1100.0,
                 pallet_height=1150.0, max_weight=1500.0,
                 num_rotations=6):
        self.samples = []
        self.pallet_length = pallet_length
        self.pallet_width = pallet_width
        self.pallet_height = pallet_height
        self.max_weight = max_weight
        self.num_rotations = num_rotations
        self._collect(data_paths, n_demos)

    def _collect(self, data_paths, n_demos):
        print(f"Collecting {n_demos} expert demos...")
        for demo_i in range(n_demos):
            data_path = data_paths[demo_i % len(data_paths)]
            with open(data_path) as f:
                data = json.load(f)
            items = expand_items(data)
            if len(items) < 2:
                continue

            env = PalletPackingEnv(
                items,
                pallet_length=self.pallet_length,
                pallet_width=self.pallet_width,
                pallet_height=self.pallet_height,
                max_pallet_weight=self.max_weight,
                num_rotations=self.num_rotations,
            )

            obs, _ = env.reset()
            done = False
            while not done:
                expert_action = self._ep_expert_action(env)
                masks = env.action_masks()

                if not masks[expert_action]:
                    valid = np.where(masks)[0]
                    if len(valid) > 0:
                        expert_action = int(valid[0])
                    else:
                        break

                valid_actions = np.where(masks)[0]
                if len(valid_actions) > 1:
                    n_valid = len(valid_actions)
                    scores = np.full(n_valid, -1.0, dtype=np.float32)
                    for i, a in enumerate(valid_actions):
                        if a == expert_action:
                            scores[i] = 1.0
                        else:
                            scores[i] = 0.0

                    self.samples.append({
                        "heightmap": obs["heightmap"].copy(),
                        "ep_obs": obs["ep_obs"].copy(),
                        "item": obs["item"].copy(),
                        "progress": obs["progress"].copy(),
                        "valid_actions": valid_actions.copy(),
                        "scores": scores.copy(),
                    })

                obs, _, terminated, truncated, _ = env.step(expert_action)
                done = terminated or truncated

        print(f"Collected {len(self.samples)} placement decisions")

    def _ep_expert_action(self, env):
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
                z = float(ep_arr[real_idx, 2])
                x = float(ep_arr[real_idx, 0])
                y = float(ep_arr[real_idx, 1])
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s["heightmap"],
            s["ep_obs"],
            s["item"],
            s["progress"],
            s["valid_actions"],
            s["scores"],
        )


class PlacementScorer(nn.Module):
    def __init__(self, grid_l=240, grid_w=220, n_actions=90):
        super().__init__()
        self.n_actions = n_actions

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
            dummy = torch.zeros(1, 1, grid_l, grid_w)
            cnn_out_dim = self.heightmap_cnn(dummy).shape[1]

        n_other = K_EP_TOP * 4 + 4 + 3
        self.other_mlp = nn.Sequential(
            nn.Linear(n_other, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.action_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
        )

        self.scorer = nn.Sequential(
            nn.Linear(cnn_out_dim + 64 + 16, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def score_actions(self, heightmap, ep_obs, item, progress, actions):
        B = heightmap.shape[0]
        hm = heightmap.unsqueeze(1)
        cnn_feat = self.heightmap_cnn(hm)
        ep_flat = ep_obs.reshape(B, -1)
        other = torch.cat([ep_flat, item, progress], dim=1)
        mlp_feat = self.other_mlp(other)
        state_feat = torch.cat([cnn_feat, mlp_feat], dim=1)

        state_expanded = state_feat.unsqueeze(1).expand(-1, actions.shape[1], -1)
        act_emb = self.action_embed(actions.float().unsqueeze(-1))
        combined = torch.cat([state_expanded, act_emb], dim=-1)
        scores = self.scorer(combined).squeeze(-1)
        return scores


def collate_fn(batch):
    heightmaps = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
    ep_obs = torch.tensor(np.stack([b[1] for b in batch]), dtype=torch.float32)
    items = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.float32)
    progress = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)

    max_valid = max(len(b[4]) for b in batch)
    n_actions = 90
    valid_actions = torch.zeros(len(batch), max_valid, dtype=torch.long)
    scores = torch.full((len(batch), max_valid), -100.0, dtype=torch.float32)
    n_valid = torch.zeros(len(batch), dtype=torch.long)

    for i, b in enumerate(batch):
        nv = len(b[4])
        valid_actions[i, :nv] = torch.tensor(b[4], dtype=torch.long)
        scores[i, :nv] = torch.tensor(b[5], dtype=torch.float32)
        n_valid[i] = nv

    return heightmaps, ep_obs, items, progress, valid_actions, scores, n_valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=None)
    parser.add_argument("--n_demos", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", default="./checkpoints/scorer")
    parser.add_argument("--pallet_length", type=float, default=1200.0)
    parser.add_argument("--pallet_width", type=float, default=1100.0)
    parser.add_argument("--pallet_height", type=float, default=1150.0)
    parser.add_argument("--max_weight", type=float, default=1500.0)
    parser.add_argument("--rotations", type=int, default=6)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

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

    dataset = PlacementDataset(
        data_paths, n_demos=args.n_demos,
        pallet_length=args.pallet_length,
        pallet_width=args.pallet_width,
        pallet_height=args.pallet_height,
        max_weight=args.max_weight,
        num_rotations=args.rotations,
    )

    if len(dataset) == 0:
        print("No data collected!")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PlacementScorer(
        grid_l=int(args.pallet_length // 5),
        grid_w=int(args.pallet_width // 5),
        n_actions=K_EP_TOP * args.rotations,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        correct = 0
        total = 0

        for heightmaps, ep_obs, items, progress, valid_actions, scores, n_valid in loader:
            heightmaps = heightmaps.to(device)
            ep_obs = ep_obs.to(device)
            items = items.to(device)
            progress = progress.to(device)
            valid_actions = valid_actions.to(device)
            scores = scores.to(device)
            n_valid = n_valid.to(device)

            B = heightmaps.shape[0]
            max_v = valid_actions.shape[1]

            pred = model.score_actions(heightmaps, ep_obs, items, progress,
                                       valid_actions)

            mask = torch.arange(max_v, device=device).unsqueeze(0) < n_valid.unsqueeze(1)
            pred_masked = pred[mask]
            target_masked = scores[mask]

            loss = criterion(pred_masked, target_masked)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            with torch.no_grad():
                for i in range(B):
                    nv = int(n_valid[i].item())
                    if nv < 2:
                        continue
                    p = pred[i, :nv]
                    expert_idx = torch.argmax(scores[i, :nv]).item()
                    pred_idx = torch.argmax(p).item()
                    if expert_idx == pred_idx:
                        correct += 1
                    total += 1

        avg_loss = total_loss / max(n_batches, 1)
        acc = correct / max(total, 1) * 100
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model_state": model.state_dict(),
                "grid_l": int(args.pallet_length // 5),
                "grid_w": int(args.pallet_width // 5),
                "n_actions": K_EP_TOP * args.rotations,
            }, save_dir / "best_scorer.pt")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"acc={acc:.1f}%  best_loss={best_loss:.4f}")

    print(f"\nBest scorer saved to {save_dir / 'best_scorer.pt'}")


if __name__ == "__main__":
    main()
