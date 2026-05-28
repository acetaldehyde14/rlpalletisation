"""
Item-Ordering Environment for pallet packing.

Instead of learning WHERE to place items, this env lets the RL agent
choose in WHAT ORDER to place them. The EP heuristic handles actual
placement decisions. The agent just picks which item to place next.

This completely avoids the spatial perception problem — EP handles all
3D geometry. The agent only needs to learn high-level packing strategy
(e.g., "place similar-sized items together", "large items first").

Action space: Discrete(max_items) with masks for already-placed items.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pallet_env import (
    Item, PlacedBox, expand_items, get_rotation_dims,
    ROTATION_LABELS, SUPPORT_THRESHOLD, SUPPORT_EPS, OVERLAP_EPS,
)

MAX_ITEMS = 300
MAX_SKUS = 50


@dataclass
class _OrderPallet:
    boxes: list
    placements: list
    used_volume: float
    used_weight: float


def _gen_eps(pallet_l, pallet_w, pallet_h, boxes):
    eps_b = 0.01
    seen = set()
    pts = []
    def add(xv, yv, zv):
        if xv > pallet_l + eps_b or yv > pallet_w + eps_b or zv > pallet_h + eps_b:
            return
        key = (round(xv * 100), round(yv * 100), round(zv * 100))
        if key not in seen:
            seen.add(key)
            pts.append((xv, yv, zv))
    add(0., 0., 0.)
    for b in boxes:
        add(b.x + b.l, b.y, b.z)
        add(b.x, b.y + b.w, b.z)
        add(b.x, b.y, b.z + b.h)
    pts.sort(key=lambda p: (p[2], p[0], p[1]))
    return pts


def _can_place(pallet, x, y, z, l, w, h, weight,
               pallet_l, pallet_w, pallet_h, max_weight):
    EPS = 1.0
    if x + l > pallet_l + EPS: return False
    if y + w > pallet_w + EPS: return False
    if z + h > pallet_h + EPS: return False
    if pallet.used_weight + weight > max_weight + 0.001: return False
    for b in pallet.boxes:
        if (x < b.x + b.l - EPS and x + l > b.x + EPS and
            y < b.y + b.w - EPS and y + w > b.y + EPS and
            z < b.z + b.h - EPS and z + h > b.z + EPS):
            return False
    if z > 1.0:
        base_area = l * w
        if base_area < 1e-6: return False
        supported = 0.0
        for b in pallet.boxes:
            if abs((b.z + b.h) - z) < 1.0:
                ox = min(x + l, b.x + b.l) - max(x, b.x)
                oy = min(y + w, b.y + b.w) - max(y, b.y)
                if ox > 0 and oy > 0:
                    supported += ox * oy
        if supported / base_area < 0.70: return False
    return True


def _ep_place(item, pallet, pallet_l, pallet_w, pallet_h, max_weight,
              num_rotations=6):
    pts = _gen_eps(pallet_l, pallet_w, pallet_h, pallet.boxes)
    rotations = list(range(num_rotations))
    base_areas = [get_rotation_dims(item, r)[0] * get_rotation_dims(item, r)[1]
                  for r in rotations]
    max_base = max(base_areas) if base_areas else 1.0
    primary = [r for r, a in zip(rotations, base_areas) if abs(a - max_base) < 1e-6]
    fallback = [r for r in rotations if r not in primary]

    for rot_group in [primary, fallback]:
        best = None
        best_score = float("inf")
        for pt in pts:
            x, y, z = pt
            for rot in rot_group:
                fp_l, fp_w, stack_h = get_rotation_dims(item, rot)
                if not _can_place(pallet, x, y, z, fp_l, fp_w, stack_h,
                                  item.weight, pallet_l, pallet_w,
                                  pallet_h, max_weight):
                    continue
                score = z * 1000.0 + x + y
                if score < best_score:
                    best_score = score
                    best = (x, y, z, fp_l, fp_w, stack_h, rot)
        if best is not None:
            x, y, z, fp_l, fp_w, stack_h, rot = best
            box = PlacedBox(x=x, y=y, z=z, l=fp_l, w=fp_w, h=stack_h,
                            weight=item.weight, sku=item.sku,
                            rotation=rot, rotation_label=ROTATION_LABELS[rot])
            pallet.boxes.append(box)
            pallet.used_volume += item.length * item.width * item.height
            pallet.used_weight += item.weight
            pallet.placements.append({
                "item": item, "x_mm": x, "y_mm": y, "z_mm": z,
                "l_mm": fp_l, "w_mm": fp_w, "h_mm": stack_h,
                "rotation": rot, "rotation_label": ROTATION_LABELS[rot],
            })
            return True
    return False


class ItemOrderEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        items: List[Item],
        pallet_length: float = 1200.0,
        pallet_width: float = 1100.0,
        pallet_height: float = 1150.0,
        max_pallet_weight: float = 1500.0,
        num_rotations: int = 6,
    ):
        super().__init__()
        self.original_items = items
        self.pallet_length = float(pallet_length)
        self.pallet_width = float(pallet_width)
        self.pallet_height = float(pallet_height)
        self.max_pallet_weight = float(max_pallet_weight)
        self.num_rotations = int(num_rotations)
        self.pallet_volume = pallet_length * pallet_width * pallet_height

        self.n_items = min(len(items), MAX_ITEMS)
        sku_set = sorted(set(it.sku.split("#")[0] for it in items))
        self.sku_to_idx = {s: i for i, s in enumerate(sku_set)}
        self.n_skus = len(sku_set)

        self.action_space = spaces.Discrete(MAX_ITEMS)
        self.observation_space = spaces.Dict({
            "item_volumes": spaces.Box(0., 1., (MAX_ITEMS,), np.float32),
            "placed_mask": spaces.Box(0, 1, (MAX_ITEMS,), np.int32),
            "pallet_utils": spaces.Box(0., 1., (10,), np.float32),
            "progress": spaces.Box(0., 1., (3,), np.float32),
        })

        self.items: List[Item] = []
        self.placed: np.ndarray = np.zeros(self.n_items, dtype=bool)
        self.pallets: List[_OrderPallet] = []
        self.step_count = 0
        self.total_volume = 0.0
        self.reset()

    def _new_pallet(self) -> _OrderPallet:
        return _OrderPallet(boxes=[], placements=[], used_volume=0.0, used_weight=0.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.items = list(self.original_items[:self.n_items])
        self.placed = np.zeros(self.n_items, dtype=bool)
        self.pallets = [self._new_pallet()]
        self.step_count = 0
        self.total_volume = sum(
            it.length * it.width * it.height for it in self.items
        )
        return self._get_obs(), {}

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(MAX_ITEMS, dtype=bool)
        mask[:self.n_items] = ~self.placed
        return mask

    def _get_obs(self) -> dict:
        item_volumes = np.zeros(MAX_ITEMS, dtype=np.float32)
        for i, item in enumerate(self.items):
            if i < MAX_ITEMS:
                vol = item.length * item.width * item.height
                item_volumes[i] = min(vol / self.pallet_volume, 1.0)

        placed_mask = np.zeros(MAX_ITEMS, dtype=np.int32)
        placed_mask[:self.n_items] = self.placed.astype(np.int32)

        pallet_utils = np.zeros(10, dtype=np.float32)
        for i, p in enumerate(self.pallets[:10]):
            pallet_utils[i] = min(p.used_volume / self.pallet_volume, 1.0)

        progress = np.array([
            self.step_count / max(self.n_items, 1),
            min(len(self.pallets) / 10.0, 1.0),
            sum(p.used_volume for p in self.pallets) / max(self.total_volume, 1.0),
        ], dtype=np.float32)

        return {
            "item_volumes": item_volumes,
            "placed_mask": placed_mask,
            "pallet_utils": pallet_utils,
            "progress": progress,
        }

    def step(self, action: int):
        info = {}

        if action >= self.n_items or self.placed[action]:
            reward = -5.0
            self.step_count += 1
            terminated = self.step_count >= self.n_items
            if terminated:
                reward += self._final_reward(info)
            return self._get_obs(), reward, terminated, False, info

        item = self.items[action]
        self.placed[action] = True

        placed_on_existing = False
        for pallet in self.pallets:
            if _ep_place(item, pallet, self.pallet_length, self.pallet_width,
                         self.pallet_height, self.max_pallet_weight,
                         self.num_rotations):
                placed_on_existing = True

                vol = item.length * item.width * item.height
                cur_pallet = self.pallets[-1]
                util_before = cur_pallet.used_volume / self.pallet_volume
                util_after = (cur_pallet.used_volume) / self.pallet_volume

                reward = (vol / self.pallet_volume) * 10.0
                if util_after > 0.7:
                    reward += 3.0
                if util_after > 0.9:
                    reward += 5.0
                break

        if not placed_on_existing:
            new_pallet = self._new_pallet()
            self.pallets.append(new_pallet)
            success = _ep_place(item, new_pallet, self.pallet_length,
                                self.pallet_width, self.pallet_height,
                                self.max_pallet_weight, self.num_rotations)
            reward = -2.0
            if success:
                vol = item.length * item.width * item.height
                reward += (vol / self.pallet_volume) * 5.0
            else:
                reward -= 5.0
                info["skipped"] = item.sku

        self.step_count += 1
        terminated = self.step_count >= self.n_items
        if terminated:
            reward += self._final_reward(info)

        return self._get_obs(), reward, terminated, False, info

    def _final_reward(self, info: dict) -> float:
        n_pallets = len([p for p in self.pallets if p.placements])
        total_vol = sum(p.used_volume for p in self.pallets)
        capacity = n_pallets * self.pallet_volume if n_pallets > 0 else 1.0
        util = total_vol / capacity

        n_placed = sum(len(p.placements) for p in self.pallets)
        n_skipped = self.n_items - n_placed

        info["num_pallets"] = n_pallets
        info["utilization"] = util
        info["placed"] = n_placed

        pallet_bonuses = 0.0
        for p in self.pallets:
            p_util = p.used_volume / self.pallet_volume
            if p_util > 0.9:
                pallet_bonuses += 50.0
            elif p_util > 0.7:
                pallet_bonuses += 25.0
            elif p_util > 0.5:
                pallet_bonuses += 10.0

        return pallet_bonuses + util * 100.0 - n_pallets * 20.0 - n_skipped * 20.0
