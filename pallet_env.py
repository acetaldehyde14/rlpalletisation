"""
Pallet Packing Environment  --  Extreme Point formulation.

Mirrors psatops ExtremePointAlgorithm (Java):

  Extreme points
    Origin (0,0,0) + after each placed box b:
      (b.x+b.l, b.y, b.z) | (b.x, b.y+b.w, b.z) | (b.x, b.y, b.z+b.h)
    Sorted by  z*1000 + x + y  -- gravity-first, matching ScoringUtils.

  Support check
    Items at z > 0 need >= 70% footprint area supported.
    Matches PlacementUtils.SUPPORT_THRESHOLD = 0.70.

  Overlap check
    Exact AABB, 1 mm tolerance. Matches PlacementUtils.overlaps3D.

  Rotations
    Six orientations matching RotationUtils.getAllowedRotations:
      0 LWH  1 WLH  2 LHW  3 HLW  4 WHL  5 HWL

Action space
    Discrete(K_EP * NUM_ROTATIONS) = 50 * 6 = 300 actions.
    ~66x smaller than old grid_l * grid_w * 6 sweep.

Speed
    Extreme points are updated incrementally (3 new points per box).
    Overlap and support checks are vectorised with numpy broadcasting.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Constants ──────────────────────────────────────────────────────────────────
K_EP           = 50
NUM_ROTATIONS  = 6
SUPPORT_THRESHOLD = 0.70
OVERLAP_EPS    = 1.0
SUPPORT_EPS    = 1.0
ROTATION_LABELS = ["LWH", "WLH", "LHW", "HLW", "WHL", "HWL"]

# ── Data types ─────────────────────────────────────────────────────────────────
@dataclass
class Item:
    sku: str
    length: float
    width: float
    height: float
    weight: float


@dataclass
class PlacedBox:
    x: float; y: float; z: float
    l: float; w: float; h: float
    weight: float
    sku: str
    rotation: int
    rotation_label: str


def expand_items(data: dict) -> List[Item]:
    items: List[Item] = []
    for entry in data["items"]:
        for i in range(math.ceil(entry["quantity"])):
            items.append(Item(
                sku=f"{entry['sku']}#{i:03d}",
                length=float(entry["length_mm"]),
                width=float(entry["width_mm"]),
                height=float(entry["height_mm"]),
                weight=float(entry["weight_kg"]),
            ))
    return items


# ── Geometry helpers ───────────────────────────────────────────────────────────
def get_rotation_dims(item: Item, rot: int) -> Tuple[float, float, float]:
    """(fp_l, fp_w, stack_h) for rotation 0-5. Matches psatops RotationUtils."""
    l, w, h = item.length, item.width, item.height
    return [(l,w,h),(w,l,h),(l,h,w),(h,l,w),(w,h,l),(h,w,l)][rot]


def support_fraction(boxes: List[PlacedBox],
                     x: float, y: float, z: float, l: float, w: float) -> float:
    """Scalar support fraction (used for eval stats reporting)."""
    if z <= SUPPORT_EPS:
        return 1.0
    base_area = l * w
    if base_area < 1e-6:
        return 1.0
    supported = 0.0
    for b in boxes:
        if abs((b.z + b.h) - z) < SUPPORT_EPS:
            ox = min(x+l, b.x+b.l) - max(x, b.x)
            oy = min(y+w, b.y+b.w) - max(y, b.y)
            if ox > 0 and oy > 0:
                supported += ox * oy
    return min(supported / base_area, 1.0)


def generate_extreme_points(pallet_l: float, pallet_w: float, pallet_h: float,
                             boxes: List[PlacedBox]) -> np.ndarray:
    """
    Build EP array from scratch (used in evaluate.py baselines).
    Returns (K_EP, 3) float32, padded with -1.
    """
    eps_b = 0.01
    seen: set = set()
    pts: list = []

    def add(xv, yv, zv):
        if xv > pallet_l+eps_b or yv > pallet_w+eps_b or zv > pallet_h+eps_b:
            return
        key = (round(xv*100), round(yv*100), round(zv*100))
        if key not in seen:
            seen.add(key)
            pts.append((xv, yv, zv))

    add(0., 0., 0.)
    for b in boxes:
        add(b.x+b.l, b.y,    b.z)
        add(b.x,    b.y+b.w, b.z)
        add(b.x,    b.y,     b.z+b.h)

    pts.sort(key=lambda p: p[2]*1000. + p[0] + p[1])
    result = np.full((K_EP, 3), -1., dtype=np.float32)
    n = min(len(pts), K_EP)
    if n:
        result[:n] = np.array(pts[:n], dtype=np.float32)
    return result


# ── Environment ────────────────────────────────────────────────────────────────
class PalletPackingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        items: List[Item],
        pallet_length:   float = 1200.0,
        pallet_width:    float = 1100.0,
        pallet_height:   float = 1150.0,
        max_pallet_weight: float = 1500.0,
        grid_resolution: float = 5.0,
        new_pallet_penalty:      float = 5.0,
        final_utilization_bonus: float = 50.0,
        pallet_count_penalty:    float = 3.0,
        skip_penalty:            float = 15.0,
        low_z_bonus_weight:      float = 0.3,
        sort_items: bool = True,
    ):
        super().__init__()
        self.original_items = items
        self.pallet_length = float(pallet_length)
        self.pallet_width  = float(pallet_width)
        self.pallet_height = float(pallet_height)
        self.max_pallet_weight  = float(max_pallet_weight)
        self.grid_resolution    = float(grid_resolution)
        self.new_pallet_penalty      = new_pallet_penalty
        self.final_utilization_bonus = final_utilization_bonus
        self.pallet_count_penalty    = pallet_count_penalty
        self.skip_penalty            = skip_penalty
        self.low_z_bonus_weight      = low_z_bonus_weight
        self.sort_items = sort_items

        self.grid_l = int(self.pallet_length // self.grid_resolution)
        self.grid_w = int(self.pallet_width  // self.grid_resolution)
        self.pallet_volume = self.pallet_length * self.pallet_width * self.pallet_height

        self.action_space = spaces.Discrete(K_EP * NUM_ROTATIONS)
        self.observation_space = spaces.Dict({
            "heightmap": spaces.Box(0., 1., (self.grid_l, self.grid_w), np.float32),
            "ep_obs":    spaces.Box(-0.1, 1., (K_EP, 4),               np.float32),
            "item":      spaces.Box(0., 1., (4,),                      np.float32),
            "progress":  spaces.Box(0., 1., (2,),                      np.float32),
        })

        self.items: List[Item] = []
        self.current_idx = 0
        self.total_items = 0
        self.pallets: List[dict] = []
        self.reset()

    # ── Pallet management ──────────────────────────────────────────────────────

    def _new_pallet(self) -> dict:
        ep_arr = np.full((K_EP, 3), -1., dtype=np.float32)
        ep_arr[0] = [0., 0., 0.]
        return {
            "boxes":        [],
            "boxes_arr":    np.empty((0, 6), dtype=np.float32),
            "ep_arr":       ep_arr,           # (K_EP, 3) sorted candidate positions
            "ep_seen":      {(0, 0, 0)},      # dedup keys
            "ep_scores":    [(0.0, 0.0, 0.0, 0.0)],  # sorted (score, x, y, z)
            "heightmap":    np.zeros((self.grid_l, self.grid_w), dtype=np.float32),
            "placements":   [],
            "used_volume":  0.0,
            "used_weight":  0.0,
        }

    def _add_ep_for_box(self, pallet: dict, box: PlacedBox) -> None:
        """Incremental EP update: insert the 3 new points generated by box."""
        seen   = pallet["ep_seen"]
        scores = pallet["ep_scores"]
        eps_b  = 0.01
        pl     = self.pallet_length; pw = self.pallet_width; ph = self.pallet_height

        for (xv, yv, zv) in (
            (box.x + box.l, box.y,       box.z),
            (box.x,         box.y + box.w, box.z),
            (box.x,         box.y,         box.z + box.h),
        ):
            if xv > pl + eps_b or yv > pw + eps_b or zv > ph + eps_b:
                continue
            key = (round(xv * 100), round(yv * 100), round(zv * 100))
            if key not in seen:
                seen.add(key)
                bisect.insort(scores, (zv * 1000.0 + xv + yv, xv, yv, zv))

        # Rebuild the fixed-size numpy array from the sorted score list
        n = min(len(scores), K_EP)
        ep_arr = np.full((K_EP, 3), -1., dtype=np.float32)
        if n:
            ep_arr[:n] = [[t[1], t[2], t[3]] for t in scores[:n]]
        pallet["ep_arr"] = ep_arr

    def _commit_box(self, pallet: dict, box: PlacedBox) -> None:
        pallet["boxes"].append(box)
        new_row = np.array([[box.x, box.y, box.z, box.l, box.w, box.h]], dtype=np.float32)
        pallet["boxes_arr"] = np.concatenate([pallet["boxes_arr"], new_row], axis=0)
        self._add_ep_for_box(pallet, box)
        # Rasterise onto heightmap for observation
        r  = self.grid_resolution
        x0 = int(box.x // r)
        y0 = int(box.y // r)
        x1 = min(int(math.ceil((box.x + box.l) / r)), self.grid_l)
        y1 = min(int(math.ceil((box.y + box.w) / r)), self.grid_w)
        if x1 > x0 and y1 > y0:
            np.maximum(pallet["heightmap"][x0:x1, y0:y1],
                       box.z + box.h,
                       out=pallet["heightmap"][x0:x1, y0:y1])

    # ── Single-placement check (called in step() only for the chosen action) ──

    def _check_ep(self, pallet: dict, item: Item, ep_idx: int, rot: int
                  ) -> Optional[Tuple[float, float, float, float, float, float]]:
        ep = pallet["ep_arr"][ep_idx]
        if ep[0] < -0.5:
            return None
        x, y, z = float(ep[0]), float(ep[1]), float(ep[2])
        fp_l, fp_w, stack_h = get_rotation_dims(item, rot)

        if x + fp_l > self.pallet_length + 0.01: return None
        if y + fp_w > self.pallet_width  + 0.01: return None
        if z + stack_h > self.pallet_height + 0.01: return None
        if pallet["used_weight"] + item.weight > self.max_pallet_weight + 0.001: return None

        ba = pallet["boxes_arr"]
        if len(ba):
            eps = OVERLAP_EPS
            ov = (
                (x < ba[:,0]+ba[:,3]-eps) & (x+fp_l > ba[:,0]+eps) &
                (y < ba[:,1]+ba[:,4]-eps) & (y+fp_w > ba[:,1]+eps) &
                (z < ba[:,2]+ba[:,5]-eps) & (z+stack_h > ba[:,2]+eps)
            )
            if ov.any():
                return None

        if z > SUPPORT_EPS:
            if not len(ba):
                return None
            contact = np.abs(ba[:,2] + ba[:,5] - z) < SUPPORT_EPS
            ox = np.maximum(0, np.minimum(x+fp_l, ba[:,0]+ba[:,3]) - np.maximum(x, ba[:,0]))
            oy = np.maximum(0, np.minimum(y+fp_w, ba[:,1]+ba[:,4]) - np.maximum(y, ba[:,1]))
            if (contact * ox * oy).sum() / (fp_l * fp_w) < SUPPORT_THRESHOLD:
                return None

        return (x, y, z, fp_l, fp_w, stack_h)

    # ── Vectorised action mask ─────────────────────────────────────────────────

    def action_masks(self) -> np.ndarray:
        """
        Vectorised check over all K_EP * NUM_ROTATIONS pairs.
        Overlap and support use numpy broadcasting -- no Python loops over boxes.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        item = self._current_item()
        if item is None:
            mask[0] = True
            return mask

        pallet = self.pallets[-1]
        if pallet["used_weight"] + item.weight > self.max_pallet_weight:
            mask[:] = True
            return mask

        eps_arr = pallet["ep_arr"]                    # (K_EP, 3)
        valid_ep = eps_arr[:, 0] >= -0.5              # (K_EP,) bool
        if not valid_ep.any():
            mask[:] = True
            return mask

        v_idx = np.where(valid_ep)[0]                 # (V,)
        xs = eps_arr[v_idx, 0]                        # (V,)
        ys = eps_arr[v_idx, 1]
        zs = eps_arr[v_idx, 2]
        ba = pallet["boxes_arr"]                      # (N, 6) or empty
        have = len(ba) > 0

        if have:
            bx = ba[:,0]; by = ba[:,1]; bz = ba[:,2]
            bl = ba[:,3]; bw = ba[:,4]; bh = ba[:,5]
            tops = bz + bh

        for rot in range(NUM_ROTATIONS):
            fp_l, fp_w, stack_h = get_rotation_dims(item, rot)

            # 1. Bounds
            ok = (
                (xs + fp_l <= self.pallet_length + 0.01) &
                (ys + fp_w <= self.pallet_width  + 0.01) &
                (zs + stack_h <= self.pallet_height + 0.01)
            )
            if not ok.any():
                continue

            # 2. 3D overlap  (V, 1) vs (1, N)
            if have:
                eps = OVERLAP_EPS
                coll = (
                    (xs[:,None] < bx+bl-eps) & (xs[:,None]+fp_l > bx+eps) &
                    (ys[:,None] < by+bw-eps) & (ys[:,None]+fp_w > by+eps) &
                    (zs[:,None] < bz+bh-eps) & (zs[:,None]+stack_h > bz+eps)
                )
                ok &= ~coll.any(axis=1)
            if not ok.any():
                continue

            # 3. Support fraction
            floor = zs <= SUPPORT_EPS
            nf = ok & ~floor                          # non-floor, passed overlap
            if nf.any():
                if not have:
                    ok[nf] = False
                else:
                    m = np.where(nf)[0]               # indices into V
                    xs_m = xs[m]; ys_m = ys[m]; zs_m = zs[m]
                    contact = np.abs(tops - zs_m[:,None]) < SUPPORT_EPS
                    ox = np.maximum(0, np.minimum(xs_m[:,None]+fp_l, bx+bl) -
                                       np.maximum(xs_m[:,None], bx))
                    oy = np.maximum(0, np.minimum(ys_m[:,None]+fp_w, by+bw) -
                                       np.maximum(ys_m[:,None], by))
                    sf = (contact * ox * oy).sum(axis=1) / (fp_l * fp_w)
                    ok[m[sf < SUPPORT_THRESHOLD]] = False

            if ok.any():
                mask[v_idx[ok] * NUM_ROTATIONS + rot] = True

        if not mask.any():
            mask[:] = True
        return mask

    # ── Gym API ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.items = list(self.original_items)
        if self.sort_items:
            self.items.sort(key=lambda it: -(it.length * it.width * it.height))
        self.current_idx = 0
        self.total_items = len(self.items)
        self.pallets = [self._new_pallet()]
        return self._get_obs(), {}

    def _current_item(self) -> Optional[Item]:
        if self.current_idx >= len(self.items):
            return None
        return self.items[self.current_idx]

    def _get_obs(self) -> dict:
        item   = self._current_item()
        pallet = self.pallets[-1]

        if item is None:
            item_feat = np.zeros(4, dtype=np.float32)
        else:
            item_feat = np.array([
                min(item.length / self.pallet_length, 1.0),
                min(item.width  / self.pallet_width,  1.0),
                min(item.height / self.pallet_height, 1.0),
                min(item.weight / self.max_pallet_weight, 1.0),
            ], dtype=np.float32)

        heightmap = (pallet["heightmap"] / self.pallet_height).astype(np.float32)

        ep_raw = pallet["ep_arr"]
        ep_obs = np.zeros((K_EP, 4), dtype=np.float32)
        valid  = ep_raw[:, 0] >= -0.1
        ep_obs[valid, 0] = ep_raw[valid, 0] / self.pallet_length
        ep_obs[valid, 1] = ep_raw[valid, 1] / self.pallet_width
        ep_obs[valid, 2] = ep_raw[valid, 2] / self.pallet_height
        ep_obs[valid, 3] = 1.0

        progress = np.array([
            self.current_idx / max(self.total_items, 1),
            min(len(self.pallets) / 20.0, 1.0),
        ], dtype=np.float32)

        return {"heightmap": heightmap, "ep_obs": ep_obs,
                "item": item_feat, "progress": progress}

    def step(self, action: int):
        info: dict = {}
        item = self._current_item()
        if item is None:
            return self._get_obs(), 0., True, False, info

        ep_idx = int(action) // NUM_ROTATIONS
        rot    = int(action) % NUM_ROTATIONS
        pallet = self.pallets[-1]
        result = self._check_ep(pallet, item, ep_idx, rot)
        reward = 0.

        if result is None:
            self.pallets.append(self._new_pallet())
            pallet = self.pallets[-1]
            reward -= self.new_pallet_penalty
            for try_rot in range(NUM_ROTATIONS):
                cand = self._check_ep(pallet, item, 0, try_rot)
                if cand is not None:
                    rot = try_rot; ep_idx = 0; result = cand; break

        if result is None:
            self.current_idx += 1
            reward -= self.skip_penalty
            info["skipped"] = item.sku
            terminated = self.current_idx >= self.total_items
            if terminated:
                self._apply_final_reward(info)
                reward += info["final_reward"]
            return self._get_obs(), reward, terminated, False, info

        x, y, z, fp_l, fp_w, stack_h = result
        box = PlacedBox(x=x, y=y, z=z, l=fp_l, w=fp_w, h=stack_h,
                        weight=item.weight, sku=item.sku,
                        rotation=rot, rotation_label=ROTATION_LABELS[rot])
        self._commit_box(pallet, box)

        vol = item.length * item.width * item.height
        pallet["used_volume"] += vol
        pallet["used_weight"] += item.weight
        pallet["placements"].append({
            "item": item, "x_mm": x, "y_mm": y, "z_mm": z,
            "l_mm": fp_l, "w_mm": fp_w, "h_mm": stack_h,
            "rotation": rot, "rotation_label": ROTATION_LABELS[rot],
        })

        vol_reward  = (vol / self.pallet_volume) * 20.0
        low_z_bonus = (1.0 - z / self.pallet_height) * self.low_z_bonus_weight
        reward += vol_reward + low_z_bonus

        self.current_idx += 1
        terminated = self.current_idx >= self.total_items
        if terminated:
            self._apply_final_reward(info)
            reward += info["final_reward"]
        return self._get_obs(), reward, terminated, False, info

    def _apply_final_reward(self, info: dict) -> None:
        used   = [p for p in self.pallets if p["placements"]]
        n_used = max(len(used), 1)
        total_vol = sum(p["used_volume"] for p in used)
        capacity  = n_used * self.pallet_volume
        util = total_vol / capacity if capacity > 0 else 0.
        info["final_reward"] = (util * self.final_utilization_bonus
                                - n_used * self.pallet_count_penalty)
        info["num_pallets"] = n_used
        info["utilization"] = util
        info["placed"] = sum(len(p["placements"]) for p in used)
