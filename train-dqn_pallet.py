import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


@dataclass
class Box:
    sku: str
    length_mm: float
    width_mm: float
    height_mm: float
    weight_kg: float


@dataclass
class Placement:
    sku: str
    x_mm: float
    y_mm: float
    z_mm: float
    length_mm: float
    width_mm: float
    height_mm: float
    weight_kg: float
    rotation: int


class PalletPackingEnv(gym.Env):
    """
    DQN palletisation environment.

    DQN action:
        choose one discrete placement for the next box:
        action = x_bin, y_bin, rotation

    Simplification:
        - The environment places boxes in a 2.5D heightmap.
        - A box can only be placed on a flat supported rectangular area.
        - This prevents floating boxes.
        - It is suitable as a first RL baseline.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        boxes: List[Box],
        pallet_length_mm: int = 1200,
        pallet_width_mm: int = 1000,
        max_height_mm: int = 1150,
        max_weight_kg: float = 1500.0,
        grid_mm: int = 50,
        max_boxes_per_episode: int = 240,
        shuffle_boxes: bool = True,
        invalid_action_penalty: float = -0.25,
    ):
        super().__init__()

        self.original_boxes = boxes
        self.pallet_length_mm = pallet_length_mm
        self.pallet_width_mm = pallet_width_mm
        self.max_height_mm = max_height_mm
        self.max_weight_kg = max_weight_kg
        self.grid_mm = grid_mm
        self.max_boxes_per_episode = max_boxes_per_episode
        self.shuffle_boxes = shuffle_boxes
        self.invalid_action_penalty = invalid_action_penalty

        self.grid_l = math.ceil(pallet_length_mm / grid_mm)
        self.grid_w = math.ceil(pallet_width_mm / grid_mm)

        self.rotations = [
            # length, width, height mappings
            (0, 1, 2),
            (1, 0, 2),
            (0, 2, 1),
            (2, 0, 1),
            (1, 2, 0),
            (2, 1, 0),
        ]

        self.num_rotations = len(self.rotations)

        self.action_space = spaces.Discrete(
            self.grid_l * self.grid_w * self.num_rotations
        )

        # Observation:
        # - normalized heightmap flattened
        # - current box dimensions/weight normalized
        # - remaining boxes ratio
        # - current total weight ratio
        obs_size = self.grid_l * self.grid_w + 6
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.boxes: List[Box] = []
        self.current_index = 0
        self.heightmap = np.zeros((self.grid_l, self.grid_w), dtype=np.float32)
        self.total_weight = 0.0
        self.placements: List[Placement] = []

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)

        self.boxes = list(self.original_boxes)

        if self.shuffle_boxes:
            self.np_random.shuffle(self.boxes)

        self.boxes = self.boxes[: self.max_boxes_per_episode]

        # Start with larger-base boxes first.
        # This makes training easier and improves stability.
        self.boxes.sort(
            key=lambda b: max(
                b.length_mm * b.width_mm,
                b.length_mm * b.height_mm,
                b.width_mm * b.height_mm,
            ),
            reverse=True,
        )

        self.current_index = 0
        self.heightmap = np.zeros((self.grid_l, self.grid_w), dtype=np.float32)
        self.total_weight = 0.0
        self.placements = []

        return self._get_obs(), {}

    def step(self, action: int):
        if self.current_index >= len(self.boxes):
            return self._get_obs(), 0.0, True, False, self._info()

        box = self.boxes[self.current_index]
        x_bin, y_bin, rot_idx = self._decode_action(action)

        dims = self._rotated_dims(box, rot_idx)
        placed, reward = self._try_place(box, dims, x_bin, y_bin, rot_idx)

        if placed:
            self.current_index += 1
        else:
            # Failed action still advances slowly to avoid getting stuck forever.
            # You can remove this if you want the agent to retry the same box.
            reward = self.invalid_action_penalty
            self.current_index += 1

        terminated = self.current_index >= len(self.boxes)
        truncated = False

        if terminated:
            reward += self._episode_end_bonus()

        return self._get_obs(), reward, terminated, truncated, self._info()

    def _decode_action(self, action: int) -> Tuple[int, int, int]:
        rot_idx = action % self.num_rotations
        tmp = action // self.num_rotations
        y_bin = tmp % self.grid_w
        x_bin = tmp // self.grid_w
        return x_bin, y_bin, rot_idx

    def _rotated_dims(self, box: Box, rot_idx: int) -> Tuple[float, float, float]:
        base = [box.length_mm, box.width_mm, box.height_mm]
        mapping = self.rotations[rot_idx]
        return base[mapping[0]], base[mapping[1]], base[mapping[2]]

    def _try_place(
        self,
        box: Box,
        dims: Tuple[float, float, float],
        x_bin: int,
        y_bin: int,
        rot_idx: int,
    ) -> Tuple[bool, float]:
        l_mm, w_mm, h_mm = dims

        l_cells = math.ceil(l_mm / self.grid_mm)
        w_cells = math.ceil(w_mm / self.grid_mm)

        if x_bin + l_cells > self.grid_l:
            return False, self.invalid_action_penalty

        if y_bin + w_cells > self.grid_w:
            return False, self.invalid_action_penalty

        x_mm = x_bin * self.grid_mm
        y_mm = y_bin * self.grid_mm

        if x_mm + l_mm > self.pallet_length_mm:
            return False, self.invalid_action_penalty

        if y_mm + w_mm > self.pallet_width_mm:
            return False, self.invalid_action_penalty

        footprint = self.heightmap[
            x_bin : x_bin + l_cells,
            y_bin : y_bin + w_cells,
        ]

        # Strict support rule:
        # entire footprint must be flat. This avoids floating / half-supported boxes.
        z_mm = float(np.max(footprint))

        if not np.allclose(footprint, z_mm):
            return False, self.invalid_action_penalty

        if z_mm + h_mm > self.max_height_mm:
            return False, self.invalid_action_penalty

        if self.total_weight + box.weight_kg > self.max_weight_kg:
            return False, self.invalid_action_penalty

        self.heightmap[
            x_bin : x_bin + l_cells,
            y_bin : y_bin + w_cells,
        ] = z_mm + h_mm

        self.total_weight += box.weight_kg

        self.placements.append(
            Placement(
                sku=box.sku,
                x_mm=x_mm,
                y_mm=y_mm,
                z_mm=z_mm,
                length_mm=l_mm,
                width_mm=w_mm,
                height_mm=h_mm,
                weight_kg=box.weight_kg,
                rotation=rot_idx,
            )
        )

        box_volume = l_mm * w_mm * h_mm
        pallet_volume = (
            self.pallet_length_mm * self.pallet_width_mm * self.max_height_mm
        )

        volume_reward = box_volume / pallet_volume

        # Prefer lower, compact stacks.
        height_penalty = 0.03 * ((z_mm + h_mm) / self.max_height_mm)

        # Small reward for valid placement.
        reward = 1.0 + 10.0 * volume_reward - height_penalty

        return True, reward

    def _episode_end_bonus(self) -> float:
        placed_count = len(self.placements)
        total_count = len(self.boxes)

        if total_count == 0:
            return 0.0

        placed_ratio = placed_count / total_count

        used_volume = sum(
            p.length_mm * p.width_mm * p.height_mm for p in self.placements
        )

        pallet_volume = (
            self.pallet_length_mm * self.pallet_width_mm * self.max_height_mm
        )

        volume_utilisation = used_volume / pallet_volume

        max_height_used = float(np.max(self.heightmap))
        height_ratio = max_height_used / self.max_height_mm

        # Big goals:
        # - place more boxes
        # - improve volume utilisation
        # - avoid unnecessarily tall stacks
        return (
            20.0 * placed_ratio
            + 50.0 * volume_utilisation
            - 3.0 * height_ratio
        )

    def _get_obs(self) -> np.ndarray:
        height_obs = (self.heightmap / self.max_height_mm).flatten()

        if self.current_index < len(self.boxes):
            box = self.boxes[self.current_index]
            box_obs = np.array(
                [
                    box.length_mm / self.pallet_length_mm,
                    box.width_mm / self.pallet_width_mm,
                    box.height_mm / self.max_height_mm,
                    box.weight_kg / self.max_weight_kg,
                ],
                dtype=np.float32,
            )
        else:
            box_obs = np.zeros(4, dtype=np.float32)

        remaining_ratio = np.array(
            [
                1.0 - (self.current_index / max(1, len(self.boxes))),
                self.total_weight / self.max_weight_kg,
            ],
            dtype=np.float32,
        )

        return np.concatenate([height_obs, box_obs, remaining_ratio]).astype(
            np.float32
        )

    def _info(self) -> Dict:
        used_volume = sum(
            p.length_mm * p.width_mm * p.height_mm for p in self.placements
        )

        pallet_volume = (
            self.pallet_length_mm * self.pallet_width_mm * self.max_height_mm
        )

        return {
            "placed_boxes": len(self.placements),
            "total_boxes": len(self.boxes),
            "volume_utilisation": used_volume / pallet_volume,
            "weight_used_kg": self.total_weight,
            "max_height_used_mm": float(np.max(self.heightmap)),
            "placements": [p.__dict__ for p in self.placements],
        }


def load_boxes_from_csv(path: str) -> List[Box]:
    df = pd.read_csv(path)

    rename_map = {
        "Product Code": "sku",
        "SKU": "sku",
        "sku": "sku",
        "Amount": "quantity",
        "quantity": "quantity",
        "Length cm": "length_cm",
        "Width cm": "width_cm",
        "Height cm": "height_cm",
        "length_mm": "length_mm",
        "width_mm": "width_mm",
        "height_mm": "height_mm",
        "weight_kg": "weight_kg",
        "Weight kg": "weight_kg",
    }

    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

    required_any = [
        {"sku", "quantity", "length_mm", "width_mm", "height_mm", "weight_kg"},
        {"sku", "quantity", "length_cm", "width_cm", "height_cm", "weight_kg"},
    ]

    cols = set(df.columns)

    if not any(req.issubset(cols) for req in required_any):
        raise ValueError(
            "CSV must contain either "
            "sku, quantity, length_mm, width_mm, height_mm, weight_kg "
            "or sku, quantity, length_cm, width_cm, height_cm, weight_kg"
        )

    boxes: List[Box] = []

    for _, row in df.iterrows():
        qty = int(row["quantity"])

        if "length_mm" in df.columns:
            length_mm = float(row["length_mm"])
            width_mm = float(row["width_mm"])
            height_mm = float(row["height_mm"])
        else:
            length_mm = float(row["length_cm"]) * 10.0
            width_mm = float(row["width_cm"]) * 10.0
            height_mm = float(row["height_cm"]) * 10.0

        weight_kg = float(row["weight_kg"])
        sku = str(row["sku"])

        for _ in range(qty):
            boxes.append(
                Box(
                    sku=sku,
                    length_mm=length_mm,
                    width_mm=width_mm,
                    height_mm=height_mm,
                    weight_kg=weight_kg,
                )
            )

    return boxes


def train(args):
    boxes = load_boxes_from_csv(args.csv)

    env = PalletPackingEnv(
        boxes=boxes,
        pallet_length_mm=args.pallet_length,
        pallet_width_mm=args.pallet_width,
        max_height_mm=args.max_height,
        max_weight_kg=args.max_weight,
        grid_mm=args.grid,
        max_boxes_per_episode=args.max_boxes,
        shuffle_boxes=True,
    )

    env = Monitor(env)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,
        gamma=0.98,
        train_freq=4,
        target_update_interval=2_000,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log=args.tensorboard_log,
    )

    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    os.makedirs(args.output_dir, exist_ok=True)

    model_path = os.path.join(args.output_dir, "dqn_pallet_model")
    model.save(model_path)

    print(f"Saved model to: {model_path}.zip")

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f"Eval mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")

    export_solution(model, boxes, args)


def export_solution(model, boxes: List[Box], args):
    env = PalletPackingEnv(
        boxes=boxes,
        pallet_length_mm=args.pallet_length,
        pallet_width_mm=args.pallet_width,
        max_height_mm=args.max_height,
        max_weight_kg=args.max_weight,
        grid_mm=args.grid,
        max_boxes_per_episode=args.max_boxes,
        shuffle_boxes=False,
    )

    obs, _ = env.reset()
    done = False
    info = {}

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

    placements = info.get("placements", [])

    df = pd.DataFrame(placements)

    output_csv = os.path.join(args.output_dir, "dqn_solution.csv")
    df.to_csv(output_csv, index=False)

    print(f"Exported DQN layout to: {output_csv}")
    print(f"Placed boxes: {info.get('placed_boxes')} / {info.get('total_boxes')}")
    print(f"Volume utilisation: {info.get('volume_utilisation'):.4f}")
    print(f"Max height used: {info.get('max_height_used_mm'):.1f} mm")
    print(f"Weight used: {info.get('weight_used_kg'):.2f} kg")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", default="runs/dqn_pallet")
    parser.add_argument("--tensorboard-log", default="runs/tensorboard")

    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)

    parser.add_argument("--pallet-length", type=int, default=1200)
    parser.add_argument("--pallet-width", type=int, default=1000)
    parser.add_argument("--max-height", type=int, default=1150)
    parser.add_argument("--max-weight", type=float, default=1500.0)

    parser.add_argument("--grid", type=int, default=50)
    parser.add_argument("--max-boxes", type=int, default=240)

    args = parser.parse_args()
    if args.tensorboard_log == "":
        args.tensorboard_log = None

    train(args)


if __name__ == "__main__":
    main()
