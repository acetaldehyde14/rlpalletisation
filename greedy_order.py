"""
Greedy ordering heuristic for pallet packing with EP placement.

At each step, evaluates all remaining items, simulates placing each one
with EP, and picks the item that leads to the best state.

Also tries multiple static orderings (volume-desc, height-desc, etc.)
and reports the best result.

Usage:
    python greedy_order.py --data data.json
"""

from __future__ import annotations

import argparse
import json
from typing import List

from pallet_env import (
    Item, expand_items, get_rotation_dims, ROTATION_LABELS,
    PlacedBox,
)
from evaluate import _new_result_pallet, _generate_eps, _can_place, _place_box_ep, report


def run_ep_with_order(items: List[Item], env_kwargs, order_indices: List[int],
                      num_rotations=6):
    pallet_length = env_kwargs["pallet_length"]
    pallet_width = env_kwargs["pallet_width"]
    pallet_height = env_kwargs["pallet_height"]
    max_weight = env_kwargs["max_pallet_weight"]

    ordered = [items[i] for i in order_indices]
    pallets = [_new_result_pallet()]
    skipped = 0

    for item in ordered:
        placed = False
        for pallet in pallets:
            if _place_box_ep(pallet, item, pallet_length, pallet_width,
                             pallet_height, max_weight, num_rotations):
                placed = True
                break
        if not placed:
            pallets.append(_new_result_pallet())
            if not _place_box_ep(pallets[-1], item, pallet_length, pallet_width,
                                  pallet_height, max_weight, num_rotations):
                skipped += 1

    from pallet_env import PalletPackingEnv
    env = PalletPackingEnv(items, **env_kwargs, sort_items=False)
    env.reset()
    used = [p for p in pallets if p["placements"]]
    env.pallets = pallets
    return env


def greedy_lookahead_order(items: List[Item], env_kwargs, num_rotations=6,
                           n_lookahead=1):
    pallet_length = env_kwargs["pallet_length"]
    pallet_width = env_kwargs["pallet_width"]
    pallet_height = env_kwargs["pallet_height"]
    max_weight = env_kwargs["max_pallet_weight"]

    n = len(items)
    placed = [False] * n
    order = []
    pallets = [_new_result_pallet()]

    for step in range(n):
        best_idx = -1
        best_score = float("inf")

        remaining = [i for i in range(n) if not placed[i]]
        if not remaining:
            break

        for idx in remaining:
            sim_pallets = []
            for p in pallets:
                sp = _new_result_pallet()
                sp["boxes"] = list(p["boxes"])
                sp["placements"] = list(p["placements"])
                sp["used_volume"] = p["used_volume"]
                sp["used_weight"] = p["used_weight"]
                sim_pallets.append(sp)

            placed_ok = False
            for pallet in sim_pallets:
                if _place_box_ep(pallet, items[idx], pallet_length, pallet_width,
                                 pallet_height, max_weight, num_rotations):
                    placed_ok = True
                    break
            if not placed_ok:
                sim_pallets.append(_new_result_pallet())
                _place_box_ep(sim_pallets[-1], items[idx], pallet_length,
                              pallet_width, pallet_height, max_weight, num_rotations)

            n_pallets = len(sim_pallets)
            total_vol = sum(p["used_volume"] for p in sim_pallets)
            capacity = n_pallets * pallet_length * pallet_width * pallet_height
            util = total_vol / capacity if capacity > 0 else 0
            score = -util * 1000 + n_pallets * 100

            if score < best_score:
                best_score = score
                best_idx = idx

        placed[best_idx] = True
        order.append(best_idx)

        placed_ok = False
        for pallet in pallets:
            if _place_box_ep(pallet, items[best_idx], pallet_length, pallet_width,
                             pallet_height, max_weight, num_rotations):
                placed_ok = True
                break
        if not placed_ok:
            pallets.append(_new_result_pallet())
            _place_box_ep(pallets[-1], items[best_idx], pallet_length,
                          pallet_width, pallet_height, max_weight, num_rotations)

        if (step + 1) % 20 == 0:
            n_p = len([p for p in pallets if p["placements"]])
            print(f"  Step {step+1}/{n}: {n_p} pallets so far")

    return order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.json")
    parser.add_argument("--rotations", type=int, default=6)
    parser.add_argument("--pallet_length", type=float, default=1200.0)
    parser.add_argument("--pallet_width", type=float, default=1100.0)
    parser.add_argument("--pallet_height", type=float, default=1150.0)
    parser.add_argument("--max_weight", type=float, default=1500.0)
    parser.add_argument("--greedy", action="store_true",
                        help="Run greedy lookahead (slow)")
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)
    items = expand_items(data)

    env_kwargs = dict(
        pallet_length=args.pallet_length,
        pallet_width=args.pallet_width,
        pallet_height=args.pallet_height,
        max_pallet_weight=args.max_weight,
        num_rotations=args.rotations,
    )

    import numpy as np

    static_orderings = {
        "Volume descending": sorted(range(len(items)),
                                     key=lambda i: -(items[i].length * items[i].width * items[i].height)),
        "Volume ascending": sorted(range(len(items)),
                                    key=lambda i: items[i].length * items[i].width * items[i].height),
        "Height descending": sorted(range(len(items)),
                                     key=lambda i: -items[i].height),
        "Base area descending": sorted(range(len(items)),
                                        key=lambda i: -(items[i].length * items[i].width)),
        "Weight descending": sorted(range(len(items)),
                                     key=lambda i: -items[i].weight),
        "Max dimension descending": sorted(range(len(items)),
                                            key=lambda i: -max(items[i].length, items[i].width, items[i].height)),
    }

    print(f"Items: {len(items)}")
    print()

    best_n = float("inf")
    best_name = ""
    for name, order in static_orderings.items():
        env = run_ep_with_order(items, env_kwargs, order, args.rotations)
        n_placed = sum(len(p["placements"]) for p in env.pallets if p["placements"])
        n_pallets = len([p for p in env.pallets if p["placements"]])
        total_vol = sum(p["used_volume"] for p in env.pallets if p["placements"])
        capacity = n_pallets * env.pallet_volume
        util = total_vol / capacity * 100 if capacity > 0 else 0
        print(f"{name}: {n_pallets} pallets, {util:.1f}% util, {n_placed} placed")
        if n_pallets < best_n:
            best_n = n_pallets
            best_name = name

    print(f"\nBest static ordering: {best_name} ({best_n} pallets)")

    for _ in range(5):
        rng = np.random.RandomState(_)
        order = list(range(len(items)))
        rng.shuffle(order)
        env = run_ep_with_order(items, env_kwargs, order, args.rotations)
        n_placed = sum(len(p["placements"]) for p in env.pallets if p["placements"])
        n_pallets = len([p for p in env.pallets if p["placements"]])
        total_vol = sum(p["used_volume"] for p in env.pallets if p["placements"])
        capacity = n_pallets * env.pallet_volume
        util = total_vol / capacity * 100 if capacity > 0 else 0
        print(f"Random {_}: {n_pallets} pallets, {util:.1f}% util, {n_placed} placed")

    if args.greedy:
        print("\nRunning greedy lookahead...")
        order = greedy_lookahead_order(items, env_kwargs, args.rotations)
        env = run_ep_with_order(items, env_kwargs, order, args.rotations)
        report(env, "Greedy Lookahead + EP")


if __name__ == "__main__":
    main()
