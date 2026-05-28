"""
Search-based pallet packing agent.

The agent keeps the proven Extreme Point placer for geometry and searches over
item orderings. This is usually a better use of compute than asking a plain RL
policy to relearn collision, support, and rotation rules from scratch.

Usage:
    python packing_agent.py --data data.json
    python packing_agent.py --data data/orders/order_0000.json --trials 300
    python packing_agent.py --benchmark data/orders --limit 50 --trials 120
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np

from evaluate import (
    _can_place,
    _generate_eps,
    _new_result_pallet,
    _place_box_ep,
    report,
    visualize,
)
from pallet_env import Item, PalletPackingEnv, PlacedBox, ROTATION_LABELS, expand_items, get_rotation_dims


@dataclass
class PackResult:
    name: str
    pallets: list
    n_items: int
    placed: int
    n_pallets: int
    util: float
    min_util: float
    score: tuple


def _pack_order(items: List[Item], env_kwargs: dict, order: List[int],
                name: str, num_rotations: int) -> PackResult:
    pallets = [_new_result_pallet()]
    skipped = 0

    for idx in order:
        item = items[idx]
        placed = False
        for pallet in pallets:
            if _place_box_ep(
                pallet, item,
                env_kwargs["pallet_length"],
                env_kwargs["pallet_width"],
                env_kwargs["pallet_height"],
                env_kwargs["max_pallet_weight"],
                num_rotations,
            ):
                placed = True
                break

        if not placed:
            pallets.append(_new_result_pallet())
            if not _place_box_ep(
                pallets[-1], item,
                env_kwargs["pallet_length"],
                env_kwargs["pallet_width"],
                env_kwargs["pallet_height"],
                env_kwargs["max_pallet_weight"],
                num_rotations,
            ):
                skipped += 1

    used = [p for p in pallets if p["placements"]]
    n_pallets = len(used)
    placed = sum(len(p["placements"]) for p in used)
    pallet_volume = (
        env_kwargs["pallet_length"] *
        env_kwargs["pallet_width"] *
        env_kwargs["pallet_height"]
    )
    total_vol = sum(p["used_volume"] for p in used)
    util = total_vol / max(n_pallets * pallet_volume, 1.0)
    min_util = min((p["used_volume"] / pallet_volume for p in used), default=0.0)
    max_height = 0.0
    for p in used:
        for b in p["boxes"]:
            max_height = max(max_height, b.z + b.h)

    score = (
        skipped,
        n_pallets,
        -placed,
        -util,
        -min_util,
        max_height / max(env_kwargs["pallet_height"], 1.0),
    )
    return PackResult(name, pallets, len(items), placed, n_pallets, util, min_util, score)


def _env_from_result(items: List[Item], env_kwargs: dict, result: PackResult) -> PalletPackingEnv:
    env = PalletPackingEnv(items, **env_kwargs, sort_items=False)
    env.reset()
    env.pallets = result.pallets
    return env


def _clone_pallet(pallet: dict) -> dict:
    return {
        "boxes": list(pallet["boxes"]),
        "placements": list(pallet["placements"]),
        "used_volume": pallet["used_volume"],
        "used_weight": pallet["used_weight"],
    }


def _candidate_placements(pallet: dict, item: Item, env_kwargs: dict,
                          num_rotations: int, per_pallet: int) -> list[tuple]:
    out = []
    points = _generate_eps(
        env_kwargs["pallet_length"],
        env_kwargs["pallet_width"],
        env_kwargs["pallet_height"],
        pallet["boxes"],
    )
    for x, y, z in points:
        for rot in range(num_rotations):
            l, w, h = get_rotation_dims(item, rot)
            if not _can_place(
                pallet, x, y, z, l, w, h, item.weight,
                env_kwargs["pallet_length"],
                env_kwargs["pallet_width"],
                env_kwargs["pallet_height"],
                env_kwargs["max_pallet_weight"],
            ):
                continue
            wall_touch = int(abs(x) < 1.0) + int(abs(y) < 1.0)
            base_area = l * w
            score = (
                z * 1000.0 + x + y,
                -base_area,
                h,
                -wall_touch,
            )
            out.append((score, x, y, z, l, w, h, rot))
    out.sort(key=lambda c: c[0])
    return out[:per_pallet]


def _place_candidate(pallet: dict, item: Item, cand: tuple) -> dict:
    _, x, y, z, l, w, h, rot = cand
    new_pallet = _clone_pallet(pallet)
    box = PlacedBox(
        x=x, y=y, z=z, l=l, w=w, h=h,
        weight=item.weight,
        sku=item.sku,
        rotation=rot,
        rotation_label=ROTATION_LABELS[rot],
    )
    new_pallet["boxes"].append(box)
    new_pallet["used_volume"] += item.length * item.width * item.height
    new_pallet["used_weight"] += item.weight
    new_pallet["placements"].append({
        "item": item,
        "x_mm": x, "y_mm": y, "z_mm": z,
        "l_mm": l, "w_mm": w, "h_mm": h,
        "rotation": rot,
        "rotation_label": ROTATION_LABELS[rot],
    })
    return new_pallet


def _state_score(pallets: list[dict], pallet_volume: float, placed: int,
                 total_items: int) -> tuple:
    used = [p for p in pallets if p["placements"]]
    n_pallets = len(used)
    utils = [p["used_volume"] / pallet_volume for p in used]
    min_util = min(utils, default=0.0)
    total_util = sum(utils) / max(n_pallets, 1)
    max_height = 0.0
    for p in used:
        for b in p["boxes"]:
            max_height = max(max_height, b.z + b.h)
    return (
        n_pallets,
        -(placed / max(total_items, 1)),
        -total_util,
        -min_util,
        max_height,
    )


def run_beam_agent(items: List[Item], env_kwargs: dict, beam_width: int = 24,
                   per_pallet: int = 5, num_rotations: int = 6) -> PackResult:
    order = _static_orders(items)["ep_volume_desc"]
    ordered = [items[i] for i in order]
    pallet_volume = (
        env_kwargs["pallet_length"] *
        env_kwargs["pallet_width"] *
        env_kwargs["pallet_height"]
    )
    beam: list[tuple[tuple, list[dict], int, int]] = [
        ((0, 0, 0, 0, 0), [_new_result_pallet()], 0, 0)
    ]

    for step, item in enumerate(ordered):
        next_beam = []
        for _, pallets, placed, skipped in beam:
            expanded = False
            for p_idx, pallet in enumerate(pallets):
                for cand in _candidate_placements(
                    pallet, item, env_kwargs, num_rotations, per_pallet
                ):
                    new_pallets = [_clone_pallet(p) for p in pallets]
                    new_pallets[p_idx] = _place_candidate(new_pallets[p_idx], item, cand)
                    score = _state_score(new_pallets, pallet_volume, placed + 1, len(items))
                    next_beam.append((score, new_pallets, placed + 1, skipped))
                    expanded = True

            new_pallet = _new_result_pallet()
            new_cands = _candidate_placements(
                new_pallet, item, env_kwargs, num_rotations, per_pallet
            )
            if new_cands:
                new_pallets = [_clone_pallet(p) for p in pallets]
                new_pallets.append(_place_candidate(new_pallet, item, new_cands[0]))
                score = _state_score(new_pallets, pallet_volume, placed + 1, len(items))
                next_beam.append((score, new_pallets, placed + 1, skipped))
            elif not expanded:
                score = _state_score(pallets, pallet_volume, placed, len(items))
                next_beam.append((score, [_clone_pallet(p) for p in pallets], placed, skipped + 1))

        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:beam_width]

    best_score, best_pallets, placed, skipped = min(beam, key=lambda x: x[0])
    used = [p for p in best_pallets if p["placements"]]
    n_pallets = len(used)
    total_vol = sum(p["used_volume"] for p in used)
    util = total_vol / max(n_pallets * pallet_volume, 1.0)
    min_util = min((p["used_volume"] / pallet_volume for p in used), default=0.0)
    score = (skipped, n_pallets, -placed, -util, -min_util, best_score[-1])
    return PackResult(
        f"beam_w{beam_width}_c{per_pallet}",
        best_pallets,
        len(items),
        placed,
        n_pallets,
        util,
        min_util,
        score,
    )


def _item_volume(item: Item) -> float:
    return item.length * item.width * item.height


def _base_area(item: Item) -> float:
    dims = sorted([item.length, item.width, item.height], reverse=True)
    return dims[0] * dims[1]


def _static_orders(items: List[Item]) -> dict[str, List[int]]:
    idxs = list(range(len(items)))
    return {
        "ep_volume_desc": sorted(idxs, key=lambda i: -_item_volume(items[i])),
        "volume_asc": sorted(idxs, key=lambda i: _item_volume(items[i])),
        "base_area_desc": sorted(idxs, key=lambda i: -_base_area(items[i])),
        "height_desc": sorted(idxs, key=lambda i: -items[i].height),
        "max_dim_desc": sorted(idxs, key=lambda i: -max(items[i].length, items[i].width, items[i].height)),
        "flat_first": sorted(idxs, key=lambda i: (items[i].height / max(items[i].length, items[i].width), -_base_area(items[i]))),
        "tall_first": sorted(idxs, key=lambda i: (-items[i].height / max(items[i].length, items[i].width), -_item_volume(items[i]))),
        "weight_desc": sorted(idxs, key=lambda i: -items[i].weight),
    }


def _score_order(items: List[Item], weights: tuple[float, ...], rng: random.Random,
                 noise: float) -> List[int]:
    scored = []
    for i, item in enumerate(items):
        dims = sorted([item.length, item.width, item.height], reverse=True)
        vol = _item_volume(item)
        features = (
            vol,
            dims[0] * dims[1],
            item.height,
            dims[0],
            item.weight,
            item.length / max(item.width, 1.0),
            item.height / max(dims[0], 1.0),
        )
        value = sum(w * f for w, f in zip(weights, features))
        value += rng.gauss(0.0, noise) * max(vol, 1.0)
        scored.append((value, i))
    scored.sort(reverse=True)
    return [i for _, i in scored]


def _mutate_order(order: List[int], rng: random.Random, strength: float) -> List[int]:
    out = list(order)
    n = len(out)
    if n < 2:
        return out

    n_ops = max(1, int(n * strength))
    for _ in range(n_ops):
        op = rng.randrange(3)
        a = rng.randrange(n)
        b = rng.randrange(n)
        if a == b:
            continue
        if op == 0:
            out[a], out[b] = out[b], out[a]
        elif op == 1:
            item = out.pop(a)
            out.insert(b, item)
        else:
            lo, hi = sorted((a, b))
            out[lo:hi] = reversed(out[lo:hi])
    return out


def run_search_agent(items: List[Item], env_kwargs: dict, trials: int = 200,
                     seed: int = 7, elite_size: int = 8,
                     num_rotations: int = 6) -> PackResult:
    rng = random.Random(seed)
    best: PackResult | None = None
    elites: list[tuple[tuple, List[int]]] = []

    def consider(name: str, order: List[int]) -> None:
        nonlocal best, elites
        result = _pack_order(items, env_kwargs, order, name, num_rotations)
        if best is None or result.score < best.score:
            best = result
        elites.append((result.score, order))
        elites.sort(key=lambda x: x[0])
        del elites[elite_size:]

    for name, order in _static_orders(items).items():
        consider(name, order)

    weight_bank = [
        (1, 0, 0, 0, 0, 0, 0),
        (0.2, 1, 0.2, 0, 0, 0, 0),
        (0.1, 0.2, 1, 0.4, 0, 0, 0),
        (0.1, 0.4, -0.8, 0.2, 0, 0, 0),
        (0.4, 0.2, 0.2, 0.1, 0, 0.5, 0),
        (0.4, 0.2, 0.2, 0.1, 0, -0.5, 0.5),
    ]

    for t in range(trials):
        if t < len(weight_bank) * 4:
            weights = weight_bank[t % len(weight_bank)]
            noise = 0.03 * (1 + t // len(weight_bank))
            order = _score_order(items, weights, rng, noise)
            name = f"weighted_{t:04d}"
        elif elites and rng.random() < 0.75:
            _, base = rng.choice(elites[:max(1, len(elites) // 2)])
            strength = rng.uniform(0.01, 0.12)
            order = _mutate_order(base, rng, strength)
            name = f"mutated_{t:04d}"
        else:
            order = list(range(len(items)))
            rng.shuffle(order)
            name = f"random_{t:04d}"
        consider(name, order)

    assert best is not None
    return best


def _load_items(path: str) -> List[Item]:
    with open(path) as f:
        return expand_items(json.load(f))


def _env_kwargs(args) -> dict:
    return {
        "pallet_length": args.pallet_length,
        "pallet_width": args.pallet_width,
        "pallet_height": args.pallet_height,
        "max_pallet_weight": args.max_weight,
        "num_rotations": args.rotations,
    }


def _run_one(path: str, args) -> tuple[PackResult, PackResult]:
    items = _load_items(path)
    env_kwargs = _env_kwargs(args)
    ep_order = _static_orders(items)["ep_volume_desc"]
    ep = _pack_order(items, env_kwargs, ep_order, "Extreme Point volume-desc", args.rotations)
    candidates = [ep]
    if args.agent in ("beam", "hybrid"):
        candidates.append(run_beam_agent(
            items, env_kwargs,
            beam_width=args.beam_width,
            per_pallet=args.per_pallet,
            num_rotations=args.rotations,
        ))
    if args.agent in ("order", "hybrid"):
        candidates.append(run_search_agent(
            items, env_kwargs,
            trials=args.trials,
            seed=args.seed,
            elite_size=args.elite_size,
            num_rotations=args.rotations,
        ))
    agent = min(candidates, key=lambda r: r.score)
    return ep, agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.json")
    parser.add_argument("--benchmark", default=None,
                        help="Directory of generated order JSON files")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--agent", choices=["order", "beam", "hybrid"], default="hybrid")
    parser.add_argument("--beam_width", type=int, default=24)
    parser.add_argument("--per_pallet", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--elite_size", type=int, default=8)
    parser.add_argument("--rotations", type=int, default=6)
    parser.add_argument("--pallet_length", type=float, default=1200.0)
    parser.add_argument("--pallet_width", type=float, default=1100.0)
    parser.add_argument("--pallet_height", type=float, default=1150.0)
    parser.add_argument("--max_weight", type=float, default=1500.0)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.benchmark:
        paths = [
            str(Path(args.benchmark) / fn)
            for fn in sorted(os.listdir(args.benchmark))
            if fn.endswith(".json") and fn != "catalog.json"
        ]
        if args.limit:
            paths = paths[:args.limit]

        wins = ties = losses = improved_layouts = 0
        pallet_delta = 0
        for path in paths:
            ep, agent = _run_one(path, args)
            delta = ep.n_pallets - agent.n_pallets
            pallet_delta += delta
            if delta > 0:
                wins += 1
            elif delta == 0:
                ties += 1
                if agent.score < ep.score:
                    improved_layouts += 1
            else:
                losses += 1
            print(
                f"{Path(path).name}: EP {ep.n_pallets} pallets/{ep.util*100:.1f}% "
                f"min {ep.min_util*100:.1f}% -> agent "
                f"{agent.n_pallets} pallets/{agent.util*100:.1f}% "
                f"min {agent.min_util*100:.1f}% "
                f"({agent.name})"
            )

        print()
        print(f"Benchmark orders : {len(paths)}")
        print(f"Agent wins/ties/losses vs EP: {wins}/{ties}/{losses}")
        print(f"Same-pallet layout improvements: {improved_layouts}")
        print(f"Net pallet reduction: {pallet_delta}")
        return

    ep, agent = _run_one(args.data, args)
    print(f"EP baseline : {ep.n_pallets} pallets, util {ep.util*100:.2f}%, placed {ep.placed}/{ep.n_items}")
    print(f"Agent best  : {agent.n_pallets} pallets, util {agent.util*100:.2f}%, placed {agent.placed}/{agent.n_items}")
    print(f"Min util    : EP {ep.min_util*100:.2f}% -> agent {agent.min_util*100:.2f}%")
    print(f"Best policy : {agent.name}")

    env_kwargs = _env_kwargs(args)
    items = _load_items(args.data)
    agent_env = _env_from_result(items, env_kwargs, agent)
    report(agent_env, "Search Agent + EP")
    if args.plot:
        visualize(agent_env, "Search Agent + EP", "search_agent_result.png")


if __name__ == "__main__":
    main()
