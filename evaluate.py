"""
Evaluate the trained RL agent against two deterministic baselines
and render the pallet layouts as 3D plots.

Baselines
    First Fit Decreasing  --  picks the first valid (ep, rot) at each step.
    Extreme Point         --  picks the (ep, rot) with lowest score
                              z * 1000 + x + y,  matching psatops
                              ScoringUtils.scoreCandidate (no CG term).
                              This is the closest Python equivalent to the
                              psatops EXTREME_POINT algorithm.

Usage
    python evaluate.py --baseline_only
    python evaluate.py --model checkpoints/best_model.zip
"""

from __future__ import annotations

import argparse
import json
from itertools import product as iproduct
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pallet_env import (
    PalletPackingEnv, expand_items,
    get_rotation_dims, support_fraction,
    NUM_ROTATIONS, K_EP, SUPPORT_THRESHOLD,
)


# ── 3D drawing helper ──────────────────────────────────────────────────────────
def _draw_box(ax, x, y, z, dx, dy, dz, color):
    verts = list(iproduct([x, x + dx], [y, y + dy], [z, z + dz]))
    faces = [
        [0, 1, 3, 2], [4, 5, 7, 6],
        [0, 1, 5, 4], [2, 3, 7, 6],
        [0, 2, 6, 4], [1, 3, 7, 5],
    ]
    poly = [[verts[i] for i in face] for face in faces]
    ax.add_collection3d(
        Poly3DCollection(poly, facecolors=color, edgecolors="black",
                         linewidths=0.25, alpha=0.85)
    )


def visualize(env: PalletPackingEnv, title: str, output_path: str) -> None:
    used = [p for p in env.pallets if p["placements"]]
    if not used:
        print("No placements to draw.")
        return

    n = len(used)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    fig.suptitle(title, fontsize=13)

    cmap = plt.cm.tab20
    sku_colors: dict = {}

    for i, pallet in enumerate(used):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        for pl in pallet["placements"]:
            base_sku = pl["item"].sku.split("#")[0]
            if base_sku not in sku_colors:
                sku_colors[base_sku] = cmap(len(sku_colors) % 20)
            _draw_box(
                ax,
                pl["x_mm"], pl["y_mm"], pl["z_mm"],
                pl["l_mm"], pl["w_mm"], pl["h_mm"],
                sku_colors[base_sku],
            )
        ax.set_xlim(0, env.pallet_length)
        ax.set_ylim(0, env.pallet_width)
        ax.set_zlim(0, env.pallet_height)
        ax.set_xlabel("Length (mm)")
        ax.set_ylabel("Width (mm)")
        ax.set_zlabel("Height (mm)")
        util   = pallet["used_volume"] / env.pallet_volume * 100
        weight = pallet["used_weight"]
        n_items = len(pallet["placements"])
        ax.set_title(
            f"Pallet {i + 1}: {n_items} items, {util:.1f}% util, {weight:.1f} kg"
        )

    handles = [
        Patch(facecolor=c, edgecolor="black", label=s)
        for s, c in sku_colors.items()
    ]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=min(len(handles), 6),
        bbox_to_anchor=(0.5, -0.01),
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


# ── Stats reporter ─────────────────────────────────────────────────────────────
def report(env: PalletPackingEnv, label: str) -> None:
    used = [p for p in env.pallets if p["placements"]]
    n_used = len(used)
    total_vol = sum(p["used_volume"] for p in used)
    capacity  = n_used * env.pallet_volume
    util      = total_vol / capacity * 100 if capacity > 0 else 0.0
    placed    = sum(len(p["placements"]) for p in used)

    # Support fraction statistics across all placed boxes
    sf_values = []
    for p in used:
        for pl in p["placements"]:
            sf = support_fraction(
                p["boxes"],
                pl["x_mm"], pl["y_mm"], pl["z_mm"],
                pl["l_mm"], pl["w_mm"],
            )
            sf_values.append(sf)
    avg_sf = np.mean(sf_values) * 100 if sf_values else 0.0

    print()
    print(f"=== {label} ===")
    print(f"Pallets used       {n_used}")
    print(f"Items placed       {placed} / {env.total_items}")
    print(f"Avg utilisation    {util:.2f}%")
    print(f"Avg support frac   {avg_sf:.1f}%  (threshold {SUPPORT_THRESHOLD*100:.0f}%)")
    for i, p in enumerate(used):
        pu = p["used_volume"] / env.pallet_volume * 100
        rots_used = {pl["rotation_label"] for pl in p["placements"]}
        print(f"  Pallet {i + 1}: {len(p['placements'])} items, "
              f"util {pu:5.1f}%, weight {p['used_weight']:.2f} kg, "
              f"rotations {sorted(rots_used)}")


# ── Baselines ──────────────────────────────────────────────────────────────────
def run_first_fit(items, env_kwargs) -> PalletPackingEnv:
    """Pick the first valid (ep, rot) action at each step."""
    env = PalletPackingEnv(items, **env_kwargs, sort_items=True)
    env.reset()
    while env.current_idx < env.total_items:
        masks = env.action_masks()
        valid = np.where(masks)[0]
        if valid.size == 0:
            break
        env.step(int(valid[0]))
    return env


def run_extreme_point(items, env_kwargs) -> PalletPackingEnv:
    """
    Greedy psatops-style Extreme Point:
    pick (ep, rot) with lowest score  z * 1000 + x + y.
    Matching ScoringUtils.scoreCandidate dominant term (no CG, no base preference).
    """
    env = PalletPackingEnv(items, **env_kwargs, sort_items=True)
    env.reset()

    while env.current_idx < env.total_items:
        masks = env.action_masks()
        valid = np.where(masks)[0]
        if valid.size == 0:
            break

        item   = env._current_item()
        pallet = env.pallets[-1]

        best_action = int(valid[0])
        best_score  = float("inf")

        for action in valid:
            ep_idx = int(action) // NUM_ROTATIONS
            rot    = int(action) % NUM_ROTATIONS
            result = env._check_ep(pallet, item, ep_idx, rot)
            if result is None:
                continue
            x, y, z, fp_l, fp_w, stack_h = result
            # psatops ScoringUtils.scoreCandidate: z*1000 + x + y
            score = z * 1000.0 + x + y
            if score < best_score:
                best_score  = score
                best_action = int(action)

        env.step(best_action)
    return env


# ── RL agent runner ────────────────────────────────────────────────────────────
def run_agent(model, items, env_kwargs) -> PalletPackingEnv:
    from sb3_contrib.common.wrappers import ActionMasker

    base_env = PalletPackingEnv(items, **env_kwargs, sort_items=True)
    wrapped  = ActionMasker(base_env, lambda e: e.action_masks())
    obs, _   = wrapped.reset()
    done = False
    while not done:
        masks = wrapped.action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, _, terminated, truncated, _ = wrapped.step(int(action))
        done = terminated or truncated
    return base_env


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",          default="data.json")
    parser.add_argument("--model",         default="./checkpoints/best_model.zip")
    parser.add_argument("--baseline_only", action="store_true")
    parser.add_argument("--pallet_length", type=float, default=1200.0)
    parser.add_argument("--pallet_width",  type=float, default=1100.0)
    parser.add_argument("--pallet_height", type=float, default=1150.0)
    parser.add_argument("--max_weight",    type=float, default=1500.0)
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)
    items = expand_items(data)

    env_kwargs = dict(
        pallet_length=args.pallet_length,
        pallet_width=args.pallet_width,
        pallet_height=args.pallet_height,
        max_pallet_weight=args.max_weight,
    )

    total_vol  = sum(it.length * it.width * it.height for it in items) / 1e9
    pallet_vol = args.pallet_length * args.pallet_width * args.pallet_height / 1e9
    print(f"Loaded {len(items)} items from {args.data}")
    print(f"Total item volume   {total_vol:.3f} m^3")
    print(f"Pallet capacity     {pallet_vol:.3f} m^3")
    print(f"Theoretical min     {int(np.ceil(total_vol / pallet_vol))} pallets")

    # First Fit Decreasing baseline
    ff_env = run_first_fit(items, env_kwargs)
    report(ff_env, "First Fit Decreasing")
    visualize(ff_env, "First Fit Decreasing", "ff_result.png")

    # Extreme Point baseline (psatops-style)
    ep_env = run_extreme_point(items, env_kwargs)
    report(ep_env, "Extreme Point (psatops-style)")
    visualize(ep_env, "Extreme Point (psatops-style)", "ep_result.png")

    if args.baseline_only:
        return

    if not Path(args.model).exists():
        print(f"\nModel not found at {args.model}.")
        print("Run train.py first, or pass --baseline_only.")
        return

    from sb3_contrib import MaskablePPO
    model     = MaskablePPO.load(args.model)
    agent_env = run_agent(model, items, env_kwargs)
    report(agent_env, "MaskablePPO RL agent")
    visualize(agent_env, "MaskablePPO RL agent", "agent_result.png")


if __name__ == "__main__":
    main()
