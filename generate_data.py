"""
Diverse training data generator for pallet packing RL.

Generates 200 randomised orders drawn from a realistic warehouse SKU catalogue.
Each order varies in:
  - Number of active SKUs (2 to 7)
  - Quantity per SKU (5 to 60)
  - Box dimensions (realistic spread covering thin panels, large cartons, cubes,
    tall narrow items, wide flat sheets)
  - Density (light vs heavy items)

The generator also includes the original PSA TOPS Case 1 and Case 2 data so the
agent trains on real observed orders as well as synthetic ones.

Output: data/orders/order_0000.json ... order_0199.json
        data/orders/catalog.json  (full SKU reference)

Usage:
    python generate_data.py                     # 200 orders (default)
    python generate_data.py --n_orders 500      # more variety
    python generate_data.py --seed 99           # reproducible
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

CATALOG = [
    {"sku": "FP-XS", "l": 300, "w": 15,  "h": 180, "wt": 0.08,  "family": "flat_panel"},
    {"sku": "FP-S",  "l": 380, "w": 20,  "h": 220, "wt": 0.12,  "family": "flat_panel"},
    {"sku": "FP-M",  "l": 430, "w": 22,  "h": 250, "wt": 0.165, "family": "flat_panel"},
    {"sku": "FP-L",  "l": 500, "w": 25,  "h": 290, "wt": 0.22,  "family": "flat_panel"},
    {"sku": "FP-XL", "l": 580, "w": 28,  "h": 330, "wt": 0.30,  "family": "flat_panel"},
    {"sku": "RD-S",  "l": 340, "w": 12,  "h": 200, "wt": 0.09,  "family": "rod"},
    {"sku": "RD-M",  "l": 410, "w": 14,  "h": 240, "wt": 0.131, "family": "rod"},
    {"sku": "RD-L",  "l": 500, "w": 18,  "h": 270, "wt": 0.18,  "family": "rod"},
    {"sku": "SH-S",  "l": 330, "w": 460, "h":  8,  "wt": 0.07,  "family": "sheet"},
    {"sku": "SH-M",  "l": 400, "w": 540, "h": 10,  "wt": 0.131, "family": "sheet"},
    {"sku": "SH-L",  "l": 480, "w": 620, "h": 12,  "wt": 0.20,  "family": "sheet"},
    {"sku": "CB-S",  "l": 300, "w": 380, "h": 180, "wt": 0.40,  "family": "carton"},
    {"sku": "CB-M",  "l": 370, "w": 520, "h": 230, "wt": 0.70,  "family": "carton"},
    {"sku": "CB-L",  "l": 420, "w": 560, "h": 260, "wt": 1.00,  "family": "carton"},
    {"sku": "CB-XL", "l": 480, "w": 600, "h": 290, "wt": 1.40,  "family": "carton"},
    {"sku": "CU-S",  "l": 120, "w": 120, "h": 120, "wt": 0.25,  "family": "cube"},
    {"sku": "CU-M",  "l": 200, "w": 200, "h": 200, "wt": 0.80,  "family": "cube"},
    {"sku": "CU-L",  "l": 280, "w": 280, "h": 280, "wt": 1.80,  "family": "cube"},
    {"sku": "TN-S",  "l": 180, "w": 160, "h": 440, "wt": 0.60,  "family": "tall"},
    {"sku": "TN-M",  "l": 240, "w": 220, "h": 560, "wt": 1.10,  "family": "tall"},
    {"sku": "TN-L",  "l": 300, "w": 260, "h": 650, "wt": 1.80,  "family": "tall"},
    {"sku": "WF-S",  "l": 550, "w": 460, "h":  80, "wt": 0.50,  "family": "wide_flat"},
    {"sku": "WF-M",  "l": 700, "w": 550, "h": 120, "wt": 0.90,  "family": "wide_flat"},
    {"sku": "WF-L",  "l": 850, "w": 650, "h": 150, "wt": 1.50,  "family": "wide_flat"},
    {"sku": "SM-A",  "l":  90, "w":  70, "h":  50, "wt": 0.04,  "family": "small"},
    {"sku": "SM-B",  "l": 130, "w": 100, "h":  80, "wt": 0.08,  "family": "small"},
    {"sku": "SM-C",  "l": 160, "w": 130, "h": 110, "wt": 0.15,  "family": "small"},
    {"sku": "HV-S",  "l": 250, "w": 200, "h": 180, "wt": 3.0,   "family": "heavy"},
    {"sku": "HV-M",  "l": 350, "w": 280, "h": 240, "wt": 6.0,   "family": "heavy"},
]

CASE1 = {
    "order_id": "TOPS-CASE1",
    "items": [
        {"sku": "C-BTA073V650G3ECE", "quantity": 67, "length_mm": 430,  "width_mm": 21.6,  "height_mm": 250.0, "weight_kg": 0.165},
        {"sku": "A-DIA-ELH13-GCE",   "quantity":  2, "length_mm": 370,  "width_mm": 520.0, "height_mm": 230.0, "weight_kg": 0.060},
        {"sku": "A-DIA-ELH15-GCE",   "quantity": 22, "length_mm": 380,  "width_mm": 520.0, "height_mm": 230.0, "weight_kg": 0.131},
        {"sku": "G-DIA-ELH17-GCE",   "quantity": 18, "length_mm": 400,  "width_mm": 540.0, "height_mm": 230.0, "weight_kg": 0.131},
        {"sku": "A-DIA-ELH19-GCE",   "quantity":  5, "length_mm": 400,  "width_mm": 540.0, "height_mm":   9.6, "weight_kg": 0.131},
        {"sku": "A-DIA-ELH21-GCE",   "quantity":  3, "length_mm": 410,  "width_mm":  14.1, "height_mm": 240.0, "weight_kg": 0.131},
    ],
}
CASE2 = {
    "order_id": "TOPS-CASE2",
    "items": [
        {"sku": "C-BTA073V650G3ECE", "quantity": 26, "length_mm": 430,  "width_mm": 21.6,  "height_mm": 250.0, "weight_kg": 0.165},
        {"sku": "G-DIA-ELH15-GCE",   "quantity":  5, "length_mm": 370,  "width_mm": 520.0, "height_mm": 230.0, "weight_kg": 0.006},
        {"sku": "G-DIA-ELH17-GCE",   "quantity":  7, "length_mm": 400,  "width_mm": 540.0, "height_mm": 230.0, "weight_kg": 0.131},
        {"sku": "G-DIA-ELH19-GCE",   "quantity":  4, "length_mm": 400,  "width_mm": 540.0, "height_mm":   9.6, "weight_kg": 0.131},
        {"sku": "G-DIA-ELH21-GCE",   "quantity":  6, "length_mm": 410,  "width_mm":  14.1, "height_mm": 240.0, "weight_kg": 0.131},
    ],
}

PALLET_VOL_MM3 = 1200 * 1100 * 1150
MAX_PALLETS    = 4


def _sku_to_entry(sku: dict, qty: int) -> dict:
    return {
        "sku":        sku["sku"],
        "quantity":   qty,
        "length_mm":  sku["l"],
        "width_mm":   sku["w"],
        "height_mm":  sku["h"],
        "weight_kg":  sku["wt"],
    }


def _item_vol(entry: dict) -> float:
    return entry["length_mm"] * entry["width_mm"] * entry["height_mm"]


def generate_order(
    rng: random.Random,
    n_skus: int | None = None,
    profile: str = "mixed",
) -> dict:
    if profile == "uniform":
        family = rng.choice(list({s["family"] for s in CATALOG}))
        pool = [s for s in CATALOG if s["family"] == family]
    elif profile == "hard":
        families = rng.sample(["flat_panel", "carton", "sheet", "tall"], k=2)
        pool = [s for s in CATALOG if s["family"] in families]
    elif profile == "small":
        pool = [s for s in CATALOG if s["family"] in ("small", "cube", "flat_panel")]
    else:
        pool = CATALOG

    k = n_skus if n_skus else rng.randint(3, 7)
    k = min(k, len(pool))
    selected = rng.sample(pool, k)

    items = []
    total_vol = 0.0
    max_vol = MAX_PALLETS * PALLET_VOL_MM3

    for sku in selected:
        item_vol = sku["l"] * sku["w"] * sku["h"]
        max_qty = max(2, int((max_vol / k) / item_vol))
        max_qty = min(max_qty, 60)
        qty = rng.randint(2, max(2, max_qty))

        added_vol = item_vol * qty
        if total_vol + added_vol > max_vol:
            qty = max(1, int((max_vol - total_vol) / item_vol))
            if qty == 0:
                continue

        items.append(_sku_to_entry(sku, qty))
        total_vol += item_vol * qty

    if not items:
        sku = next(s for s in CATALOG if s["sku"] == "CB-M")
        items.append(_sku_to_entry(sku, 10))

    return {"items": items}


def generate_dataset(n_orders: int = 200, seed: int = 42) -> list:
    rng = random.Random(seed)
    orders = []

    orders.append(CASE1)
    orders.append(CASE2)

    for _ in range(8):
        variant = {"items": [
            dict(e, quantity=max(1, round(e["quantity"] * rng.uniform(0.4, 1.8))))
            for e in CASE1["items"]
        ]}
        orders.append(variant)

    profiles = ["mixed", "mixed", "mixed", "uniform", "hard", "small"]
    for i in range(n_orders - len(orders)):
        profile = rng.choice(profiles)
        orders.append(generate_order(rng, profile=profile))

    rng.shuffle(orders)
    return orders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_orders", type=int, default=200)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--out_dir",  default="data/orders")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    orders = generate_dataset(args.n_orders, args.seed)

    for i, order in enumerate(orders):
        path = os.path.join(args.out_dir, f"order_{i:04d}.json")
        with open(path, "w") as f:
            json.dump(order, f, indent=2)

    with open(os.path.join(args.out_dir, "catalog.json"), "w") as f:
        json.dump(CATALOG, f, indent=2)

    total_items = sum(
        sum(e["quantity"] for e in o["items"])
        for o in orders
    )
    avg_items = total_items / len(orders)
    skus_per_order = [len(o["items"]) for o in orders]
    print(f"Generated {len(orders)} orders in {args.out_dir}/")
    print(f"  Avg items per order : {avg_items:.1f}")
    print(f"  Avg SKUs per order  : {sum(skus_per_order)/len(skus_per_order):.1f}")
    print(f"  SKU range           : {min(skus_per_order)} to {max(skus_per_order)}")

    all_skus = {e["sku"] for o in orders for e in o["items"]}
    print(f"  Unique SKUs used    : {len(all_skus)}")


if __name__ == "__main__":
    main()
