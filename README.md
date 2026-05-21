# Pallet Packing RL Agent  --  v2 (psatops-aligned)

A MaskablePPO reinforcement learning agent for 3D pallet packing that mirrors
the placement logic in the **psatops** Java backend.

---

## What changed from v1

| Aspect | v1 (grid sweep) | v2 (Extreme Point) |
|---|---|---|
| Action space | grid_l * grid_w * 6 ~ 19 800 | K_EP * 6 = **300** |
| Placement candidates | every grid cell | extreme points only |
| Placement coordinates | grid-snapped (20 mm) | **exact mm** |
| Support check | flat-surface only | **70% footprint** (psatops) |
| Bottom-up stacking | via heightmap max | gravity-first EP sort |
| Rotations | 6 | 6 (matching RotationUtils) |
| Training speed | baseline | **~10-20x faster** |

### Extreme Point algorithm (mirrors psatops)

After each placed box `b`, three new candidate positions are generated:

```
(b.x + b.l, b.y,       b.z)
(b.x,       b.y + b.w, b.z)
(b.x,       b.y,       b.z + b.h)
```

Plus the origin `(0, 0, 0)`.  Candidates are sorted by
`z * 1000 + x + y` -- the same gravity-first score used by
`ScoringUtils.scoreCandidate` in psatops.  This guarantees items
pack from the bottom up.

### 70% support fraction (mirrors psatops)

Any item placed at `z > 0` must have at least 70% of its footprint
area physically resting on the top surfaces of already-placed boxes.
This matches `PlacementUtils.SUPPORT_THRESHOLD = 0.70`.

### Rotations (mirrors psatops RotationUtils)

```
0  LWH   (l, w, h)   -- original
1  WLH   (w, l, h)   -- horizontal 90 degree
2  LHW   (l, h, w)   -- tipped onto width face
3  HLW   (h, l, w)
4  WHL   (w, h, l)   -- tipped onto length face
5  HWL   (h, w, l)
```

---

## Quick start

```bash
pip install stable-baselines3 sb3-contrib torch numpy matplotlib gymnasium
```

### Run baselines only (no training needed)

```bash
python evaluate.py --baseline_only
```

Produces `ff_result.png` (First Fit) and `ep_result.png` (Extreme Point).
The EP baseline is the closest Python equivalent to psatops `EXTREME_POINT`.

### Train

```bash
# Standard  (~5 min, 4 CPU cores)
python train.py --timesteps 500000 --n_envs 4 --no_tensorboard

# Thorough  (~20 min, 8 cores, real parallelism)
python train.py --timesteps 2000000 --n_envs 8 --subproc

# Quick smoke test
python train.py --timesteps 20000 --n_envs 2 --no_tensorboard
```

### Evaluate trained model

```bash
python evaluate.py --model checkpoints/best_model.zip
```

Produces `agent_result.png` alongside the two baselines.

---

## Custom pallet / input

```bash
python train.py --data my_order.json \
    --pallet_length 1200 --pallet_width 1000 \
    --pallet_height 1150 --max_weight 1500
```

Input JSON format:

```json
{
  "items": [
    {
      "sku": "C-BTA073V650G3ECE",
      "quantity": 10,
      "length_mm": 430,
      "width_mm": 216,
      "height_mm": 250,
      "weight_kg": 0.165
    }
  ]
}
```

---

## Architecture

### Observation space

| Key | Shape | Description |
|---|---|---|
| `heightmap` | (60, 55) | Rasterised top-surface heights, normalised to [0, 1] |
| `ep_obs` | (50, 4) | EP candidates: x, y, z normalised + valid flag |
| `item` | (4,) | Next item: l, w, h, weight normalised |
| `progress` | (2,) | Fraction placed, pallet count norm |

### Action space

`Discrete(300)` -- decoded as `ep_idx * 6 + rot`.

EP index 0 is always the psatops-optimal position (lowest z, then x, then y).
The agent learns to follow or override it based on global context.

### Reward

| Event | Reward |
|---|---|
| Place item | `(item_vol / pallet_vol) * 20` |
| Low placement bonus | `(1 - z / H) * 0.3` |
| Open new pallet | `-5` |
| Skip item (oversized) | `-15` |
| Terminal | `util * 50 - n_pallets * 3` |

### Hyperparameters

| Param | Value | Rationale |
|---|---|---|
| `n_steps` | 1024 | Longer rollouts reduce per-update overhead |
| `batch_size` | 256 | Larger batches for stable gradients |
| `n_epochs` | 5 | Sufficient with small action space |
| `gamma` | 0.995 | High discount for 117-step episodes |
| `ent_coef` | 0.005 | Lower exploration needed vs v1 |
| `lr` | 3e-4 -> 3e-5 | Linear decay |

---

## Files

```
pallet_env.py   Environment (psatops-aligned, vectorised masks)
train.py        MaskablePPO training script
evaluate.py     Baselines + trained agent comparison + 3D plots
data.json       Sample order (117 items, 6 SKUs)
```
