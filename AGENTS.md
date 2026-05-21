# AGENTS.md

## Commands

```bash
pip install -r requirements.txt

# Quick smoke test (~1 min)
python train.py --timesteps 20000 --n_envs 2 --no_tensorboard

# Standard training (~15-30 min, 4 cores)
python train.py --timesteps 500000 --n_envs 4 --no_tensorboard

# Thorough training (best results, 1-2 h)
python train.py --timesteps 2000000 --n_envs 8 --subproc

# Baselines only (no model needed)
python evaluate.py --baseline_only

# Evaluate trained model
python evaluate.py --model checkpoints/best_model.zip
```

No test suite, linter, or typechecker is configured.

## Architecture

Three files, no package structure:

- **`pallet_env.py`** — `PalletPackingEnv` (Gymnasium env). Extreme Point formulation (v2) mirroring the psatops Java backend. Action space `Discrete(300)` = 50 candidate positions × 6 rotations. Vectorized action masking via numpy broadcasting.
- **`train.py`** — `MaskablePPO` from `sb3-contrib` with `MultiInputPolicy`. Outputs to `checkpoints/` and `logs/`.
- **`evaluate.py`** — Runs two deterministic baselines (First Fit Decreasing, Extreme Point) and optionally a trained agent. Produces 3D matplotlib PNGs (`ff_result.png`, `ep_result.png`, `agent_result.png`).

## Critical gotchas

- **CLAUDE.md describes the old v1 architecture** (grid sweep, 4800 actions, 2 rotations, flat support). The actual code is v2 (Extreme Point, 300 actions, 6 rotations, 70% support threshold). README.md is accurate; trust it over CLAUDE.md.
- **Default pallet dimensions**: 1200×1100×1150 mm, max 1500 kg — not the EUR pallet specs in CLAUDE.md.
- **Heightmap shape is dynamic**: `(grid_l, grid_w)` derived from `pallet_length / grid_resolution` and `pallet_width / grid_resolution`, not a fixed 60×40.
- **`--subproc` uses `SubprocVecEnv`** which requires the `if __name__ == "__main__"` guard (already present in `train.py`).
- **Action masking fallback**: when no valid placements exist on the current pallet, `action_masks()` returns all-True, which triggers the env to open a new pallet in `step()`.
- **Items sorted by volume descending** at reset when `sort_items=True` (default).

## Key hyperparameters (v2 defaults)

`n_steps=1024`, `batch_size=256`, `n_epochs=5`, `gamma=0.995`, `ent_coef=0.005`, `lr=3e-4` (linear decay to 10%), `clip_range=0.2`, `gae_lambda=0.95`.

## Custom pallet / input data

Both `train.py` and `evaluate.py` accept `--pallet_length`, `--pallet_width`, `--pallet_height`, `--max_weight`. Input JSON via `--data` with format: `{"items": [{"sku": ..., "quantity": ..., "length_mm": ..., "width_mm": ..., "height_mm": ..., "weight_kg": ...}]}`.
