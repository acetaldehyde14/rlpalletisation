"""
Run 6 ablation experiments: base + 5 individual reward improvements.

Each run: 1M steps max, 16 envs, plateau early stopping.
"""

import subprocess
import sys
import time
from pathlib import Path

experiments = [
    {
        "name": "run0_base",
        "label": "Base (vol reward only)",
        "extra": [],
    },
    {
        "name": "run1_util_bonus",
        "label": "+ Per-pallet utilization bonus (20/10/3 for >90/70/50%)",
        "extra": ["--reward_util_bonus"],
    },
    {
        "name": "run2_topk_ep",
        "label": "+ Top-K EP (K_EP_TOP=15, action space 90)",
        "extra": ["--k_ep_top", "15"],
    },
    {
        "name": "run3_lookahead",
        "label": "+ Shaped reward with look-ahead (+1.0 next fits)",
        "extra": ["--reward_lookahead"],
    },
    {
        "name": "run4_compactness",
        "label": "+ Compactness/low-z bonuses",
        "extra": ["--reward_compactness"],
    },
    {
        "name": "run5_penalties",
        "label": "+ Penalties (new pallet -3.0, skip -10.0)",
        "extra": ["--reward_penalties"],
    },
]

for exp in experiments:
    print(f"\n{'='*60}")
    print(f"Starting: {exp['name']}")
    print(f"  {exp['label']}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "train.py",
        "--timesteps", "1000000",
        "--n_envs", "16",
        "--subproc",
        "--no_tensorboard",
        "--no_bc",
        "--skip_stage1",
        "--patience", "20",
        "--save_dir", f"checkpoints/ablation/{exp['name']}",
        "--log_dir", f"logs/ablation/{exp['name']}",
    ] + exp["extra"]

    start = time.time()
    result = subprocess.run(cmd, capture_output=False, timeout=7200)
    elapsed = time.time() - start
    print(f"\n  Completed in {elapsed/60:.1f} min (exit code: {result.returncode})")

print("\n\nAll experiments done. Evaluating...")

eval_cmd = [
    sys.executable, "-c",
    """
import json
from pathlib import Path
results = []
for exp in ["run0_base","run1_util_bonus","run2_topk_ep","run3_lookahead","run4_compactness","run5_penalties"]:
    model_path = Path(f"checkpoints/ablation/{exp}/best_model.zip")
    if model_path.exists():
        results.append(f"--model\\n{model_path}")
    else:
        print(f"  {exp}: NO MODEL FOUND")

cmd_parts = ["python", "evaluate.py", "--data", "data.json"] + [x for pair in [r.split("\\n") for r in results] for x in pair]
import subprocess
subprocess.run(cmd_parts)
""",
]

subprocess.run(eval_cmd, shell=False)
