"""
Debug: Check what colors are in the diff masks for aa18de87.
"""
import sys
import json
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_core.types import Pixel
from arc_core.present import build_present
from arc_core.palette import canonicalize_palette_for_task

# Load task
data_file = Path('/Users/ravishq/code/arc-agi-opoch-toe/data/arc-agi_training_challenges.json')
with open(data_file) as f:
    all_tasks = json.load(f)

task_data = all_tasks["aa18de87"]
train_pairs = [(pair["input"], pair["output"]) for pair in task_data["train"]]

print("=" * 80)
print("MASK COLOR DETECTION DEBUG")
print("=" * 80)

# Canonicalize
train_inputs = [inp for inp, _ in train_pairs]
test_input = task_data["test"][0]["input"]
palette_map = canonicalize_palette_for_task(train_inputs, test_input)

def apply_palette(grid, pmap):
    if not pmap:
        return grid
    return [[pmap.get(grid[r][c], grid[r][c]) for c in range(len(grid[0]))]
            for r in range(len(grid))]

canonicalized_train_pairs = [
    (apply_palette(inp, palette_map), out)
    for inp, out in train_pairs
]

# Check each train
for train_idx, (X_i, Y_i) in enumerate(canonicalized_train_pairs):
    print(f"\n{'='*80}")
    print(f"TRAIN {train_idx}")
    print(f"{'='*80}")

    rows, cols = len(X_i), len(X_i[0])
    print(f"Grid size: {rows}x{cols}")

    # Compute diff
    diff_mask = set()
    for r in range(rows):
        for c in range(cols):
            if Y_i[r][c] != X_i[r][c]:
                diff_mask.add(Pixel(r, c))

    print(f"Diff pixels: {len(diff_mask)}")

    # Get colors in diff region
    diff_colors = set()
    for p in diff_mask:
        c = Y_i[p.row][p.col]
        diff_colors.add(c)

    print(f"Colors in diff region: {sorted(diff_colors)}")

    # Sample pixels
    sample = list(diff_mask)[:5]
    print(f"Sample diff pixels:")
    for p in sample:
        print(f"  {p}: X={X_i[p.row][p.col]} -> Y={Y_i[p.row][p.col]}")
