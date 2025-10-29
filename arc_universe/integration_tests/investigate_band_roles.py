"""
Phase 1 Investigation: What are band roles and how do they work?

Per engineering_spec.md line 108:
"Masks are present-definable (band roles, component classes, periodic classes)"

Per engineering_spec.md line 180:
"1-D WL bands yields band roles (masks)"

Let me check what shape_result actually contains.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_compile import compile_theta

# Load task aa18de87
data_file = Path('/Users/ravishq/code/arc-agi-opoch-toe/data/arc-agi_training_challenges.json')
with open(data_file) as f:
    all_tasks = json.load(f)

task_data = all_tasks["aa18de87"]
train_pairs = [(pair["input"], pair["output"]) for pair in task_data["train"]]
test_input = task_data["test"][0]["input"]

print("=" * 80)
print("PHASE 1: INVESTIGATING BAND ROLES")
print("=" * 80)

# Compile
theta, U0, closures, receipts = compile_theta(train_pairs, test_input)

# Check shape_result
shape_result = theta.get("shape")

print("\n[1] Shape Result Structure:")
if shape_result:
    print(f"  Type: {type(shape_result)}")
    print(f"  Fields: {dir(shape_result)}")

    print(f"\n[2] Row partition (R):")
    print(f"  Type: {type(shape_result.R)}")
    print(f"  Contents: {shape_result.R}")
    print(f"  Num row classes: {shape_result.num_row_classes}")

    print(f"\n[3] Column partition (C):")
    print(f"  Type: {type(shape_result.C)}")
    print(f"  Contents: {shape_result.C}")
    print(f"  Num col classes: {shape_result.num_col_classes}")

    print(f"\n[4] Per-train partitions:")
    print(f"  Row partitions: {len(shape_result.row_partitions)} trains")
    for i, R_i in enumerate(shape_result.row_partitions):
        print(f"    Train {i}: {R_i}")

    print(f"  Col partitions: {len(shape_result.col_partitions)} trains")
    for i, C_i in enumerate(shape_result.col_partitions):
        print(f"    Train {i}: {C_i}")

    print(f"\n[5] Understanding band roles:")
    print(f"  A 'band role' is a (row_class, col_class) pair from the unified domain U = R×C")
    print(f"  For aa18de87:")
    print(f"    - {shape_result.num_row_classes} row classes")
    print(f"    - {shape_result.num_col_classes} col classes")
    print(f"    - Total band roles: {shape_result.num_row_classes * shape_result.num_col_classes}")

    print(f"\n[6] Example: What pixels have band role (row_class=0, col_class=0)?")
    test_present = theta["test_present"]
    test_grid = test_present.grid
    rows, cols = len(test_grid), len(test_grid[0])

    # For test grid, we need to compute its 1D WL and map to unified domain
    # For now, just show the unified R and C
    print(f"  Test grid shape: {rows}x{cols}")
    print(f"  Unified R (row classes): {shape_result.R}")
    print(f"  Unified C (col classes): {shape_result.C}")

    print(f"\n[7] Key insight:")
    print(f"  - Band roles are SEMANTIC (row_class, col_class)")
    print(f"  - They work on ANY grid size")
    print(f"  - To get pixels: evaluate R and C on target grid, then select pixels matching the band")
else:
    print("  No shape_result found!")

# Check role_map
print(f"\n[8] WL role_map:")
role_map = theta.get("role_map", {})
print(f"  Total (grid_id, pixel) entries: {len(role_map)}")

# Sample entries
sample_entries = list(role_map.items())[:10]
print(f"  Sample entries:")
for (grid_id, pixel), role_id in sample_entries:
    print(f"    Grid {grid_id}, Pixel {pixel} → Role {role_id}")

print(f"\n[9] Comparing band roles vs WL roles:")
print(f"  - WL roles: Per-pixel from 2-D WL (finer grain)")
print(f"  - Band roles: Per-(row_class, col_class) from 1-D WL (coarser grain)")
print(f"  - Band roles are SUBSETS of pixels sharing same (R[r], C[c])")
