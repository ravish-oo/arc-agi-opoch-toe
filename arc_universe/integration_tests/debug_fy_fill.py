"""
Debug: Why do REGION_FILL laws fail FY verification?
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_compile import compile_theta
from arc_laws.selectors import apply_selector_on_test

# Load task
data_file = Path('/Users/ravishq/code/arc-agi-opoch-toe/data/arc-agi_training_challenges.json')
with open(data_file) as f:
    all_tasks = json.load(f)

task_data = all_tasks["aa18de87"]
train_pairs = [(pair["input"], pair["output"]) for pair in task_data["train"]]
test_input = task_data["test"][0]["input"]

print("=" * 80)
print("REGION_FILL FY VERIFICATION DEBUG")
print("=" * 80)

# Compile
theta, U0, closures, receipts = compile_theta(train_pairs, test_input)

# Get masks
masks = theta.get("masks", [])
print(f"\n[1] Testing {len(masks)} masks:")

# Get train pairs
train_pairs_canon = theta.get("train_pairs", [])

for mask_idx, mask_spec in enumerate(masks):
    print(f"\n{'='*80}")
    print(f"MASK {mask_idx}")
    print(f"{'='*80}")

    mask_pixels = mask_spec["pixels"]
    selector_type = mask_spec["selector"]

    print(f"  Pixels: {len(mask_pixels)}")
    print(f"  Selector: {selector_type}")

    # Try FY verification manually
    for train_idx, (X, Y) in enumerate(train_pairs_canon):
        print(f"\n  Train {train_idx} ({len(X)}x{len(X[0])}):")

        # Compute selector on this train
        fill_color, empty_mask = apply_selector_on_test(
            selector_type=selector_type,
            mask=mask_pixels,
            X_test=X,
            k=None
        )

        print(f"    Selector result: color={fill_color}, empty={empty_mask}")

        if empty_mask or fill_color is None:
            print(f"    → SKIP (empty mask)")
            continue

        # Apply fill
        Y_pred = [row[:] for row in X]
        for p in mask_pixels:
            if 0 <= p.row < len(Y_pred) and 0 <= p.col < len(Y_pred[0]):
                Y_pred[p.row][p.col] = fill_color

        # Check match
        matches = (Y_pred == Y)
        print(f"    → {'PASS' if matches else 'FAIL'}")

        if not matches:
            # Find mismatches
            diffs = []
            for r in range(min(len(Y), len(Y_pred))):
                for c in range(min(len(Y[0]), len(Y_pred[0]))):
                    if Y[r][c] != Y_pred[r][c]:
                        diffs.append((r, c, Y_pred[r][c], Y[r][c]))

            print(f"    Mismatches: {len(diffs)}")
            for r, c, pred, exp in diffs[:5]:
                print(f"      ({r},{c}): pred={pred}, expected={exp}")
