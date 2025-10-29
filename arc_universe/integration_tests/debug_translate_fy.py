"""
Detailed diagnostic: WHY does TRANSLATE delta=(0,2) fail FY check on train pair 0?
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load task
data_file = Path('/Users/ravishq/code/arc-agi-opoch-toe/data/arc-agi_training_challenges.json')
with open(data_file) as f:
    all_tasks = json.load(f)

task_data = all_tasks["aa18de87"]
train_pairs = [(pair["input"], pair["output"]) for pair in task_data["train"]]

X_i, Y_i = train_pairs[0]

print("=" * 80)
print("TRANSLATE FY CHECK - TRAIN PAIR 0")
print("=" * 80)

# Show grids
print(f"\nX_i (input):")
for row in X_i:
    print(f"  {row}")

print(f"\nY_i (output):")
for row in Y_i:
    print(f"  {row}")

# Test TRANSLATE with delta=(0, 2)
dr, dc = 0, 2

print(f"\n\nTesting TRANSLATE with delta=({dr}, {dc}):")
print(f"  This reads from X_i[r-{dr}][c-{dc}] = X_i[r][c-2]")

rows_y, cols_y = len(Y_i), len(Y_i[0])
rows_x, cols_x = len(X_i), len(X_i[0])

mismatches = []
for r in range(rows_y):
    for c in range(cols_y):
        src_r, src_c = r - dr, c - dc

        if 0 <= src_r < rows_x and 0 <= src_c < cols_x:
            src_color = X_i[src_r][src_c]
            tgt_color = Y_i[r][c]

            if src_color != 0 and src_color != tgt_color:
                mismatches.append((r, c, src_r, src_c, src_color, tgt_color))

if mismatches:
    print(f"\n  MISMATCHES FOUND: {len(mismatches)}")
    for r, c, src_r, src_c, src_color, tgt_color in mismatches[:10]:
        print(f"    ({r},{c}): X_i[{src_r}][{src_c}]={src_color} but Y_i[{r}][{c}]={tgt_color}")
else:
    print(f"\n  ✓ NO MISMATCHES - FY check should PASS!")

# Check all deltas
print(f"\n\n" + "=" * 80)
print("Testing ALL deltas:")
print("=" * 80)

for delta in [(0, 2), (0, 4), (0, 6), (0, 0)]:
    dr, dc = delta
    mismatches = 0
    for r in range(rows_y):
        for c in range(cols_y):
            src_r, src_c = r - dr, c - dc
            if 0 <= src_r < rows_x and 0 <= src_c < cols_x:
                src_color = X_i[src_r][src_c]
                tgt_color = Y_i[r][c]
                if src_color != 0 and src_color != tgt_color:
                    mismatches += 1
    print(f"  delta={delta}: {mismatches} mismatches {'✓' if mismatches == 0 else '✗'}")

