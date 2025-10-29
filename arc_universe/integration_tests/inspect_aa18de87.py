"""
Inspect task aa18de87 to understand the pattern and what masks should exist.
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

print("=" * 80)
print("TASK aa18de87 INSPECTION")
print("=" * 80)

for idx, (X, Y) in enumerate(train_pairs):
    print(f"\n{'='*80}")
    print(f"TRAIN PAIR {idx}")
    print(f"{'='*80}")

    print(f"\nInput shape: {len(X)}x{len(X[0])}")
    print(f"Output shape: {len(Y)}x{len(Y[0])}")

    # Show input
    print(f"\nInput:")
    for row in X:
        print(f"  {row}")

    # Show output
    print(f"\nOutput:")
    for row in Y:
        print(f"  {row}")

    # Count colors in input
    input_colors = set()
    for row in X:
        for c in row:
            input_colors.add(c)
    print(f"\nInput colors: {sorted(input_colors)}")

    # Count colors in output
    output_colors = set()
    for row in Y:
        for c in row:
            output_colors.add(c)
    print(f"Output colors: {sorted(output_colors)}")

    # Find differences
    diff_pixels = []
    for r in range(len(X)):
        for c in range(len(X[0])):
            if X[r][c] != Y[r][c]:
                diff_pixels.append((r, c, X[r][c], Y[r][c]))

    print(f"\nDifferences: {len(diff_pixels)} pixels changed")
    if len(diff_pixels) <= 50:
        for r, c, x_val, y_val in diff_pixels[:10]:
            print(f"  ({r},{c}): {x_val} -> {y_val}")

    # Check if there are background (0) pixels in input that become non-zero in output
    background_filled = []
    for r in range(len(X)):
        for c in range(len(X[0])):
            if X[r][c] == 0 and Y[r][c] != 0:
                background_filled.append((r, c, Y[r][c]))

    print(f"\nBackground pixels filled: {len(background_filled)}")
    if background_filled:
        print(f"  First few: {background_filled[:10]}")

    # Check for holes in colored components
    print(f"\nChecking for holes in colored components...")
    for color in sorted(input_colors):
        if color == 0:
            continue

        # Find all pixels of this color
        color_pixels = set()
        for r in range(len(X)):
            for c in range(len(X[0])):
                if X[r][c] == color:
                    color_pixels.add((r, c))

        if not color_pixels:
            continue

        # Find bbox
        min_r = min(r for r, c in color_pixels)
        max_r = max(r for r, c in color_pixels)
        min_c = min(c for r, c in color_pixels)
        max_c = max(c for r, c in color_pixels)

        # Find pixels in bbox but not in color
        inside = []
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if (r, c) not in color_pixels:
                    inside.append((r, c, X[r][c]))

        if inside:
            print(f"  Color {color}: bbox has {len(inside)} non-{color} pixels inside")
            print(f"    Sample: {inside[:5]}")
