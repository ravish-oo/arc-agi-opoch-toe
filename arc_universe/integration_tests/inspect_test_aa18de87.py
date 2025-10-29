"""
Inspect the TEST input for aa18de87 to see what needs to be filled.
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
test_input = task_data["test"][0]["input"]

print("=" * 80)
print("TEST INPUT INSPECTION")
print("=" * 80)

print(f"\nTest shape: {len(test_input)}x{len(test_input[0])}")

print(f"\nTest input:")
for row in test_input:
    print(f"  {row}")

# Count colors
from collections import Counter
all_pixels = []
for row in test_input:
    all_pixels.extend(row)

color_counts = Counter(all_pixels)
print(f"\nColor counts:")
for color, count in sorted(color_counts.items()):
    print(f"  Color {color}: {count} pixels")

# Find background pixels
background = []
for r in range(len(test_input)):
    for c in range(len(test_input[0])):
        if test_input[r][c] == 0:
            background.append((r, c))

print(f"\nBackground (0) pixels: {len(background)}")
if len(background) <= 30:
    print(f"  Positions: {background}")
