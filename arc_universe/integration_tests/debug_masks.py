"""
Diagnostic: Test mask extraction for task aa18de87.

Expected: Should detect masks that cover the 24 empty pixels.
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
print("MASK EXTRACTION DIAGNOSTIC - TASK aa18de87")
print("=" * 80)

# Compile theta
print("\n[1] Compiling theta with mask extraction...")
theta, U0, closures, receipts = compile_theta(train_pairs, test_input)

# Check what masks were detected
print(f"\n[2] Masks detected:")
masks = theta.get("masks", [])
print(f"  Number of masks: {len(masks)}")

for idx, mask_spec in enumerate(masks):
    mask_pixels = mask_spec["pixels"]
    selector = mask_spec["selector"]
    print(f"    Mask {idx}: {len(mask_pixels)} pixels, selector={selector}")
    # Show first few pixels
    sample = list(mask_pixels)[:5]
    print(f"      Sample pixels: {sample}")

# Check if REGION_FILL laws were created
print(f"\n[4] REGION_FILL laws:")
connect_fill_laws = theta.get("connect_fill_laws", [])
print(f"  Number of laws: {len(connect_fill_laws)}")

# Check U0 - do we still have empty pixels?
print(f"\n[5] U0 state:")
total_exprs = sum(len(exprs) for exprs in U0.values())
empty_pixels = [q for q, exprs in U0.items() if len(exprs) == 0]
singleton_pixels = [q for q, exprs in U0.items() if len(exprs) == 1]
multi_pixels = [(q, len(exprs)) for q, exprs in U0.items() if len(exprs) > 1]

print(f"  Total expressions: {total_exprs}")
print(f"  Empty pixels: {len(empty_pixels)}")
print(f"  Singleton pixels: {len(singleton_pixels)}")
print(f"  Multiple expression pixels: {len(multi_pixels)}")

if empty_pixels:
    print(f"\n  Sample empty pixels (first 5):")
    for q in empty_pixels[:5]:
        print(f"    {q}")

# Check expression types
print(f"\n[6] Expression types in U0:")
expr_types = {}
for q, exprs in U0.items():
    for expr in exprs:
        kind = expr.kind
        expr_types[kind] = expr_types.get(kind, 0) + 1

for kind, count in sorted(expr_types.items()):
    print(f"  {kind}: {count}")

# Show basis used
print(f"\n[7] Basis used:")
basis = receipts.get("basis_used", [])
print(f"  {basis}")
