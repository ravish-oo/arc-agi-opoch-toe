"""
Debug mask extraction for aa18de87.

Check:
1. How many masks are extracted
2. What selectors are assigned
3. How many FILL laws pass FY verification
"""

import sys
sys.path.append("/Users/ravishq/code/arc-agi-opoch-toe/arc_universe")

from arc_compile.compile_theta import compile_theta
from utils import load_arc_tasks
import json

# Load task
tasks = load_arc_tasks("training")
task = tasks["aa18de87"]
train_pairs = [(ex['input'], ex['output']) for ex in task['train']]
test_input = task['test'][0]['input']

print("=" * 80)
print("DEBUG: Mask Extraction for aa18de87")
print("=" * 80)

# Compile theta
theta, U0, closures, receipts = compile_theta(train_pairs, test_input)

# Check masks
masks = theta.get("masks", [])
print(f"\n1. Extracted {len(masks)} mask specs")
if masks:
    print(f"\n   Sample masks (first 5):")
    for i, mask_dict in enumerate(masks[:5]):
        mask_spec = mask_dict["mask_spec"]
        selector = mask_dict["selector_type"]
        print(f"   - Mask {i}: type={mask_spec.mask_type}, params={mask_spec.params}, selector={selector}")

# Check connect_fill_laws
fill_laws = theta.get("connect_fill_laws", [])
print(f"\n2. {len(fill_laws)} FILL laws passed FY verification")

if fill_laws:
    from arc_laws.connect_fill import FillLaw
    fill_only = [law for law in fill_laws if isinstance(law, FillLaw)]
    print(f"   ({len(fill_only)} are FillLaw, {len(fill_laws) - len(fill_only)} are ConnectLaw)")

    if fill_only:
        print(f"\n   Sample FillLaw:")
        law = fill_only[0]
        print(f"   - mask_spec: type={law.mask_spec.mask_type}, params={law.mask_spec.params}")
        print(f"   - selector_type: {law.selector_type}")

# Check region_fills (for init_expressions)
region_fills = theta.get("region_fills", [])
print(f"\n3. {len(region_fills)} region_fills for init_expressions")
if region_fills:
    print(f"\n   Sample region_fill (first one):")
    rf = region_fills[0]
    print(f"   - mask size: {len(rf['mask'])} pixels")
    print(f"   - fill_color: {rf['fill_color']}")

print("\n" + "=" * 80)
