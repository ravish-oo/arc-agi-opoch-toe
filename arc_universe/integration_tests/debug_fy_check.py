"""
Diagnostic: Why do all TRANSLATE expressions fail FY check for task aa18de87?
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_compile import compile_theta
from arc_core.types import Pixel

# Load task
data_file = Path('/Users/ravishq/code/arc-agi-opoch-toe/data/arc-agi_training_challenges.json')
with open(data_file) as f:
    all_tasks = json.load(f)

task_data = all_tasks["aa18de87"]
train_pairs = [(pair["input"], pair["output"]) for pair in task_data["train"]]
test_input = task_data["test"][0]["input"]

print("=" * 80)
print("FY CHECK DIAGNOSTIC FOR TASK aa18de87")
print("=" * 80)

# Compile
theta, U0, closures, receipts = compile_theta(train_pairs, test_input)

print(f"\n[1] Theta contents:")
print(f"  deltas (list): {theta.get('deltas', [])}")
print(f"  components (dict, test): {list(theta.get('components', {}).keys())[:10]}")
print(f"  components_per_grid (list, train): {len(theta.get('components_per_grid', []))} grids")

# Get a sample TRANSLATE expression
sample_expr = None
for pixel, exprs in U0.items():
    for expr in exprs:
        if expr.kind == "TRANSLATE":
            sample_expr = expr
            break
    if sample_expr:
        break

print(f"\n[2] Sample TRANSLATE expression:")
print(f"  {sample_expr}")

# Check what goes wrong in FY check
print(f"\n[3] Checking FY for this expression on train pair 0:")

X_i, Y_i = train_pairs[0]
print(f"  X_i shape: {len(X_i)}x{len(X_i[0])}")
print(f"  Y_i shape: {len(Y_i)}x{len(Y_i[0])}")

# Line 442-448 check
print(f"\n[4] Checking delta validation (lines 442-448):")
compiled_deltas = theta.get("deltas", {})
print(f"  compiled_deltas type: {type(compiled_deltas)}")
print(f"  compiled_deltas value: {compiled_deltas}")
print(f"  expr.component_id: {sample_expr.component_id}")
print(f"  expr.component_id in compiled_deltas: {sample_expr.component_id in compiled_deltas}")

# Line 458 check - Dom()
print(f"\n[5] Checking Dom() for pixel (0,0) on training grid:")
q_test = Pixel(0, 0)
try:
    is_defined = sample_expr.Dom(q_test, X_i, theta)
    print(f"  is_defined: {is_defined}")
    
    if is_defined:
        eval_result = sample_expr.eval(q_test, X_i, theta)
        expected = Y_i[0][0]
        print(f"  eval_result: {eval_result}")
        print(f"  expected: {expected}")
        print(f"  match: {eval_result == expected}")
except Exception as e:
    print(f"  ERROR: {e}")

# Check what components are available
print(f"\n[6] Components available in theta:")
test_components = theta.get("components", {})
print(f"  Number of test components: {len(test_components)}")
for comp_id, comp in list(test_components.items())[:3]:
    print(f"    Component {comp_id}: {len(comp.pixels)} pixels")

print(f"\n[7] Train components (from components_per_grid):")
train_comps_per_grid = theta.get("components_per_grid", [])
if train_comps_per_grid:
    print(f"  Train pair 0 has {len(train_comps_per_grid[0])} components")
    for i, comp in enumerate(train_comps_per_grid[0][:3]):
        print(f"    Component {i}: {len(comp.pixels)} pixels")

