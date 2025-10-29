"""
Diagnostic script to debug LFP convergence for task aa18de87.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_compile import compile_theta
from arc_fixedpoint.lfp import compute_lfp

# Load task aa18de87
data_file = Path('/Users/ravishq/code/arc-agi-opoch-toe/data/arc-agi_training_challenges.json')
with open(data_file) as f:
    all_tasks = json.load(f)

task_data = all_tasks["aa18de87"]
train_pairs = [(pair["input"], pair["output"]) for pair in task_data["train"]]
test_input = task_data["test"][0]["input"]

print("=" * 80)
print("TASK aa18de87 - DIAGNOSTIC")
print("=" * 80)

# Step 1: Compile theta
print("\n[1] Compiling theta...")
theta, U0, closures, receipts = compile_theta(train_pairs, test_input)

print(f"  train_constant_roles: {len(theta.get('train_constant_roles', {}))}")
print(f"  deltas: {theta.get('deltas', [])}")
print(f"  basis_used: {receipts.get('basis_used', [])}")
print(f"  unseen_roles: {receipts['wl'].get('unseen_roles', 0)}")

# Step 2: Check U0
print(f"\n[2] U0 initial state:")
print(f"  Total pixels: {len(U0)}")
print(f"  Empty pixels: {sum(1 for exprs in U0.values() if len(exprs) == 0)}")
print(f"  Total expressions: {sum(len(exprs) for exprs in U0.values())}")
print(f"  Expressions per pixel (sample):")
for i, (pixel, exprs) in enumerate(list(U0.items())[:5]):
    print(f"    Pixel {pixel}: {len(exprs)} expressions")
    for expr in list(exprs)[:3]:
        print(f"      - {expr.kind}: {expr}")

# Step 3: Run LFP with detailed logging
print(f"\n[3] Running LFP...")
X_test_present = theta["test_present"].grid

try:
    U_star, lfp_receipt = compute_lfp(U0, theta, X_test_present)
    
    print(f"  Passes: {lfp_receipt.passes}")
    print(f"  Total removals: {lfp_receipt.total_removals}")
    print(f"  Removals per stage:")
    for stage, count in lfp_receipt.removals_per_stage.items():
        print(f"    {stage}: {count}")
    print(f"  Singletons: {lfp_receipt.singletons}/{len(U_star)}")
    
    print(f"\n[4] SUCCESS!")
    
except AssertionError as e:
    print(f"\n[4] LFP FAILED TO CONVERGE")
    print(f"  Error: {e}")
    
    # Check what went wrong
    print(f"\n[5] Post-mortem analysis:")
    
    # Re-run LFP manually to see what's happening
    from arc_fixedpoint.closures import (
        apply_definedness_closure,
        apply_canvas_closure,
        apply_lattice_closure,
        apply_block_closure,
        apply_object_closure,
        apply_selector_closure,
        apply_local_paint_closure,
        apply_interface_closure,
    )
    
    U = {q: set(exprs) for q, exprs in U0.items()}
    
    print(f"\n  Initial state: {sum(len(exprs) for exprs in U.values())} total expressions")
    
    # Pass 1
    print(f"\n  Pass 1:")
    U, r1 = apply_definedness_closure(U, X_test_present, theta)
    print(f"    After T_def: {sum(len(exprs) for exprs in U.values())} exprs (removed {r1})")
    
    U, r2 = apply_canvas_closure(U, theta)
    print(f"    After T_canvas: {sum(len(exprs) for exprs in U.values())} exprs (removed {r2})")
    
    U, r3 = apply_lattice_closure(U, theta)
    print(f"    After T_lattice: {sum(len(exprs) for exprs in U.values())} exprs (removed {r3})")
    
    U, r4 = apply_block_closure(U, theta)
    print(f"    After T_block: {sum(len(exprs) for exprs in U.values())} exprs (removed {r4})")
    
    U, r5 = apply_object_closure(U, theta)
    print(f"    After T_object: {sum(len(exprs) for exprs in U.values())} exprs (removed {r5})")
    
    U, r6 = apply_selector_closure(U, theta)
    print(f"    After T_select: {sum(len(exprs) for exprs in U.values())} exprs (removed {r6})")
    
    U, r7 = apply_local_paint_closure(U, theta)
    print(f"    After T_local: {sum(len(exprs) for exprs in U.values())} exprs (removed {r7})")
    
    U, r8 = apply_interface_closure(U, theta)
    print(f"    After T_Gamma: {sum(len(exprs) for exprs in U.values())} exprs (removed {r8})")
    
    print(f"\n  Total removals pass 1: {r1+r2+r3+r4+r5+r6+r7+r8}")
    
    # Check which pixels have multiple expressions
    multi_expr_pixels = [(q, len(exprs)) for q, exprs in U.items() if len(exprs) > 1]
    print(f"\n  Pixels with multiple expressions: {len(multi_expr_pixels)}")
    if multi_expr_pixels:
        print(f"  Sample (first 5):")
        for q, count in multi_expr_pixels[:5]:
            print(f"    {q}: {count} expressions")
            for expr in list(U[q])[:3]:
                print(f"      - {expr}")


    # Better diagnostics
    empty_pixels = [q for q, exprs in U.items() if len(exprs) == 0]
    singleton_pixels = [q for q, exprs in U.items() if len(exprs) == 1]
    multi_pixels = [(q, len(exprs)) for q, exprs in U.items() if len(exprs) > 1]
    
    print(f"\n  Pixel distribution:")
    print(f"    Empty: {len(empty_pixels)}")
    print(f"    Singleton: {len(singleton_pixels)}")
    print(f"    Multiple: {len(multi_pixels)}")
    
    if empty_pixels:
        print(f"\n  Sample empty pixels (first 3):")
        for q in empty_pixels[:3]:
            print(f"    {q}")
