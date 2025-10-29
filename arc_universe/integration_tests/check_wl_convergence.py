#!/usr/bin/env python3
"""
Diagnostic: Check WL actual convergence iterations.

This script verifies that WL early stopping is working correctly
by tracking actual iteration counts across tasks.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_core.present import build_present

from diagnostic_wl import wl_union_instrumented
from utils import grid_to_type, parse_task, random_sample_tasks

# Load 50 tasks
print("Loading 50 random tasks for WL convergence diagnostics...")
tasks = random_sample_tasks(dataset="training", n=50, seed=42)

convergence_counts = {}
tasks_hitting_cap = []

for task_id, task_dict in tasks.items():
    # Parse task
    train_pairs, test_inputs = parse_task(task_dict)

    # Convert to Grid type
    train_inputs = [grid_to_type(inp) for inp, _ in train_pairs]
    test_inputs = [grid_to_type(inp) for inp in test_inputs]
    all_inputs = train_inputs + test_inputs

    # Build presents
    presents_all = [build_present(grid) for grid in all_inputs]

    # Run instrumented WL
    role_map, actual_iters = wl_union_instrumented(presents_all, escalate=False, max_iters=12)

    # Track convergence
    if actual_iters not in convergence_counts:
        convergence_counts[actual_iters] = 0
    convergence_counts[actual_iters] += 1

    # Check if hitting cap
    if actual_iters == 12:
        tasks_hitting_cap.append(task_id)

    print(f"Task {task_id}: converged at iteration {actual_iters}")

print("\n" + "="*80)
print("CONVERGENCE DISTRIBUTION")
print("="*80)

for iters in sorted(convergence_counts.keys()):
    count = convergence_counts[iters]
    pct = (count / 50) * 100
    bar = "█" * int(pct / 2)
    print(f"Iteration {iters:2d}: {count:2d} tasks ({pct:5.1f}%) {bar}")

print("\n" + "="*80)
print("CRITICAL CHECK: Tasks hitting max_iters cap (potential bug)")
print("="*80)

if tasks_hitting_cap:
    print(f"⚠️  {len(tasks_hitting_cap)} tasks hit the max_iters=12 cap:")
    for task_id in tasks_hitting_cap[:10]:  # Show first 10
        print(f"  - {task_id}")
    if len(tasks_hitting_cap) > 10:
        print(f"  ... and {len(tasks_hitting_cap) - 10} more")
    print("\n❌ POTENTIAL BUG: If many tasks hit cap, WL may not be converging!")
    print("   Either increase max_iters or investigate convergence logic.")
else:
    print("✅ All tasks converged before hitting max_iters cap!")
    print("   Early stopping is working correctly.")

print("\n" + "="*80)
print("STATISTICS")
print("="*80)

avg_iters = sum(iters * count for iters, count in convergence_counts.items()) / 50
print(f"Average convergence iteration: {avg_iters:.2f}")
print(f"Min convergence iteration: {min(convergence_counts.keys())}")
print(f"Max convergence iteration: {max(convergence_counts.keys())}")
print(f"Tasks at max_iters: {len(tasks_hitting_cap)}/50 ({len(tasks_hitting_cap)/50*100:.1f}%)")
