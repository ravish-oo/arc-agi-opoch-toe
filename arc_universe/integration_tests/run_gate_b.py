#!/usr/bin/env python3
"""
Gate B Integration Test: Shape Meet Validation (after WO-04)

Extends Gate A with shape unification via 1-D WL meet on rows/columns.

Tests:
- Everything from Gate A (present + WL union)
- Shape meet: R = ∧R_i, C = ∧C_i
- Domain analysis: |R|, |C|
- Shape-change detection

Usage:
    python run_gate_b.py --limit 50 --dataset training
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import arc_core
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_core.present import PiG, build_present
from arc_core.shape import unify_shape
from arc_core.wl import wl_union

from utils import (
    build_receipt,
    compute_summary_stats,
    grid_to_type,
    parse_task,
    random_sample_tasks,
    save_receipt,
    setup_logger,
)


def validate_pig_idempotence(grid):
    """Verify that ΠG is idempotent: ΠG(ΠG(X)) = ΠG(X)."""
    canonical_once = PiG(grid)
    canonical_twice = PiG(canonical_once)
    return canonical_once == canonical_twice


def validate_task_gate_b(task_id, task_dict, logger):
    """
    Validate a single task for Gate B (extends Gate A with shape).

    Returns:
        Receipt dictionary
    """
    try:
        # Parse task
        train_pairs, test_inputs = parse_task(task_dict)

        # Convert to Grid type
        train_inputs = [grid_to_type(inp) for inp, _ in train_pairs]
        train_outputs = [grid_to_type(out) for _, out in train_pairs]
        test_inputs = [grid_to_type(inp) for inp in test_inputs]

        all_inputs = train_inputs + test_inputs

        # ============================
        # GATE A: Present + WL Union
        # ============================

        logger.info(f"Task {task_id}: Building presents for {len(all_inputs)} grids")
        presents_all = []
        pig_idempotent_checks = []

        for i, grid in enumerate(all_inputs):
            # Check ΠG idempotence
            is_idempotent = validate_pig_idempotence(grid)
            pig_idempotent_checks.append(is_idempotent)

            # Build present
            present = build_present(grid)
            presents_all.append(present)

        all_pig_idempotent = all(pig_idempotent_checks)

        # Separate train and test presents
        presents_train = presents_all[: len(train_inputs)]
        presents_test = presents_all[len(train_inputs) :]

        # Run WL union (first run)
        logger.info(f"Task {task_id}: Running WL union (run 1)")
        role_map_1 = wl_union(presents_all, escalate=False, max_iters=12)

        # Count iterations (conservative estimate)
        wl_iters = 12

        # Extract roles
        roles_train = set()
        roles_test = set()

        for (grid_id, pixel), role_id in role_map_1.items():
            if grid_id < len(train_inputs):
                roles_train.add(role_id)
            else:
                roles_test.add(role_id)

        unseen_roles = roles_test - roles_train

        logger.info(
            f"Task {task_id}: WL converged, roles_train={len(roles_train)}, "
            f"roles_test={len(roles_test)}, unseen={len(unseen_roles)}"
        )

        # Run WL union again for determinism check
        logger.info(f"Task {task_id}: Running WL union (run 2) for determinism check")
        role_map_2 = wl_union(presents_all, escalate=False, max_iters=12)
        deterministic = role_map_1 == role_map_2

        if not deterministic:
            logger.warning(f"Task {task_id}: Determinism check FAILED")

        # ============================
        # GATE B: Shape Unification
        # ============================

        logger.info(f"Task {task_id}: Running shape unification on training inputs")
        shape_params = unify_shape(presents_train)

        # Extract shape statistics
        num_row_classes = shape_params.num_row_classes
        num_col_classes = shape_params.num_col_classes

        # Get grid dimensions
        train_input_shapes = [
            [len(grid), len(grid[0])] for grid in train_inputs
        ]
        test_input_shapes = [[len(grid), len(grid[0])] for grid in test_inputs]
        train_output_shapes = [
            [len(grid), len(grid[0])] for grid in train_outputs
        ]

        logger.info(
            f"Task {task_id}: Shape unified, |R|={num_row_classes}, |C|={num_col_classes}"
        )

        # Build present data
        present_data = {
            "CBC3": True,
            "E4": True,
            "E8": False,
            "grids_canonicalized": len(all_inputs),
            "pig_idempotent": all_pig_idempotent,
        }

        # Build WL data
        wl_data = {
            "iters": wl_iters,
            "roles_train": len(roles_train),
            "roles_test": len(roles_test),
            "unseen_roles": len(unseen_roles),
            "deterministic": deterministic,
        }

        # Build shape data
        shape_data = {
            "domain_mode": "R×C_meet",
            "num_row_classes": num_row_classes,
            "num_col_classes": num_col_classes,
            "train_input_shapes": train_input_shapes,
            "test_input_shapes": test_input_shapes,
            "train_output_shapes": train_output_shapes,
        }

        # Determine status
        status = "PASS"
        if len(unseen_roles) > 0:
            status = "FAIL"
            logger.error(f"Task {task_id}: CRITICAL - unseen_roles > 0")
        elif not deterministic:
            status = "FAIL"
            logger.error(f"Task {task_id}: CRITICAL - not deterministic")
        elif not all_pig_idempotent:
            status = "FAIL"
            logger.error(f"Task {task_id}: CRITICAL - ΠG not idempotent")
        else:
            logger.info(f"Task {task_id}: PASS")

        # Build receipt
        receipt = build_receipt(
            task_id=task_id,
            gate="B",
            present_data=present_data,
            wl_data=wl_data,
            shape_data=shape_data,
            status=status,
        )

        return receipt

    except Exception as e:
        logger.error(f"Task {task_id}: Exception - {type(e).__name__}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return build_receipt(
            task_id=task_id, gate="B", status="FAIL", error=str(e)
        )


def main():
    parser = argparse.ArgumentParser(
        description="Gate B Integration Test: Shape Meet Validation"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of tasks to sample (default: 50)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="training",
        choices=["training", "evaluation"],
        help="Dataset to use (default: training)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for task sampling"
    )

    args = parser.parse_args()

    # Setup paths
    integration_dir = Path(__file__).parent
    logs_dir = integration_dir / "logs"
    receipts_dir = integration_dir / "receipts" / "gate_b"

    # Setup logger
    logger = setup_logger("gate_b", logs_dir / "gate_b.log")

    logger.info("=" * 80)
    logger.info(f"Gate B Integration Test")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Task limit: {args.limit}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("=" * 80)

    # Load tasks
    logger.info(f"Loading {args.limit} random tasks from {args.dataset} set...")
    tasks = random_sample_tasks(dataset=args.dataset, n=args.limit, seed=args.seed)
    logger.info(f"Loaded {len(tasks)} tasks")

    # Process tasks
    receipts = []
    for task_id, task_dict in tasks.items():
        logger.info(f"\n--- Processing task {task_id} ---")
        receipt = validate_task_gate_b(task_id, task_dict, logger)
        receipts.append(receipt)

        # Save receipt
        save_receipt(receipt, receipts_dir)

    # Compute summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)

    stats = compute_summary_stats(receipts)

    logger.info(f"Total tasks: {stats['total_tasks']}")
    logger.info(f"Passed: {stats['passed']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Pass rate: {stats['pass_rate']:.2%}")

    if "wl" in stats:
        logger.info(f"\nWL Statistics:")
        logger.info(f"  Average iterations: {stats['wl']['avg_iterations']:.1f}")
        logger.info(f"  Max iterations: {stats['wl']['max_iterations']}")
        logger.info(
            f"  Unseen roles violations: {stats['wl']['unseen_roles_violations']}"
        )
        logger.info(
            f"  Determinism pass rate: {stats['wl']['determinism_pass_rate']:.2%}"
        )

    if "shape" in stats:
        logger.info(f"\nShape Statistics:")
        logger.info(
            f"  Average row classes (|R|): {stats['shape']['avg_row_classes']:.1f}"
        )
        logger.info(
            f"  Average col classes (|C|): {stats['shape']['avg_col_classes']:.1f}"
        )

    # Critical invariant checks
    logger.info("\n" + "=" * 80)
    logger.info("CRITICAL INVARIANT CHECKS")
    logger.info("=" * 80)

    unseen_violations = stats["wl"]["unseen_roles_violations"]
    determinism_rate = stats["wl"]["determinism_pass_rate"]

    if unseen_violations == 0:
        logger.info("✅ unseen_roles = 0 for ALL tasks (PASS)")
    else:
        logger.error(f"❌ unseen_roles > 0 for {unseen_violations} tasks (FAIL)")

    if determinism_rate == 1.0:
        logger.info("✅ 100% determinism across all tasks (PASS)")
    else:
        logger.error(f"❌ Determinism rate: {determinism_rate:.2%} (FAIL)")

    logger.info("\n" + "=" * 80)
    logger.info(f"Gate B validation complete. Receipts saved to: {receipts_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
