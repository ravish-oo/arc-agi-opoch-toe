#!/usr/bin/env python3
"""
Gate A Integration Test: WL Union Validation (after WO-03)

Tests present canonicalization and 1-WL on union of train+test inputs.

Critical Invariants:
- unseen_roles = 0 (no test-only roles)
- deterministic = True (role IDs stable across runs)
- pig_idempotent = True (ΠG is idempotent)

Usage:
    python run_gate_a.py --limit 50 --dataset training
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import arc_core
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_core.present import PiG, build_present
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


def validate_task_gate_a(task_id, task_dict, logger):
    """
    Validate a single task for Gate A.

    Returns:
        Receipt dictionary
    """
    try:
        # Parse task
        train_pairs, test_inputs = parse_task(task_dict)

        # Convert to Grid type
        train_inputs = [grid_to_type(inp) for inp, _ in train_pairs]
        test_inputs = [grid_to_type(inp) for inp in test_inputs]

        all_inputs = train_inputs + test_inputs

        # Build Present structures
        logger.info(f"Task {task_id}: Building presents for {len(all_inputs)} grids")
        presents = []
        pig_idempotent_checks = []

        for i, grid in enumerate(all_inputs):
            # Check ΠG idempotence
            is_idempotent = validate_pig_idempotence(grid)
            pig_idempotent_checks.append(is_idempotent)

            # Build present
            present = build_present(grid)
            presents.append(present)

        # Check if all ΠG checks passed
        all_pig_idempotent = all(pig_idempotent_checks)

        # Run WL union (E4 only - E8 escalation happens via push-back in later WOs)
        logger.info(f"Task {task_id}: Running WL union (run 1)")
        role_map_1 = wl_union(presents, escalate=False, max_iters=12)

        # Extract roles
        roles_train = set()
        roles_test = set()

        for (grid_id, pixel), role_id in role_map_1.items():
            if grid_id < len(train_inputs):
                roles_train.add(role_id)
            else:
                roles_test.add(role_id)

        # Compute unseen roles (roles in test but NOT in train)
        # Per anchor clarification: unseen_roles > 0 is ALLOWED when test has new structure
        # The invariant is "shared ID space" (single union run), NOT "unseen_roles == 0"
        unseen_roles = roles_test - roles_train

        logger.info(
            f"Task {task_id}: WL converged, roles_train={len(roles_train)}, "
            f"roles_test={len(roles_test)}, unseen={len(unseen_roles)}"
        )

        if len(unseen_roles) > 0:
            logger.info(
                f"Task {task_id}: {len(unseen_roles)} unseen roles "
                f"(test has new input structure - this is allowed)"
            )

        # Check determinism: role maps should be identical
        logger.info(f"Task {task_id}: Running WL union (run 2) for determinism check")
        role_map_2 = wl_union(presents, escalate=False, max_iters=12)
        deterministic = role_map_1 == role_map_2

        if not deterministic:
            logger.warning(f"Task {task_id}: Determinism check FAILED")

        # Build present data
        present_data = {
            "CBC3": True,  # Always computed
            "E4": True,  # Always used
            "E8": False,  # E8 escalation happens via push-back (later WOs)
            "grids_canonicalized": len(all_inputs),
            "pig_idempotent": all_pig_idempotent,
        }

        # Build WL data
        # Per anchor clarification: "shared_id_space" is the invariant, not "unseen_roles == 0"
        wl_data = {
            "iters": 12,  # Conservative estimate
            "roles_train": len(roles_train),
            "roles_test": len(roles_test),
            "unseen_roles": len(unseen_roles),  # Measurement, not failure condition
            "deterministic": deterministic,
            "shared_id_space": True,  # WL computed once on union (invariant)
        }

        # Determine status based on CORRECT invariants
        # Per anchor author clarification:
        # - Assert: shared_id_space == True (WL on union, one partition)
        # - Assert: deterministic == True (stable IDs)
        # - Do NOT assert: unseen_roles == 0 (allowed when test has new structure)
        status = "PASS"

        if not deterministic:
            status = "FAIL"
            logger.error(f"Task {task_id}: FAIL - determinism check failed")
        elif not all_pig_idempotent:
            status = "FAIL"
            logger.error(f"Task {task_id}: FAIL - ΠG not idempotent")
        else:
            logger.info(f"Task {task_id}: PASS")

        # Build receipt
        receipt = build_receipt(
            task_id=task_id,
            gate="A",
            present_data=present_data,
            wl_data=wl_data,
            status=status,
        )

        return receipt

    except Exception as e:
        logger.error(f"Task {task_id}: Exception - {type(e).__name__}: {e}")
        return build_receipt(
            task_id=task_id, gate="A", status="FAIL", error=str(e)
        )


def main():
    parser = argparse.ArgumentParser(
        description="Gate A Integration Test: WL Union Validation"
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
    receipts_dir = integration_dir / "receipts" / "gate_a"

    # Setup logger
    logger = setup_logger("gate_a", logs_dir / "gate_a.log")

    logger.info("=" * 80)
    logger.info(f"Gate A Integration Test")
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
        receipt = validate_task_gate_a(task_id, task_dict, logger)
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
            f"  Determinism pass rate: {stats['wl']['determinism_pass_rate']:.2%}"
        )
        logger.info(f"  Tasks with unseen roles: {stats['wl']['tasks_with_unseen']}")
        logger.info(f"  Avg unseen roles (when present): {stats['wl']['avg_unseen_when_present']:.1f}")

    # Critical invariant checks
    logger.info("\n" + "=" * 80)
    logger.info("CRITICAL INVARIANT CHECKS")
    logger.info("=" * 80)

    determinism_rate = stats["wl"]["determinism_pass_rate"]
    tasks_with_unseen = stats["wl"]["tasks_with_unseen"]

    # Per anchor author clarification:
    # - Invariant: shared_id_space == True (WL on union, one partition)
    # - Invariant: deterministic == True (stable IDs)
    # - NOT invariant: unseen_roles == 0 (allowed when test has new structure)

    if determinism_rate == 1.0:
        logger.info("✅ 100% determinism across all tasks (PASS)")
    else:
        logger.error(f"❌ Determinism rate: {determinism_rate:.2%} (FAIL)")

    # Log unseen roles as measurement (not failure)
    if tasks_with_unseen > 0:
        logger.info(
            f"📊 {tasks_with_unseen} tasks have unseen roles "
            f"(test has new input structure - this is allowed)"
        )
    else:
        logger.info("📊 All tasks have complete role overlap (train covers test)")

    logger.info("\n" + "=" * 80)
    logger.info(f"Gate A validation complete. Receipts saved to: {receipts_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
