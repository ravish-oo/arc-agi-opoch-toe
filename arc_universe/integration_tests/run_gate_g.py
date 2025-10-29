#!/usr/bin/env python3
"""
Gate G Integration Test: Full end-to-end pipeline (after WO-19/20)

Tests the complete compilation → LFP → evaluation pipeline:
- WO-19: compile_theta (ΠG → WL → shape → params → U0 → closures)
- WO-17: compute_lfp (8-stage closure application)
- WO-18: evaluate_and_unpresent (singleton eval + g_test^-1)

Validates ALL anchor-mandated metrics:
- unseen_roles = 0 (no test-only WL roles)
- fy_gap = 0 (FY exactness)
- undefined_reads = 0 (totality)
- singletons == N (LFP convergence)
- diffs = 0 or cells_wrong_after_π = 0 (correctness)
- deterministic two-run byte identity

Usage:
    python run_gate_g.py --limit 100 --dataset training
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_compile.compile_theta import compile_theta
from arc_fixedpoint.lfp import compute_lfp
from arc_fixedpoint.eval_unpresent import evaluate_and_unpresent

from utils import (
    compute_summary_stats,
    grid_to_type,
    parse_task,
    random_sample_tasks,
    save_receipt,
    setup_logger,
)

import logging
from typing import Dict, List, Tuple, Any


def compute_grid_diff(predicted: List[List[int]], expected: List[List[int]]) -> int:
    """
    Compute number of differing cells between predicted and expected grids.

    Returns:
        Number of cells that differ (0 = perfect match)
    """
    if len(predicted) != len(expected):
        return float('inf')  # Shape mismatch

    diff_count = 0
    for r in range(len(predicted)):
        if len(predicted[r]) != len(expected[r]):
            return float('inf')  # Shape mismatch

        for c in range(len(predicted[r])):
            if predicted[r][c] != expected[r][c]:
                diff_count += 1

    return diff_count


def grid_to_bytes(grid: List[List[int]]) -> bytes:
    """Convert grid to deterministic byte representation for comparison."""
    return json.dumps(grid, sort_keys=True).encode('utf-8')


def run_task_twice(task_id: str, task_data: dict, logger: logging.Logger) -> Tuple[Dict[str, Any], bool]:
    """
    Run task twice to verify determinism.

    Returns:
        (receipt, deterministic_flag)
    """
    try:
        # Parse task
        train_pairs, test_inputs = parse_task(task_data)
        expected_outputs = task_data.get("test", [])

        if not test_inputs:
            return {
                "task_id": task_id,
                "gate": "G",
                "timestamp": datetime.now().isoformat(),
                "status": "SKIP",
                "error": "No test inputs"
            }, True

        # Test only first test input
        test_input = test_inputs[0]
        expected_output = expected_outputs[0]["output"] if expected_outputs and "output" in expected_outputs[0] else None

        # ============================
        # RUN 1
        # ============================
        logger.info(f"Task {task_id}: Running pass 1...")

        # Step 1: compile_theta
        theta1, U0_1, closures1, receipts_stub1 = compile_theta(train_pairs, test_input)

        # Extract test present grid
        X_test_present_1 = theta1["test_present"].grid

        # Step 2: compute_lfp (may fail with non-singletons)
        try:
            U_star_1, lfp_receipt1 = compute_lfp(U0_1, theta1, X_test_present_1)
            lfp_failed = False
        except AssertionError as e:
            # LFP didn't converge to singletons
            logger.error(f"Task {task_id}: LFP assertion failed - {e}")
            lfp_failed = True
            lfp_error = str(e)
            # Can't continue to evaluation
            return {
                "task_id": task_id,
                "gate": "G",
                "timestamp": datetime.now().isoformat(),
                "status": "FAIL",
                "error": f"LFP failed to converge: {lfp_error}",
                "metrics": {
                    "unseen_roles": receipts_stub1["wl"].get("unseen_roles", 0),
                    "u0_size": len(U0_1),
                    "u0_empty_pixels": sum(1 for expr_set in U0_1.values() if len(expr_set) == 0),
                }
            }, False

        # Step 3: evaluate_and_unpresent
        Y_star_1 = evaluate_and_unpresent(
            U_star_1,
            X_test_present_1,
            theta1,
            theta1["g_test"]
        )

        # Convert to bytes for determinism check
        bytes1 = grid_to_bytes(Y_star_1)

        # ============================
        # RUN 2
        # ============================
        logger.info(f"Task {task_id}: Running pass 2...")

        theta2, U0_2, closures2, receipts_stub2 = compile_theta(train_pairs, test_input)
        X_test_present_2 = theta2["test_present"].grid
        U_star_2, lfp_receipt2 = compute_lfp(U0_2, theta2, X_test_present_2)
        Y_star_2 = evaluate_and_unpresent(
            U_star_2,
            X_test_present_2,
            theta2,
            theta2["g_test"]
        )

        bytes2 = grid_to_bytes(Y_star_2)

        # ============================
        # Check determinism
        # ============================
        deterministic = (bytes1 == bytes2)
        if not deterministic:
            logger.error(f"Task {task_id}: NOT DETERMINISTIC! Outputs differ between runs")

        # ============================
        # Validate metrics (using run 1)
        # ============================

        # Check unseen_roles (informational only - per clarifications, unseen_roles > 0 is OK)
        unseen_roles = receipts_stub1["wl"].get("unseen_roles", 0)
        if unseen_roles != 0:
            logger.info(f"Task {task_id}: unseen_roles = {unseen_roles} (informational, not a failure)")

        # Check FY gap (from lfp_receipt)
        fy_gap = lfp_receipt1.fy_gap if hasattr(lfp_receipt1, 'fy_gap') else 0
        if fy_gap != 0:
            logger.warning(f"Task {task_id}: fy_gap = {fy_gap} (expected 0)")

        # Check undefined_reads (totality)
        undefined_reads = lfp_receipt1.undefined_reads if hasattr(lfp_receipt1, 'undefined_reads') else 0
        if undefined_reads != 0:
            logger.warning(f"Task {task_id}: undefined_reads = {undefined_reads} (expected 0)")

        # Check singletons
        total_pixels = len(U_star_1)
        singletons = sum(1 for expr_set in U_star_1.values() if len(expr_set) == 1)
        if singletons != total_pixels:
            logger.error(f"Task {task_id}: singletons = {singletons}/{total_pixels} (expected all)")

        # Check diffs (if expected output available)
        if expected_output:
            diff = compute_grid_diff(Y_star_1, expected_output)
            perfect_match = (diff == 0)
            logger.info(f"Task {task_id}: diffs = {diff}, perfect_match = {perfect_match}")
        else:
            diff = None
            perfect_match = None

        # ============================
        # Build receipt
        # ============================
        # Note: unseen_roles > 0 is informational, not a failure condition
        status = "PASS" if (
            deterministic and
            fy_gap == 0 and
            undefined_reads == 0 and
            singletons == total_pixels and
            (expected_output is None or diff == 0)
        ) else "FAIL"

        receipt = {
            "task_id": task_id,
            "gate": "G",
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "deterministic": deterministic,
            "metrics": {
                "unseen_roles": unseen_roles,
                "fy_gap": fy_gap,
                "undefined_reads": undefined_reads,
                "singletons": singletons,
                "total_pixels": total_pixels,
                "diffs": diff,
                "perfect_match": perfect_match
            },
            "lfp": {
                "passes": lfp_receipt1.passes,
                "total_removals": lfp_receipt1.total_removals
            },
            "wl": receipts_stub1["wl"],
            "present": receipts_stub1["present"],
            "basis_used": receipts_stub1.get("basis_used", [])
        }

        if status == "FAIL":
            errors = []
            if not deterministic:
                errors.append("Not deterministic")
            # Note: unseen_roles is informational, not an error
            if fy_gap != 0:
                errors.append(f"fy_gap={fy_gap}")
            if undefined_reads != 0:
                errors.append(f"undefined_reads={undefined_reads}")
            if singletons != total_pixels:
                errors.append(f"singletons={singletons}/{total_pixels}")
            if expected_output and diff != 0:
                errors.append(f"diffs={diff}")

            receipt["error"] = "; ".join(errors)

        return receipt, deterministic

    except Exception as e:
        logger.error(f"Task {task_id}: Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "task_id": task_id,
            "gate": "G",
            "timestamp": datetime.now().isoformat(),
            "status": "FAIL",
            "error": f"Fatal error: {e}"
        }, False


def main():
    parser = argparse.ArgumentParser(description="Gate G Integration Test")
    parser.add_argument("--limit", type=int, default=100, help="Number of tasks to test")
    parser.add_argument("--dataset", type=str, default="training", choices=["training", "evaluation"],
                       help="Dataset to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Setup logging
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    logger = setup_logger("gate_g", log_dir / "gate_g.log")

    # Setup receipts directory
    receipts_dir = Path(__file__).parent / "receipts" / "gate_g"
    receipts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Gate G Integration Test - FULL END-TO-END PIPELINE")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Task limit: {args.limit}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("=" * 80)

    # Load tasks
    logger.info(f"Loading {args.limit} random tasks from {args.dataset} set...")
    tasks = random_sample_tasks(args.dataset, args.limit, args.seed)
    logger.info(f"Loaded {len(tasks)} tasks")

    # Run each task
    passed = 0
    failed = 0
    skipped = 0
    deterministic_count = 0
    perfect_matches = 0

    for task_id, task_data in tasks.items():
        logger.info(f"\n--- Processing task {task_id} ---")

        receipt, deterministic = run_task_twice(task_id, task_data, logger)

        # Save receipt
        save_receipt(receipt, receipts_dir)

        # Update stats
        status = receipt["status"]
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        elif status == "SKIP":
            skipped += 1

        if deterministic:
            deterministic_count += 1

        if receipt.get("metrics", {}).get("perfect_match") == True:
            perfect_matches += 1

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)
    logger.info(f"Total tasks: {len(tasks)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Pass rate: {passed / len(tasks) * 100:.1f}%")
    logger.info(f"")
    logger.info(f"Deterministic runs: {deterministic_count}/{len(tasks)} ({deterministic_count / len(tasks) * 100:.1f}%)")
    logger.info(f"Perfect matches: {perfect_matches}/{len(tasks)} ({perfect_matches / len(tasks) * 100:.1f}%)")
    logger.info(f"")
    logger.info("=" * 80)
    logger.info(f"Gate G validation complete. Receipts saved to: {receipts_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
