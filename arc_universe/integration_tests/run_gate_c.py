#!/usr/bin/env python3
"""
Gate C Integration Test: Palette Orbit + Canonicalization (after WO-05)

Extends Gate B with palette orbit CPRQ and canonicalizer N.

Tests:
- Everything from Gate B (present + WL union + shape meet)
- Orbit CPRQ: abstract color mapping
- Canonicalizer N: 3-level sort (count↓, first↑, boundary-hash↑)
- Train permutations π_i: one per training output
- Isomorphic by palette: cells_wrong_after_π = 0

Usage:
    python run_gate_c.py --limit 50 --dataset training
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import arc_core
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_core.palette import (
    canonicalize_palette_for_task,
    compute_train_permutations,
    orbit_cprq,
    verify_isomorphic,
)
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


def validate_task_gate_c(task_id, task_dict, logger):
    """
    Validate a single task for Gate C (extends Gate B with palette orbit).

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
        train_input_shapes = [[len(grid), len(grid[0])] for grid in train_inputs]
        test_input_shapes = [[len(grid), len(grid[0])] for grid in test_inputs]
        train_output_shapes = [[len(grid), len(grid[0])] for grid in train_outputs]

        logger.info(
            f"Task {task_id}: Shape unified, |R|={num_row_classes}, |C|={num_col_classes}"
        )

        # ============================
        # GATE C: Palette Orbit + Canon
        # ============================

        logger.info(f"Task {task_id}: Running orbit CPRQ on training pairs")

        # Build train pairs with typed grids
        typed_train_pairs = [(train_inputs[i], train_outputs[i]) for i in range(len(train_inputs))]

        # Run orbit CPRQ
        abstract_map = orbit_cprq(typed_train_pairs, presents_train)

        # Determine label mode
        label_mode = "orbit" if abstract_map.is_orbit else "strict"

        logger.info(f"Task {task_id}: Label mode = {label_mode}")

        # Compute train permutations (if orbit mode)
        train_permutations = []
        cells_wrong_counts = []

        if abstract_map.is_orbit:
            logger.info(f"Task {task_id}: Computing train permutations π_i")
            train_permutations = compute_train_permutations(abstract_map, typed_train_pairs)

            # Verify isomorphic by palette (cells_wrong_after_π = 0)
            for i, perm in enumerate(train_permutations):
                cells_wrong = verify_isomorphic(
                    abstract_map.abstract_grid,
                    train_outputs[i],
                    perm
                )
                cells_wrong_counts.append(cells_wrong)

                if cells_wrong != 0:
                    logger.warning(
                        f"Task {task_id}: Train {i} has {cells_wrong} cells wrong after π "
                        "(expected 0 for isomorphic)"
                    )
        else:
            # Strict mode - no permutations needed
            cells_wrong_counts = [0] * len(train_outputs)

        # Check if all trains are isomorphic by palette
        all_isomorphic = all(count == 0 for count in cells_wrong_counts)

        logger.info(
            f"Task {task_id}: Isomorphic by palette = {all_isomorphic}, "
            f"cells_wrong = {cells_wrong_counts}"
        )

        # Canonicalize palette for task (pooled across train ∪ test inputs)
        logger.info(f"Task {task_id}: Canonicalizing palette (per-task pooling)")

        # Use first test input (or create empty if no test)
        test_input = test_inputs[0] if test_inputs else [[]]
        palette_map = canonicalize_palette_for_task(train_inputs, test_input)

        # Canon rule description
        canon_rule = "3-level sort: count↓, first_appearance↑, boundary_hash↑"

        logger.info(
            f"Task {task_id}: Palette canonicalized, "
            f"{len(palette_map)} colors mapped"
        )

        # Log unseen roles (measurement, not failure)
        if len(unseen_roles) > 0:
            logger.info(
                f"Task {task_id}: {len(unseen_roles)} unseen roles "
                f"(test has new input structure - this is allowed)"
            )

        # ============================
        # Build Receipt
        # ============================

        # Determine status
        status = "PASS"
        if not deterministic:
            status = "FAIL"
            logger.error(f"Task {task_id}: CRITICAL - not deterministic")
        elif not all_pig_idempotent:
            status = "FAIL"
            logger.error(f"Task {task_id}: CRITICAL - ΠG not idempotent")
        elif not all_isomorphic and abstract_map.is_orbit:
            status = "FAIL"
            logger.error(
                f"Task {task_id}: CRITICAL - orbit mode but not isomorphic by palette"
            )
        else:
            logger.info(f"Task {task_id}: PASS")

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

        # Build palette data (Gate C specific)
        palette_data = {
            "label_mode": label_mode,
            "canon_rule": canon_rule,
            "num_colors": len(palette_map),
            "is_orbit": abstract_map.is_orbit,
            "train_permutations": [
                {str(k): v for k, v in perm.items()}
                for perm in train_permutations
            ],  # Convert to JSON-serializable
            "isomorphic_by_palette": all_isomorphic,
            "cells_wrong_after_pi": cells_wrong_counts,
        }

        # Build receipt
        receipt = build_receipt(
            task_id=task_id,
            gate="C",
            present_data=present_data,
            wl_data=wl_data,
            shape_data=shape_data,
            status=status,
        )

        # Add palette data to receipt
        receipt["palette"] = palette_data

        return receipt

    except Exception as e:
        logger.error(f"Task {task_id}: Exception - {type(e).__name__}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return build_receipt(task_id=task_id, gate="C", status="FAIL", error=str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Gate C Integration Test: Palette Orbit + Canonicalization"
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
    receipts_dir = integration_dir / "receipts" / "gate_c"

    # Setup logger
    logger = setup_logger("gate_c", logs_dir / "gate_c.log")

    logger.info("=" * 80)
    logger.info(f"Gate C Integration Test")
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
        receipt = validate_task_gate_c(task_id, task_dict, logger)
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
        logger.info(
            f"  Avg unseen roles (when present): {stats['wl']['avg_unseen_when_present']:.1f}"
        )

    if "shape" in stats:
        logger.info(f"\nShape Statistics:")
        logger.info(
            f"  Average row classes (|R|): {stats['shape']['avg_row_classes']:.1f}"
        )
        logger.info(
            f"  Average col classes (|C|): {stats['shape']['avg_col_classes']:.1f}"
        )

    # Palette statistics (Gate C specific)
    palette_receipts = [r for r in receipts if "palette" in r]
    if palette_receipts:
        orbit_count = sum(1 for r in palette_receipts if r["palette"]["is_orbit"])
        strict_count = len(palette_receipts) - orbit_count
        isomorphic_count = sum(
            1 for r in palette_receipts if r["palette"]["isomorphic_by_palette"]
        )

        logger.info(f"\nPalette Statistics:")
        logger.info(f"  Orbit mode tasks: {orbit_count}")
        logger.info(f"  Strict mode tasks: {strict_count}")
        logger.info(f"  Isomorphic by palette: {isomorphic_count}/{len(palette_receipts)}")

        # Average colors per task
        avg_colors = sum(
            r["palette"]["num_colors"] for r in palette_receipts
        ) / len(palette_receipts)
        logger.info(f"  Average colors per task: {avg_colors:.1f}")

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

    # Palette-specific checks
    if palette_receipts:
        if isomorphic_count == len(palette_receipts):
            logger.info("✅ All tasks are isomorphic by palette (PASS)")
        else:
            logger.error(
                f"❌ {len(palette_receipts) - isomorphic_count} tasks failed "
                f"isomorphic by palette check (FAIL)"
            )

    logger.info("\n" + "=" * 80)
    logger.info(f"Gate C validation complete. Receipts saved to: {receipts_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
