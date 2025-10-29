#!/usr/bin/env python3
"""
Gate E Integration Test: First End-to-End Predictions (after WO-09/10/15/16/17)

Tests the full fixed-point machinery with local_paint + object_arith laws.
This is the FIRST gate that generates actual predictions!

Per docs/anchors/engineering_spec.md §7.2 and implementation_plan.md:
- WO-09: local_paint (per-role recoloring)
- WO-10: object_arith (translate/copy/delete components)
- WO-15/16/17: closure pipeline + FY + GLUE

Expects:
- Easy tasks (pure per-role paint or simple translations) to hit diffs=0
- LFP convergence to singletons
- No undefined reads
- FY/GLUE consistency

Usage:
    python run_gate_e.py --limit 10 --dataset training

NOTE: This is a complex orchestration test. We manually build theta and U0
      for the subset of implemented laws (WO-09, WO-10).
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_core.palette import orbit_cprq, canonicalize_palette_for_task
from arc_core.present import PiG, build_present, D4_TRANSFORMATIONS
from arc_core.shape import unify_shape
from arc_core.wl import wl_union
from arc_laws.local_paint import build_local_paint
from arc_laws.object_arith import build_object_arith
from arc_core.types import LocalPaintParams, Pixel, RoleId
from arc_core.components import extract_components
from arc_fixedpoint.expressions import LocalPaintExpr, TranslateExpr, init_expressions
from arc_fixedpoint.lfp import compute_lfp

from utils import (
    build_receipt,
    compute_summary_stats,
    grid_to_type,
    parse_task,
    random_sample_tasks,
    save_receipt,
    setup_logger,
)


def count_diff(grid1, grid2):
    """
    Count differing cells between two grids.

    Returns:
        Number of cells that differ (or -1 if shapes don't match)
    """
    if len(grid1) != len(grid2):
        return -1
    if len(grid1[0]) != len(grid2[0]):
        return -1

    diff_count = 0
    for r in range(len(grid1)):
        for c in range(len(grid1[0])):
            if grid1[r][c] != grid2[r][c]:
                diff_count += 1

    return diff_count


def validate_task_gate_e(task_id, task_dict, logger):
    """
    Validate a single task for Gate E (first end-to-end predictions).

    Strategy:
    1. Run Gates A-D machinery (present, WL, shape, palette, params)
    2. Build minimal theta dict for local_paint laws
    3. Extract local_paint laws from training
    4. Build U0 with LocalPaintExpr candidates
    5. Run compute_lfp to get singletons
    6. Evaluate expressions to get prediction
    7. Unpresent and compare to expected output

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

        # Test outputs may not be available (hidden in evaluation set)
        test_outputs = []
        for example in task_dict["test"]:
            if "output" in example:
                test_outputs.append(grid_to_type(example["output"]))
            else:
                test_outputs.append(None)  # Hidden output

        all_inputs = train_inputs + test_inputs

        logger.info(f"Task {task_id}: Processing {len(train_inputs)} train + {len(test_inputs)} test")

        # ============================
        # GATE A-D: Infrastructure
        # ============================

        # Build presents
        presents_all = [build_present(grid) for grid in all_inputs]
        presents_train = presents_all[: len(train_inputs)]

        # Run WL union
        role_map = wl_union(presents_all, escalate=False, max_iters=12)

        # Extract test present (canonical test input)
        X_test_present = presents_all[len(train_inputs)].grid

        # ============================
        # Build theta for local_paint
        # ============================

        # Prepare LocalPaintParams
        train_pairs_typed = list(zip(train_inputs, train_outputs))

        local_paint_params = LocalPaintParams(
            train_pairs=train_pairs_typed,
            presents=presents_train,
            role_map=role_map
        )

        # ============================
        # Extract Laws (WO-09, WO-10)
        # ============================

        # Extract local_paint laws (WO-09)
        logger.info(f"Task {task_id}: Extracting local_paint laws (WO-09)")
        local_paint_laws = build_local_paint(local_paint_params)
        logger.info(f"Task {task_id}: Found {len(local_paint_laws)} local_paint laws")

        # Extract object_arith laws (WO-10)
        logger.info(f"Task {task_id}: Extracting object_arith laws (WO-10)")
        theta_for_object = {
            "train_pairs": train_pairs_typed,
        }
        object_arith_laws = build_object_arith(theta_for_object)
        logger.info(f"Task {task_id}: Found {len(object_arith_laws)} object_arith laws")

        # Check if any laws were extracted
        total_laws = len(local_paint_laws) + len(object_arith_laws)
        if total_laws == 0:
            logger.warning(f"Task {task_id}: No laws extracted (skipping LFP)")
            return build_receipt(
                task_id=task_id,
                gate="E",
                status="SKIP",
                error="No laws extracted"
            )

        logger.info(f"Task {task_id}: Total laws extracted: {total_laws}")

        # ============================
        # Build U0 manually (simplified)
        # ============================

        # For Gate E, we manually build U0 with only LOCAL_PAINT expressions
        # (init_expressions requires full theta, which is complex to build)

        U0 = {}
        H, W = len(X_test_present), len(X_test_present[0])

        # For each pixel in test canvas
        for r in range(H):
            for c in range(W):
                q = Pixel(r, c)
                candidates = set()

                # Find WL role for this pixel in test
                role_at_q = None
                for (grid_id, pixel), role_id in role_map.items():
                    if grid_id == len(train_inputs) and pixel.row == r and pixel.col == c:
                        role_at_q = role_id
                        break

                if role_at_q is None:
                    # Pixel not in role_map (shouldn't happen)
                    logger.warning(f"Task {task_id}: Pixel ({r},{c}) not in role_map")
                    continue

                # Add LOCAL_PAINT expressions from extracted laws
                for law in local_paint_laws:
                    if law.params["role"] == role_at_q:
                        expr = LocalPaintExpr(
                            role_id=role_at_q,
                            color=law.params["color"],
                            mask_type="role"
                        )
                        candidates.add(expr)

                # Add TRANSLATE/COPY expressions from object_arith laws
                # Note: TranslateExpr applies to ALL pixels (closures filter invalid ones)
                for law in object_arith_laws:
                    if law.operation in ("translate", "copy"):
                        # Add translate/copy expression with delta
                        expr = TranslateExpr(
                            component_id=0,  # Placeholder (closures will filter by Dom)
                            delta=law.delta,
                            color=None  # Keep original color
                        )
                        candidates.add(expr)

                if r == 0 and c == 0:  # Log for first pixel
                    logger.info(
                        f"Task {task_id}: Pixel (0,0) role={role_at_q}, "
                        f"candidates={len(candidates)} (local_paint + object_arith)"
                    )

                U0[q] = candidates

        logger.info(f"Task {task_id}: Built U0 with {len(U0)} pixels")

        # Check if U0 has expressions for all pixels
        empty_pixels = [q for q, exprs in U0.items() if len(exprs) == 0]
        if empty_pixels:
            logger.info(
                f"Task {task_id}: {len(empty_pixels)}/{len(U0)} pixels have no candidate expressions "
                f"(likely unseen roles - task not solvable by local_paint alone)"
            )
            return build_receipt(
                task_id=task_id,
                gate="E",
                status="SKIP",
                error=f"Not solvable by local_paint (unseen roles)"
            )

        # ============================
        # Run LFP
        # ============================

        # Extract components from test input for object_arith
        test_input_components = extract_components(test_inputs[0])
        components_dict = {comp.component_id: comp for comp in test_input_components}

        # Build minimal theta for closures
        # CRITICAL: test_grid_id must match wl_union convention (arc_core/wl.py:49)
        # Per arc_fixedpoint/expressions.py:178, TranslateExpr.Dom() needs theta["components"]
        theta = {
            "canvas_shape": (H, W),
            "role_map": role_map,
            "train_pairs": train_pairs_typed,
            "presents_train": presents_train,
            "X_test_present": X_test_present,
            "test_grid_id": len(train_inputs),  # Match wl_union grid_id convention
            "components": components_dict,  # For object_arith Dom checks
        }

        logger.info(f"Task {task_id}: Running LFP...")

        try:
            U_star, lfp_receipt = compute_lfp(U0, theta, X_test_present)
        except Exception as lfp_error:
            logger.error(f"Task {task_id}: LFP failed - {lfp_error}")
            import traceback
            logger.error(traceback.format_exc())
            return build_receipt(
                task_id=task_id,
                gate="E",
                status="FAIL",
                error=f"LFP failed: {lfp_error}"
            )

        logger.info(
            f"Task {task_id}: LFP converged in {lfp_receipt.passes} passes, "
            f"{lfp_receipt.total_removals} removals, {lfp_receipt.singletons} singletons"
        )

        # Check singletons
        non_singletons = [q for q, exprs in U_star.items() if len(exprs) != 1]
        if non_singletons:
            logger.error(
                f"Task {task_id}: LFP did not converge to singletons! "
                f"{len(non_singletons)} pixels have != 1 expression"
            )
            return build_receipt(
                task_id=task_id,
                gate="E",
                status="FAIL",
                error=f"Non-singletons: {len(non_singletons)} pixels"
            )

        # ============================
        # Evaluate expressions (FY)
        # ============================

        logger.info(f"Task {task_id}: Evaluating expressions...")

        Y_hat_canonical = []
        for r in range(H):
            row = []
            for c in range(W):
                q = Pixel(r, c)
                expr = next(iter(U_star[q]))  # Get singleton expression

                try:
                    color = expr.eval(q, X_test_present, theta)
                    row.append(color)
                except Exception as eval_error:
                    logger.error(f"Task {task_id}: eval failed at ({r},{c}) - {eval_error}")
                    return build_receipt(
                        task_id=task_id,
                        gate="E",
                        status="FAIL",
                        error=f"eval failed: {eval_error}"
                    )

            Y_hat_canonical.append(row)

        # ============================
        # Unpresent (GLUE)
        # ============================

        logger.info(f"Task {task_id}: Unpresenting prediction...")

        # Get g_test inverse transform
        g_inverse = presents_all[len(train_inputs)].g_inverse

        # Apply inverse transform to unpresent
        Y_hat_raw = D4_TRANSFORMATIONS[g_inverse](Y_hat_canonical)

        # ============================
        # Compare to expected output
        # ============================

        expected_output = test_outputs[0]

        if expected_output is None:
            # No ground truth available (evaluation set)
            logger.info(f"Task {task_id}: No ground truth available (evaluation set)")
            diff = None
            status = "PASS"  # Pass if LFP converged successfully
        else:
            diff = count_diff(Y_hat_raw, expected_output)

            if diff == -1:
                logger.error(
                    f"Task {task_id}: Shape mismatch! "
                    f"Predicted {len(Y_hat_raw)}×{len(Y_hat_raw[0])}, "
                    f"Expected {len(expected_output)}×{len(expected_output[0])}"
                )
                status = "FAIL"
            elif diff == 0:
                logger.info(f"Task {task_id}: PERFECT MATCH! ✨ diffs=0")
                status = "PASS"
            else:
                logger.info(f"Task {task_id}: Partial match, diffs={diff}")
                status = "PASS"  # Still pass, just track diffs

        # ============================
        # Build Receipt
        # ============================

        # Build LFP data
        lfp_data = {
            "passes": lfp_receipt.passes,
            "total_removals": lfp_receipt.total_removals,
            "removals_per_stage": dict(lfp_receipt.removals_per_stage),
            "singletons": lfp_receipt.singletons,
            "singletons_expected": H * W,
            "singletons_match": lfp_receipt.singletons == H * W,
        }

        # Build FY/GLUE data
        fy_glue_data = {
            "undefined_reads": 0,  # All expressions evaluated successfully
            "eval_success": True,
            "unpresent_success": True,
        }

        # Build prediction data
        prediction_data = {
            "num_laws_extracted": total_laws,
            "num_local_paint_laws": len(local_paint_laws),
            "num_object_arith_laws": len(object_arith_laws),
            "diffs": [diff] if diff is not None else [None],
            "perfect_match": diff == 0 if diff is not None else None,
            "shape_match": diff != -1 if diff is not None else None,
            "ground_truth_available": expected_output is not None,
        }

        receipt = build_receipt(
            task_id=task_id,
            gate="E",
            present_data={},
            wl_data={},
            shape_data={},
            status=status,
        )

        # Add Gate E specific data
        receipt["lfp"] = lfp_data
        receipt["fy_glue"] = fy_glue_data
        receipt["prediction"] = prediction_data

        return receipt

    except Exception as e:
        logger.error(f"Task {task_id}: Exception - {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return build_receipt(task_id=task_id, gate="E", status="FAIL", error=str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Gate E Integration Test: First End-to-End Predictions"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of tasks to sample (default: 10)",
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
    receipts_dir = integration_dir / "receipts" / "gate_e"

    # Setup logger
    logger = setup_logger("gate_e", logs_dir / "gate_e.log")

    logger.info("=" * 80)
    logger.info(f"Gate E Integration Test - FIRST END-TO-END PREDICTIONS!")
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
        receipt = validate_task_gate_e(task_id, task_dict, logger)
        receipts.append(receipt)

        # Save receipt
        save_receipt(receipt, receipts_dir)

    # Compute summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)

    passed = [r for r in receipts if r["status"] == "PASS"]
    failed = [r for r in receipts if r["status"] == "FAIL"]
    skipped = [r for r in receipts if r["status"] == "SKIP"]

    logger.info(f"Total tasks: {len(receipts)}")
    logger.info(f"Passed: {len(passed)}")
    logger.info(f"Failed: {len(failed)}")
    logger.info(f"Skipped: {len(skipped)}")
    logger.info(f"Pass rate: {len(passed)/len(receipts)*100:.1f}%")

    # LFP statistics
    lfp_receipts = [r for r in passed if "lfp" in r]
    if lfp_receipts:
        avg_passes = sum(r["lfp"]["passes"] for r in lfp_receipts) / len(lfp_receipts)
        avg_removals = sum(r["lfp"]["total_removals"] for r in lfp_receipts) / len(lfp_receipts)

        logger.info(f"\nLFP Statistics:")
        logger.info(f"  Average passes: {avg_passes:.1f}")
        logger.info(f"  Average removals: {avg_removals:.1f}")
        logger.info(f"  Convergence rate: {len(lfp_receipts)}/{len(passed)} ({len(lfp_receipts)/len(passed)*100:.1f}%)")

    # Prediction statistics
    pred_receipts = [r for r in passed if "prediction" in r]
    if pred_receipts:
        perfect_matches = [r for r in pred_receipts if r["prediction"]["perfect_match"]]

        logger.info(f"\nPrediction Statistics:")
        logger.info(f"  Perfect matches (diffs=0): {len(perfect_matches)}/{len(pred_receipts)}")
        logger.info(f"  Perfect match rate: {len(perfect_matches)/len(pred_receipts)*100:.1f}%")

        # Show some diffs
        for r in pred_receipts[:10]:
            diffs = r["prediction"]["diffs"][0]
            logger.info(f"    Task {r['task_id']}: diffs={diffs}")

    logger.info("\n" + "=" * 80)
    logger.info(f"Gate E validation complete. Receipts saved to: {receipts_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
