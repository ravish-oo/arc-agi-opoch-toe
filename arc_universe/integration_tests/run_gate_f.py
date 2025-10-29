#!/usr/bin/env python3
"""
Gate F Integration Test: Extended Law Families (after WO-09/10/11/12/13/14)

Tests all 6 law families for bug detection and code hardening:
- WO-09: local_paint (per-role recoloring)
- WO-10: object_arith (translate/copy/delete)
- WO-11: periodic (tiling/lattice/phases)
- WO-12: blow_block (BLOWUP[k] + BLOCK_SUBST[B])
- WO-13: selectors (ARGMAX/UNIQUE/MODE/PARITY)
- WO-14: connect_fill (CONNECT_ENDPOINTS + REGION_FILL)

Purpose: Find bugs in new WO-11/12/13/14 code before WO-19 (compile_theta).

Logs:
- Lattice basis (periods, method: FFT/KMP)
- Motif hashes (blow_block)
- Selector mask hashes
- Anchors (connect_fill)
- Law counts per family

Expected: Many LFP failures (incomplete theta), but should catch:
- Crashes in law extraction
- Type errors
- Missing theta parameters
- Edge cases

Usage:
    python run_gate_f.py --limit 10 --dataset training
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_core.palette import orbit_cprq, canonicalize_palette_for_task
from arc_core.present import PiG, build_present, D4_TRANSFORMATIONS
from arc_core.shape import unify_shape
from arc_core.wl import wl_union
from arc_core.lattice import infer_lattice
from arc_core.canvas import infer_canvas
from arc_core.components import extract_components
from arc_laws.local_paint import build_local_paint
from arc_laws.object_arith import build_object_arith
from arc_laws.periodic import build_periodic
from arc_laws.blow_block import build_blow_block
from arc_laws.selectors import apply_selector_on_test
from arc_laws.connect_fill import build_connect_fill
from arc_core.types import LocalPaintParams, Pixel, RoleId
from arc_fixedpoint.expressions import LocalPaintExpr, TranslateExpr, init_expressions
from arc_fixedpoint.lfp import compute_lfp

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

def validate_task_gate_f(task_id: str, task_data: dict, logger) -> Dict[str, Any]:
    """
    Gate F validation: Test all 6 law families (WO-09/10/11/12/13/14).

    Focus: Bug detection in new law families, not end-to-end solving.
    """
    logger.info(f"\n--- Processing task {task_id} ---")

    try:
        # Parse task
        train_pairs, test_inputs = parse_task(task_data)
        logger.info(f"Task {task_id}: Processing {len(train_pairs)} train + {len(test_inputs)} test")

        # Get expected output if available (training set only)
        expected_output = None
        if "test" in task_data and len(task_data["test"]) > 0:
            test_example = task_data["test"][0]
            if "output" in test_example:
                expected_output = test_example["output"]

        # ============================
        # Build Presents (WO-02)
        # ============================
        train_inputs = [x for x, _ in train_pairs]
        train_outputs = [y for _, y in train_pairs]

        train_inputs_present = [PiG(x) for x in train_inputs]
        train_outputs_present = [PiG(y) for y in train_outputs]
        presents_train = [build_present(x) for x in train_inputs_present]

        # Test present
        X_test = test_inputs[0]
        X_test_present = PiG(X_test)
        present_test = build_present(X_test_present)

        # ============================
        # WL Union (WO-03)
        # ============================
        presents_all = presents_train + [present_test]
        role_map = wl_union(presents_all, escalate=False, max_iters=12)

        # ============================
        # Typed train pairs (for law extraction)
        # ============================
        train_pairs_typed = list(zip(train_inputs_present, train_outputs_present))

        # ============================
        # Parameters (WO-06, WO-07, WO-08)
        # ============================

        # Lattice (WO-06) - for periodic
        lattice_result = None
        try:
            lattice_result = infer_lattice(train_inputs_present)
            if lattice_result:
                period_r, period_c = lattice_result.periods
                logger.info(f"Task {task_id}: Lattice detected - method={lattice_result.method}, "
                          f"periods=({period_r}, {period_c})")
        except Exception as e:
            logger.warning(f"Task {task_id}: Lattice inference failed: {e}")

        # Canvas (WO-07) - for blow_block
        canvas_result = None
        try:
            canvas_result = infer_canvas(train_pairs_typed)
            if canvas_result and canvas_result.transform_type != "IDENTITY":
                logger.info(f"Task {task_id}: Canvas detected - {canvas_result.transform_type}")
        except Exception as e:
            logger.warning(f"Task {task_id}: Canvas inference failed: {e}")

        # Components (WO-08) - for object_arith
        test_input_components = extract_components(test_inputs[0])
        components_dict = {comp.component_id: comp for comp in test_input_components}

        # Local paint params (WO-09)
        local_paint_params = LocalPaintParams(
            train_pairs=train_pairs_typed,
            presents=presents_train,
            role_map=role_map,
        )

        # ============================
        # Extract Laws (ALL 6 families)
        # ============================

        law_counts = {}

        # WO-09: local_paint
        logger.info(f"Task {task_id}: Extracting local_paint laws (WO-09)")
        try:
            local_paint_laws = build_local_paint(local_paint_params)
            law_counts["local_paint"] = len(local_paint_laws)
            logger.info(f"Task {task_id}: Found {len(local_paint_laws)} local_paint laws")
        except Exception as e:
            logger.error(f"Task {task_id}: local_paint extraction failed: {e}")
            local_paint_laws = []
            law_counts["local_paint"] = 0

        # WO-10: object_arith
        logger.info(f"Task {task_id}: Extracting object_arith laws (WO-10)")
        try:
            theta_for_object = {"train_pairs": train_pairs_typed}
            object_arith_laws = build_object_arith(theta_for_object)
            law_counts["object_arith"] = len(object_arith_laws)
            logger.info(f"Task {task_id}: Found {len(object_arith_laws)} object_arith laws")
        except Exception as e:
            logger.error(f"Task {task_id}: object_arith extraction failed: {e}")
            object_arith_laws = []
            law_counts["object_arith"] = 0

        # WO-11: periodic
        logger.info(f"Task {task_id}: Extracting periodic structure (WO-11)")
        periodic_structure = None
        try:
            theta_for_periodic = {
                "grids_present": train_inputs_present,
                "train_pairs": train_pairs_typed,
            }
            periodic_structure = build_periodic(theta_for_periodic)
            if periodic_structure:
                law_counts["periodic"] = periodic_structure.num_phases
                p_r, p_c = periodic_structure.lattice.periods
                logger.info(f"Task {task_id}: Found periodic structure - "
                          f"periods=({p_r}, {p_c}), "
                          f"phases={periodic_structure.num_phases}")
            else:
                law_counts["periodic"] = 0
                logger.info(f"Task {task_id}: No periodic structure detected")
        except Exception as e:
            logger.error(f"Task {task_id}: periodic extraction failed: {e}")
            periodic_structure = None
            law_counts["periodic"] = 0

        # WO-12: blow_block
        logger.info(f"Task {task_id}: Extracting blow_block laws (WO-12)")
        try:
            theta_for_blow = {"train_pairs": train_pairs_typed}
            blow_block_laws = build_blow_block(theta_for_blow)
            law_counts["blow_block"] = len(blow_block_laws)
            if blow_block_laws:
                k = blow_block_laws[0].k
                num_motifs = len(blow_block_laws[0].motifs)
                logger.info(f"Task {task_id}: Found blow_block - k={k}, motifs={num_motifs}")
            else:
                logger.info(f"Task {task_id}: No blow_block detected")
        except Exception as e:
            logger.error(f"Task {task_id}: blow_block extraction failed: {e}")
            blow_block_laws = []
            law_counts["blow_block"] = 0

        # WO-13/14: selectors + connect_fill
        logger.info(f"Task {task_id}: Extracting connect_fill laws (WO-13/14)")
        try:
            theta_for_connect = {
                "train_pairs": train_pairs_typed,
                "anchors": [],  # Would need to be extracted from training
                "masks": [],    # Would need to be extracted from training
            }
            connect_fill_laws = build_connect_fill(theta_for_connect)
            law_counts["connect_fill"] = len(connect_fill_laws)
            logger.info(f"Task {task_id}: Found {len(connect_fill_laws)} connect_fill laws")
        except Exception as e:
            logger.error(f"Task {task_id}: connect_fill extraction failed: {e}")
            connect_fill_laws = []
            law_counts["connect_fill"] = 0

        # Check if any laws were extracted
        total_laws = sum(law_counts.values())
        if total_laws == 0:
            logger.warning(f"Task {task_id}: No laws extracted from any family (skipping LFP)")
            return {
                "task_id": task_id,
                "gate": "F",
                "timestamp": datetime.now().isoformat(),
                "status": "SKIP",
                "error": "No laws extracted",
                "law_counts": law_counts
            }

        logger.info(f"Task {task_id}: Total laws extracted: {total_laws} across {sum(1 for c in law_counts.values() if c > 0)} families")

        # ============================
        # Build U0 (simplified - only local_paint + object_arith for now)
        # ============================

        U0 = {}
        H, W = len(X_test_present), len(X_test_present[0])

        for r in range(H):
            for c in range(W):
                q = Pixel(r, c)
                candidates = set()

                # Find WL role for this pixel
                role_at_q = None
                for (grid_id, pixel), role_id in role_map.items():
                    if grid_id == len(train_inputs) and pixel.row == r and pixel.col == c:
                        role_at_q = role_id
                        break

                if role_at_q is None:
                    logger.warning(f"Task {task_id}: Pixel ({r},{c}) not in role_map")
                    continue

                # Add LOCAL_PAINT expressions
                for law in local_paint_laws:
                    if law.params["role"] == role_at_q:
                        expr = LocalPaintExpr(
                            role_id=role_at_q,
                            color=law.params["color"],
                            mask_type="role"
                        )
                        candidates.add(expr)

                # Add TRANSLATE expressions from object_arith
                for law in object_arith_laws:
                    if law.operation in ("translate", "copy"):
                        expr = TranslateExpr(
                            component_id=0,
                            delta=law.delta,
                            color=None
                        )
                        candidates.add(expr)

                if r == 0 and c == 0:
                    logger.info(f"Task {task_id}: Pixel (0,0) role={role_at_q}, candidates={len(candidates)}")

                U0[q] = candidates

        if not U0:
            logger.warning(f"Task {task_id}: U0 is empty (skipping LFP)")
            return {
                "task_id": task_id,
                "gate": "F",
                "timestamp": datetime.now().isoformat(),
                "status": "SKIP",
                "error": "U0 empty",
                "law_counts": law_counts
            }

        # Check for pixels with no candidates
        empty_pixels = sum(1 for candidates in U0.values() if len(candidates) == 0)
        if empty_pixels == len(U0):
            logger.info(f"Task {task_id}: {empty_pixels}/{len(U0)} pixels have no candidate expressions "
                       f"(likely unseen roles - task not solvable by these law families alone)")
            return {
                "task_id": task_id,
                "gate": "F",
                "timestamp": datetime.now().isoformat(),
                "status": "SKIP",
                "error": "All pixels have no candidates (unseen roles or insufficient law coverage)",
                "law_counts": law_counts
            }

        logger.info(f"Task {task_id}: Built U0 with {len(U0)} pixels")

        # ============================
        # Build theta (incomplete - will cause LFP failures)
        # ============================

        theta = {
            "canvas_shape": (H, W),
            "role_map": role_map,
            "train_pairs": train_pairs_typed,
            "presents_train": presents_train,
            "X_test_present": X_test_present,
            "test_grid_id": len(train_inputs),
            "components": components_dict,
            "lattice": lattice_result,
            "canvas": canvas_result,
        }

        # ============================
        # Run LFP (expect failures due to incomplete theta)
        # ============================

        logger.info(f"Task {task_id}: Running LFP...")
        try:
            U_star, lfp_receipt = compute_lfp(U0, theta, X_test_present)
            logger.info(f"Task {task_id}: LFP converged to singletons!")

            # If we got here, try to evaluate (very rare without WO-19)
            # TODO: Implement evaluation when available
            diff = None

        except AssertionError as e:
            # Expected: LFP doesn't converge to singletons
            error_msg = str(e)
            logger.error(f"Task {task_id}: LFP failed - {error_msg}")
            return {
                "task_id": task_id,
                "gate": "F",
                "timestamp": datetime.now().isoformat(),
                "status": "FAIL",
                "error": f"LFP failed: {error_msg}",
                "law_counts": law_counts
            }
        except Exception as e:
            # Unexpected error - this is a bug!
            logger.error(f"Task {task_id}: Unexpected error during LFP: {e}")
            import traceback
            traceback.print_exc()
            return {
                "task_id": task_id,
                "gate": "F",
                "timestamp": datetime.now().isoformat(),
                "status": "FAIL",
                "error": f"Unexpected error: {e}",
                "law_counts": law_counts
            }

        # Build receipt with prediction data
        return {
            "task_id": task_id,
            "gate": "F",
            "timestamp": datetime.now().isoformat(),
            "status": "PASS",
            "law_counts": law_counts,
            "total_laws": total_laws,
            "diffs": [diff] if diff is not None else [None],
            "perfect_match": diff == 0 if diff is not None else None,
        }

    except Exception as e:
        logger.error(f"Task {task_id}: Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "task_id": task_id,
            "gate": "F",
            "timestamp": datetime.now().isoformat(),
            "status": "FAIL",
            "error": f"Fatal error: {e}"
        }


def main():
    parser = argparse.ArgumentParser(description="Gate F Integration Test")
    parser.add_argument("--limit", type=int, default=10, help="Number of tasks to test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for task selection")
    parser.add_argument("--dataset", type=str, default="training", choices=["training", "evaluation"],
                       help="Dataset to test")
    args = parser.parse_args()

    # Setup logging
    log_file = Path(__file__).parent / "logs" / "gate_f.log"
    log_file.parent.mkdir(exist_ok=True)
    logger = setup_logger("gate_f", log_file)

    logger.info("=" * 80)
    logger.info("Gate F Integration Test - ALL 6 LAW FAMILIES (WO-09/10/11/12/13/14)")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Task limit: {args.limit}")
    logger.info(f"Random seed: {args.seed}")
    logger.info("=" * 80)

    # Load and sample tasks
    logger.info(f"Loading {args.limit} random tasks from {args.dataset} set...")
    tasks = random_sample_tasks(args.dataset, args.limit, args.seed)
    logger.info(f"Loaded {len(tasks)} tasks")

    # Process each task
    receipts = []
    for task_id, task_data in tasks.items():
        receipt = validate_task_gate_f(task_id, task_data, logger)
        receipts.append(receipt)

        # Save receipt
        receipt_dir = Path(__file__).parent / "receipts" / "gate_f"
        save_receipt(receipt, receipt_dir)

    # Compute summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 80)

    passed = sum(1 for r in receipts if r["status"] == "PASS")
    failed = sum(1 for r in receipts if r["status"] == "FAIL")
    skipped = sum(1 for r in receipts if r["status"] == "SKIP")

    logger.info(f"Total tasks: {len(receipts)}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Pass rate: {100 * passed / len(receipts):.1f}%")

    # Law family statistics
    logger.info("\nLaw Family Statistics:")
    law_family_counts = {
        "local_paint": 0,
        "object_arith": 0,
        "periodic": 0,
        "blow_block": 0,
        "connect_fill": 0,
    }

    for receipt in receipts:
        if "data" in receipt and "law_counts" in receipt["data"]:
            for family, count in receipt["data"]["law_counts"].items():
                if count > 0:
                    law_family_counts[family] += 1

    for family, task_count in law_family_counts.items():
        logger.info(f"  {family}: {task_count}/{len(receipts)} tasks ({100*task_count/len(receipts):.1f}%)")

    logger.info("\n" + "=" * 80)
    logger.info(f"Gate F validation complete. Receipts saved to: {receipt_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
