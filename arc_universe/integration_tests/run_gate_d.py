#!/usr/bin/env python3
"""
Gate D Integration Test: Parameter Extraction (after WO-06/07/08)

Tests parameter compilation for lattice, canvas, and components.
No predictions yet - just deterministic parameter extraction and forward verification.

Tests:
- Everything from Gate C (present + WL + shape + palette)
- Lattice: FFT/HNF + KMP fallback, deterministic basis
- Canvas: RESIZE/CONCAT/FRAME with exact forward verification
- Components: 8-CC + Hungarian matching with lex ties

Usage:
    python run_gate_d.py --limit 50 --dataset training
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import arc_core
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_core.canvas import infer_canvas
from arc_core.components import extract_components, match_components
from arc_core.lattice import infer_lattice
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


def validate_task_gate_d(task_id, task_dict, logger):
    """
    Validate a single task for Gate D (extends Gate C with parameter extraction).

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
        deterministic_wl = role_map_1 == role_map_2

        if not deterministic_wl:
            logger.warning(f"Task {task_id}: WL determinism check FAILED")

        # ============================
        # GATE B: Shape Unification
        # ============================

        logger.info(f"Task {task_id}: Running shape unification on training inputs")
        shape_params = unify_shape(presents_train)

        # Extract shape statistics
        num_row_classes = shape_params.num_row_classes
        num_col_classes = shape_params.num_col_classes

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
            train_permutations = compute_train_permutations(abstract_map, typed_train_pairs)

            # Verify isomorphic by palette
            for i, perm in enumerate(train_permutations):
                cells_wrong = verify_isomorphic(
                    abstract_map.abstract_grid,
                    train_outputs[i],
                    perm
                )
                cells_wrong_counts.append(cells_wrong)
        else:
            cells_wrong_counts = [0] * len(train_outputs)

        # Check if all trains are isomorphic by palette
        all_isomorphic = all(count == 0 for count in cells_wrong_counts)

        # Canonicalize palette for task
        test_input = test_inputs[0] if test_inputs else [[]]
        palette_map = canonicalize_palette_for_task(train_inputs, test_input)

        logger.info(
            f"Task {task_id}: Palette canonicalized, {len(palette_map)} colors"
        )

        # ============================
        # GATE D: Parameter Extraction
        # ============================

        # (1) Lattice extraction (WO-06)
        logger.info(f"Task {task_id}: Running lattice detection (WO-06)")

        lattice_result_1 = infer_lattice(train_inputs)
        lattice_result_2 = infer_lattice(train_inputs)  # Determinism check

        deterministic_lattice = (lattice_result_1 == lattice_result_2)

        if lattice_result_1 is not None:
            logger.info(
                f"Task {task_id}: Lattice detected via {lattice_result_1.method}, "
                f"periods={lattice_result_1.periods}"
            )
            lattice_data = {
                "detected": True,
                "basis": lattice_result_1.basis,
                "periods": lattice_result_1.periods,
                "method": lattice_result_1.method,
                "deterministic": deterministic_lattice,
            }
        else:
            logger.info(f"Task {task_id}: No lattice detected")
            lattice_data = {
                "detected": False,
                "deterministic": True,  # None == None is deterministic
            }

        # (2) Canvas extraction (WO-07)
        logger.info(f"Task {task_id}: Running canvas inference (WO-07)")

        canvas_result_1 = infer_canvas(typed_train_pairs)
        canvas_result_2 = infer_canvas(typed_train_pairs)  # Determinism check

        deterministic_canvas = (canvas_result_1 == canvas_result_2)

        if canvas_result_1 is not None:
            logger.info(
                f"Task {task_id}: Canvas detected: {canvas_result_1.operation}"
            )
            canvas_data = {
                "detected": True,
                "operation": canvas_result_1.operation,
                "verified_exact": canvas_result_1.verified_exact,
                "deterministic": deterministic_canvas,
                # Include params
                "pads_crops": canvas_result_1.pads_crops,
                "pad_color": canvas_result_1.pad_color,
                "axis": canvas_result_1.axis,
                "k": canvas_result_1.k,
                "gap": canvas_result_1.gap,
                "gap_color": canvas_result_1.gap_color,
                "frame_color": canvas_result_1.frame_color,
                "frame_thickness": canvas_result_1.frame_thickness,
            }
        else:
            logger.info(f"Task {task_id}: No canvas transform detected")
            canvas_data = {
                "detected": False,
                "deterministic": True,  # None == None is deterministic
            }

        # (3) Component extraction (WO-08)
        logger.info(f"Task {task_id}: Running component extraction (WO-08)")

        # Extract components from first train pair
        comps_X_1 = extract_components(train_inputs[0])
        comps_X_2 = extract_components(train_inputs[0])  # Determinism check
        deterministic_comps_X = (comps_X_1 == comps_X_2)

        comps_Y_1 = extract_components(train_outputs[0])
        comps_Y_2 = extract_components(train_outputs[0])  # Determinism check
        deterministic_comps_Y = (comps_Y_1 == comps_Y_2)

        # Match components
        if comps_X_1 and comps_Y_1:
            matches_1 = match_components(comps_X_1, comps_Y_1)
            matches_2 = match_components(comps_X_1, comps_Y_1)  # Determinism check
            deterministic_matches = (matches_1 == matches_2)

            logger.info(
                f"Task {task_id}: Components extracted, "
                f"|X|={len(comps_X_1)}, |Y|={len(comps_Y_1)}, matches={len(matches_1)}"
            )

            # Extract deltas
            deltas = [m.delta for m in matches_1 if m.comp_X_id != -1 and m.comp_Y_id != -1]

            components_data = {
                "extracted": True,
                "num_comps_X": len(comps_X_1),
                "num_comps_Y": len(comps_Y_1),
                "num_matches": len(matches_1),
                "deltas": deltas,
                "deterministic_extract_X": deterministic_comps_X,
                "deterministic_extract_Y": deterministic_comps_Y,
                "deterministic_matches": deterministic_matches,
            }
        else:
            logger.info(f"Task {task_id}: No components extracted (empty grids)")
            components_data = {
                "extracted": False,
                "deterministic_extract_X": deterministic_comps_X,
                "deterministic_extract_Y": deterministic_comps_Y,
                "deterministic_matches": True,
            }

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

        if not deterministic_wl:
            status = "FAIL"
            logger.error(f"Task {task_id}: CRITICAL - WL not deterministic")
        elif not all_pig_idempotent:
            status = "FAIL"
            logger.error(f"Task {task_id}: CRITICAL - ΠG not idempotent")
        elif not all_isomorphic and abstract_map.is_orbit:
            status = "FAIL"
            logger.error(
                f"Task {task_id}: CRITICAL - orbit mode but not isomorphic by palette"
            )
        elif lattice_data.get("detected") and not lattice_data.get("deterministic"):
            status = "FAIL"
            logger.error(f"Task {task_id}: CRITICAL - lattice not deterministic")
        elif canvas_data.get("detected") and not canvas_data.get("deterministic"):
            status = "FAIL"
            logger.error(f"Task {task_id}: CRITICAL - canvas not deterministic")
        elif components_data.get("extracted") and not all([
            components_data.get("deterministic_extract_X"),
            components_data.get("deterministic_extract_Y"),
            components_data.get("deterministic_matches"),
        ]):
            status = "FAIL"
            logger.error(f"Task {task_id}: CRITICAL - components not deterministic")
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
            "deterministic": deterministic_wl,
        }

        # Build shape data
        shape_data = {
            "domain_mode": "R×C_meet",
            "num_row_classes": num_row_classes,
            "num_col_classes": num_col_classes,
        }

        # Build palette data
        palette_data = {
            "label_mode": label_mode,
            "num_colors": len(palette_map),
            "is_orbit": abstract_map.is_orbit,
            "isomorphic_by_palette": all_isomorphic,
        }

        # Build receipt
        receipt = build_receipt(
            task_id=task_id,
            gate="D",
            present_data=present_data,
            wl_data=wl_data,
            shape_data=shape_data,
            status=status,
        )

        # Add palette data
        receipt["palette"] = palette_data

        # Add Gate D specific data (parameter extraction)
        receipt["params"] = {
            "lattice": lattice_data,
            "canvas": canvas_data,
            "components": components_data,
        }

        return receipt

    except Exception as e:
        logger.error(f"Task {task_id}: Exception - {type(e).__name__}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return build_receipt(task_id=task_id, gate="D", status="FAIL", error=str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Gate D Integration Test: Parameter Extraction"
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
    receipts_dir = integration_dir / "receipts" / "gate_d"

    # Setup logger
    logger = setup_logger("gate_d", logs_dir / "gate_d.log")

    logger.info("=" * 80)
    logger.info(f"Gate D Integration Test")
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
        receipt = validate_task_gate_d(task_id, task_dict, logger)
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

    # Parameter extraction statistics (Gate D specific)
    params_receipts = [r for r in receipts if "params" in r]
    if params_receipts:
        logger.info(f"\nParameter Extraction Statistics:")

        # Lattice stats
        lattice_detected = sum(
            1 for r in params_receipts if r["params"]["lattice"]["detected"]
        )
        lattice_fft = sum(
            1 for r in params_receipts
            if r["params"]["lattice"].get("method") == "FFT"
        )
        lattice_kmp = sum(
            1 for r in params_receipts
            if r["params"]["lattice"].get("method") == "KMP"
        )
        lattice_deterministic = sum(
            1 for r in params_receipts
            if r["params"]["lattice"]["deterministic"]
        )

        logger.info(f"  Lattice:")
        logger.info(f"    Detected: {lattice_detected}/{len(params_receipts)}")
        logger.info(f"    FFT method: {lattice_fft}")
        logger.info(f"    KMP method: {lattice_kmp}")
        logger.info(f"    Deterministic: {lattice_deterministic}/{len(params_receipts)}")

        # Canvas stats
        canvas_detected = sum(
            1 for r in params_receipts if r["params"]["canvas"]["detected"]
        )
        canvas_verified = sum(
            1 for r in params_receipts
            if r["params"]["canvas"].get("verified_exact", False)
        )
        canvas_deterministic = sum(
            1 for r in params_receipts
            if r["params"]["canvas"]["deterministic"]
        )

        logger.info(f"  Canvas:")
        logger.info(f"    Detected: {canvas_detected}/{len(params_receipts)}")
        logger.info(f"    Verified exact: {canvas_verified}/{canvas_detected if canvas_detected > 0 else 1}")
        logger.info(f"    Deterministic: {canvas_deterministic}/{len(params_receipts)}")

        # Components stats
        components_extracted = sum(
            1 for r in params_receipts if r["params"]["components"]["extracted"]
        )
        components_deterministic = sum(
            1 for r in params_receipts
            if all([
                r["params"]["components"]["deterministic_extract_X"],
                r["params"]["components"]["deterministic_extract_Y"],
                r["params"]["components"]["deterministic_matches"],
            ])
        )

        logger.info(f"  Components:")
        logger.info(f"    Extracted: {components_extracted}/{len(params_receipts)}")
        logger.info(f"    Deterministic: {components_deterministic}/{len(params_receipts)}")

    # Critical invariant checks
    logger.info("\n" + "=" * 80)
    logger.info("CRITICAL INVARIANT CHECKS")
    logger.info("=" * 80)

    determinism_rate = stats["wl"]["determinism_pass_rate"]
    tasks_with_unseen = stats["wl"]["tasks_with_unseen"]

    if determinism_rate == 1.0:
        logger.info("✅ 100% WL determinism across all tasks (PASS)")
    else:
        logger.error(f"❌ WL Determinism rate: {determinism_rate:.2%} (FAIL)")

    # Parameter determinism checks
    if params_receipts:
        all_params_deterministic = all(
            r["params"]["lattice"]["deterministic"] and
            r["params"]["canvas"]["deterministic"] and
            all([
                r["params"]["components"]["deterministic_extract_X"],
                r["params"]["components"]["deterministic_extract_Y"],
                r["params"]["components"]["deterministic_matches"],
            ])
            for r in params_receipts
        )

        if all_params_deterministic:
            logger.info("✅ 100% parameter determinism (lattice, canvas, components) (PASS)")
        else:
            logger.error("❌ Some parameters not deterministic (FAIL)")

    # Log unseen roles as measurement (not failure)
    if tasks_with_unseen > 0:
        logger.info(
            f"📊 {tasks_with_unseen} tasks have unseen roles "
            f"(test has new input structure - this is allowed)"
        )
    else:
        logger.info("📊 All tasks have complete role overlap (train covers test)")

    logger.info("\n" + "=" * 80)
    logger.info(f"Gate D validation complete. Receipts saved to: {receipts_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
