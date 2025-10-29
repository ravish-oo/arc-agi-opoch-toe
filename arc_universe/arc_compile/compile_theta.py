"""
End-to-end compilation: ΠG → WL union → shape meet → param extraction → U0 → closures (WO-19).

Per implementation_plan.md lines 480-487, math_spec.md §8, and engineering_spec.md §8.

API:
    compile_theta(train_pairs, test_input) -> (theta, U0, closures, receipts_stub)

Deterministic compilation with no search:
- Step 1: Present all inputs (train ∪ test)
- Step 2: WL union (shared roles, no test-only roles)
- Step 3: Shape meet (trains only)
- Step 4: Palette canonicalization (train∪test inputs)
- Step 5: Parameter extraction (all 8 law families)
- Step 6: Build U0 via init_expressions
- Step 7: Build closures list (fixed order)
- Step 8: Create receipts stub

All functions deterministic, total, and grounded in anchor docs.
"""

from typing import Dict, List, Set, Tuple

from arc_core.canvas import infer_canvas
from arc_core.components import extract_components
from arc_core.lattice import infer_lattice
from arc_core.palette import canonicalize_palette_for_task
from arc_core.present import build_present
from arc_core.shape import unify_shape
from arc_core.types import Grid, Pixel
from arc_core.wl import wl_union
from arc_fixedpoint.closures import (
    apply_block_closure,
    apply_canvas_closure,
    apply_definedness_closure,
    apply_interface_closure,
    apply_lattice_closure,
    apply_local_paint_closure,
    apply_object_closure,
    apply_selector_closure,
)
from arc_fixedpoint.expressions import Expr, init_expressions
from arc_laws.blow_block import build_blow_block
from arc_laws.connect_fill import build_connect_fill
from arc_laws.local_paint import build_local_paint
from arc_laws.object_arith import build_object_arith
from arc_laws.periodic import build_periodic


def compile_theta(
    train_pairs: List[Tuple[Grid, Grid]],
    test_input: Grid
) -> Tuple[dict, Dict[Pixel, Set[Expr]], list, dict]:
    """
    End-to-end compilation from train_pairs + test_input.

    Per math_spec.md §8 lines 148-151 and clarifications:
    1. Present all inputs (ΠG)
    2. WL union on {train inputs} ∪ {test input}
    3. Shape meet (trains only)
    4. Palette canonicalization (train∪test inputs)
    5. Parameter extraction (all 8 families)
    6. Build U0 (init expressions)
    7. Build closures (fixed order)
    8. Create receipts stub

    Args:
        train_pairs: List of (input_grid, output_grid) tuples
        test_input: Test input grid X*

    Returns:
        Tuple of (theta, U0, closures, receipts_stub) where:
        - theta: Compiled parameters for all law families
        - U0: Initial expression sets Dict[Pixel, Set[Expr]]
        - closures: List of 8 closure functions in fixed order
        - receipts_stub: Partial receipt with present/WL/basis stats

    Acceptance:
        - Deterministic (same inputs → same outputs)
        - No test-only roles (WL union guarantees)
        - All 8 law families extracted where applicable
        - Closures in fixed order: T_def → T_canvas → T_lattice → T_block → T_object → T_select → T_local → T_Γ

    Examples:
        >>> train_pairs = [([[0,1],[2,3]], [[4,5],[6,7]])]
        >>> test_input = [[0,1],[2,3]]
        >>> theta, U0, closures, receipts = compile_theta(train_pairs, test_input)
        >>> len(closures)
        8
    """
    # =========================================================================
    # Step 1: Present all inputs (ΠG)
    # =========================================================================
    # Per math_spec.md §1 line 22: "store g_test so that the final output is unpresented"

    train_inputs = [inp for inp, _ in train_pairs]
    train_outputs = [out for _, out in train_pairs]
    all_inputs = train_inputs + [test_input]

    # Build presents for all inputs
    presents_all = [build_present(grid) for grid in all_inputs]
    presents_train = presents_all[:len(train_inputs)]
    test_present = presents_all[len(train_inputs)]

    # Extract g_test for unpresenting
    g_test = test_present.g_inverse

    # =========================================================================
    # Step 2: WL union on {train inputs} ∪ {test input}
    # =========================================================================
    # Per implementation_clarifications.md §1 lines 26-31:
    # "Run 1-WL on: 𝒱 = {X₁, X₂, ..., Xₘ} ∪ {X*}"
    # Per math_spec.md §3 line 48-55: "1-WL on the disjoint union (train ∪ test)"

    role_map = wl_union(presents_all, escalate=False, max_iters=12)

    # Count roles
    train_roles = set()
    test_roles = set()
    for (grid_id, pixel), role_id in role_map.items():
        if grid_id < len(train_inputs):
            train_roles.add(role_id)
        else:
            test_roles.add(role_id)

    roles_train = len(train_roles)
    roles_test = len(test_roles)
    unseen_roles = len(test_roles - train_roles)

    # Count WL iterations (simplified - actual count from wl_union implementation)
    wl_iters = 12  # Placeholder - actual value from wl_union

    # =========================================================================
    # Step 3: Shape meet (trains only)
    # =========================================================================
    # Per math_spec.md §4 lines 61-63: "For each train input, run 1-D WL on rows/cols"
    # Shape meet uses train evidence only

    shape_result = unify_shape(presents_train)

    # =========================================================================
    # Step 4: Palette canonicalization (train∪test inputs)
    # =========================================================================
    # Per implementation_clarifications.md §2 line 58:
    # "Per-task, pooled across all inputs (train ∪ test), not outputs"

    palette_map = canonicalize_palette_for_task(train_inputs, test_input)

    # =========================================================================
    # Step 5: Parameter extraction (all 8 families)
    # =========================================================================
    # Per math_spec.md §7 lines 100-108: "Deterministic parameter extraction (compile-time)"

    # Initialize theta dict
    # Add canvas_shape from test present (needed by init_expressions)
    test_grid = test_present.grid
    canvas_shape = (len(test_grid), len(test_grid[0]))

    theta = {
        "role_map": role_map,
        "presents": presents_all,
        "presents_train": presents_train,
        "test_present": test_present,
        "g_test": g_test,
        "palette_map": palette_map,
        "shape": shape_result,
        "train_pairs": train_pairs,
        "canvas_shape": canvas_shape,
    }

    # Track which law families are used
    basis_used = set()

    # === Family 1: Local Paint (WO-09) ===
    # Per engineering_spec.md §5.1: "Paint/repaint present roles, component classes"
    from arc_core.types import LocalPaintParams

    local_paint_params = LocalPaintParams(
        train_pairs=train_pairs,
        presents=presents_train,
        role_map=role_map
    )
    local_paint_laws = build_local_paint(local_paint_params)

    if local_paint_laws:
        theta["local_paint_laws"] = local_paint_laws
        basis_used.add("LOCAL_PAINT")

    # === Family 2: Object Arithmetic (WO-10) ===
    # Per engineering_spec.md §5.2: "Components & Δ / matching: Hungarian + lex tie"

    # Extract components for each train input
    components_per_grid = []
    for present in presents_train:
        comps = extract_components(present.grid)
        components_per_grid.append(comps)

    # Store as list for law extraction
    theta["components"] = components_per_grid

    object_arith_laws = build_object_arith(theta)

    if object_arith_laws:
        theta["object_arith_laws"] = object_arith_laws
        basis_used.add("TRANSLATE")

    # For init_expressions, we need a dict of components from test
    # Build from test present (overwrite the list with dict for init_expressions)
    test_components = extract_components(test_present.grid)
    # Convert list to dict by component_id
    components_dict = {comp.component_id: comp for comp in test_components}

    # Store both: keep list in "components_per_grid" for reference
    theta["components_per_grid"] = components_per_grid
    # Overwrite "components" with dict for init_expressions
    theta["components"] = components_dict
    theta["deltas"] = []  # Placeholder for deltas extracted from laws

    # === Family 3: Periodic/Tiling (WO-11) ===
    # Per engineering_spec.md §5.3: "Lattice: FFT ACF → HNF → D8 canonical"

    # Try to detect lattice from train inputs
    # infer_lattice takes a list of grids
    lattice_result = infer_lattice(train_inputs)

    if lattice_result is not None:
        theta["lattice"] = lattice_result

        # Build periodic structure
        periodic_structure = build_periodic(theta)
        if periodic_structure is not None:
            theta["periodic"] = periodic_structure
            basis_used.add("PERIODIC")

    # === Family 4: Blowup/Block Substitution (WO-12) ===
    # Per engineering_spec.md §5.4: "BLOWUP[k], BLOCK_SUBST[B]"

    blow_block_laws = build_blow_block(theta)

    if blow_block_laws:
        theta["blow_block_laws"] = blow_block_laws
        basis_used.add("BLOWUP")
        basis_used.add("BLOCK_SUBST")

    # === Family 5: Histogram/Selector (WO-13) ===
    # Per engineering_spec.md §5.5: "ARGMAX_COLOR, UNIQUE_COLOR, MODE_k×k"
    # Selectors are compiled on-demand by closures, not pre-extracted
    # We just flag that selectors are available
    basis_used.add("SELECTOR")

    # === Family 6: Canvas Arithmetic (WO-07) ===
    # Per engineering_spec.md §5.6: "RESIZE, CONCAT, FRAME"

    canvas_map = infer_canvas(train_pairs)

    if canvas_map is not None:
        theta["canvas_map"] = canvas_map
        basis_used.add("CANVAS")

    # === Family 7: Connect Endpoints (WO-14) ===
    # === Family 8: Region Fill (WO-14) ===
    # Per engineering_spec.md §5.7-5.8: "CONNECT_ENDPOINTS, REGION_FILL"

    connect_fill_laws = build_connect_fill(theta)

    if connect_fill_laws:
        theta["connect_fill_laws"] = connect_fill_laws
        basis_used.add("CONNECT")
        basis_used.add("FILL")

    # =========================================================================
    # Step 6: Build U0 (init expressions)
    # =========================================================================
    # Per math_spec.md §8 lines 113-114:
    # "Let E_q(θ) be the finite set of candidates permitted at pixel q,
    #  and U_0=∏_q P(E_q)"

    U0 = init_expressions(theta)

    # =========================================================================
    # Step 7: Build closures (fixed order)
    # =========================================================================
    # Per implementation_clarifications.md §5 lines 271-278:
    # Fixed order: T_def → T_canvas → T_lattice → T_block → T_object → T_select → T_local → T_Γ

    closures = [
        apply_definedness_closure,
        apply_canvas_closure,
        apply_lattice_closure,
        apply_block_closure,
        apply_object_closure,
        apply_selector_closure,
        apply_local_paint_closure,
        apply_interface_closure,
    ]

    # =========================================================================
    # Step 8: Create receipts stub
    # =========================================================================
    # Per implementation_plan.md lines 502-539: Receipt structure

    receipts_stub = {
        "present": {
            "CBC3": True,  # Always computed
            "E4": True,    # Always used
            "E8": False,   # Would be True if escalated
            "Row1D": False,  # Placeholder
            "Col1D": False,  # Placeholder
        },
        "wl": {
            "iters": wl_iters,
            "roles_train": roles_train,
            "roles_test": roles_test,
            "unseen_roles": unseen_roles,
        },
        "basis_used": sorted(list(basis_used)),
    }

    # Add lattice info if detected
    if lattice_result is not None:
        receipts_stub["lattice"] = {
            "method": lattice_result.method,
            "periods": lattice_result.periods,
        }

    return theta, U0, closures, receipts_stub
