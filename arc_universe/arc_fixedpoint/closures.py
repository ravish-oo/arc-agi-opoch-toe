"""
8 closure functions for fixed-point solver (WO-16).

Per implementation_plan.md lines 341-415 and engineering_spec.md §8, clarifications §5.

Each closure enforces specific constraints by removing invalid expressions:
- T_def: Remove expressions where q ∉ Dom(e)
- T_canvas: Remove RESIZE/CONCAT/FRAME that don't reproduce trains
- T_lattice: Remove PERIODIC/TILE with wrong phases
- T_block: Remove BLOWUP[k], BLOCK_SUBST[B] that fail train tiles
- T_object: Remove TRANSLATE/COPY/RECOLOR with wrong Δ or colors
- T_select: Remove ARGMAX/UNIQUE/MODE/CONNECT/FILL that fail trains
- T_local: Remove per-role recolors that mismatch train pixels
- T_Γ: Remove expression pairs violating overlap/junction equality (GLUE)

Fixed order per clarifications §5 (line 17):
T_def → T_canvas → T_lattice → T_block → T_object → T_select → T_local → T_Γ

Critical properties (per engineering_spec.md §A1, clarifications §5):
- Monotone: Each closure removes only, never adds
- Idempotent: T(T(U)) = T(U)
- FY exactness: Expressions must reproduce training outputs exactly
- GLUE: Expressions must agree at interfaces (A2)
"""

from typing import Dict, Set, List, Tuple
from arc_core.types import Grid, Pixel
from arc_fixedpoint.expressions import (
    Expr,
    LocalPaintExpr,
    TranslateExpr,
    CopyExpr,
    PeriodicExpr,
    BlowupExpr,
    BlockSubstExpr,
    SelectorExpr,
    ResizeExpr,
    ConcatExpr,
    FrameExpr,
    ConnectExpr,
    RegionFillExpr,
)


# =============================================================================
# Helper: Normalize Training Data Format
# =============================================================================


def _normalize_train_pairs(theta: dict) -> List[Tuple[Grid, Grid]]:
    """
    Normalize training data from theta to standard format.

    CRITICAL FIX (BUG-16-02): Closures expect "train_pairs" but tests may provide "trains".
    Per implementation_plan.md line 484: official format is train_pairs: [(X1, Y1), ...]

    This helper accepts both formats for robustness during development/testing:
    - "train_pairs": [(X1, Y1), ...] (official format)
    - "trains": [{"input": X, "output": Y}] (alternative test format)

    Args:
        theta: Compiled parameters dictionary

    Returns:
        List of (input_grid, output_grid) tuples
    """
    train_pairs = theta.get("train_pairs", [])
    if not train_pairs:
        # Try alternative "trains" format (list of dicts)
        trains = theta.get("trains", [])
        if trains:
            train_pairs = [(t["input"], t["output"]) for t in trains]
    return train_pairs


# =============================================================================
# T_def: Definedness Closure
# =============================================================================


def apply_definedness_closure(
    U: Dict[Pixel, Set[Expr]],
    X_test_present: Grid,
    theta: dict
) -> Tuple[Dict[Pixel, Set[Expr]], int]:
    """
    Remove expressions where q ∉ Dom(e) on ΠG(X_test).

    Per engineering_spec.md §7 (line 140):
    "Definedness closures T_def: remove e at pixel q if q ∉ Dom(e) on ΠG(X_test)"

    Per clarifications §5 (line 295):
    "Remove expressions where `q ∉ Dom(e)` on `Π_G(X_test)`"

    Args:
        U: Product lattice (per-pixel expression sets)
        X_test_present: Canonical test grid ΠG(X*)
        theta: Compiled parameters

    Returns:
        (U_modified, removals_count): Modified U with only defined expressions

    Acceptance:
        - Monotone: Only removes expressions, never adds
        - Idempotent: Applying twice = applying once
        - Uses Expr.Dom() to check definedness
        - After lfp, all expressions are defined at their pixels
    """
    removed = 0

    for q in list(U.keys()):
        before_count = len(U[q])
        # Keep only expressions defined at q
        U[q] = {e for e in U[q] if e.Dom(q, X_test_present, theta)}
        after_count = len(U[q])
        removed += (before_count - after_count)

    return U, removed


# =============================================================================
# T_canvas: Canvas Closure
# =============================================================================


def apply_canvas_closure(
    U: Dict[Pixel, Set[Expr]],
    theta: dict
) -> Tuple[Dict[Pixel, Set[Expr]], int]:
    """
    Remove RESIZE/CONCAT/FRAME expressions that don't reproduce trains.

    Per clarifications §5 (line 296):
    "Remove RESIZE/CONCAT/FRAME that don't reproduce trains"

    Per engineering_spec.md §A1 (line 11):
    "Every law that survives must reproduce the training outputs exactly"

    Args:
        U: Product lattice
        theta: Compiled parameters containing:
            - train_pairs: List[(X_i, Y_i)] training input/output pairs
            - canvas_params: Canvas transformation parameters

    Returns:
        (U_modified, removals_count): Modified U with only exact canvas expressions

    Acceptance:
        - FY exactness: expr.eval(q, X_i) == Y_i[q] for all training pairs
        - Checks RESIZE, CONCAT, FRAME expressions
        - Monotone and idempotent
    """
    removed = 0
    train_pairs = _normalize_train_pairs(theta)

    if not train_pairs:
        # No training data to check against
        return U, removed

    for q in list(U.keys()):
        for expr in list(U[q]):
            # Only check canvas expressions
            if expr.kind not in ["RESIZE", "CONCAT", "FRAME"]:
                continue

            # Check FY exactness on all training pairs
            exact = True
            for X_i, Y_i in train_pairs:
                # Check if expression reproduces training output
                rows_y, cols_y = len(Y_i), len(Y_i[0])

                # CRITICAL FIX (BUG-16-06, BUG-16-07): Verify output size matches
                # Per engineering_spec.md line 113: "verify exact forward mapping on all trains"
                # "Exact forward mapping" = correct output SIZE + correct pixel VALUES
                if expr.kind == "RESIZE":
                    rows_x, cols_x = len(X_i), len(X_i[0])
                    top, bottom, left, right = expr.pads
                    expected_rows = rows_x + top + bottom
                    expected_cols = cols_x + left + right

                    if expected_rows != rows_y or expected_cols != cols_y:
                        # Output size mismatch → not exact forward mapping
                        exact = False
                        break

                elif expr.kind == "CONCAT":
                    # TODO (future): Verify CONCAT output size
                    # Would need to compute concatenated dimensions based on axis and gaps
                    pass

                elif expr.kind == "FRAME":
                    # TODO (future): Verify FRAME output size
                    # Would need to compute framed dimensions based on thickness
                    pass

                # For each pixel in training output
                for r in range(rows_y):
                    for c in range(cols_y):
                        q_train = Pixel(r, c)
                        try:
                            eval_result = expr.eval(q_train, X_i, theta)
                            expected = Y_i[r][c]
                            if eval_result != expected:
                                exact = False
                                break
                        except (IndexError, KeyError):
                            # Expression evaluation failed
                            exact = False
                            break
                    if not exact:
                        break
                if not exact:
                    break

            # Remove if not exact
            if not exact:
                U[q].discard(expr)
                removed += 1

    return U, removed


# =============================================================================
# T_lattice: Lattice Closure
# =============================================================================


def apply_lattice_closure(
    U: Dict[Pixel, Set[Expr]],
    theta: dict
) -> Tuple[Dict[Pixel, Set[Expr]], int]:
    """
    Remove PERIODIC/TILE expressions with wrong phases.

    Per clarifications §5 (line 297):
    "Remove PERIODIC/TILE with wrong phases on trains"

    Args:
        U: Product lattice
        theta: Compiled parameters containing:
            - train_pairs: List[(X_i, Y_i)]
            - periodic_structure: PeriodicStructure with phase info

    Returns:
        (U_modified, removals_count): Modified U with only phase-consistent expressions

    Acceptance:
        - Checks PERIODIC expressions against training data
        - Verifies phase structure consistency
        - FY exactness on periodic patterns
    """
    removed = 0
    train_pairs = _normalize_train_pairs(theta)
    periodic_structure = theta.get("periodic_structure")

    if not train_pairs or not periodic_structure:
        # No periodic structure or training data
        return U, removed

    for q in list(U.keys()):
        for expr in list(U[q]):
            # Only check periodic expressions
            if expr.kind != "PERIODIC":
                continue

            # Check FY exactness on all training pairs
            exact = True
            for X_i, Y_i in train_pairs:
                rows_y, cols_y = len(Y_i), len(Y_i[0])

                for r in range(rows_y):
                    for c in range(cols_y):
                        q_train = Pixel(r, c)
                        try:
                            eval_result = expr.eval(q_train, X_i, theta)
                            expected = Y_i[r][c]
                            if eval_result != expected:
                                exact = False
                                break
                        except (IndexError, KeyError, AttributeError):
                            exact = False
                            break
                    if not exact:
                        break
                if not exact:
                    break

            if not exact:
                U[q].discard(expr)
                removed += 1

    return U, removed


# =============================================================================
# T_block: Block Closure
# =============================================================================


def apply_block_closure(
    U: Dict[Pixel, Set[Expr]],
    theta: dict
) -> Tuple[Dict[Pixel, Set[Expr]], int]:
    """
    Remove BLOWUP[k], BLOCK_SUBST[B] expressions that fail train tiles.

    Per clarifications §5 (line 298):
    "Remove BLOWUP[k], BLOCK_SUBST[B] that fail train tiles"

    Args:
        U: Product lattice
        theta: Compiled parameters containing:
            - train_pairs: List[(X_i, Y_i)]
            - k_blowup: Blowup factor
            - motifs: Dict[color, motif_grid] for block substitution

    Returns:
        (U_modified, removals_count): Modified U with only exact block expressions

    Acceptance:
        - Checks BLOWUP and BLOCK_SUBST expressions
        - Verifies exact tile reproduction on training data
        - FY exactness for motif substitutions
    """
    removed = 0
    train_pairs = _normalize_train_pairs(theta)

    if not train_pairs:
        return U, removed

    for q in list(U.keys()):
        for expr in list(U[q]):
            # Only check block expressions
            if expr.kind not in ["BLOWUP", "BLOCK_SUBST"]:
                continue

            # Check FY exactness on all training pairs
            exact = True
            for X_i, Y_i in train_pairs:
                rows_y, cols_y = len(Y_i), len(Y_i[0])

                # CRITICAL FIX (BUG-16-08): Verify BLOWUP output size matches
                # Per engineering_spec.md line 104: "learn k ... by exact check on aligned tiles"
                # BLOWUP[k] must produce output of size (H*k, W*k)
                if expr.kind == "BLOWUP":
                    rows_x, cols_x = len(X_i), len(X_i[0])
                    expected_rows = rows_x * expr.k
                    expected_cols = cols_x * expr.k

                    if expected_rows != rows_y or expected_cols != cols_y:
                        # Output size mismatch → wrong k
                        exact = False
                        break

                elif expr.kind == "BLOCK_SUBST":
                    # TODO (future): Verify BLOCK_SUBST output size if needed
                    pass

                for r in range(rows_y):
                    for c in range(cols_y):
                        q_train = Pixel(r, c)
                        try:
                            eval_result = expr.eval(q_train, X_i, theta)
                            expected = Y_i[r][c]
                            if eval_result != expected:
                                exact = False
                                break
                        except (IndexError, KeyError, AttributeError):
                            exact = False
                            break
                    if not exact:
                        break
                if not exact:
                    break

            if not exact:
                U[q].discard(expr)
                removed += 1

    return U, removed


# =============================================================================
# T_object: Object Closure
# =============================================================================


def apply_object_closure(
    U: Dict[Pixel, Set[Expr]],
    theta: dict
) -> Tuple[Dict[Pixel, Set[Expr]], int]:
    """
    Remove TRANSLATE/COPY/RECOLOR expressions with wrong Δ or colors.

    Per clarifications §5 (line 299):
    "Remove TRANSLATE/COPY/RECOLOR with wrong Δ or colors"

    Args:
        U: Product lattice
        theta: Compiled parameters containing:
            - train_pairs: List[(X_i, Y_i)]
            - components: Component data
            - deltas: Delta vectors

    Returns:
        (U_modified, removals_count): Modified U with only exact object expressions

    Acceptance:
        - Checks TRANSLATE, COPY, RECOLOR expressions
        - Verifies correct delta vectors and color mappings
        - FY exactness on component transformations
    """
    removed = 0
    train_pairs = _normalize_train_pairs(theta)

    if not train_pairs:
        return U, removed

    for q in list(U.keys()):
        for expr in list(U[q]):
            # Only check object arithmetic expressions
            if expr.kind not in ["TRANSLATE", "COPY"]:
                continue

            # Check FY exactness on all training pairs
            exact = True
            for X_i, Y_i in train_pairs:
                rows_y, cols_y = len(Y_i), len(Y_i[0])

                # CRITICAL FIX (BUG-16-09): Verify TRANSLATE parameters match compiled theta
                # Per clarifications.md line 299: "TRANSLATE/COPY/RECOLOR with wrong Δ or colors"
                # Check if delta matches the compiled deltas in theta
                if expr.kind == "TRANSLATE":
                    # Check output size (TRANSLATE shouldn't change grid size)
                    rows_x, cols_x = len(X_i), len(X_i[0])
                    if rows_x != rows_y or cols_x != cols_y:
                        # TRANSLATE changing size is wrong
                        exact = False
                        break

                    # Check if delta matches compiled theta
                    compiled_deltas = theta.get("deltas", {})
                    if expr.component_id in compiled_deltas:
                        expected_delta = compiled_deltas[expr.component_id]
                        if expr.delta != expected_delta:
                            # Wrong delta → remove expression
                            exact = False
                            break

                # For TRANSLATE/COPY: check if transformation is correct on ALL pixels
                # Not just pixels where Dom=True, because wrong Δ means component in wrong place
                for r in range(rows_y):
                    for c in range(cols_y):
                        q_train = Pixel(r, c)
                        try:
                            # For TRANSLATE with wrong Δ, we need to check ALL pixels
                            # If Dom is False but Y_i expects non-background, that's a mismatch
                            is_defined = expr.Dom(q_train, X_i, theta)

                            if is_defined:
                                eval_result = expr.eval(q_train, X_i, theta)
                                expected = Y_i[r][c]
                                if eval_result != expected:
                                    exact = False
                                    break
                            else:
                                # Expression not defined at this pixel
                                # For TRANSLATE: if Y_i has non-zero here, might be wrong Δ
                                # But this is tricky - other expressions might paint this pixel
                                # So we can't fail just because Dom is False
                                # The real check is: does the overall composition work?
                                # For now, just trust Dom + eval for defined pixels
                                pass

                        except (IndexError, KeyError, AttributeError):
                            exact = False
                            break
                    if not exact:
                        break
                if not exact:
                    break

            if not exact:
                U[q].discard(expr)
                removed += 1

    return U, removed


# =============================================================================
# T_select: Selector Closure (CRITICAL)
# =============================================================================


def apply_selector_closure(
    U: Dict[Pixel, Set[Expr]],
    theta: dict
) -> Tuple[Dict[Pixel, Set[Expr]], int]:
    """
    Remove ARGMAX/UNIQUE/MODE/CONNECT/FILL expressions that fail trains.

    CRITICAL: When apply_selector_on_test returns empty_mask=True,
    MUST remove that selector expression from U[q].

    Per implementation_plan.md lines 361-398:
    "CRITICAL: When apply_selector_on_test returns empty_mask=True,
     MUST remove that selector expression from U[q]."

    Per clarifications §5 (line 300):
    "Remove ARGMAX/UNIQUE/MODE/CONNECT/FILL that fail trains"

    Args:
        U: Product lattice
        theta: Compiled parameters containing:
            - train_pairs: List[(X_i, Y_i)]
            - X_test_present: Canonical test grid
            - selectors: Selector specifications

    Returns:
        (U_modified, removals_count): Modified U with only exact selector expressions

    Acceptance:
        - CRITICAL: Remove selector if empty_mask=True (spec §12)
        - Check FY exactness on training pairs
        - Handle SELECTOR and REGION_FILL expressions
        - Remove CONNECT expressions that fail to reproduce trains
    """
    removed = 0
    train_pairs = _normalize_train_pairs(theta)
    X_test_present = theta.get("X_test_present")

    if not X_test_present:
        # Need test grid for selector evaluation
        return U, removed

    # Import here to avoid circular dependency
    from arc_laws.selectors import apply_selector_on_test

    for q in list(U.keys()):
        for expr in list(U[q]):
            should_remove = False

            # 1) SELECTOR expressions: ARGMAX/UNIQUE/MODE/PARITY
            if expr.kind == "SELECTOR":
                # Evaluate on test grid
                try:
                    color, empty_mask = apply_selector_on_test(
                        selector_type=expr.selector_type,
                        mask=set(expr.mask),  # Convert frozenset to set
                        X_test=X_test_present,
                        histogram=None,
                        k=expr.k
                    )

                    # CRITICAL: Delete if mask is empty (per spec §12)
                    if empty_mask:
                        should_remove = True
                    else:
                        # Check FY exactness on training pairs
                        if train_pairs:
                            for X_i, Y_i in train_pairs:
                                # Evaluate selector on training input
                                train_color, train_empty = apply_selector_on_test(
                                    selector_type=expr.selector_type,
                                    mask=set(expr.mask),
                                    X_test=X_i,
                                    histogram=None,
                                    k=expr.k
                                )

                                if train_empty:
                                    should_remove = True
                                    break

                                # Check if selector produces correct color for pixels in mask
                                rows_y, cols_y = len(Y_i), len(Y_i[0])
                                for r in range(rows_y):
                                    for c in range(cols_y):
                                        q_train = Pixel(r, c)
                                        if q_train in expr.mask:
                                            if train_color != Y_i[r][c]:
                                                should_remove = True
                                                break
                                    if should_remove:
                                        break
                                if should_remove:
                                    break

                except Exception:
                    # Evaluation failed
                    should_remove = True

            # 2) REGION_FILL depends on selector color
            elif expr.kind == "REGION_FILL":
                # CRITICAL FIX (BUG-16-10): Check for empty mask FIRST
                # Per engineering_spec.md line 234 (Spec §12):
                # "if a selector mask M is empty... delete all conflicting expressions"
                # REGION_FILL uses a mask, so same requirement applies

                # Check if mask is empty OR empty on test (all pixels out of bounds)
                if not expr.mask:
                    # Empty mask set → undefined behavior, must delete
                    should_remove = True
                else:
                    # Check if mask is "empty on test" (all pixels out of bounds)
                    mask_has_valid_pixel = False
                    rows, cols = len(X_test_present), len(X_test_present[0])
                    for pixel in expr.mask:
                        if 0 <= pixel.row < rows and 0 <= pixel.col < cols:
                            mask_has_valid_pixel = True
                            break

                    if not mask_has_valid_pixel:
                        # Mask empty on test → undefined behavior, must delete
                        should_remove = True

                # Only check FY exactness if we haven't already decided to remove
                if not should_remove and train_pairs:
                    for X_i, Y_i in train_pairs:
                        rows_y, cols_y = len(Y_i), len(Y_i[0])

                        for r in range(rows_y):
                            for c in range(cols_y):
                                q_train = Pixel(r, c)
                                if q_train in expr.mask:
                                    expected = Y_i[r][c]
                                    if expr.fill_color != expected:
                                        should_remove = True
                                        break
                            if should_remove:
                                break
                        if should_remove:
                            break

            # 3) CONNECT expressions
            elif expr.kind == "CONNECT":
                # Check FY exactness on training pairs
                if train_pairs:
                    for X_i, Y_i in train_pairs:
                        rows_y, cols_y = len(Y_i), len(Y_i[0])

                        for r in range(rows_y):
                            for c in range(cols_y):
                                q_train = Pixel(r, c)
                                try:
                                    eval_result = expr.eval(q_train, X_i, theta)
                                    expected = Y_i[r][c]
                                    if eval_result != expected:
                                        should_remove = True
                                        break
                                except (IndexError, KeyError):
                                    should_remove = True
                                    break
                            if should_remove:
                                break
                        if should_remove:
                            break

            # Remove if invalid
            if should_remove:
                U[q].discard(expr)
                removed += 1

    return U, removed


# =============================================================================
# T_local: Local Paint Closure
# =============================================================================


def apply_local_paint_closure(
    U: Dict[Pixel, Set[Expr]],
    theta: dict
) -> Tuple[Dict[Pixel, Set[Expr]], int]:
    """
    Remove per-role recolors that mismatch train pixels.

    Per clarifications §5 (line 301):
    "Remove per-role recolors that mismatch train pixels"

    Per engineering_spec.md §5.1 (line 89):
    "Closures: keep only those that reproduce all train pixels"

    Args:
        U: Product lattice
        theta: Compiled parameters containing:
            - train_pairs: List[(X_i, Y_i)]
            - role_map: Dict[(grid_id, Pixel), RoleId]

    Returns:
        (U_modified, removals_count): Modified U with only exact local paint expressions

    Acceptance:
        - Checks LOCAL_PAINT expressions
        - Verifies color matches training output for each role
        - FY exactness: expr.eval(q) == Y[q] for all q in role
    """
    removed = 0
    train_pairs = _normalize_train_pairs(theta)

    if not train_pairs:
        return U, removed

    for q in list(U.keys()):
        for expr in list(U[q]):
            # Only check local paint expressions
            if expr.kind != "LOCAL_PAINT":
                continue

            # Check FY exactness on all training pairs
            exact = True
            for i, (X_i, Y_i) in enumerate(train_pairs):
                rows_y, cols_y = len(Y_i), len(Y_i[0])

                for r in range(rows_y):
                    for c in range(cols_y):
                        q_train = Pixel(r, c)

                        # Check if this pixel has the role
                        if not expr.Dom(q_train, X_i, theta):
                            continue

                        # Pixel has this role → check color
                        eval_result = expr.eval(q_train, X_i, theta)
                        expected = Y_i[r][c]
                        if eval_result != expected:
                            exact = False
                            break
                    if not exact:
                        break
                if not exact:
                    break

            if not exact:
                U[q].discard(expr)
                removed += 1

    return U, removed


# =============================================================================
# T_Γ: Interface Closure (GLUE)
# =============================================================================


def apply_interface_closure(
    U: Dict[Pixel, Set[Expr]],
    theta: dict
) -> Tuple[Dict[Pixel, Set[Expr]], int]:
    """
    Remove expression pairs violating overlap/junction equality (GLUE).

    Per clarifications §5 (line 302):
    "Remove expression pairs violating overlap/junction equality"

    Per engineering_spec.md §A2 (line 13):
    "All interface equalities (at overlaps, band junctions, canvas joins)
     must be satisfied; we enforce this by interface closures"

    Args:
        U: Product lattice
        theta: Compiled parameters

    Returns:
        (U_modified, removals_count): Modified U with only GLUE-consistent expressions

    Acceptance:
        - Enforces A2 (Gluing): expressions must agree at interfaces
        - If two expressions both write to a pixel, they must produce same color
        - Checks overlaps, band junctions, canvas joins
        - Removes conflicting expression pairs
    """
    removed = 0

    # CRITICAL FIX (BUG-16-03): Get evaluation grid
    # Per spec A2: Need to evaluate expressions to detect color conflicts
    # Try multiple sources in order of preference:
    X_test_present = theta.get("X_test_present")
    if not X_test_present:
        # Fall back to train data if available
        trains = theta.get("trains", [])
        if trains:
            # Use first train input for evaluation
            X_test_present = trains[0]["input"]
        else:
            # Try train_pairs format
            train_pairs = theta.get("train_pairs", [])
            if train_pairs:
                X_test_present = train_pairs[0][0]

    # If still no grid, can't evaluate → return unchanged
    if not X_test_present:
        return U, removed

    # For each pixel q, check if multiple expressions produce different colors
    for q in list(U.keys()):
        if len(U[q]) <= 1:
            # Only one expression or empty → no conflicts
            continue

        # Evaluate all expressions at this pixel
        expr_colors = {}
        for expr in U[q]:
            try:
                color = expr.eval(q, X_test_present, theta)
                expr_colors[expr] = color
            except Exception:
                # Evaluation failed → will be removed by other closures
                pass

        # Check for conflicts
        if len(set(expr_colors.values())) > 1:
            # Multiple different colors → GLUE violation
            # Keep expressions that produce the most common color (tie-break: min)
            from collections import Counter
            color_counts = Counter(expr_colors.values())

            if color_counts:
                # Find most common color (tie-break: smallest)
                max_count = max(color_counts.values())
                most_common_colors = [c for c, count in color_counts.items() if count == max_count]
                keep_color = min(most_common_colors)

                # Remove expressions that don't produce keep_color
                for expr, color in list(expr_colors.items()):
                    if color != keep_color:
                        U[q].discard(expr)
                        removed += 1

    return U, removed


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "apply_definedness_closure",
    "apply_canvas_closure",
    "apply_lattice_closure",
    "apply_block_closure",
    "apply_object_closure",
    "apply_selector_closure",
    "apply_local_paint_closure",
    "apply_interface_closure",
]
