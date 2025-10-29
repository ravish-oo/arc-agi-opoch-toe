"""
Local paint: per-role recolor law extraction (WO-09).

Per implementation_plan.md lines 271-274 and engineering_spec.md §5.1.

Extracts role→color mappings from training data. Each law instance:
- Maps a present role (from WL) to an output color
- Must be exact on all training pixels
- Is deterministic (consistent across all trains)

Closures (T_local) will later remove any recolor that mismatches train pixels.
"""

from typing import Dict, List, Tuple

from arc_core.present import D4_TRANSFORMATIONS, PiG_with_inverse
from arc_core.types import (
    Grid,
    LawInstance,
    LocalPaintParams,
    Pixel,
    RoleId,
)


def build_local_paint(params: LocalPaintParams) -> List[LawInstance]:
    """
    Build local paint law instances from training data.

    Per engineering_spec.md §5.1:
    "Paint/repaint present roles, component classes, row/col bands, periodic residue classes.
     Closures: keep only those that reproduce all train pixels."

    Algorithm:
    1. For each training pair, extract role→color mappings
    2. Find roles that map consistently to the same color across ALL trains
    3. Generate one law instance per consistent role→color mapping
    4. Verify exactness on trains (every pixel with that role maps to that color)

    Args:
        params: LocalPaintParams containing:
            - train_pairs: [(X_1, Y_1), ..., (X_m, Y_m)]
            - presents: Present structures for each X_i
            - role_map: Dict[(grid_id, pixel), RoleId] from wl_union

    Returns:
        List of LawInstance objects, each representing a role→color mapping.
        Empty list if no consistent mappings found.

    Acceptance:
        - Deterministic (same inputs → same outputs)
        - Exact on trains (reproduce all train pixels via role→color map)
        - No test-only roles (WL runs on union, so all test roles in train support)

    Notes:
        - This is the simplest law family (direct recolor by role)
        - Component classes, row/col bands, periodic residue are extensions (future WOs)
        - Per clarifications §1: WL runs on train inputs ∪ test input (no outputs)
          so role_map contains roles for inputs only
    """
    if not params.train_pairs:
        # No training data → no laws to extract
        return []

    # Step 1: Extract role→color mappings from each training pair
    # role_colors[role] = {colors seen across all trains}
    role_colors: Dict[RoleId, set[int]] = {}

    for grid_id, (X_i, Y_i) in enumerate(params.train_pairs):
        present_i = params.presents[grid_id]
        # Use canonical grid from present (not raw input)
        # WL roles are computed on canonical grids
        X_canonical = present_i.grid

        # Apply the SAME canonicalization transform to the output
        # to ensure role→color mappings are consistent in canonical space
        transform_name = present_i.g_inverse
        # g_inverse is the name of the transform needed to UNpresent
        # To present the output, we need to apply the FORWARD transform
        # which is stored in D4_INVERSES[g_inverse]
        from arc_core.present import D4_INVERSES

        forward_transform_name = D4_INVERSES[transform_name]
        forward_transform = D4_TRANSFORMATIONS[forward_transform_name]
        Y_canonical = forward_transform(Y_i)

        rows_in, cols_in = len(X_canonical), len(X_canonical[0])
        rows_out, cols_out = len(Y_canonical), len(Y_canonical[0])

        # Per clarifications §1: present is applied to inputs only (for WL)
        # But for local paint law extraction, we need output in canonical space too
        # Output Y_i may have different dimensions (canvas transform)
        # For local paint, we assume input/output dimensions match in canonical space
        # (canvas transforms are handled separately in WO-07)

        if rows_in != rows_out or cols_in != cols_out:
            # Dimension mismatch → local paint doesn't apply here
            # (This case is handled by canvas laws in WO-07)
            # Skip this training pair for role→color extraction
            continue

        # Extract role→color mapping for this training pair
        for r in range(rows_in):
            for c in range(cols_in):
                pixel = Pixel(r, c)
                role = params.role_map.get((grid_id, pixel))

                if role is None:
                    # Pixel not in role map (shouldn't happen if WL ran correctly)
                    continue

                output_color = Y_canonical[r][c]

                # Record this role→color mapping
                if role not in role_colors:
                    role_colors[role] = set()
                role_colors[role].add(output_color)

    # Step 2: Find consistent mappings (roles that map to exactly one color)
    consistent_mappings: Dict[RoleId, int] = {}

    for role, colors in role_colors.items():
        if len(colors) == 1:
            # This role consistently maps to a single color across all trains
            color = next(iter(colors))
            consistent_mappings[role] = color

    # Step 3: Generate law instances
    law_instances: List[LawInstance] = []

    for role, color in consistent_mappings.items():
        # Create a law instance for this role→color mapping
        law = LawInstance(
            law_type="LOCAL_PAINT_ROLE",
            params={"role": int(role), "color": color},
            domain_desc=f"All pixels with role {role}",
        )
        law_instances.append(law)

    # Step 4: Verify exactness on trains
    # For each law instance, check that it reproduces all train pixels
    verified_laws: List[LawInstance] = []

    for law in law_instances:
        role = RoleId(law.params["role"])
        expected_color = law.params["color"]
        is_exact = True

        # Check all training pairs
        for grid_id, (X_i, Y_i) in enumerate(params.train_pairs):
            present_i = params.presents[grid_id]
            X_canonical = present_i.grid

            # Canonicalize output too
            from arc_core.present import D4_INVERSES

            transform_name = present_i.g_inverse
            forward_transform_name = D4_INVERSES[transform_name]
            forward_transform = D4_TRANSFORMATIONS[forward_transform_name]
            Y_canonical = forward_transform(Y_i)

            rows_in, cols_in = len(X_canonical), len(X_canonical[0])
            rows_out, cols_out = len(Y_canonical), len(Y_canonical[0])

            if rows_in != rows_out or cols_in != cols_out:
                # Skip dimension-mismatched pairs
                continue

            for r in range(rows_in):
                for c in range(cols_in):
                    pixel = Pixel(r, c)
                    pixel_role = params.role_map.get((grid_id, pixel))

                    if pixel_role == role:
                        # This pixel has the role we're checking
                        actual_color = Y_canonical[r][c]
                        if actual_color != expected_color:
                            # Mismatch! This law is not exact on trains
                            is_exact = False
                            break

                if not is_exact:
                    break

            if not is_exact:
                break

        if is_exact:
            verified_laws.append(law)

    return verified_laws


def apply_local_paint(
    law: LawInstance, grid: Grid, role_map: Dict[Pixel, RoleId]
) -> Grid:
    """
    Apply a local paint law to a grid.

    Args:
        law: LawInstance with law_type="LOCAL_PAINT_ROLE"
        grid: Input grid to transform
        role_map: Mapping from pixel to role (for this grid only)

    Returns:
        Transformed grid with role→color mapping applied.

    Note:
        This is a helper function for testing/evaluation.
        During lfp, law instances are kept as expressions and evaluated
        only at the fixed point (per WO-15, WO-18).
    """
    if law.law_type != "LOCAL_PAINT_ROLE":
        raise ValueError(f"Expected LOCAL_PAINT_ROLE, got {law.law_type}")

    role = RoleId(law.params["role"])
    color = law.params["color"]

    rows, cols = len(grid), len(grid[0])
    result = [row[:] for row in grid]  # Deep copy

    for r in range(rows):
        for c in range(cols):
            pixel = Pixel(r, c)
            if role_map.get(pixel) == role:
                result[r][c] = color

    return result
