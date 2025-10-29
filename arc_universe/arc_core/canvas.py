"""
Canvas arithmetic: RESIZE/CONCAT/FRAME transforms (WO-07).

Per implementation_plan.md lines 243-253 and engineering_spec.md §5.6, §6:
- RESIZE[pads/crops]: Add padding or remove pixels
- CONCAT[axis, k, gap]: Repeat grid k times with uniform gap
- FRAME[color, thickness]: Draw uniform border
- Compositions: Base operation + optional FRAME wrapper

Key principles (from clarifications):
1. Finite candidate enumeration (no search, no ILP)
2. Exact forward verification on ALL trains (pixel-perfect)
3. Lex-min tie-breaking using global order §2
4. Y only for verification (no leakage into parameters)
5. Return None if no valid transform

CONCAT semantics (locked):
- Repeats same Π_G(X) exactly k times
- Uniform gap with single gap_color
- No per-tile transformations (those belong to other law families)

Enumeration order (fixed):
IDENTITY → RESIZE → CONCAT → (optional FRAME wrapper)
First exact match wins; lex-min if multiple valid.
"""

from dataclasses import dataclass
from typing import Literal

from .order_hash import hash64, lex_min
from .types import Grid


@dataclass(frozen=True)
class CanvasMap:
    """
    Canvas transformation parameters.

    Per clarifications: Canvas ops are finite, deterministic layout transforms.
    Separation of concerns: Content transformations belong to other law families.
    """
    operation: str  # "identity" | "resize" | "concat" | "frame" |
                    # "resize+frame" | "concat+frame"

    # RESIZE params
    pads_crops: tuple[int, int, int, int] | None = None  # (top, bottom, left, right)
    pad_color: int | None = None  # For positive pads only (derived from observation)

    # CONCAT params
    axis: Literal["rows", "cols"] | None = None
    k: int | None = None  # Number of copies (≥2)
    gap: int | None = None  # Uniform gap width (≥0)
    gap_color: int | None = None

    # FRAME params
    frame_color: int | None = None
    frame_thickness: int | None = None

    # Verification metadata
    verified_exact: bool = False

    def to_tuple(self) -> tuple:
        """
        Convert to tuple for lex-min comparison (global order §2).

        Order: operation → axis → k → gap → gap_color →
               pads_crops → pad_color → frame_color → frame_thickness
        """
        return (
            self.operation,
            self.axis or "",
            self.k or 0,
            self.gap or 0,
            self.gap_color or 0,
            self.pads_crops or (0, 0, 0, 0),
            self.pad_color or 0,
            self.frame_color or 0,
            self.frame_thickness or 0,
        )


# =============================================================================
# Transform Application Functions
# =============================================================================

def _apply_resize(
    grid: Grid,
    pads_crops: tuple[int, int, int, int],
    pad_color: int = 0
) -> Grid:
    """
    Apply RESIZE transform: add padding or crop.

    Args:
        grid: Input grid
        pads_crops: (top, bottom, left, right) - positive=pad, negative=crop
        pad_color: Color for padding (default 0)

    Returns:
        Transformed grid

    Per spec: Padding uses pad_color, cropping removes pixels.
    """
    top, bottom, left, right = pads_crops
    H, W = len(grid), len(grid[0])

    # Handle cropping (negative values)
    crop_top = abs(min(top, 0))
    crop_bottom = abs(min(bottom, 0))
    crop_left = abs(min(left, 0))
    crop_right = abs(min(right, 0))

    # Crop first
    cropped = [
        row[crop_left : W - crop_right]
        for row in grid[crop_top : H - crop_bottom]
    ]

    if not cropped or not cropped[0]:
        # Degenerate case: cropped to empty
        return [[]]

    # Handle padding (positive values)
    pad_top = max(top, 0)
    pad_bottom = max(bottom, 0)
    pad_left = max(left, 0)
    pad_right = max(right, 0)

    # Add padding
    H_cropped, W_cropped = len(cropped), len(cropped[0])

    result = []

    # Top padding
    for _ in range(pad_top):
        result.append([pad_color] * (W_cropped + pad_left + pad_right))

    # Middle rows with left/right padding
    for row in cropped:
        padded_row = [pad_color] * pad_left + row + [pad_color] * pad_right
        result.append(padded_row)

    # Bottom padding
    for _ in range(pad_bottom):
        result.append([pad_color] * (W_cropped + pad_left + pad_right))

    return result


def _apply_concat(
    grid: Grid,
    axis: Literal["rows", "cols"],
    k: int,
    gap: int = 0,
    gap_color: int = 0
) -> Grid:
    """
    Apply CONCAT transform: repeat grid k times with uniform gap.

    Args:
        grid: Input grid (Π_G(X))
        axis: "rows" (vertical stack) or "cols" (horizontal stack)
        k: Number of copies (≥2)
        gap: Uniform gap width (≥0)
        gap_color: Single color for gap area

    Returns:
        Concatenated grid

    Per spec: CONCAT repeats the SAME grid k times (no per-tile transforms).
    """
    if k < 2:
        return grid

    H, W = len(grid), len(grid[0])

    if axis == "rows":
        # Vertical concatenation
        result = []
        for i in range(k):
            # Add copy of grid
            for row in grid:
                result.append(row[:])  # Copy row

            # Add gap (except after last copy)
            if i < k - 1 and gap > 0:
                for _ in range(gap):
                    result.append([gap_color] * W)

        return result

    else:  # axis == "cols"
        # Horizontal concatenation
        result = []
        for row in grid:
            new_row = []
            for i in range(k):
                new_row.extend(row)
                # Add gap (except after last copy)
                if i < k - 1 and gap > 0:
                    new_row.extend([gap_color] * gap)
            result.append(new_row)

        return result


def _apply_frame(
    grid: Grid,
    frame_color: int,
    frame_thickness: int
) -> Grid:
    """
    Apply FRAME transform: draw uniform border.

    Args:
        grid: Input grid
        frame_color: Border color
        frame_thickness: Border width in pixels

    Returns:
        Grid with frame

    Per spec: Thickness ∈ [1, min(H,W)//2], uniform color.
    """
    if frame_thickness <= 0:
        return grid

    H, W = len(grid), len(grid[0])
    t = frame_thickness

    # Create result with frame
    result = []

    for r in range(H + 2 * t):
        row = []
        for c in range(W + 2 * t):
            # Check if in frame region
            if r < t or r >= H + t or c < t or c >= W + t:
                row.append(frame_color)
            else:
                # Interior: copy from original grid
                row.append(grid[r - t][c - t])
        result.append(row)

    return result


def _grids_equal(g1: Grid, g2: Grid) -> bool:
    """Pixel-perfect equality check."""
    if len(g1) != len(g2):
        return False
    if not g1 or not g2:
        return len(g1) == len(g2) == 0
    if len(g1[0]) != len(g2[0]):
        return False

    for r in range(len(g1)):
        for c in range(len(g1[0])):
            if g1[r][c] != g2[r][c]:
                return False

    return True


# =============================================================================
# Candidate Enumeration (Observation-Guided, Finite)
# =============================================================================

def _enumerate_identity_candidate(train_pairs: list[tuple[Grid, Grid]]) -> CanvasMap | None:
    """Check if IDENTITY (no transform) verifies on all trains."""
    for X, Y in train_pairs:
        if not _grids_equal(X, Y):
            return None

    return CanvasMap(
        operation="identity",
        verified_exact=True
    )


def _enumerate_resize_candidates(train_pairs: list[tuple[Grid, Grid]]) -> list[CanvasMap]:
    """
    Enumerate RESIZE candidates from size differences.

    Observation-guided (still finite):
    - Derive size differences from trains
    - For padding: observe pad colors from Y border regions
    - For cropping: enumerate crop distributions

    Per spec: Finite enumeration, exact verification required.
    """
    candidates = []

    if not train_pairs:
        return candidates

    # Analyze first train pair to get size difference
    X0, Y0 = train_pairs[0]
    H_x, W_x = len(X0), len(X0[0]) if X0 else 0
    H_y, W_y = len(Y0), len(Y0[0]) if Y0 else 0

    dH = H_y - H_x
    dW = W_y - W_x

    if dH == 0 and dW == 0:
        # No size change
        return candidates

    # Case 1: Padding (positive differences)
    if dH > 0 or dW > 0:
        # Observe pad color from Y0 borders
        observed_colors = set()

        # Top/bottom borders
        if dH > 0:
            for c in range(W_y):
                observed_colors.add(Y0[0][c])  # Top
                observed_colors.add(Y0[H_y - 1][c])  # Bottom

        # Left/right borders
        if dW > 0:
            for r in range(H_y):
                observed_colors.add(Y0[r][0])  # Left
                observed_colors.add(Y0[r][W_y - 1])  # Right

        # Try observed pad colors
        for pad_color in sorted(observed_colors):
            # Enumerate padding distributions
            # Try: symmetric, top-heavy, bottom-heavy, left-heavy, right-heavy
            distributions = []

            if dH > 0:
                # Vertical padding distributions
                for top in range(dH + 1):
                    bottom = dH - top
                    distributions.append((top, bottom))
            else:
                distributions.append((0, 0))

            if dW > 0:
                # Horizontal padding distributions
                h_distributions = []
                for left in range(dW + 1):
                    right = dW - left
                    h_distributions.append((left, right))
            else:
                h_distributions = [(0, 0)]

            # Combine vertical and horizontal
            for (top, bottom) in distributions:
                for (left, right) in h_distributions:
                    candidates.append(CanvasMap(
                        operation="resize",
                        pads_crops=(top, bottom, left, right),
                        pad_color=pad_color
                    ))

    # Case 2: Cropping (negative differences)
    elif dH < 0 or dW < 0:
        # Enumerate crop distributions
        distributions = []

        if dH < 0:
            for top in range(abs(dH) + 1):
                bottom = abs(dH) - top
                distributions.append((top, bottom))
        else:
            distributions.append((0, 0))

        if dW < 0:
            h_distributions = []
            for left in range(abs(dW) + 1):
                right = abs(dW) - left
                h_distributions.append((left, right))
        else:
            h_distributions = [(0, 0)]

        # Combine (negative for cropping)
        for (top, bottom) in distributions:
            for (left, right) in h_distributions:
                candidates.append(CanvasMap(
                    operation="resize",
                    pads_crops=(-top, -bottom, -left, -right),
                    pad_color=None  # Not used for cropping
                ))

    return candidates


def _enumerate_concat_candidates(train_pairs: list[tuple[Grid, Grid]]) -> list[CanvasMap]:
    """
    Enumerate CONCAT candidates from size ratios.

    Per spec: Repeat Π_G(X) k times with uniform gap.
    Observation-guided: derive k and gap_color from Y.
    """
    candidates = []

    if not train_pairs:
        return candidates

    X0, Y0 = train_pairs[0]
    H_x, W_x = len(X0), len(X0[0]) if X0 else 0
    H_y, W_y = len(Y0), len(Y0[0]) if Y0 else 0

    if H_x == 0 or W_x == 0:
        return candidates

    # Try vertical concatenation (rows)
    if H_y >= H_x:
        for k in range(2, 11):  # k ∈ [2, 10] (reasonable bound)
            for gap in range(0, 11):  # gap ∈ [0, 10] (reasonable bound)
                expected_height = k * H_x + (k - 1) * gap

                if expected_height == H_y and W_x == W_y:
                    # Possible match - observe gap color from Y0
                    gap_colors = set()

                    if gap > 0:
                        # Extract gap region colors
                        gap_start = H_x
                        gap_end = H_x + gap
                        if gap_end <= H_y:
                            for r in range(gap_start, gap_end):
                                for c in range(W_y):
                                    gap_colors.add(Y0[r][c])
                    else:
                        gap_colors.add(0)  # Default

                    for gap_color in sorted(gap_colors):
                        candidates.append(CanvasMap(
                            operation="concat",
                            axis="rows",
                            k=k,
                            gap=gap,
                            gap_color=gap_color
                        ))

    # Try horizontal concatenation (cols)
    if W_y >= W_x:
        for k in range(2, 11):
            for gap in range(0, 11):
                expected_width = k * W_x + (k - 1) * gap

                if expected_width == W_y and H_x == H_y:
                    # Observe gap color
                    gap_colors = set()

                    if gap > 0:
                        gap_start = W_x
                        gap_end = W_x + gap
                        if gap_end <= W_y:
                            for r in range(H_y):
                                for c in range(gap_start, gap_end):
                                    gap_colors.add(Y0[r][c])
                    else:
                        gap_colors.add(0)

                    for gap_color in sorted(gap_colors):
                        candidates.append(CanvasMap(
                            operation="concat",
                            axis="cols",
                            k=k,
                            gap=gap,
                            gap_color=gap_color
                        ))

    return candidates


def _enumerate_frame_candidates(
    train_pairs: list[tuple[Grid, Grid]],
    base_operation: str | None = None
) -> list[CanvasMap]:
    """
    Enumerate FRAME candidates (standalone or wrapper).

    Per spec: thickness ∈ [1, min(H,W)//2], observe frame_color from Y.
    """
    candidates = []

    if not train_pairs:
        return candidates

    X0, Y0 = train_pairs[0]
    H_x, W_x = len(X0), len(X0[0]) if X0 else 0
    H_y, W_y = len(Y0), len(Y0[0]) if Y0 else 0

    if H_y == 0 or W_y == 0:
        return candidates

    # Determine max thickness
    max_thickness = min(H_y, W_y) // 2

    if max_thickness < 1:
        return candidates

    # Detect uniform frame in Y0
    for thickness in range(1, max_thickness + 1):
        # Extract border colors at this thickness
        border_colors = set()

        # Top and bottom borders
        for r in range(thickness):
            for c in range(W_y):
                border_colors.add(Y0[r][c])
                border_colors.add(Y0[H_y - 1 - r][c])

        # Left and right borders
        for r in range(H_y):
            for c in range(thickness):
                border_colors.add(Y0[r][c])
                border_colors.add(Y0[r][W_y - 1 - c])

        # If uniform border (single color), it's a frame candidate
        if len(border_colors) == 1:
            frame_color = next(iter(border_colors))

            # Check if interior matches X (standalone frame)
            interior_H = H_y - 2 * thickness
            interior_W = W_y - 2 * thickness

            if base_operation is None and interior_H == H_x and interior_W == W_x:
                candidates.append(CanvasMap(
                    operation="frame",
                    frame_color=frame_color,
                    frame_thickness=thickness
                ))

    return candidates


# =============================================================================
# Verification and Selection
# =============================================================================

def _verify_canvas_map(candidate: CanvasMap, train_pairs: list[tuple[Grid, Grid]]) -> bool:
    """
    Verify that candidate transforms ALL train inputs to outputs exactly.

    Per spec: Pixel-perfect forward check on every train pair.
    """
    for X, Y in train_pairs:
        try:
            # Apply transform
            if candidate.operation == "identity":
                computed = X

            elif candidate.operation == "resize":
                computed = _apply_resize(
                    X,
                    candidate.pads_crops,
                    candidate.pad_color or 0
                )

            elif candidate.operation == "concat":
                computed = _apply_concat(
                    X,
                    candidate.axis,
                    candidate.k,
                    candidate.gap,
                    candidate.gap_color or 0
                )

            elif candidate.operation == "frame":
                computed = _apply_frame(
                    X,
                    candidate.frame_color,
                    candidate.frame_thickness
                )

            elif candidate.operation == "resize+frame":
                # Apply resize first, then frame
                temp = _apply_resize(
                    X,
                    candidate.pads_crops,
                    candidate.pad_color or 0
                )
                computed = _apply_frame(
                    temp,
                    candidate.frame_color,
                    candidate.frame_thickness
                )

            elif candidate.operation == "concat+frame":
                # Apply concat first, then frame
                temp = _apply_concat(
                    X,
                    candidate.axis,
                    candidate.k,
                    candidate.gap,
                    candidate.gap_color or 0
                )
                computed = _apply_frame(
                    temp,
                    candidate.frame_color,
                    candidate.frame_thickness
                )

            else:
                return False

            # Pixel-perfect comparison
            if not _grids_equal(computed, Y):
                return False

        except (IndexError, ValueError, TypeError):
            # Malformed transform
            return False

    return True


def _select_lex_min_canvas(candidates: list[CanvasMap]) -> CanvasMap:
    """
    Select lex-min candidate using global order §2.

    Per spec: If multiple valid transforms exist, choose lex-min parameter tuple.
    """
    if not candidates:
        raise ValueError("Cannot select from empty candidate list")

    return lex_min(candidates, key=lambda c: c.to_tuple())


# =============================================================================
# Main API
# =============================================================================

def infer_canvas(train_pairs: list[tuple[Grid, Grid]]) -> CanvasMap | None:
    """
    Infer canvas transformation from training pairs.

    Per implementation_plan.md lines 243-253 and clarifications:
    - Enumerate finite candidates (observation-guided, deterministic)
    - Verify exact forward mapping on ALL trains (pixel-perfect)
    - Select lex-min WITHIN each operation type
    - Return FIRST valid candidate in enumeration order (early return)
    - Return None if no valid transform

    Enumeration order (fixed):
        IDENTITY → RESIZE → CONCAT → RESIZE+FRAME → CONCAT+FRAME → FRAME

    Per anchor docs (math_spec.md §7, engineering_spec.md §6):
    - "choose lex-min parameter tuple" means lex-min WITHIN operation type
    - "Choose the first candidate" means first in enumeration order
    - Enumeration order establishes precedence (RESIZE before FRAME)

    Args:
        train_pairs: List of (input, output) grid pairs

    Returns:
        CanvasMap if valid transform found, None otherwise

    Acceptance:
        - Only exact forward maps accepted
        - Lex-min within operation type (not globally)
        - Enumeration order precedence (first valid wins)
        - Y used only for verification (no leakage)

    Examples:
        >>> # Padding example
        >>> X = [[1, 2], [3, 4]]
        >>> Y = [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]]
        >>> result = infer_canvas([(X, Y)])
        >>> result.operation
        'resize'
        >>> result.pads_crops
        (1, 1, 1, 1)
    """
    if not train_pairs:
        return None

    # Phase 1: IDENTITY (early return if valid)
    identity_candidate = _enumerate_identity_candidate(train_pairs)
    if identity_candidate is not None:
        return identity_candidate

    # Phase 2: RESIZE (standalone) - early return with lex-min within RESIZE
    resize_candidates = _enumerate_resize_candidates(train_pairs)
    valid_resize = [c for c in resize_candidates if _verify_canvas_map(c, train_pairs)]
    if valid_resize:
        selected = _select_lex_min_canvas(valid_resize)
        return CanvasMap(
            operation=selected.operation,
            pads_crops=selected.pads_crops,
            pad_color=selected.pad_color,
            verified_exact=True
        )

    # Phase 3: CONCAT (standalone) - early return with lex-min within CONCAT
    concat_candidates = _enumerate_concat_candidates(train_pairs)
    valid_concat = [c for c in concat_candidates if _verify_canvas_map(c, train_pairs)]
    if valid_concat:
        selected = _select_lex_min_canvas(valid_concat)
        return CanvasMap(
            operation=selected.operation,
            axis=selected.axis,
            k=selected.k,
            gap=selected.gap,
            gap_color=selected.gap_color,
            verified_exact=True
        )

    # Phase 4: RESIZE + FRAME (composition) - early return with lex-min
    # Enumerate RESIZE+FRAME independently by working backward from Y (like CONCAT+FRAME)
    valid_resize_frame = []
    X0, Y0 = train_pairs[0]
    H_x, W_x = len(X0), len(X0[0]) if X0 else 0
    H_y, W_y = len(Y0), len(Y0[0]) if Y0 else 0

    if H_x > 0 and W_x > 0 and H_y > 0 and W_y > 0:
        # Try different frame thicknesses
        max_frame_thickness = min(H_y, W_y) // 2

        for thickness in range(1, max_frame_thickness + 1):
            # Interior size after removing frame
            interior_H = H_y - 2 * thickness
            interior_W = W_y - 2 * thickness

            if interior_H <= 0 or interior_W <= 0:
                continue

            # Observe frame color from Y border
            border_colors = set()
            for r in range(thickness):
                for c in range(W_y):
                    border_colors.add(Y0[r][c])

            # Compute padding/cropping needed to go from X to interior
            dH = interior_H - H_x
            dW = interior_W - W_x

            # Case 1: Padding (positive differences)
            if dH >= 0 and dW >= 0:
                # Observe pad color from Y interior borders (near frame boundary)
                pad_colors = set()
                if dH > 0 or dW > 0:
                    # Sample from interior edges
                    for r in range(thickness, thickness + min(1, interior_H)):
                        for c in range(thickness, thickness + interior_W):
                            pad_colors.add(Y0[r][c])
                    for r in range(thickness, thickness + interior_H):
                        for c in range(thickness, thickness + min(1, interior_W)):
                            pad_colors.add(Y0[r][c])
                else:
                    pad_colors.add(0)  # Default

                # Enumerate padding distributions
                for pad_color in sorted(pad_colors):
                    # Try different padding distributions
                    if dH > 0:
                        h_distributions = [(top, dH - top) for top in range(dH + 1)]
                    else:
                        h_distributions = [(0, 0)]

                    if dW > 0:
                        w_distributions = [(left, dW - left) for left in range(dW + 1)]
                    else:
                        w_distributions = [(0, 0)]

                    for (top, bottom) in h_distributions:
                        for (left, right) in w_distributions:
                            for frame_color in sorted(border_colors):
                                composed = CanvasMap(
                                    operation="resize+frame",
                                    pads_crops=(top, bottom, left, right),
                                    pad_color=pad_color,
                                    frame_color=frame_color,
                                    frame_thickness=thickness
                                )

                                if _verify_canvas_map(composed, train_pairs):
                                    valid_resize_frame.append(composed)

            # Case 2: Cropping (negative differences)
            elif dH < 0 or dW < 0:
                # Enumerate cropping distributions
                if dH < 0:
                    h_distributions = [(top, abs(dH) - top) for top in range(abs(dH) + 1)]
                else:
                    h_distributions = [(0, 0)]

                if dW < 0:
                    w_distributions = [(left, abs(dW) - left) for left in range(abs(dW) + 1)]
                else:
                    w_distributions = [(0, 0)]

                for (top, bottom) in h_distributions:
                    for (left, right) in w_distributions:
                        for frame_color in sorted(border_colors):
                            composed = CanvasMap(
                                operation="resize+frame",
                                pads_crops=(-top, -bottom, -left, -right),
                                pad_color=None,  # Not used for cropping
                                frame_color=frame_color,
                                frame_thickness=thickness
                            )

                            if _verify_canvas_map(composed, train_pairs):
                                valid_resize_frame.append(composed)

    if valid_resize_frame:
        selected = _select_lex_min_canvas(valid_resize_frame)
        return CanvasMap(
            operation=selected.operation,
            pads_crops=selected.pads_crops,
            pad_color=selected.pad_color,
            frame_color=selected.frame_color,
            frame_thickness=selected.frame_thickness,
            verified_exact=True
        )

    # Phase 5: CONCAT + FRAME (composition) - early return with lex-min
    # Enumerate CONCAT+FRAME independently by considering frame removal
    valid_concat_frame = []
    X0, Y0 = train_pairs[0]
    H_x, W_x = len(X0), len(X0[0]) if X0 else 0
    H_y, W_y = len(Y0), len(Y0[0]) if Y0 else 0

    if H_x > 0 and W_x > 0 and H_y > 0 and W_y > 0:
        # Try different frame thicknesses
        max_frame_thickness = min(H_y, W_y) // 2

        for thickness in range(1, max_frame_thickness + 1):
            # Interior size after removing frame
            interior_H = H_y - 2 * thickness
            interior_W = W_y - 2 * thickness

            if interior_H <= 0 or interior_W <= 0:
                continue

            # Check if interior could be concat of X
            # Try vertical concat
            if interior_W == W_x and interior_H >= H_x:
                for k in range(2, 11):
                    for gap in range(0, 11):
                        expected_height = k * H_x + (k - 1) * gap
                        if expected_height == interior_H:
                            # Observe frame color from Y border
                            border_colors = set()
                            for r in range(thickness):
                                for c in range(W_y):
                                    border_colors.add(Y0[r][c])

                            # Observe gap color (if gap > 0)
                            gap_colors = {0}
                            if gap > 0 and H_x + gap <= interior_H:
                                gap_start = thickness + H_x
                                gap_end = gap_start + gap
                                if gap_end <= H_y - thickness:
                                    for r in range(gap_start, gap_end):
                                        for c in range(thickness, W_y - thickness):
                                            gap_colors.add(Y0[r][c])

                            for frame_color in sorted(border_colors):
                                for gap_color in sorted(gap_colors):
                                    composed = CanvasMap(
                                        operation="concat+frame",
                                        axis="rows",
                                        k=k,
                                        gap=gap,
                                        gap_color=gap_color,
                                        frame_color=frame_color,
                                        frame_thickness=thickness
                                    )

                                    if _verify_canvas_map(composed, train_pairs):
                                        valid_concat_frame.append(composed)

            # Try horizontal concat
            if interior_H == H_x and interior_W >= W_x:
                for k in range(2, 11):
                    for gap in range(0, 11):
                        expected_width = k * W_x + (k - 1) * gap
                        if expected_width == interior_W:
                            # Observe frame color
                            border_colors = set()
                            for r in range(thickness):
                                for c in range(W_y):
                                    border_colors.add(Y0[r][c])

                            # Observe gap color
                            gap_colors = {0}
                            if gap > 0 and W_x + gap <= interior_W:
                                gap_start = thickness + W_x
                                gap_end = gap_start + gap
                                if gap_end <= W_y - thickness:
                                    for r in range(thickness, H_y - thickness):
                                        for c in range(gap_start, gap_end):
                                            gap_colors.add(Y0[r][c])

                            for frame_color in sorted(border_colors):
                                for gap_color in sorted(gap_colors):
                                    composed = CanvasMap(
                                        operation="concat+frame",
                                        axis="cols",
                                        k=k,
                                        gap=gap,
                                        gap_color=gap_color,
                                        frame_color=frame_color,
                                        frame_thickness=thickness
                                    )

                                    if _verify_canvas_map(composed, train_pairs):
                                        valid_concat_frame.append(composed)

    if valid_concat_frame:
        selected = _select_lex_min_canvas(valid_concat_frame)
        return CanvasMap(
            operation=selected.operation,
            axis=selected.axis,
            k=selected.k,
            gap=selected.gap,
            gap_color=selected.gap_color,
            frame_color=selected.frame_color,
            frame_thickness=selected.frame_thickness,
            verified_exact=True
        )

    # Phase 6: FRAME (standalone) - early return with lex-min
    frame_candidates = _enumerate_frame_candidates(train_pairs)
    valid_frame = [c for c in frame_candidates if _verify_canvas_map(c, train_pairs)]
    if valid_frame:
        selected = _select_lex_min_canvas(valid_frame)
        return CanvasMap(
            operation=selected.operation,
            frame_color=selected.frame_color,
            frame_thickness=selected.frame_thickness,
            verified_exact=True
        )

    # No valid transform found
    return None
