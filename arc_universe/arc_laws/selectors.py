"""
Histogram/selector laws (WO-13).

Per implementation_plan.md lines 300-318 and engineering_spec.md §5.5.

Selectors compute colors from histograms over present-definable masks:
- ARGMAX_COLOR: Most frequent color in mask
- ARGMIN_NONZERO: Least frequent non-zero color
- UNIQUE_COLOR: Single unique color (None if multiple)
- MODE_k×k: Mode in k×k neighborhood around pixel
- PARITY_COLOR: Color based on even/odd count

CRITICAL: Empty mask handling per implementation_plan.md lines 306-307:
"If mask is empty, returns (None, True) signaling T_select closure to remove this expression."

All tie-breaks use global order (smallest canonical color).
"""

from typing import Dict, Optional, Set, Tuple

from arc_core.types import Grid, Pixel


def apply_selector_on_test(
    selector_type: str,
    mask: Set[Pixel],
    X_test: Grid,
    histogram: Optional[Dict[int, int]] = None,
    k: Optional[int] = None,
) -> Tuple[Optional[int], bool]:
    """
    Apply selector on test grid.

    Per implementation_plan.md lines 304-308:
    "Returns (color_or_None, empty_mask_flag).
     If mask is empty, returns (None, True) signaling T_select closure to remove this expression."

    Args:
        selector_type: One of "ARGMAX", "ARGMIN_NONZERO", "UNIQUE", "MODE_kxk", "PARITY"
        mask: Set of pixels defining the mask (from present roles/components/bands/residues)
        X_test: Canonical test grid (ΠG(X*))
        histogram: Optional pre-computed histogram from training (for reference, not used)
        k: Window size for MODE_kxk (required if selector_type="MODE_kxk")

    Returns:
        (color_or_None, empty_mask_flag) where:
        - color_or_None: Selected color, or None if mask empty or no valid selection
        - empty_mask_flag: True if mask is empty on test, False otherwise

    Acceptance:
        - Recompute histogram on ΠG(X_test) for non-empty masks
        - When empty_mask=True, T_select closure MUST remove that selector expression from U[q]
        - Tie-breaks (histogram modes) use global order (smallest canonical color)
        - Deterministic (no randomness)

    Examples:
        >>> # Empty mask case
        >>> apply_selector_on_test("ARGMAX", set(), [[1,2],[3,4]])
        (None, True)

        >>> # ARGMAX case
        >>> mask = {Pixel(0,0), Pixel(0,1), Pixel(1,0)}
        >>> apply_selector_on_test("ARGMAX", mask, [[1,1],[1,2]])
        (1, False)  # Color 1 appears 3 times, color 2 appears 1 time
    """
    # CRITICAL: Check for empty mask FIRST (per spec lines 306-307)
    if not mask:
        return (None, True)

    # Handle CONSTANT:c format (deterministic color from training)
    if selector_type.startswith("CONSTANT:"):
        constant_color = int(selector_type.split(":")[1])
        return (constant_color, False)

    # Validate selector type
    valid_types = ["ARGMAX", "ARGMIN_NONZERO", "UNIQUE", "MODE_kxk", "PARITY"]
    if selector_type not in valid_types:
        raise ValueError(f"Invalid selector_type '{selector_type}'. Must be one of {valid_types}")

    # MODE_kxk requires k parameter
    if selector_type == "MODE_kxk" and k is None:
        raise ValueError("MODE_kxk requires k parameter")

    # Recompute histogram on X_test for this mask (per spec line 311)
    hist = _compute_histogram(mask, X_test)

    # CRITICAL FIX (BUG-16-01): Check if histogram is empty
    # Per engineering_spec.md §12 line 234: "prevents undefined histograms"
    # When mask is "empty on test" (all pixels out of bounds), signal deletion
    # This handles case where mask set is non-empty but produces no histogram entries
    if not hist:
        return (None, True)

    # Apply selector logic based on type
    if selector_type == "ARGMAX":
        return _argmax_color(hist), False
    elif selector_type == "ARGMIN_NONZERO":
        return _argmin_nonzero(hist), False
    elif selector_type == "UNIQUE":
        return _unique_color(hist), False
    elif selector_type == "MODE_kxk":
        # For MODE_kxk, mask should be a single pixel
        # But we still compute from histogram if mask is provided
        return _mode_kxk(mask, X_test, k), False
    elif selector_type == "PARITY":
        return _parity_color(hist), False
    else:
        # Should never reach here due to validation above
        raise ValueError(f"Unhandled selector_type: {selector_type}")


def _compute_histogram(mask: Set[Pixel], grid: Grid) -> Dict[int, int]:
    """
    Compute color histogram over mask on grid.

    Args:
        mask: Set of pixels
        grid: Grid to sample from

    Returns:
        Dictionary mapping color -> count

    Note:
        - Only counts pixels that are within grid bounds
        - Empty histogram if no valid pixels
    """
    histogram: Dict[int, int] = {}
    rows, cols = len(grid), len(grid[0])

    for pixel in mask:
        r, c = pixel.row, pixel.col
        # Check bounds
        if 0 <= r < rows and 0 <= c < cols:
            color = grid[r][c]
            histogram[color] = histogram.get(color, 0) + 1

    return histogram


def _argmax_color(histogram: Dict[int, int]) -> Optional[int]:
    """
    ARGMAX_COLOR: Most frequent color in mask.

    Per spec line 313: "Tie-breaks (histogram modes) use global order (smallest canonical color)"

    Args:
        histogram: Color -> count mapping

    Returns:
        Color with highest count, or None if histogram empty.
        Tie-break: smallest color.
    """
    if not histogram:
        return None

    # Find maximum count
    max_count = max(histogram.values())

    # Find all colors with max count
    max_colors = [color for color, count in histogram.items() if count == max_count]

    # Tie-break: smallest color (global order)
    return min(max_colors)


def _argmin_nonzero(histogram: Dict[int, int]) -> Optional[int]:
    """
    ARGMIN_NONZERO: Least frequent non-zero color.

    Args:
        histogram: Color -> count mapping

    Returns:
        Non-zero color with lowest count, or None if no non-zero colors.
        Tie-break: smallest color.
    """
    if not histogram:
        return None

    # Filter out color 0 (background)
    nonzero_hist = {color: count for color, count in histogram.items() if color != 0}

    if not nonzero_hist:
        return None

    # Find minimum count among non-zero colors
    min_count = min(nonzero_hist.values())

    # Find all non-zero colors with min count
    min_colors = [color for color, count in nonzero_hist.items() if count == min_count]

    # Tie-break: smallest color (global order)
    return min(min_colors)


def _unique_color(histogram: Dict[int, int]) -> Optional[int]:
    """
    UNIQUE_COLOR: Single unique color in mask.

    Args:
        histogram: Color -> count mapping

    Returns:
        The single color if mask contains exactly one color, None otherwise.
    """
    if not histogram:
        return None

    # Check if exactly one color
    if len(histogram) == 1:
        return next(iter(histogram.keys()))

    # Multiple colors → no unique color
    return None


def _mode_kxk(mask: Set[Pixel], grid: Grid, k: int) -> Optional[int]:
    """
    MODE_k×k: Mode in k×k neighborhood.

    For each pixel in mask, compute mode in k×k window centered on that pixel.

    Args:
        mask: Set of pixels (typically a single pixel for MODE)
        grid: Grid to sample from
        k: Window size (k×k neighborhood)

    Returns:
        Most frequent color in k×k windows, or None if no valid pixels.
        Tie-break: smallest color.

    Note:
        - Window is centered on pixel
        - If mask has multiple pixels, aggregate histogram across all windows
        - Out-of-bounds pixels are ignored
    """
    if not mask:
        return None

    rows, cols = len(grid), len(grid[0])
    aggregated_hist: Dict[int, int] = {}

    # For each pixel in mask, compute k×k neighborhood histogram
    for pixel in mask:
        r_center, c_center = pixel.row, pixel.col

        # Check if center pixel is in bounds
        if not (0 <= r_center < rows and 0 <= c_center < cols):
            continue

        # Compute k×k window bounds
        half_k = k // 2
        r_start = r_center - half_k
        r_end = r_center + half_k + 1  # Inclusive of k pixels
        c_start = c_center - half_k
        c_end = c_center + half_k + 1

        # Sample k×k window
        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                # Check bounds
                if 0 <= r < rows and 0 <= c < cols:
                    color = grid[r][c]
                    aggregated_hist[color] = aggregated_hist.get(color, 0) + 1

    # Return ARGMAX of aggregated histogram
    return _argmax_color(aggregated_hist)


def _parity_color(histogram: Dict[int, int]) -> Optional[int]:
    """
    PARITY_COLOR: Color based on even/odd count.

    Per engineering_spec.md §5.5: "PARITY_COLOR"

    Algorithm:
    - Compute total count of all colors in mask
    - If total count is even: return color 0 (even parity)
    - If total count is odd: return color 1 (odd parity)

    Args:
        histogram: Color -> count mapping

    Returns:
        0 if total count is even, 1 if odd, None if empty.

    Note:
        This is a simple parity function. The spec doesn't specify
        exact behavior, so we implement the most straightforward version:
        even count → 0, odd count → 1.
    """
    if not histogram:
        return None

    # Compute total count
    total_count = sum(histogram.values())

    # Return parity: 0 for even, 1 for odd
    return total_count % 2


# Export public API
__all__ = ["apply_selector_on_test"]
