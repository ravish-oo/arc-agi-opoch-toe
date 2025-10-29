"""
Role-based mask evaluation for REGION_FILL (WO-19).

Per engineering_spec.md §5.8 and line 108:
"Masks are present-definable (band roles, component classes, periodic classes)."

Key principle: Masks are SEMANTIC definitions (not absolute pixels) that work
across any grid size.

Grounding:
- engineering_spec.md line 108: Three mask types
- engineering_spec.md line 121: "closures drop any variant not matching train"
- math_spec.md line 106: "Selectors: histogram on ΠG(X_train) over present masks"
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Optional

from arc_core.types import Grid, Pixel
from arc_core.components import extract_components


@dataclass(frozen=True)
class MaskSpec:
    """
    Role-based mask specification (present-definable).

    Per engineering_spec.md line 108, three mask types:
    1. band_role: From 1-D WL (row_class, col_class) pairs
    2. component_interior: Holes inside components of a color
    3. periodic_class: Residue classes from lattice (if detected)

    Grounding: engineering_spec.md §5.5, §5.8
    """
    mask_type: str  # "band_role", "component_interior", "periodic_class"
    params: dict    # Type-specific parameters


def evaluate_mask(
    mask_spec: MaskSpec,
    grid: Grid,
    theta: dict,
    grid_id: Optional[int] = None
) -> Set[Pixel]:
    """
    Evaluate semantic mask on specific grid to get pixel set.

    Per engineering_spec.md line 108: "Masks are present-definable"
    This means they work on ANY grid size by re-evaluating the semantic definition.

    Args:
        mask_spec: Semantic mask specification
        grid: Target grid to evaluate on
        theta: Compiled parameters (contains shape, components, etc.)
        grid_id: Grid ID (for WL role lookups, if needed)

    Returns:
        Set of pixels matching the mask definition on this grid

    Grounding:
        - engineering_spec.md line 108: Present-definable masks
        - math_spec.md §4: Shape meet U = R×C
        - WO-08: Component extraction
    """
    if mask_spec.mask_type == "band_role":
        return _evaluate_band_role_mask(mask_spec, grid, theta)

    elif mask_spec.mask_type == "component_interior":
        return _evaluate_component_interior_mask(mask_spec, grid, theta)

    elif mask_spec.mask_type == "periodic_class":
        return _evaluate_periodic_class_mask(mask_spec, grid, theta)

    else:
        raise ValueError(f"Unknown mask_type: {mask_spec.mask_type}")


def _evaluate_band_role_mask(
    mask_spec: MaskSpec,
    grid: Grid,
    theta: dict
) -> Set[Pixel]:
    """
    Evaluate band role mask: pixels with (row_class, col_class) = (r_c, c_c).

    Per engineering_spec.md line 180:
    "1-D WL bands yields band roles (masks)"

    Per math_spec.md §4:
    "Unified domain U = R×C" from shape meet

    Algorithm:
    1. Get unified partitions R and C from theta["shape"]
    2. For each pixel (r, c), check if R[r] == r_class and C[c] == c_class
    3. Return matching pixels

    Grounding: math_spec.md §4, engineering_spec.md line 180
    """
    shape_result = theta.get("shape")
    if not shape_result:
        return set()  # No shape info, empty mask

    R = shape_result.R  # {row_idx -> row_class}
    C = shape_result.C  # {col_idx -> col_class}

    r_class, c_class = mask_spec.params["band"]

    rows, cols = len(grid), len(grid[0]) if grid else 0
    pixels = set()

    for r in range(rows):
        for c in range(cols):
            # Check if this pixel's row/col classes match the band
            if R.get(r) == r_class and C.get(c) == c_class:
                pixels.add(Pixel(r, c))

    return pixels


def _evaluate_component_interior_mask(
    mask_spec: MaskSpec,
    grid: Grid,
    theta: dict
) -> Set[Pixel]:
    """
    Evaluate component interior mask: holes inside components of a color.

    Per engineering_spec.md line 108:
    "Masks are present-definable (band roles, component classes, ...)"

    Algorithm (per user clarifications lines 113-122):
    1. Extract components from grid (8-CC)
    2. Find components with matching color
    3. For each component, find interior = pixels in bbox not in component
    4. Return union of all interiors

    This implements "hole detection" - bounded faces inside components.

    Grounding: engineering_spec.md line 108, user clarifications (hole algorithm)
    """
    color = mask_spec.params.get("color")
    region = mask_spec.params.get("region", "interior")  # "interior", "boundary", "all"

    # Extract components from grid
    components = extract_components(grid)

    # Find components with matching color
    matching = [comp for comp in components if comp.color == color]

    if region == "interior":
        # Find holes (interior pixels not in component)
        interior = set()
        for comp in matching:
            # Get bbox
            r_min, r_max, c_min, c_max = comp.bbox

            # Interior = pixels in bbox not in component
            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    p = Pixel(r, c)
                    if p not in comp.pixels:
                        interior.add(p)

        return interior

    elif region == "boundary":
        # 4-conn boundary of component
        boundary = set()
        rows, cols = len(grid), len(grid[0]) if grid else 0
        for comp in matching:
            for p in comp.pixels:
                # Check if this pixel has a 4-neighbor outside the component
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = p.row + dr, p.col + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbor = Pixel(nr, nc)
                        if neighbor not in comp.pixels:
                            boundary.add(p)
                            break
        return boundary

    elif region == "all":
        # All pixels in components of this color
        all_pixels = set()
        for comp in matching:
            all_pixels.update(comp.pixels)
        return all_pixels

    else:
        return set()


def _evaluate_periodic_class_mask(
    mask_spec: MaskSpec,
    grid: Grid,
    theta: dict
) -> Set[Pixel]:
    """
    Evaluate periodic class mask: pixels at residue class modulo lattice.

    Per engineering_spec.md line 108:
    "Masks are present-definable (band roles, component classes, periodic classes)"

    Algorithm (if lattice detected):
    1. Get lattice basis from theta["lattice"]
    2. For each pixel (r, c), compute residue class: (r mod period_r, c mod period_c)
    3. Return pixels matching target residue class

    Grounding: engineering_spec.md line 108, WO-11 periodic/tiling
    """
    lattice_result = theta.get("lattice")
    if not lattice_result:
        return set()  # No lattice, empty mask

    # Get periods from lattice
    periods = lattice_result.periods  # (period_r, period_c)
    residue_class = mask_spec.params.get("residue_class")  # (r_offset, c_offset)

    if not periods or not residue_class:
        return set()

    period_r, period_c = periods
    r_offset, c_offset = residue_class

    rows, cols = len(grid), len(grid[0]) if grid else 0
    pixels = set()

    for r in range(rows):
        for c in range(cols):
            # Check if (r, c) is in this residue class
            if (r % period_r) == r_offset and (c % period_c) == c_offset:
                pixels.add(Pixel(r, c))

    return pixels


__all__ = [
    "MaskSpec",
    "evaluate_mask",
]
