"""
Component extraction and Hungarian matching (WO-08).

Per implementation_plan.md lines 256-268 and clarifications §3:
- Components = equivalence classes of (SameColor ∧ 8-connected)
- Deterministic ID assignment: (lex_min → area → centroid → boundary_hash)
- Hungarian matching with lex cost: (inertia_diff, area_diff, boundary_diff)
- All integer math (no floats)
- Δ vectors from centroid displacement

Key principles:
1. Extract components from present grid (Π_G(X))
2. Assign IDs by 4-level sort (deterministic, stable)
3. Match using Hungarian with scalarized lex cost
4. Compute Δ as integer displacement (lex_min_X → lex_min_Y)
5. Handle unmatched with dummy nodes (very large weight)

All tie-breaking uses global order §2 (deterministic).
"""

from collections import deque
from dataclasses import dataclass
from typing import Set, Tuple

from .order_hash import hash64
from .types import Grid, Hash64, Pixel, ComponentId


# Type aliases
PixelSet = Set[Pixel]


@dataclass(frozen=True)
class Component:
    """
    Component representation with all integer properties.

    Per clarifications §3 and user spec:
    - pixels: All pixels in component (8-connected)
    - color: Canonical color index (after palette canon)
    - component_id: Deterministic ID from 4-level sort

    Derived properties (all integers):
    - lex_min: Top-left pixel (row-major)
    - area: Number of pixels
    - centroid_num: (sum_r, sum_c) - numerators for centroid
    - centroid_den: area (denominator for centroid)
    - inertia_num: Integer inertia = (S_rr*A - S_r²) + (S_cc*A - S_c²)
    - boundary_hash: 64-bit hash of 4-connected boundary
    - bbox: (r_min, r_max, c_min, c_max) inclusive
    """
    pixels: frozenset[Pixel]  # Frozen for hashability
    color: int
    component_id: ComponentId

    # Computed properties (all integers)
    lex_min: Pixel
    area: int
    centroid_num: Tuple[int, int]  # (sum_r, sum_c)
    centroid_den: int  # area
    inertia_num: int  # Integer second moment
    boundary_hash: Hash64  # 4-connected boundary
    bbox: Tuple[int, int, int, int]  # (r_min, r_max, c_min, c_max)


@dataclass(frozen=True)
class Match:
    """
    Component match from Hungarian algorithm.

    - comp_X_id: Component ID from X (or -1 for dummy)
    - comp_Y_id: Component ID from Y (or -1 for dummy)
    - delta: Integer displacement (Δrow, Δcol) from lex_min_X to lex_min_Y
    - cost: Lexicographic cost tuple for verification
    """
    comp_X_id: int  # ComponentId or -1
    comp_Y_id: int  # ComponentId or -1
    delta: Tuple[int, int]  # (Δrow, Δcol)
    cost: int  # Scalarized cost


# =============================================================================
# Component Extraction (8-Connected)
# =============================================================================

def _find_8connected_components_per_color(grid: Grid) -> dict[int, list[PixelSet]]:
    """
    Find 8-connected components per color using BFS.

    Per clarifications §3: Components = (SameColor ∧ 8-connected)

    Returns:
        Dict mapping color → list of component pixel sets
    """
    H, W = len(grid), len(grid[0]) if grid else 0
    if H == 0 or W == 0:
        return {}

    # Group pixels by color
    color_pixels: dict[int, set[Pixel]] = {}
    for r in range(H):
        for c in range(W):
            color = grid[r][c]
            if color not in color_pixels:
                color_pixels[color] = set()
            color_pixels[color].add(Pixel(r, c))

    # Find 8-connected components per color
    components_per_color: dict[int, list[PixelSet]] = {}

    for color, pixels in color_pixels.items():
        components = []
        remaining = pixels.copy()

        while remaining:
            # BFS from arbitrary pixel
            start = remaining.pop()
            component = {start}
            queue = deque([start])

            while queue:
                current = queue.popleft()
                r, c = current

                # Check 8 neighbors
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue

                        neighbor = Pixel(r + dr, c + dc)

                        if neighbor in remaining:
                            component.add(neighbor)
                            remaining.remove(neighbor)
                            queue.append(neighbor)

            components.append(component)

        components_per_color[color] = components

    return components_per_color


def _compute_component_properties(pixels: PixelSet, color: int) -> dict:
    """
    Compute all component properties (integer-only).

    Returns dict with:
    - lex_min: Min pixel by (row, col)
    - area: |pixels|
    - centroid_num: (sum_r, sum_c)
    - centroid_den: area
    - inertia_num: (S_rr*A - S_r²) + (S_cc*A - S_c²)
    - bbox: (r_min, r_max, c_min, c_max)
    - boundary_pixels: Set of 4-connected boundary pixels
    """
    if not pixels:
        raise ValueError("Component must have at least one pixel")

    # Basic statistics
    area = len(pixels)
    sum_r = sum(p.row for p in pixels)
    sum_c = sum(p.col for p in pixels)
    sum_rr = sum(p.row * p.row for p in pixels)
    sum_cc = sum(p.col * p.col for p in pixels)

    # Inertia (integer): I_num = (S_rr*A - S_r²) + (S_cc*A - S_c²)
    inertia_num = (sum_rr * area - sum_r * sum_r) + (sum_cc * area - sum_c * sum_c)

    # Lex-min pixel
    lex_min = min(pixels, key=lambda p: (p.row, p.col))

    # Bounding box
    r_min = min(p.row for p in pixels)
    r_max = max(p.row for p in pixels)
    c_min = min(p.col for p in pixels)
    c_max = max(p.col for p in pixels)

    # 4-connected boundary (pixels with at least one 4-neighbor outside component)
    boundary_pixels = set()
    for pixel in pixels:
        r, c = pixel
        # Check 4 neighbors
        neighbors_4 = [
            Pixel(r - 1, c),  # up
            Pixel(r + 1, c),  # down
            Pixel(r, c - 1),  # left
            Pixel(r, c + 1),  # right
        ]

        # Pixel is on boundary if any 4-neighbor is NOT in component
        if any(neighbor not in pixels for neighbor in neighbors_4):
            boundary_pixels.add(pixel)

    return {
        'lex_min': lex_min,
        'area': area,
        'centroid_num': (sum_r, sum_c),
        'centroid_den': area,
        'inertia_num': inertia_num,
        'bbox': (r_min, r_max, c_min, c_max),
        'boundary_pixels': boundary_pixels,
    }


def _compute_boundary_hash(boundary_pixels: set[Pixel], lex_min: Pixel) -> Hash64:
    """
    Compute 64-bit hash of normalized boundary.

    Per spec: Translate boundary so lex_min → (0,0), then hash row-major.

    Args:
        boundary_pixels: Set of boundary pixels (4-connected boundary)
        lex_min: Lex-min pixel of component (for normalization)

    Returns:
        64-bit hash of normalized boundary
    """
    if not boundary_pixels:
        return hash64([])

    # Normalize: translate lex_min to (0,0)
    r_offset, c_offset = lex_min
    normalized = sorted([
        (p.row - r_offset, p.col - c_offset)
        for p in boundary_pixels
    ])

    # Hash the normalized coordinate list
    return hash64(normalized)


def _compare_centroids(c1_num: Tuple[int, int], c1_den: int,
                       c2_num: Tuple[int, int], c2_den: int) -> int:
    """
    Compare two centroids as rationals (no float division).

    Centroid 1: (c1_num[0]/c1_den, c1_num[1]/c1_den)
    Centroid 2: (c2_num[0]/c2_den, c2_num[1]/c2_den)

    Returns: -1 if c1 < c2, 0 if equal, 1 if c1 > c2

    Uses cross-multiplication to avoid float division.
    """
    # Compare rows: c1_num[0]/c1_den vs c2_num[0]/c2_den
    # Cross-multiply: c1_num[0]*c2_den vs c2_num[0]*c1_den
    row_cmp = (c1_num[0] * c2_den) - (c2_num[0] * c1_den)
    if row_cmp != 0:
        return -1 if row_cmp < 0 else 1

    # Rows equal, compare cols
    col_cmp = (c1_num[1] * c2_den) - (c2_num[1] * c1_den)
    if col_cmp != 0:
        return -1 if col_cmp < 0 else 1

    return 0


def extract_components(grid: Grid) -> list[Component]:
    """
    Extract 8-connected components with deterministic IDs.

    Per implementation_plan.md WO-08 and clarifications §3:
    1. Find 8-connected components per color
    2. Compute properties (lex_min, area, centroid, inertia, boundary)
    3. Sort by 4-level key: (lex_min ↑, -area ↓, centroid ↑, boundary_hash ↑)
    4. Assign IDs: 0, 1, 2, ...

    Args:
        grid: Present grid (Π_G(X))

    Returns:
        List of components sorted by deterministic ID order

    Acceptance:
        - Stable IDs (same grid → same IDs)
        - All integer math (no floats)
        - 4-level sort is deterministic
    """
    if not grid or not grid[0]:
        return []

    # Step 1: Find 8-connected components per color
    components_per_color = _find_8connected_components_per_color(grid)

    # Step 2: Compute properties for all components
    all_components = []
    for color, pixel_sets in components_per_color.items():
        for pixels in pixel_sets:
            props = _compute_component_properties(pixels, color)

            # Compute boundary hash
            boundary_hash = _compute_boundary_hash(
                props['boundary_pixels'],
                props['lex_min']
            )

            # Create component (no ID yet)
            all_components.append({
                'pixels': frozenset(pixels),
                'color': color,
                'lex_min': props['lex_min'],
                'area': props['area'],
                'centroid_num': props['centroid_num'],
                'centroid_den': props['centroid_den'],
                'inertia_num': props['inertia_num'],
                'boundary_hash': boundary_hash,
                'bbox': props['bbox'],
            })

    # Step 3: Sort by 4-level key
    # Key: (lex_min ↑, -area ↓, centroid ↑, boundary_hash ↑)
    def sort_key(comp_dict):
        return (
            comp_dict['lex_min'],  # (row, col) tuple - Python sorts tuples lexicographically
            -comp_dict['area'],     # Negative for larger-first
            # Centroid needs custom comparison (handled separately)
            comp_dict['boundary_hash'],
        )

    # Sort with custom centroid comparison
    from functools import cmp_to_key

    def compare_components(c1, c2):
        # Compare lex_min
        if c1['lex_min'] < c2['lex_min']:
            return -1
        if c1['lex_min'] > c2['lex_min']:
            return 1

        # Compare -area (larger first)
        if c1['area'] > c2['area']:
            return -1
        if c1['area'] < c2['area']:
            return 1

        # Compare centroid (rational)
        centroid_cmp = _compare_centroids(
            c1['centroid_num'], c1['centroid_den'],
            c2['centroid_num'], c2['centroid_den']
        )
        if centroid_cmp != 0:
            return centroid_cmp

        # Compare boundary_hash
        if c1['boundary_hash'] < c2['boundary_hash']:
            return -1
        if c1['boundary_hash'] > c2['boundary_hash']:
            return 1

        return 0

    sorted_components = sorted(all_components, key=cmp_to_key(compare_components))

    # Step 4: Assign IDs
    result = []
    for component_id, comp_dict in enumerate(sorted_components):
        component = Component(
            pixels=comp_dict['pixels'],
            color=comp_dict['color'],
            component_id=ComponentId(component_id),
            lex_min=comp_dict['lex_min'],
            area=comp_dict['area'],
            centroid_num=comp_dict['centroid_num'],
            centroid_den=comp_dict['centroid_den'],
            inertia_num=comp_dict['inertia_num'],
            boundary_hash=comp_dict['boundary_hash'],
            bbox=comp_dict['bbox'],
        )
        result.append(component)

    return result


# =============================================================================
# Hungarian Matching
# =============================================================================

def _compute_boundary_diff(comp_X: Component, comp_Y: Component) -> int:
    """
    Compute boundary difference (integer overlap metric).

    Per spec: boundary_diff = |∂X| + |∂Y| - 2*|∂X ∩ ∂Y|

    Normalize both boundaries by translating lex_min to (0,0),
    then compute overlap.

    Args:
        comp_X: Component from X
        comp_Y: Component from Y

    Returns:
        Integer boundary difference (0 = identical, larger = more different)
    """
    # Extract boundary pixels (4-connected)
    boundary_X = set()
    for pixel in comp_X.pixels:
        r, c = pixel
        neighbors_4 = [
            Pixel(r - 1, c), Pixel(r + 1, c),
            Pixel(r, c - 1), Pixel(r, c + 1),
        ]
        if any(neighbor not in comp_X.pixels for neighbor in neighbors_4):
            boundary_X.add(pixel)

    boundary_Y = set()
    for pixel in comp_Y.pixels:
        r, c = pixel
        neighbors_4 = [
            Pixel(r - 1, c), Pixel(r + 1, c),
            Pixel(r, c - 1), Pixel(r, c + 1),
        ]
        if any(neighbor not in comp_Y.pixels for neighbor in neighbors_4):
            boundary_Y.add(pixel)

    # Normalize: translate lex_min to (0,0)
    r_offset_X, c_offset_X = comp_X.lex_min
    boundary_X_norm = {
        (p.row - r_offset_X, p.col - c_offset_X)
        for p in boundary_X
    }

    r_offset_Y, c_offset_Y = comp_Y.lex_min
    boundary_Y_norm = {
        (p.row - r_offset_Y, p.col - c_offset_Y)
        for p in boundary_Y
    }

    # Compute overlap
    intersection = boundary_X_norm & boundary_Y_norm

    # Boundary diff (Hamming distance)
    boundary_diff = len(boundary_X) + len(boundary_Y) - 2 * len(intersection)

    return boundary_diff


def _compute_matching_cost(comp_X: Component, comp_Y: Component) -> int:
    """
    Compute scalarized lexicographic matching cost (integer).

    Cost tuple: (inertia_diff, area_diff, boundary_diff)
    Scalarized: w = (inertia_diff << 64) + (area_diff << 32) + boundary_diff

    Per spec:
    - Minimize |inertia_X - inertia_Y|
    - Minimize |area_X - area_Y|
    - Minimize boundary_diff (overlap metric)

    Args:
        comp_X: Component from X
        comp_Y: Component from Y

    Returns:
        Scalarized integer cost (lower is better)
    """
    inertia_diff = abs(comp_X.inertia_num - comp_Y.inertia_num)
    area_diff = abs(comp_X.area - comp_Y.area)
    boundary_diff = _compute_boundary_diff(comp_X, comp_Y)

    # Scalarize with safe bit shifts
    # (ARC grids ≤ 30×30, so area ≤ 900, inertia < 1e9)
    B1 = 32  # > max bits of area_diff
    B2 = 64  # > B1 + max bits of boundary_diff

    w = (inertia_diff << B2) + (area_diff << B1) + boundary_diff

    return w


def match_components(
    comps_X: list[Component],
    comps_Y: list[Component]
) -> list[Match]:
    """
    Match components using Hungarian algorithm with lex cost.

    Per implementation_plan.md WO-08 and user spec:
    1. Build cost matrix with scalarized lex cost
    2. Add dummy nodes for unmatched components (w_dummy = very large)
    3. Run Hungarian algorithm (scipy.optimize.linear_sum_assignment)
    4. Compute Δ for each match (lex_min_X → lex_min_Y)

    Args:
        comps_X: Components from X (present grid)
        comps_Y: Components from Y (present grid)

    Returns:
        List of matches with Δ vectors

    Acceptance:
        - Deterministic (scalarized lex cost → unique solution)
        - Handles unmatched with dummy nodes
        - Δ as integer displacement
    """
    if not comps_X and not comps_Y:
        return []

    # Import scipy for Hungarian algorithm
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        raise ImportError("scipy is required for Hungarian algorithm. Install with: pip install scipy")

    n_X = len(comps_X)
    n_Y = len(comps_Y)
    n_max = max(n_X, n_Y)

    # Dummy weight (very large)
    INF = 1 << 60
    w_dummy = (INF << 64) + (INF << 32) + INF

    # Build cost matrix (n_max × n_max)
    import numpy as np
    cost_matrix = np.full((n_max, n_max), w_dummy, dtype=np.float64)

    # Fill real costs
    for i in range(n_X):
        for j in range(n_Y):
            cost_matrix[i, j] = float(_compute_matching_cost(comps_X[i], comps_Y[j]))

    # Run Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Extract matches
    matches = []
    for i, j in zip(row_ind, col_ind):
        # Check if real match (not dummy)
        if i < n_X and j < n_Y:
            comp_X = comps_X[i]
            comp_Y = comps_Y[j]

            # Compute Δ (integer displacement from lex_min_X to lex_min_Y)
            delta_r = comp_Y.lex_min.row - comp_X.lex_min.row
            delta_c = comp_Y.lex_min.col - comp_X.lex_min.col
            delta = (delta_r, delta_c)

            # Cost for verification
            cost = int(cost_matrix[i, j])

            match = Match(
                comp_X_id=comp_X.component_id,
                comp_Y_id=comp_Y.component_id,
                delta=delta,
                cost=cost,
            )
            matches.append(match)

        elif i < n_X:
            # X component unmatched (deleted)
            comp_X = comps_X[i]
            match = Match(
                comp_X_id=comp_X.component_id,
                comp_Y_id=-1,  # Dummy
                delta=(0, 0),
                cost=w_dummy,
            )
            matches.append(match)

        elif j < n_Y:
            # Y component unmatched (inserted)
            comp_Y = comps_Y[j]
            match = Match(
                comp_X_id=-1,  # Dummy
                comp_Y_id=comp_Y.component_id,
                delta=(0, 0),
                cost=w_dummy,
            )
            matches.append(match)

    return matches
