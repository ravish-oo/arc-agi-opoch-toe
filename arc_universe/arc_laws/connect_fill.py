"""
CONNECT_ENDPOINTS and REGION_FILL laws (WO-14).

Per implementation_plan.md lines 322-325 and engineering_spec.md §5.7-5.8:
- CONNECT_ENDPOINTS: Shortest 4/8-conn path between two present-definable anchors
- REGION_FILL[mask, color]: Flood fill region with selector color

Key principles:
1. CONNECT: BFS shortest path with lex-min tie-breaking (per clarifications §7)
2. FILL: Mask from present roles + selector from WO-13
3. FY exactness: All laws must reproduce training outputs exactly
4. Determinism: All paths/fills are deterministic (no randomness)

All operations use global order for tie-breaking.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from collections import deque

from arc_core.types import Grid, Pixel
from arc_laws.selectors import apply_selector_on_test


# =============================================================================
# Types
# =============================================================================


@dataclass(frozen=True)
class ConnectLaw:
    """
    CONNECT_ENDPOINTS law instance.

    Represents a shortest path between two anchors with specific metric.
    Path is precomputed and guaranteed to be lex-min among all shortest paths.

    Per engineering_spec.md §5.7:
    "Unique shortest 4/8-connected path between two present-definable anchors.
     Tie-break: lex-min path under the global order."
    """
    anchor1: Pixel              # Start point (from present)
    anchor2: Pixel              # End point (from present)
    metric: str                 # "4conn" or "8conn"
    path: Tuple[Pixel, ...]     # Computed shortest path (lex-min), as tuple for hashing
    line_color: int             # Color to draw the line


@dataclass(frozen=True)
class FillLaw:
    """
    REGION_FILL law instance.

    Represents flood fill operation with selector-determined color.
    Selector is recomputed on test input (per WO-13 spec).

    Per engineering_spec.md §5.8:
    "Fill/patched flood inside a present-definable mask with the selector color (from 5.5).
     Compile: masks/anchors from present; color law from selector."
    """
    mask_pixels: frozenset[Pixel]  # Region to fill (from present roles/components)
    selector_type: str             # "UNIQUE_COLOR", "ARGMAX", "ARGMIN_NONZERO", etc.
    selector_k: Optional[int] = None  # For MODE_kxk selector


# =============================================================================
# BFS Shortest Path Algorithms
# =============================================================================


def _get_4neighbors(pixel: Pixel, H: int, W: int) -> List[Pixel]:
    """
    Get 4-connected neighbors (orthogonal only).

    Order: up, right, down, left (for determinism).
    """
    r, c = pixel.row, pixel.col
    neighbors = []

    # Orthogonal directions in fixed order
    for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            neighbors.append(Pixel(nr, nc))

    return neighbors


def _get_8neighbors(pixel: Pixel, H: int, W: int) -> List[Pixel]:
    """
    Get 8-connected neighbors (orthogonal + diagonal).

    Order: N, NE, E, SE, S, SW, W, NW (clockwise from top).
    """
    r, c = pixel.row, pixel.col
    neighbors = []

    # Clockwise from top
    for dr, dc in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            neighbors.append(Pixel(nr, nc))

    return neighbors


def shortest_path_bfs(
    grid: Grid,
    anchor1: Pixel,
    anchor2: Pixel,
    metric: str
) -> Optional[List[Pixel]]:
    """
    Compute shortest path between two anchors using BFS.

    Per clarifications §7: "Path selection: Lex-min geodesic (CONNECT_ENDPOINTS)"

    Algorithm:
    1. BFS to find all shortest paths
    2. Among equal-length paths, select lex-min by global order
    3. Global order for paths: compare pixel tuples lexicographically

    Args:
        grid: Grid for bounds checking
        anchor1: Start pixel
        anchor2: End pixel
        metric: "4conn" or "8conn"

    Returns:
        Shortest lex-min path as list of pixels (including endpoints).
        Returns None if no path exists.

    Acceptance:
        - Deterministic (same inputs → same path)
        - Lex-min among all shortest paths
        - 4-conn uses orthogonal neighbors only
        - 8-conn uses orthogonal + diagonal neighbors
    """
    if anchor1 == anchor2:
        return [anchor1]

    H, W = len(grid), len(grid[0]) if grid else 0

    # Validate anchors are in bounds
    if not (0 <= anchor1.row < H and 0 <= anchor1.col < W):
        return None
    if not (0 <= anchor2.row < H and 0 <= anchor2.col < W):
        return None

    # Choose neighbor function based on metric
    if metric == "4conn":
        get_neighbors = lambda p: _get_4neighbors(p, H, W)
    elif metric == "8conn":
        get_neighbors = lambda p: _get_8neighbors(p, H, W)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # BFS to find all shortest paths
    queue = deque([(anchor1, [anchor1])])
    visited: Dict[Pixel, int] = {anchor1: 0}  # pixel -> shortest distance to reach
    all_paths: List[List[Pixel]] = []
    shortest_length: Optional[int] = None

    while queue:
        current, path = queue.popleft()

        # If we've already found shortest paths and this is longer, skip
        if shortest_length is not None and len(path) > shortest_length:
            continue

        # Check if we reached the target
        if current == anchor2:
            if shortest_length is None:
                shortest_length = len(path)
            if len(path) == shortest_length:
                all_paths.append(path)
            continue

        # Explore neighbors
        for neighbor in get_neighbors(current):
            new_distance = len(path)

            # If we haven't visited this pixel, or we're visiting via same-length path
            if neighbor not in visited or visited[neighbor] == new_distance:
                visited[neighbor] = new_distance
                queue.append((neighbor, path + [neighbor]))

    # No path found
    if not all_paths:
        return None

    # Select lex-min path among all shortest paths
    # Per clarifications §7: use global order (row-major pixel comparison)
    # Convert paths to tuples for comparison, then select min
    lex_min_path = min(all_paths, key=lambda p: tuple(p))

    return lex_min_path


# =============================================================================
# Flood Fill Algorithm
# =============================================================================


def flood_fill(grid: Grid, mask: Set[Pixel], color: int) -> Grid:
    """
    Flood fill region defined by mask with specified color.

    Args:
        grid: Input grid
        mask: Set of pixels to fill
        color: Color to fill with

    Returns:
        Grid with mask pixels set to color.

    Note:
        This is a simple direct fill (all pixels in mask set to color).
        More sophisticated flood fills (connectivity-based) would start from
        a seed pixel and expand. For ARC tasks, mask is typically pre-defined
        from present roles, so direct fill is appropriate.
    """
    result = [row[:] for row in grid]  # Deep copy
    H, W = len(grid), len(grid[0]) if grid else 0

    for pixel in mask:
        r, c = pixel.row, pixel.col
        # Check bounds
        if 0 <= r < H and 0 <= c < W:
            result[r][c] = color

    return result


# =============================================================================
# Law Application (for testing/verification)
# =============================================================================


def apply_connect_law(law: ConnectLaw, grid: Grid) -> Grid:
    """
    Apply CONNECT_ENDPOINTS law to grid.

    Draws precomputed path on grid with specified color.

    Per A1 (FY exactness) from math_spec.md:
    "Every surviving law instance reproduces training pixels exactly."

    Training data shows anchors (endpoints) must be PRESERVED:
    - Input:  [[0, 1, 0, 0, 2, 0]]  ← anchors at (0,1) and (0,4)
    - Output: [[0, 1, 5, 5, 2, 0]]  ← anchors preserved, only middle painted

    Therefore, paint only pixels BETWEEN anchors (skip first and last).

    Args:
        law: ConnectLaw instance
        grid: Input grid

    Returns:
        Grid with path drawn in line_color (anchors preserved).
    """
    result = [row[:] for row in grid]  # Deep copy
    H, W = len(grid), len(grid[0]) if grid else 0

    # Paint only pixels BETWEEN anchors (per FY exactness requirement)
    # Skip first and last pixels (the anchor endpoints)
    for i, pixel in enumerate(law.path):
        # Preserve anchor endpoints (per A1: reproduce training exactly)
        if i == 0 or i == len(law.path) - 1:
            continue

        r, c = pixel.row, pixel.col
        if 0 <= r < H and 0 <= c < W:
            result[r][c] = law.line_color

    return result


def apply_fill_law(law: FillLaw, grid: Grid, X_test_present: Grid) -> Grid:
    """
    Apply REGION_FILL law to grid.

    Computes fill color using selector on test input (recomputed per WO-13 spec),
    then floods mask with that color.

    Args:
        law: FillLaw instance
        grid: Input grid to transform
        X_test_present: Test input in canonical form (for selector evaluation)

    Returns:
        Grid with mask filled by selector color.

    Note:
        Per implementation_plan.md line 311:
        "Recompute histogram on ΠG(X_test) for non-empty masks"
    """
    # Recompute selector on test input
    fill_color, empty_mask = apply_selector_on_test(
        selector_type=law.selector_type,
        mask=set(law.mask_pixels),  # Convert frozenset to set
        X_test=X_test_present,
        k=law.selector_k
    )

    # Handle empty mask case
    if empty_mask or fill_color is None:
        # Per spec: closures will remove this expression
        # For direct application, return unchanged grid
        return grid

    # Flood fill with computed color
    return flood_fill(grid, set(law.mask_pixels), fill_color)


# =============================================================================
# Parameter Extraction and Law Building
# =============================================================================


def build_connect_fill(theta: dict) -> List[ConnectLaw | FillLaw]:
    """
    Build CONNECT_ENDPOINTS and REGION_FILL law instances from compiled parameters.

    Extracts:
    1. CONNECT laws from anchor pairs (computed shortest paths)
    2. FILL laws from masks with selectors

    All laws are verified for FY exactness on training pairs.

    Per engineering_spec.md §5.7-5.8:
    - CONNECT: "Unique shortest 4/8-connected path between two present-definable anchors"
    - FILL: "Fill/patched flood inside a present-definable mask with the selector color"

    Args:
        theta: Compiled parameters containing:
            - train_pairs: List[(Grid, Grid)] - training input/output pairs
            - anchors: List[dict] - anchor pair specifications (optional)
                Each dict: {"anchor1": Pixel, "anchor2": Pixel, "metric": str, "color": int}
            - masks: List[dict] - mask specifications (optional)
                Each dict: {"pixels": Set[Pixel], "selector": str, "k": Optional[int]}

    Returns:
        List of ConnectLaw and FillLaw instances that passed FY verification.
        Empty list if no laws could be extracted.

    Acceptance:
        - Deterministic (same theta → same laws)
        - FY exactness verified on all training pairs
        - Paths are lex-min among all shortest paths
        - Selectors use global order for tie-breaking
    """
    laws: List[ConnectLaw | FillLaw] = []

    train_pairs = theta.get("train_pairs", [])
    if not train_pairs:
        return laws

    # Get first training input for bounds/reference
    # (All train inputs should have been canonicalized to same shape via WO-04)
    X_ref = train_pairs[0][0] if train_pairs else None

    # =========================================================================
    # Build CONNECT_ENDPOINTS laws
    # =========================================================================

    anchors = theta.get("anchors", [])
    for anchor_spec in anchors:
        anchor1 = anchor_spec["anchor1"]
        anchor2 = anchor_spec["anchor2"]
        metric = anchor_spec["metric"]
        line_color = anchor_spec["color"]

        # Compute shortest path (lex-min)
        path = shortest_path_bfs(X_ref, anchor1, anchor2, metric)

        if path is None:
            # No path exists between anchors (shouldn't happen with valid anchors)
            continue

        # Create law instance
        law = ConnectLaw(
            anchor1=anchor1,
            anchor2=anchor2,
            metric=metric,
            path=tuple(path),  # Convert to tuple for immutability/hashing
            line_color=line_color
        )

        # Verify FY exactness on training pairs
        is_exact = _verify_connect_law(law, train_pairs)
        if is_exact:
            laws.append(law)

    # =========================================================================
    # Build REGION_FILL laws
    # =========================================================================

    masks = theta.get("masks", [])
    for mask_spec in masks:
        mask_pixels = mask_spec["pixels"]
        selector_type = mask_spec["selector"]
        selector_k = mask_spec.get("k")  # Optional, for MODE_kxk

        # Create law instance
        law = FillLaw(
            mask_pixels=frozenset(mask_pixels),  # Immutable for hashing
            selector_type=selector_type,
            selector_k=selector_k
        )

        # Verify FY exactness on training pairs
        is_exact = _verify_fill_law(law, train_pairs)
        if is_exact:
            laws.append(law)

    return laws


# =============================================================================
# FY Exactness Verification
# =============================================================================


def _verify_connect_law(law: ConnectLaw, train_pairs: List[Tuple[Grid, Grid]]) -> bool:
    """
    Verify CONNECT law reproduces all training outputs exactly.

    FY exactness: Drawing the path on input should match output.

    Args:
        law: ConnectLaw instance to verify
        train_pairs: List of (input, output) training pairs

    Returns:
        True if law is exact on all trains, False otherwise.
    """
    for X, Y in train_pairs:
        # Apply connect law to input
        Y_pred = apply_connect_law(law, X)

        # Check if prediction matches expected output
        if Y_pred != Y:
            return False

    return True


def _verify_fill_law(law: FillLaw, train_pairs: List[Tuple[Grid, Grid]]) -> bool:
    """
    Verify REGION_FILL law reproduces all training outputs exactly.

    FY exactness: Filling mask with selector color should match output.

    Args:
        law: FillLaw instance to verify
        train_pairs: List of (input, output) training pairs

    Returns:
        True if law is exact on all trains, False otherwise.

    Note:
        For REGION_FILL, we need to evaluate selector on each training input
        to determine the fill color, then verify the fill matches the output.
    """
    for X, Y in train_pairs:
        # Compute selector color on this training input
        fill_color, empty_mask = apply_selector_on_test(
            selector_type=law.selector_type,
            mask=set(law.mask_pixels),
            X_test=X,  # Use training input as "test" for verification
            k=law.selector_k
        )

        # Handle empty mask
        if empty_mask or fill_color is None:
            # If mask is empty, law doesn't apply to this training pair
            # Skip verification for this pair
            continue

        # Apply fill law
        Y_pred = flood_fill(X, set(law.mask_pixels), fill_color)

        # Check if prediction matches expected output
        if Y_pred != Y:
            return False

    return True


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "ConnectLaw",
    "FillLaw",
    "shortest_path_bfs",
    "flood_fill",
    "apply_connect_law",
    "apply_fill_law",
    "build_connect_fill",
]
