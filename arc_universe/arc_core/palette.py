"""
Palette canonicalization and orbit kernel (WO-05).

Per implementation_plan.md lines 173-185 and clarifications §2.

Provides:
- orbit_cprq: Detect palette clashes and return abstract color map
- canonicalize_palette: Input-lawful canonical palette (count↓, first↑, boundary-hash↑)
- Helper functions for boundary hash and color statistics

All functions deterministic, per-task scope.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from .order_hash import hash64
from .types import Grid, Hash64, Pixel, Present


@dataclass
class ColorStats:
    """Statistics for a color across all task inputs."""

    count: int  # Total pixel count across all inputs
    first_appearance: int  # Earliest scanline index (row-major)
    boundary_hash: Hash64  # Hash of E4 boundary pixels


@dataclass
class AbstractMap:
    """
    Abstract color mapping (orbit kernel result).

    When trainings recolor roles inconsistently, this holds the abstract
    mapping before canonical palette assignment.
    """

    abstract_grid: Grid  # Grid with abstract color IDs
    is_orbit: bool  # True if palette-orbit kernel was used
    role_to_abstract_color: Dict[int, int]  # Mapping from WL roles to abstract colors


def find_8connected_components(grid: Grid, target_color: int) -> List[Set[Pixel]]:
    """
    Find all 8-connected components of a specific color.

    Args:
        grid: Input grid
        target_color: Color to find components for

    Returns:
        List of components (each component is a set of pixels)
    """
    rows, cols = len(grid), len(grid[0])
    visited = set()
    components = []

    def flood_fill_8(start_pixel: Pixel) -> Set[Pixel]:
        """8-connected flood fill from start pixel."""
        component = set()
        stack = [start_pixel]

        while stack:
            pixel = stack.pop()

            if pixel in visited:
                continue

            r, c = pixel.row, pixel.col

            # Check bounds and color
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            if grid[r][c] != target_color:
                continue

            visited.add(pixel)
            component.add(pixel)

            # Add all 8 neighbors
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    neighbor = Pixel(r + dr, c + dc)
                    if neighbor not in visited:
                        stack.append(neighbor)

        return component

    # Find all components
    for r in range(rows):
        for c in range(cols):
            pixel = Pixel(r, c)
            if grid[r][c] == target_color and pixel not in visited:
                component = flood_fill_8(pixel)
                if component:
                    components.append(component)

    return components


def compute_boundary_pixels_for_color(grid: Grid, color: int) -> List[Pixel]:
    """
    Compute boundary pixels for all components of a color.

    Per clarifications §2:
    - Find all 8-connected components of this color
    - For each component, find boundary pixels (E4 neighbors differ)
    - Return sorted list of boundary pixels

    A pixel is boundary if ANY of its 4-connected neighbors (up/down/left/right)
    has a different color, OR if it's on the grid edge.

    Args:
        grid: Input grid
        color: Color to compute boundary pixels for

    Returns:
        List of boundary pixels sorted in row-major order
    """
    if not grid or len(grid) == 0:
        return []

    rows, cols = len(grid), len(grid[0])
    components = find_8connected_components(grid, color)
    boundary_pixels = []

    for comp in components:
        for pixel in comp:
            r, c = pixel.row, pixel.col

            # Check if any E4 neighbor has different color
            is_boundary = False
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                # Edge pixels are always boundary (implicit different color outside)
                if not (0 <= nr < rows and 0 <= nc < cols):
                    is_boundary = True
                    break
                if grid[nr][nc] != color:
                    is_boundary = True
                    break

            if is_boundary:
                boundary_pixels.append(pixel)

    # Sort in row-major order for determinism
    boundary_pixels.sort(key=lambda p: (p.row, p.col))

    return boundary_pixels


def compute_boundary_hash_for_color(grid: Grid, color: int) -> Hash64:
    """
    Compute boundary hash for all components of a color.

    Per clarifications §2:
    - Find all 8-connected components of this color
    - For each component, find boundary pixels (E4 neighbors differ)
    - Sort boundary pixels in row-major order
    - Hash the sorted coordinate list

    Args:
        grid: Input grid
        color: Color to compute boundary hash for

    Returns:
        64-bit hash of boundary pixels
    """
    boundary_pixels = compute_boundary_pixels_for_color(grid, color)

    # Hash the sorted coordinate list
    coord_list = [(p.row, p.col) for p in boundary_pixels]
    return Hash64(hash64(tuple(coord_list)))


def compute_color_stats(grids: List[Grid]) -> Dict[int, ColorStats]:
    """
    Compute color statistics across all grids in a task.

    Per clarifications §2:
    - Pool statistics across all grids (train ∪ test inputs)
    - For each color: count, first_appearance, boundary_hash

    Args:
        grids: List of grids (train inputs + test input)

    Returns:
        Dict mapping color -> ColorStats
    """
    color_stats: Dict[int, ColorStats] = {}

    # Global scanline index across all grids
    global_scanline_idx = 0

    for grid in grids:
        rows, cols = len(grid), len(grid[0])

        # Track which colors we've seen in this grid for boundary hash
        colors_in_grid = set()

        # Scan grid in row-major order
        for r in range(rows):
            for c in range(cols):
                color = grid[r][c]
                colors_in_grid.add(color)

                if color not in color_stats:
                    # Initialize stats for this color
                    # We'll compute boundary hash after seeing all grids
                    color_stats[color] = ColorStats(
                        count=0,
                        first_appearance=global_scanline_idx,
                        boundary_hash=Hash64(0),  # Placeholder
                    )
                else:
                    # Update first appearance (already have min from earlier grid)
                    color_stats[color].first_appearance = min(
                        color_stats[color].first_appearance, global_scanline_idx
                    )

                # Increment count
                color_stats[color].count += 1

                global_scanline_idx += 1

    # Compute boundary hashes
    # Note: Boundary hash is computed per-grid and we need to aggregate
    # For determinism, we compute boundary hash from the first grid where color appears
    for grid in grids:
        colors_in_grid = set()
        for row in grid:
            for color in row:
                colors_in_grid.add(color)

        for color in colors_in_grid:
            if color_stats[color].boundary_hash == 0:  # Not yet computed
                color_stats[color] = ColorStats(
                    count=color_stats[color].count,
                    first_appearance=color_stats[color].first_appearance,
                    boundary_hash=compute_boundary_hash_for_color(grid, color),
                )

    return color_stats


def canonicalize_palette(grid: Grid, color_stats: Dict[int, ColorStats]) -> Tuple[Grid, Dict[int, int]]:
    """
    Canonicalize palette for a grid using input-only observables.

    Per clarifications §2 and implementation_plan.md lines 180:
    - Sort colors by (count ↓, first_appearance ↑, boundary_hash ↑)
    - Assign canonical digits 0, 1, 2, ... in sorted order
    - Apply mapping to grid

    Args:
        grid: Input grid (possibly with arbitrary color indices)
        color_stats: Color statistics from compute_color_stats

    Returns:
        Tuple of (canonical_grid, palette_map) where:
        - canonical_grid: Grid with canonical color indices
        - palette_map: Dict mapping old color -> new canonical index
    """
    # Get colors present in this grid
    colors_in_grid = set()
    for row in grid:
        for color in row:
            colors_in_grid.add(color)

    # Sort colors by canonical order
    sorted_colors = sorted(
        colors_in_grid,
        key=lambda c: (
            -color_stats[c].count,  # Count descending
            color_stats[c].first_appearance,  # First appearance ascending
            color_stats[c].boundary_hash,  # Boundary hash ascending
        ),
    )

    # Create palette map
    palette_map = {old: new for new, old in enumerate(sorted_colors)}

    # Apply palette map to grid
    canonical_grid = [[palette_map[grid[r][c]] for c in range(len(grid[0]))] for r in range(len(grid))]

    return canonical_grid, palette_map


def orbit_cprq(
    train_pairs: List[Tuple[Grid, Grid]], presents: List[Present]
) -> AbstractMap:
    """
    Compute orbit kernel and abstract color map.

    Per math_spec.md §5 and implementation_plan.md lines 179:
    - Build unified shape U from training inputs (via shape maps s_i)
    - For each position u ∈ U, check training label c_i(u) = Y_i(s_i(u))
    - If c_i consistent across all i → strict kernel
    - If c_i varies by palette permutation → orbit kernel
    - Return abstract color mapping ρ̃: U/E* → Σ̄

    Args:
        train_pairs: List of (input, output) training pairs
        presents: List of Present structures for training inputs

    Returns:
        AbstractMap with abstract color mapping and orbit flag

    Algorithm:
    1. Unify shape across training inputs → U = R × C
    2. For each unified position u, collect colors c_i(u) from all training outputs
    3. Detect if palette clash exists (same structural position, different colors)
    4. If clash: compute orbit kernel (abstract colors based on equivalence)
    5. If no clash: use strict kernel (direct color mapping)
    """
    if not train_pairs:
        return AbstractMap(
            abstract_grid=[], is_orbit=False, role_to_abstract_color={}
        )

    # Import shape unification
    from .shape import unify_shape

    # Step 1: Build unified shape U from training inputs
    shape_params = unify_shape(presents)

    # For WO-05, we implement a position-based orbit detection:
    # - Map each position in outputs to a structural equivalence class
    # - Check if different training outputs assign different colors to equivalent positions

    # Step 2: Check if all training outputs have the same dimensions
    output_shapes = [(len(pair[1]), len(pair[1][0])) for pair in train_pairs]
    all_same_output_shape = len(set(output_shapes)) == 1

    if not all_same_output_shape:
        # Different output shapes → can't directly compare positions
        # Use first output as abstract grid (simplified for WO-05)
        first_output = train_pairs[0][1]
        return AbstractMap(
            abstract_grid=first_output,
            is_orbit=False,
            role_to_abstract_color={},
        )

    # Step 3: For same-shaped outputs, check position-by-position consistency
    first_output = train_pairs[0][1]
    rows, cols = len(first_output), len(first_output[0])

    # Build position-to-colors mapping
    position_colors: Dict[Tuple[int, int], List[int]] = {}

    for r in range(rows):
        for c in range(cols):
            colors_at_position = []
            for _, output_grid in train_pairs:
                colors_at_position.append(output_grid[r][c])
            position_colors[(r, c)] = colors_at_position

    # Step 4: Detect palette clash
    # If any position has different colors across training outputs → orbit kernel
    is_orbit = False
    for colors in position_colors.values():
        if len(set(colors)) > 1:
            # This position gets different colors in different training outputs
            is_orbit = True
            break

    # Step 5: Build abstract color mapping
    if is_orbit:
        # Orbit kernel: create abstract color IDs
        # Group positions by their color pattern across training outputs
        color_pattern_to_abstract: Dict[Tuple, int] = {}
        abstract_grid = [[0 for _ in range(cols)] for _ in range(rows)]
        next_abstract_id = 0

        for r in range(rows):
            for c in range(cols):
                pattern = tuple(position_colors[(r, c)])

                if pattern not in color_pattern_to_abstract:
                    color_pattern_to_abstract[pattern] = next_abstract_id
                    next_abstract_id += 1

                abstract_grid[r][c] = color_pattern_to_abstract[pattern]

        # Build role_to_abstract_color mapping
        # (For WO-05, we use position-based patterns as abstract colors)
        role_to_abstract = {}
        for (r, c), pattern in zip(
            [(r, c) for r in range(rows) for c in range(cols)],
            [position_colors[(r, c)] for r in range(rows) for c in range(cols)]
        ):
            # Map position to its abstract color
            role_id = r * cols + c  # Simple position-based role ID
            role_to_abstract[role_id] = color_pattern_to_abstract[tuple(pattern)]

        return AbstractMap(
            abstract_grid=abstract_grid,
            is_orbit=True,
            role_to_abstract_color=role_to_abstract,
        )
    else:
        # Strict kernel: all outputs consistent, use first output directly
        return AbstractMap(
            abstract_grid=first_output,
            is_orbit=False,
            role_to_abstract_color={},
        )


# ============================================================================
# Test API Wrappers
# These functions provide the API expected by tests as per test plan
# ============================================================================


def canonicalize_palette_for_task(
    train_inputs: List[Grid], test_input: Grid
) -> Dict[int, int]:
    """
    Per-task palette canonicalization (pools train∪test inputs).

    Per clarifications.md §2:
    - Pools statistics across ALL inputs (not per-grid)
    - Uses only inputs (NOT outputs)
    - Deterministic 3-level sort

    Args:
        train_inputs: List of training input grids
        test_input: Test input grid

    Returns:
        palette_map: {old_color -> canonical_index}
    """
    # Pool all inputs
    all_grids = train_inputs + [test_input]

    # Handle empty case
    if not all_grids or all(len(g) == 0 for g in all_grids):
        return {}

    # Compute color statistics across all inputs
    color_stats = compute_color_stats(all_grids)

    # Sort ALL colors from all inputs by canonical order
    sorted_colors = sorted(
        color_stats.keys(),
        key=lambda c: (
            -color_stats[c].count,  # Count descending
            color_stats[c].first_appearance,  # First appearance ascending
            color_stats[c].boundary_hash,  # Boundary hash ascending
        ),
    )

    # Create palette map for ALL colors
    palette_map = {old: new for new, old in enumerate(sorted_colors)}

    return palette_map


def compute_boundary_hash(grid: Grid, color: int) -> int:
    """
    Compute 64-bit hash of boundary pixels for a color.

    Per clarifications.md §2:
    - Components: 8-connected (8-CC)
    - Boundary detection: 4-connected (E4)
    - Pixel is boundary if ANY 4-neighbor has different color
    - Hash: SHA-256 of sorted (row, col) coordinates, truncated to 64-bit

    Args:
        grid: Input grid
        color: Color to compute boundary hash for

    Returns:
        int: 64-bit hash value
    """
    if not grid or len(grid) == 0:
        return 0

    hash_obj = compute_boundary_hash_for_color(grid, color)
    return int(hash_obj)  # Convert Hash64 to int


def compute_boundary_pixels(grid: Grid, color: int) -> List[Tuple[int, int]]:
    """
    Compute boundary pixels for a color (test API).

    Returns boundary pixels as list of (row, col) tuples for easy test verification.

    Args:
        grid: Input grid
        color: Color to compute boundary pixels for

    Returns:
        List of (row, col) tuples sorted in row-major order
    """
    if not grid or len(grid) == 0:
        return []

    pixels = compute_boundary_pixels_for_color(grid, color)
    return [(p.row, p.col) for p in pixels]


def compute_train_permutations(
    abstract_map: AbstractMap, trains: List[Tuple[Grid, Grid]]
) -> List[Dict[int, int]]:
    """
    Compute πᵢ for each train pair.

    Per math_spec.md §8 line 130:
    - For each training output Y_i, compute permutation πᵢ
    - πᵢ maps abstract colors to actual train colors
    - Verify cells_wrong_after_π = 0 (isomorphic by palette)

    Args:
        abstract_map: Abstract color mapping (from orbit_cprq)
        trains: List of (input, output) train pairs

    Returns:
        List of permutations: [{abstract_color -> train_color}, ...] one per train
    """
    if not abstract_map.is_orbit:
        # Strict kernel: no permutation needed (identity)
        return []

    if not trains:
        return []

    # For each training output, compute the permutation from abstract to actual colors
    permutations = []

    abstract_grid = abstract_map.abstract_grid
    rows, cols = len(abstract_grid), len(abstract_grid[0]) if abstract_grid else 0

    for _, train_output in trains:
        # Verify same dimensions
        if len(train_output) != rows or (rows > 0 and len(train_output[0]) != cols):
            # Dimension mismatch - skip this train
            permutations.append({})
            continue

        # Build permutation by matching abstract colors to actual colors
        perm: Dict[int, int] = {}

        for r in range(rows):
            for c in range(cols):
                abstract_color = abstract_grid[r][c]
                actual_color = train_output[r][c]

                if abstract_color in perm:
                    # Verify consistency: same abstract color should always map to same actual color
                    if perm[abstract_color] != actual_color:
                        # Inconsistent permutation - this shouldn't happen with correct orbit kernel
                        # Mark as invalid
                        perm[abstract_color] = -1  # Invalid marker
                else:
                    perm[abstract_color] = actual_color

        permutations.append(perm)

    return permutations


def verify_isomorphic(grid1: Grid, grid2: Grid, perm: Dict[int, int]) -> int:
    """
    Count cells wrong after applying permutation.

    Args:
        grid1: First grid (abstract)
        grid2: Second grid (train output)
        perm: Permutation mapping {abstract_color -> train_color}

    Returns:
        int: Number of cells that differ (should be 0 for isomorphic)
    """
    if len(grid1) != len(grid2):
        return -1  # Different dimensions

    if not grid1:
        return 0  # Empty grids are isomorphic

    if len(grid1[0]) != len(grid2[0]):
        return -1  # Different dimensions

    rows, cols = len(grid1), len(grid1[0])
    cells_wrong = 0

    for r in range(rows):
        for c in range(cols):
            abstract_color = grid1[r][c]
            expected_color = perm.get(abstract_color, abstract_color)
            actual_color = grid2[r][c]

            if expected_color != actual_color:
                cells_wrong += 1

    return cells_wrong
