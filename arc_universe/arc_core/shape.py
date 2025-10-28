"""
Shape unification via 1-D WL meet (WO-04).

Per implementation_plan.md lines 159-169 and math_spec.md §4.

Provides:
- unify_shape: Compute unified shape U = R×C from training inputs
- 1-D WL on rows and columns
- Meet operation on partitions

All functions deterministic, no P-catalog search.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .order_hash import hash64
from .types import Grid, Present

# Partition: maps element index -> equivalence class ID
Partition = Dict[int, int]


@dataclass
class ShapeParams:
    """
    Shape unification result (U = R×C).

    Per math_spec.md §4:
    - R: unified row partition (meet of all R_i)
    - C: unified column partition (meet of all C_i)
    - shape_maps: for each training input, maps from unified domain to input domain
    """

    R: Partition  # Row partition: row_idx -> row_class_id
    C: Partition  # Column partition: col_idx -> col_class_id
    num_row_classes: int  # Number of equivalence classes in R
    num_col_classes: int  # Number of equivalence classes in C
    # Shape maps s_i for each training input (projections R_i -> R, C_i -> C)
    # For now, we store the original partitions R_i, C_i for each training input
    row_partitions: List[Partition]  # R_i for each training input
    col_partitions: List[Partition]  # C_i for each training input


def wl_1d_on_rows(grid: Grid, max_iters: int = 12) -> Partition:
    """
    Run 1-D WL on rows of a grid.

    Each row is treated as a node. Initial color is hash of row contents.
    WL iterations refine based on adjacent rows (neighbors in 1-D).

    Args:
        grid: Input grid
        max_iters: Maximum WL iterations

    Returns:
        Partition: row_idx -> equivalence_class_id
    """
    rows = len(grid)

    # Initialize: hash of row contents
    colors: Dict[int, int] = {}
    sig2id: Dict[Tuple, int] = {}
    next_id = 0

    for r in range(rows):
        row_tuple = tuple(grid[r])
        row_hash = hash64(row_tuple)

        if row_hash not in sig2id:
            sig2id[row_hash] = next_id
            next_id += 1

        colors[r] = sig2id[row_hash]

    # WL iterations: refine based on adjacent rows
    for iteration in range(max_iters):
        new_colors: Dict[int, int] = {}
        sig2id = {}
        next_id = 0
        changed = False

        for r in range(rows):
            current_color = colors[r]

            # Neighbors: adjacent rows (r-1, r+1 in 1-D)
            neighbors = []
            if r > 0:
                neighbors.append(colors[r - 1])
            if r < rows - 1:
                neighbors.append(colors[r + 1])

            # Signature: (current_color, sorted_neighbor_colors)
            sig = (current_color, tuple(sorted(neighbors)))

            if sig not in sig2id:
                sig2id[sig] = next_id
                next_id += 1

            new_colors[r] = sig2id[sig]

            if sig2id[sig] != current_color:
                changed = True

        colors = new_colors

        if not changed:
            break

    return colors


def wl_1d_on_cols(grid: Grid, max_iters: int = 12) -> Partition:
    """
    Run 1-D WL on columns of a grid.

    Each column is treated as a node. Initial color is hash of column contents.
    WL iterations refine based on adjacent columns (neighbors in 1-D).

    Args:
        grid: Input grid
        max_iters: Maximum WL iterations

    Returns:
        Partition: col_idx -> equivalence_class_id
    """
    rows, cols = len(grid), len(grid[0])

    # Initialize: hash of column contents
    colors: Dict[int, int] = {}
    sig2id: Dict[Tuple, int] = {}
    next_id = 0

    for c in range(cols):
        col_tuple = tuple(grid[r][c] for r in range(rows))
        col_hash = hash64(col_tuple)

        if col_hash not in sig2id:
            sig2id[col_hash] = next_id
            next_id += 1

        colors[c] = sig2id[col_hash]

    # WL iterations: refine based on adjacent columns
    for iteration in range(max_iters):
        new_colors: Dict[int, int] = {}
        sig2id = {}
        next_id = 0
        changed = False

        for c in range(cols):
            current_color = colors[c]

            # Neighbors: adjacent columns (c-1, c+1 in 1-D)
            neighbors = []
            if c > 0:
                neighbors.append(colors[c - 1])
            if c < cols - 1:
                neighbors.append(colors[c + 1])

            # Signature: (current_color, sorted_neighbor_colors)
            sig = (current_color, tuple(sorted(neighbors)))

            if sig not in sig2id:
                sig2id[sig] = next_id
                next_id += 1

            new_colors[c] = sig2id[sig]

            if sig2id[sig] != current_color:
                changed = True

        colors = new_colors

        if not changed:
            break

    return colors


def meet_partitions(partitions: List[Partition]) -> Partition:
    """
    Compute the meet of multiple partitions.

    The meet is the finest partition such that if two elements are in the same
    class in the meet, they must be in the same class in ALL input partitions.

    Args:
        partitions: List of partitions (each is dict: element -> class_id)

    Returns:
        Partition: element -> meet_class_id

    Note: All partitions must have the same elements (same keys).
    """
    if not partitions:
        return {}

    if len(partitions) == 1:
        # Re-normalize to start from 0
        p = partitions[0]
        unique_classes = sorted(set(p.values()))
        class_map = {old: new for new, old in enumerate(unique_classes)}
        return {k: class_map[v] for k, v in p.items()}

    # Get all elements (should be same across all partitions)
    elements = sorted(partitions[0].keys())

    # For each element, create a signature: tuple of its class in each partition
    signatures: Dict[int, Tuple] = {}
    for elem in elements:
        sig = tuple(p[elem] for p in partitions)
        signatures[elem] = sig

    # Assign meet class IDs based on unique signatures
    sig2id: Dict[Tuple, int] = {}
    next_id = 0
    meet: Dict[int, int] = {}

    for elem in elements:
        sig = signatures[elem]
        if sig not in sig2id:
            sig2id[sig] = next_id
            next_id += 1
        meet[elem] = sig2id[sig]

    return meet


def unify_shape(presents_train: List[Present]) -> ShapeParams:
    """
    Unify shape across training inputs via 1-D WL meet.

    Per math_spec.md §4:
    - For each train input, run 1-D WL on rows/cols → R_i, C_i
    - Compute meet: R = ∧R_i, C = ∧C_i
    - Result: U = R × C (unified domain)

    Args:
        presents_train: List of Present structures for training inputs

    Returns:
        ShapeParams: Unified shape with row/col partitions and shape maps

    Acceptance:
        - Deterministic (same result for permuted trains)
        - Meet operation correct (finest common coarsening)
        - No P-catalog search (all input-derived)
    """
    if not presents_train:
        # Empty input case
        return ShapeParams(R={}, C={}, num_row_classes=0, num_col_classes=0, row_partitions=[], col_partitions=[])

    # For each training input, run 1-D WL on rows and columns
    row_partitions: List[Partition] = []
    col_partitions: List[Partition] = []

    for present in presents_train:
        grid = present.grid

        # Run 1-D WL on rows
        R_i = wl_1d_on_rows(grid)
        row_partitions.append(R_i)

        # Run 1-D WL on columns
        C_i = wl_1d_on_cols(grid)
        col_partitions.append(C_i)

    # Compute meet of row partitions
    # Note: Different grids may have different numbers of rows!
    # We need to handle this carefully.

    # Strategy: Only compute meet for grids with same dimensions
    # If all grids have same shape, compute meet normally
    # Otherwise, return identity partitions (each row/col is its own class)

    # Check if all grids have same dimensions
    shapes = [(len(p.grid), len(p.grid[0])) for p in presents_train]
    all_same_shape = len(set(shapes)) == 1

    if all_same_shape:
        # All grids have same shape, compute meet
        R = meet_partitions(row_partitions)
        C = meet_partitions(col_partitions)
    else:
        # Different shapes: use identity partitions for the first grid's dimensions
        # This is a fallback - the spec assumes same-shaped inputs for unification
        first_rows, first_cols = shapes[0]
        R = {r: r for r in range(first_rows)}
        C = {c: c for c in range(first_cols)}

    num_row_classes = len(set(R.values()))
    num_col_classes = len(set(C.values()))

    return ShapeParams(
        R=R, C=C, num_row_classes=num_row_classes, num_col_classes=num_col_classes, row_partitions=row_partitions, col_partitions=col_partitions
    )
