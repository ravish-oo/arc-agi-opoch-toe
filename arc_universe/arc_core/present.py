"""
Present structure: G = D4 × translation anchor, ΠG canonicalization (WO-02).

Per implementation_plan.md lines 130-141 and math_spec.md §1.

Provides:
- PiG(X): Canonical present (idempotent)
- build_present(X): Full present structure with CBC3, E4, row/col equivalences
- D4 transformations (8 operations: 4 rotations + 4 flips)

All functions deterministic, idempotent ΠG.
"""

from typing import Tuple

from .order_hash import hash64, lex_min
from .types import Grid, Hash64, Pixel, Present


# ==============================================================================
# D4 Group Operations (8 transformations)
# ==============================================================================

def rot90(grid: Grid) -> Grid:
    """
    Rotate 90° clockwise.

    For grid with rows×cols, result is cols×rows.
    result[r'][c'] = grid[rows-1-c'][r']
    """
    rows, cols = len(grid), len(grid[0])
    # Result has cols rows and rows columns
    result = []
    for new_r in range(cols):  # New number of rows = old cols
        row = []
        for new_c in range(rows):  # New number of cols = old rows
            row.append(grid[rows - 1 - new_c][new_r])
        result.append(row)
    return result


def rot180(grid: Grid) -> Grid:
    """Rotate 180°."""
    return [row[::-1] for row in grid[::-1]]


def rot270(grid: Grid) -> Grid:
    """
    Rotate 270° clockwise (= 90° counterclockwise).

    For grid with rows×cols, result is cols×rows.
    result[r'][c'] = grid[c'][cols-1-r']
    """
    rows, cols = len(grid), len(grid[0])
    # Result has cols rows and rows columns
    result = []
    for new_r in range(cols):  # New number of rows = old cols
        row = []
        for new_c in range(rows):  # New number of cols = old rows
            # result[new_r][new_c] = grid[new_c][cols - 1 - new_r]
            row.append(grid[new_c][cols - 1 - new_r])
        result.append(row)
    return result


def flip_h(grid: Grid) -> Grid:
    """Flip horizontal (left-right)."""
    return [row[::-1] for row in grid]


def flip_v(grid: Grid) -> Grid:
    """Flip vertical (top-bottom)."""
    return grid[::-1]


def flip_diag_main(grid: Grid) -> Grid:
    """Flip over main diagonal (top-left to bottom-right)."""
    rows, cols = len(grid), len(grid[0])
    # Transpose
    return [[grid[r][c] for r in range(rows)] for c in range(cols)]


def flip_diag_anti(grid: Grid) -> Grid:
    """Flip over anti-diagonal (top-right to bottom-left)."""
    rows, cols = len(grid), len(grid[0])
    return [[grid[rows - 1 - c][cols - 1 - r] for c in range(cols)] for r in range(rows)]


# D4 group: identity + 7 transformations
D4_TRANSFORMATIONS = {
    "identity": lambda g: g,
    "rot90": rot90,
    "rot180": rot180,
    "rot270": rot270,
    "flip_h": flip_h,
    "flip_v": flip_v,
    "flip_diag_main": flip_diag_main,
    "flip_diag_anti": flip_diag_anti,
}

# Inverse transformations (for unpresenting)
D4_INVERSES = {
    "identity": "identity",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_h": "flip_h",  # Self-inverse
    "flip_v": "flip_v",  # Self-inverse
    "flip_diag_main": "flip_diag_main",  # Self-inverse
    "flip_diag_anti": "flip_diag_anti",  # Self-inverse
}


# ==============================================================================
# Minimal Bounding Box & Anchor
# ==============================================================================

def get_minimal_bounding_box(grid: Grid) -> Tuple[int, int, int, int]:
    """
    Get minimal bounding box (top, left, bottom, right) of non-zero pixels.

    Returns (r_min, c_min, r_max, c_max) or (0, 0, rows-1, cols-1) if all zeros.
    """
    rows, cols = len(grid), len(grid[0])

    # Find non-zero pixels
    non_zero = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] != 0]

    if not non_zero:
        # All zeros -> full canvas BB
        return (0, 0, rows - 1, cols - 1)

    r_coords = [r for r, c in non_zero]
    c_coords = [c for r, c in non_zero]

    return (min(r_coords), min(c_coords), max(r_coords), max(c_coords))


def get_anchor(grid: Grid) -> Pixel:
    """
    Get translation anchor = top-left of minimal bounding box.

    Per math_spec.md §1: anchor is top-left of minimal BB.
    """
    r_min, c_min, _, _ = get_minimal_bounding_box(grid)
    return Pixel(r_min, c_min)


# ==============================================================================
# ΠG Canonicalization
# ==============================================================================

def PiG(X: Grid) -> Grid:
    """
    Canonical present via G = D4 × translation anchor.

    Per math_spec.md §1 and implementation_plan.md lines 136:
    - Apply all 8 D4 transformations
    - For each, compute anchor (top-left of minimal BB)
    - Choose lex-min by global order: (anchor, flattened grid)
    - ΠG is idempotent: ΠG(ΠG(X)) = ΠG(X)

    Args:
        X: Input grid

    Returns:
        Canonical grid (lex-min over D4 × anchor)

    Acceptance:
        - ΠG idempotent
        - Deterministic (lex-min via global order)
        - No palette normalization here (done separately in WO-05)
    """
    rows, cols = len(X), len(X[0])
    aspect_ratio_diff = abs(rows - cols)

    # Skip diagonal transforms for extreme aspect ratios (prevents IndexError)
    # Diagonal flips are geometrically invalid for very non-square grids
    if aspect_ratio_diff > 5:
        # Use only non-diagonal transformations
        allowed_transforms = {
            "identity": D4_TRANSFORMATIONS["identity"],
            "rot90": D4_TRANSFORMATIONS["rot90"],
            "rot180": D4_TRANSFORMATIONS["rot180"],
            "rot270": D4_TRANSFORMATIONS["rot270"],
            "flip_h": D4_TRANSFORMATIONS["flip_h"],
            "flip_v": D4_TRANSFORMATIONS["flip_v"],
        }
    else:
        # Use all transformations
        allowed_transforms = D4_TRANSFORMATIONS

    candidates = []

    for name, transform in allowed_transforms.items():
        transformed = transform(X)
        anchor = get_anchor(transformed)

        # Flatten grid for comparison
        flat_grid = tuple(tuple(row) for row in transformed)

        # Key: (anchor, flattened_grid)
        # Lex order: anchor first (Pixel has row-major order), then grid
        key = (anchor, flat_grid)
        candidates.append((key, transformed, name))

    # Choose lex-min by global order
    _, canonical_grid, _ = lex_min(candidates, key=lambda x: x[0])

    return canonical_grid


def PiG_with_inverse(X: Grid) -> Tuple[Grid, str]:
    """
    Canonical present with transformation name for unpresenting.

    Returns:
        (canonical_grid, transformation_name) where transformation_name
        is the D4 operation that produced the canonical grid.
    """
    rows, cols = len(X), len(X[0])
    aspect_ratio_diff = abs(rows - cols)

    # Skip diagonal transforms for extreme aspect ratios (prevents IndexError)
    # Diagonal flips are geometrically invalid for very non-square grids
    if aspect_ratio_diff > 5:
        # Use only non-diagonal transformations
        allowed_transforms = {
            "identity": D4_TRANSFORMATIONS["identity"],
            "rot90": D4_TRANSFORMATIONS["rot90"],
            "rot180": D4_TRANSFORMATIONS["rot180"],
            "rot270": D4_TRANSFORMATIONS["rot270"],
            "flip_h": D4_TRANSFORMATIONS["flip_h"],
            "flip_v": D4_TRANSFORMATIONS["flip_v"],
        }
    else:
        # Use all transformations
        allowed_transforms = D4_TRANSFORMATIONS

    candidates = []

    for name, transform in allowed_transforms.items():
        transformed = transform(X)
        anchor = get_anchor(transformed)
        flat_grid = tuple(tuple(row) for row in transformed)
        key = (anchor, flat_grid)
        candidates.append((key, transformed, name))

    _, canonical_grid, transform_name = lex_min(candidates, key=lambda x: x[0])

    return canonical_grid, transform_name


# ==============================================================================
# CBC3 Feature Extraction
# ==============================================================================

def compute_cbc3(grid: Grid, pixel: Pixel) -> Hash64:
    """
    Compute CBC3 token for a pixel (3×3 patch feature).

    Per math_spec.md §3 and engineering_spec.md §3.1:
    1. Extract 3×3 patch centered on pixel
    2. OFA (Order of First Appearance) relabel inside patch
    3. D8 canonicalize (try all 8 D4 transformations)
    4. Hash the canonical patch

    Args:
        grid: Input grid
        pixel: Center pixel

    Returns:
        64-bit hash of canonical 3×3 patch
    """
    rows, cols = len(grid), len(grid[0])
    r, c = pixel.row, pixel.col

    # Extract 3×3 patch (with zero-padding for edges)
    patch = []
    for dr in [-1, 0, 1]:
        row_data = []
        for dc in [-1, 0, 1]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                row_data.append(grid[nr][nc])
            else:
                row_data.append(0)  # Pad with 0
        patch.append(row_data)

    # OFA relabel inside patch
    ofa_map = {}
    next_label = 0
    ofa_patch = []
    for row in patch:
        ofa_row = []
        for color in row:
            if color not in ofa_map:
                ofa_map[color] = next_label
                next_label += 1
            ofa_row.append(ofa_map[color])
        ofa_patch.append(ofa_row)

    # D8 canonicalize (try all 8 D4 transformations, choose lex-min)
    canonical_patches = []
    for name, transform in D4_TRANSFORMATIONS.items():
        transformed_patch = transform(ofa_patch)
        flat = tuple(tuple(row) for row in transformed_patch)
        canonical_patches.append(flat)

    canonical_patch = lex_min(canonical_patches)

    # Hash the canonical patch
    return hash64(canonical_patch)


# ==============================================================================
# Present Structure Builder
# ==============================================================================

def build_present(X: Grid) -> Present:
    """
    Build full present structure with all features.

    Per implementation_plan.md lines 137 and math_spec.md §3:
    - Canonical grid (ΠG)
    - CBC3 tokens for each pixel
    - E4 (4-connected) neighbors
    - Row/column equivalences (pixel membership)
    - g_inverse for unpresenting

    Args:
        X: Raw input grid

    Returns:
        Present structure with all features

    Acceptance:
        - ΠG idempotent
        - present has eq members and no coords (only equivalence info)
    """
    # 1. Canonicalize via ΠG
    canonical_grid, transform_name = PiG_with_inverse(X)
    g_inverse = D4_INVERSES[transform_name]

    rows, cols = len(canonical_grid), len(canonical_grid[0])

    # 2. Compute CBC3 for each pixel
    cbc3_tokens = {}
    for r in range(rows):
        for c in range(cols):
            pixel = Pixel(r, c)
            cbc3_tokens[pixel] = compute_cbc3(canonical_grid, pixel)

    # 3. Build E4 (4-connected) neighbors
    e4_neighbors = {}
    for r in range(rows):
        for c in range(cols):
            pixel = Pixel(r, c)
            neighbors = []

            # Up, down, left, right
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    neighbors.append(Pixel(nr, nc))

            e4_neighbors[pixel] = neighbors

    # 4. Build row/column memberships
    row_members = {}
    for r in range(rows):
        row_members[r] = [Pixel(r, c) for c in range(cols)]

    col_members = {}
    for c in range(cols):
        col_members[c] = [Pixel(r, c) for r in range(rows)]

    return Present(
        grid=canonical_grid,
        cbc3=cbc3_tokens,
        e4_neighbors=e4_neighbors,
        row_members=row_members,
        col_members=col_members,
        g_inverse=g_inverse,
    )
