"""
1-WL on disjoint union (train ∪ test) (WO-03).

Per implementation_plan.md lines 145-155 and clarifications §1.

Provides:
- wl_union: 1-WL fixed point on union of presents
- E8 escalation toggle
- Shared role IDs (no test-only roles)

All functions deterministic, stable role IDs.
"""

from typing import Dict, List, Tuple

from .order_hash import hash64
from .types import Hash64, Pixel, Present, RoleId

# RoleMap: maps (grid_id, pixel) -> role_id
RoleMap = Dict[Tuple[int, Pixel], RoleId]


def wl_union(presents: List[Present], escalate: bool = False, max_iters: int = 12) -> RoleMap:
    """
    1-WL fixed point on disjoint union of presents.

    Per clarifications §1 and partition refinement:
    - Seed: hash(RAW[p], CBC3[p])
    - Update: signature-based partition refinement (no hash collisions)
    - BAG = sorted list of colors
    - Stop when stable (guaranteed convergence via monotone refinement)

    Args:
        presents: List of Present structures (train inputs + test input)
        escalate: If True, include E8 (8-connected) neighbors in WL update
        max_iters: Maximum iterations (default 12)

    Returns:
        RoleMap: Dict mapping (grid_id, pixel) -> role_id
        Role IDs are shared across all grids (no test-only roles)

    Acceptance:
        - Deterministic (stable role IDs)
        - E8 only when escalate=True
        - No coords in features (only bag-hashes)
        - Monotone partition refinement (classes only split, never merge)
    """
    # Build pixel set for each grid
    grid_pixels: Dict[int, List[Pixel]] = {}
    for grid_id, present in enumerate(presents):
        rows, cols = len(present.grid), len(present.grid[0])
        grid_pixels[grid_id] = [Pixel(r, c) for r in range(rows) for c in range(cols)]

    # Initialize colors: seed = hash(RAW[p], CBC3[p])
    # Use partition refinement from the start
    colors: Dict[Tuple[int, Pixel], int] = {}
    sig2id: Dict[Tuple, int] = {}
    next_id = 0

    for grid_id, present in enumerate(presents):
        for pixel in grid_pixels[grid_id]:
            raw_color = present.grid[pixel.row][pixel.col]
            cbc3_token = present.cbc3[pixel]
            # Signature for seed (not hash - use actual values)
            seed_sig = (raw_color, cbc3_token)

            if seed_sig not in sig2id:
                sig2id[seed_sig] = next_id
                next_id += 1

            colors[(grid_id, pixel)] = sig2id[seed_sig]

    # Iterate until stable (partition refinement)
    for iteration in range(max_iters):
        new_colors: Dict[Tuple[int, Pixel], int] = {}
        sig2id = {}  # Reset signature mapping each iteration
        next_id = 0
        changed = False

        for grid_id, present in enumerate(presents):
            for pixel in grid_pixels[grid_id]:
                current_color = colors[(grid_id, pixel)]

                # Collect neighbor bags
                # 1. E4 neighbors
                e4_neighbors = present.e4_neighbors.get(pixel, [])
                e4_bag = sorted([colors[(grid_id, nbr)] for nbr in e4_neighbors])

                # 2. E8 neighbors (if escalate=True)
                e8_bag = []
                if escalate:
                    # E8 = all 8 neighbors (E4 + diagonals)
                    rows, cols = len(present.grid), len(present.grid[0])
                    r, c = pixel.row, pixel.col

                    # Get all 8 neighbors (including diagonals)
                    all_8_neighbors = []
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue  # Skip self
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                all_8_neighbors.append(Pixel(nr, nc))

                    e8_bag = sorted([colors[(grid_id, nbr)] for nbr in all_8_neighbors])

                # 3. Row equivalence bag (exclude pixel itself)
                row_idx = pixel.row
                row_members = present.row_members.get(row_idx, [])
                row_bag = sorted([colors[(grid_id, p)] for p in row_members if p != pixel])

                # 4. Column equivalence bag (exclude pixel itself)
                col_idx = pixel.col
                col_members = present.col_members.get(col_idx, [])
                col_bag = sorted([colors[(grid_id, p)] for p in col_members if p != pixel])

                # Build signature (full tuple, not hash!)
                if escalate:
                    sig = (current_color, tuple(e4_bag), tuple(row_bag), tuple(col_bag), tuple(e8_bag))
                else:
                    sig = (current_color, tuple(e4_bag), tuple(row_bag), tuple(col_bag))

                # Assign fresh ID to unique signatures (partition refinement)
                if sig not in sig2id:
                    sig2id[sig] = next_id
                    next_id += 1

                new_colors[(grid_id, pixel)] = sig2id[sig]

                if sig2id[sig] != current_color:
                    changed = True

        # Update colors
        colors = new_colors

        # Check stability
        if not changed:
            break

    # Colors are already 0..N-1, just convert to RoleId
    role_map: RoleMap = {key: RoleId(color) for key, color in colors.items()}

    return role_map
