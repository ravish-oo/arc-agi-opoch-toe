#!/usr/bin/env python3
"""
Diagnostic WL convergence checker.

This is a wrapper around wl_union that tracks actual convergence iteration
to verify early stopping is working correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Tuple

from arc_core.order_hash import hash64
from arc_core.types import Hash64, Pixel, Present, RoleId


def wl_union_instrumented(
    presents: List[Present], escalate: bool = False, max_iters: int = 12
) -> Tuple[Dict[Tuple[int, Pixel], RoleId], int]:
    """
    1-WL with iteration tracking for diagnostics.

    Returns:
        (role_map, actual_iterations)
    """
    # Build pixel set for each grid
    grid_pixels: Dict[int, List[Pixel]] = {}
    for grid_id, present in enumerate(presents):
        rows, cols = len(present.grid), len(present.grid[0])
        grid_pixels[grid_id] = [Pixel(r, c) for r in range(rows) for c in range(cols)]

    # Initialize colors
    colors: Dict[Tuple[int, Pixel], int] = {}
    sig2id: Dict[Tuple, int] = {}
    next_id = 0

    for grid_id, present in enumerate(presents):
        for pixel in grid_pixels[grid_id]:
            raw_color = present.grid[pixel.row][pixel.col]
            cbc3_token = present.cbc3[pixel]
            seed_sig = (raw_color, cbc3_token)

            if seed_sig not in sig2id:
                sig2id[seed_sig] = next_id
                next_id += 1

            colors[(grid_id, pixel)] = sig2id[seed_sig]

    # Iterate until stable
    actual_iters = 0
    for iteration in range(max_iters):
        actual_iters = iteration + 1
        new_colors: Dict[Tuple[int, Pixel], int] = {}
        sig2id = {}
        next_id = 0
        changed = False

        for grid_id, present in enumerate(presents):
            for pixel in grid_pixels[grid_id]:
                current_color = colors[(grid_id, pixel)]

                # E4 neighbors
                e4_neighbors = present.e4_neighbors.get(pixel, [])
                e4_bag = sorted([colors[(grid_id, nbr)] for nbr in e4_neighbors])

                # E8 neighbors
                e8_bag = []
                if escalate:
                    rows, cols = len(present.grid), len(present.grid[0])
                    r, c = pixel.row, pixel.col

                    all_8_neighbors = []
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                all_8_neighbors.append(Pixel(nr, nc))

                    e8_bag = sorted([colors[(grid_id, nbr)] for nbr in all_8_neighbors])

                # Row bag
                row_idx = pixel.row
                row_members = present.row_members.get(row_idx, [])
                row_bag = sorted([colors[(grid_id, p)] for p in row_members if p != pixel])

                # Column bag
                col_idx = pixel.col
                col_members = present.col_members.get(col_idx, [])
                col_bag = sorted([colors[(grid_id, p)] for p in col_members if p != pixel])

                # Build signature
                if escalate:
                    sig = (current_color, tuple(e4_bag), tuple(row_bag), tuple(col_bag), tuple(e8_bag))
                else:
                    sig = (current_color, tuple(e4_bag), tuple(row_bag), tuple(col_bag))

                # Assign ID
                if sig not in sig2id:
                    sig2id[sig] = next_id
                    next_id += 1

                new_colors[(grid_id, pixel)] = sig2id[sig]

                if sig2id[sig] != current_color:
                    changed = True

        # Update
        colors = new_colors

        # Check stability - THIS IS THE KEY CHECK
        if not changed:
            # Early convergence detected!
            break

    # Convert to RoleId
    role_map: Dict[Tuple[int, Pixel], RoleId] = {
        key: RoleId(color) for key, color in colors.items()
    }

    return role_map, actual_iters
