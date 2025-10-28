"""
Core type definitions for the ARC-AGI solver.

Per implementation_plan.md WO-01 and clarifications §2.
"""

from dataclasses import dataclass
from typing import NewType

# Grid representation
Grid = list[list[int]]  # Grid[r][c] = color ∈ {0..9}

# Pixel coordinates (row, col)
@dataclass(frozen=True, order=True)
class Pixel:
    """Pixel coordinates in row-major order (per clarifications §2)."""
    row: int
    col: int

    def __iter__(self):
        """Allow tuple unpacking: r, c = pixel"""
        return iter((self.row, self.col))


# Role IDs from WL
RoleId = NewType("RoleId", int)

# Equivalence class IDs
EquivId = NewType("EquivId", int)

# Component ID (deterministic ordering per clarifications §3)
ComponentId = NewType("ComponentId", int)

# Hash type (64-bit from SHA-256)
Hash64 = NewType("Hash64", int)


@dataclass
class Present:
    """
    Present structure for a grid (per WO-02).

    Contains the canonicalized grid and all features needed for WL:
    - grid: The canonical grid (after ΠG)
    - cbc3: CBC3 tokens for each pixel (3×3 patch features)
    - e4_neighbors: E4 (4-connected) adjacency
    - row_members: Pixels grouped by row
    - col_members: Pixels grouped by column
    - g_inverse: Transformation to unpresent (g_test^{-1})
    """
    grid: Grid
    cbc3: dict[Pixel, Hash64]  # CBC3 token per pixel
    e4_neighbors: dict[Pixel, list[Pixel]]  # 4-connected neighbors
    row_members: dict[int, list[Pixel]]  # Pixels per row
    col_members: dict[int, list[Pixel]]  # Pixels per column
    g_inverse: str  # Transformation name to unpresent (e.g., "rot90", "flip_h")
