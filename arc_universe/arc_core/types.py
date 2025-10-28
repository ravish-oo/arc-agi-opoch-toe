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
