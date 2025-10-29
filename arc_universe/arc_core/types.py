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


@dataclass
class Lattice:
    """
    Lattice structure for periodic/tiling detection (per WO-06).

    Contains the canonical lattice basis and periods:
    - basis: 2×2 HNF basis [[v1_r, v1_c], [v2_r, v2_c]] after D8 canonicalization
    - periods: [period_row, period_col] derived from HNF
    - method: "FFT" or "KMP" indicating detection method used
    """
    basis: list[list[int]]  # 2×2 HNF basis in canonical form
    periods: list[int]  # [period_row, period_col]
    method: str  # "FFT" or "KMP"


@dataclass
class LawInstance:
    """
    Law instance representing a transformation rule (per WO-09+).

    Each law instance is a parameterized transformation that:
    - Has a specific law type (LOCAL_PAINT, TRANSLATE, PERIODIC, etc.)
    - Contains parameters (e.g., role→color mapping, Δ vector, etc.)
    - Applies to a domain of pixels
    - Must be exact on training data (checked by closures)
    """
    law_type: str  # "LOCAL_PAINT_ROLE", "TRANSLATE", "PERIODIC", etc.
    params: dict  # Law-specific parameters (e.g., {RoleId: color})
    domain_desc: str  # Human-readable domain description


@dataclass
class LocalPaintParams:
    """
    Parameters for local paint law extraction (WO-09).

    Contains all data needed to extract role→color mappings from training:
    - train_pairs: List of (input_grid, output_grid) tuples
    - presents: List of Present structures for each input grid
    - role_map: WL role assignments from wl_union (maps (grid_id, pixel) -> role)
    """
    train_pairs: list[tuple[Grid, Grid]]  # [(X_1, Y_1), ..., (X_m, Y_m)]
    presents: list[Present]  # Present structures for each X_i
    role_map: dict[tuple[int, Pixel], RoleId]  # From wl_union


# Selector types for WO-13
SelectorType = str  # "ARGMAX", "ARGMIN_NONZERO", "UNIQUE", "MODE_kxk", "PARITY"
