"""
Periodic/tiling laws (WO-11).

Per implementation_plan.md lines 285-288 and engineering_spec.md §5.3.

Provides:
- build_periodic(theta): Discovers periodic lattice structure and phase roles
- PeriodicStructure: Lattice, phase_table, phase_masks, T_periodic closure
- Optional writers (Tile/PeriodicRewrite) if proven exact on trains

Key principle (per clarification):
- WO-11 = Structure discovery (lattice + phase roles) + T_periodic constraint
- WO-09/WO-05 = Color painting (using phase_masks from WO-11)
- Separation of concerns: structure vs coloring

Algorithm:
1. Integer ACF (pooled across train ∪ test) - no floats, deterministic
2. Tier-1: Axis DC check for diagonal HNF
3. Tier-2: 2-D basis extraction (minimal norm, lex-min, HNF, D8)
4. Build phase_table: φ(r,c) = (r mod p_r, c mod p_c)
5. Create phase_masks (boolean masks per phase ID)
6. T_periodic closure: structural constraint (removes phase-breaking expressions)
7. Optional writers: Tile(m) / PeriodicRewrite(L,h) if proven on trains

Per A1 (FY exactness): All structures verified exact on training data.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from arc_core.types import Grid, Lattice
from arc_core.order_hash import lex_min
from arc_core.lattice import infer_lattice

# STUB FIX: Import from WO-15 and WO-16 for stub resolution
from arc_fixedpoint.expressions import PeriodicExpr
from arc_fixedpoint.closures import apply_lattice_closure


# =============================================================================
# Types
# =============================================================================


@dataclass(frozen=True)
class PeriodicStructure:
    """
    Periodic structure for a task (WO-11 output).

    Contains lattice, phase roles, structural closure, and optional writers.

    Per clarification:
    - lattice: The periodic basis (from WO-06 style detection)
    - phase_table: Phase ID for each pixel position
    - phase_masks: Boolean masks per phase (for WO-09/05 to use)
    - closure: T_periodic structural constraint (apply_lattice_closure from WO-16)
    - writers: PeriodicExpr instances for consistent phases (from WO-15, proven exact)
    """
    lattice: Lattice                    # From WO-06 (basis, periods, method)
    phase_table: np.ndarray             # H×W int array, phase_id per pixel
    phase_masks: List[np.ndarray]       # List of H×W bool masks per phase_id
    num_phases: int                     # Total number of phases (p_r × p_c)

    # WO-16 complete: T_periodic structural constraint (apply_lattice_closure)
    closure: Any  # Callable: apply_lattice_closure from WO-16

    # WO-15 complete: List of PeriodicExpr instances for consistent phases
    writers: List[Any]  # List[PeriodicExpr] from WO-15 (proven exact on trains)


# =============================================================================
# Note: Lattice detection delegated to WO-06 (arc_core.lattice)
# =============================================================================
#
# WO-11 builds on top of WO-06's lattice detection:
# - WO-06: Discovers lattice structure (ACF → HNF → D8 canonical)
# - WO-11: Adds phase roles, T_periodic closure, optional writers
#
# This follows the separation of concerns principle:
# - WO-06: General lattice detection utility
# - WO-11: Periodic law specialization with phase structure
#
# Per implementation_plan.md: WO-06 (completed) provides full HNF/D8 logic.


# =============================================================================
# Phase Table and Masks
# =============================================================================


def build_phase_table(H: int, W: int, p_r: int, p_c: int) -> np.ndarray:
    """
    Build phase table: φ(r,c) = (r mod p_r, c mod p_c).

    Flatten to phase_id = r_mod * p_c + c_mod for storage.

    Args:
        H: Grid height
        W: Grid width
        p_r: Row period
        p_c: Column period

    Returns:
        H×W array of phase IDs in range [0, p_r*p_c-1]
    """
    phase_table = np.zeros((H, W), dtype=np.int32)

    for r in range(H):
        for c in range(W):
            r_mod = r % p_r
            c_mod = c % p_c
            phase_id = r_mod * p_c + c_mod
            phase_table[r, c] = phase_id

    return phase_table


def build_phase_masks(phase_table: np.ndarray, num_phases: int) -> List[np.ndarray]:
    """
    Build boolean masks for each phase ID.

    Args:
        phase_table: H×W array of phase IDs
        num_phases: Total number of phases (p_r × p_c)

    Returns:
        List of H×W boolean masks, one per phase_id
    """
    H, W = phase_table.shape
    phase_masks = []

    for phase_id in range(num_phases):
        mask = (phase_table == phase_id)
        phase_masks.append(mask)

    return phase_masks


# =============================================================================
# Helper: Infer Phase Colors
# =============================================================================


def _infer_phase_colors(
    phase_table: np.ndarray,
    num_phases: int,
    train_pairs: List[Tuple[Grid, Grid]]
) -> Dict[int, int]:
    """
    Infer color for each phase from training data.

    Per A1 (FY exactness): Only create writers for phases that are consistent
    across all training pairs (all pixels in same phase → same output color).

    Args:
        phase_table: H×W array of phase IDs
        num_phases: Total number of phases
        train_pairs: Training input/output pairs

    Returns:
        Dictionary mapping phase_id -> color for consistent phases only.
        Phases with inconsistent colors are excluded.

    Algorithm:
        1. For each phase_id, collect all (input_color, output_color) pairs
        2. If all pixels in phase have same output color → consistent
        3. If different output colors → inconsistent, exclude from writers
    """
    if not train_pairs:
        return {}

    phase_colors: Dict[int, int] = {}

    # For each phase, check consistency across training pairs
    for phase_id in range(num_phases):
        # Find pixels belonging to this phase
        phase_pixels = np.argwhere(phase_table == phase_id)

        if len(phase_pixels) == 0:
            continue

        # Collect output colors for this phase across all training pairs
        output_colors_seen = set()

        for X_i, Y_i in train_pairs:
            rows_y, cols_y = len(Y_i), len(Y_i[0])

            for r, c in phase_pixels:
                # Check if pixel is within training output bounds
                if r < rows_y and c < cols_y:
                    output_color = Y_i[r][c]
                    output_colors_seen.add(output_color)

        # Check consistency: all pixels in phase should map to same output color
        if len(output_colors_seen) == 1:
            # Consistent phase → can create writer
            phase_colors[phase_id] = output_colors_seen.pop()
        # If len > 1: inconsistent phase → skip (no writer for this phase)

    return phase_colors


# =============================================================================
# Main Entry Point
# =============================================================================


def build_periodic(theta: dict) -> Optional[PeriodicStructure]:
    """
    Build periodic structure from compiled parameters.

    Per clarification:
    "WO-11 discovers structure: a 2-D lattice L and the phase roles (residue classes)
     each pixel belongs to. It does not paint by itself."

    Algorithm:
    1. Use WO-06's infer_lattice to discover periodic structure (ACF → HNF → D8)
    2. Extract periods from lattice
    3. Build phase_table: φ(r,c) = (r mod p_r, c mod p_c)
    4. Create phase_masks (boolean masks per phase)
    5. T_periodic closure: apply_lattice_closure (WO-16) - structural constraint
    6. Optional writers: PeriodicExpr instances (WO-15) - if proven exact on trains

    Args:
        theta: Compiled parameters containing:
            - grids_present: List[Grid] - canonical inputs (train ∪ test)
            - train_pairs: List[(Grid, Grid)] - for verification (optional)

    Returns:
        PeriodicStructure if lattice found, None otherwise

    Acceptance:
        - Deterministic (delegates to WO-06's deterministic lattice detection)
        - No floats in comparisons
        - Reproducible across runs
        - Full HNF/D8 canonicalization (via WO-06)
    """
    # Extract canonical grids from theta
    grids_present = theta.get("grids_present", [])
    if not grids_present:
        # No grids → no periodic structure
        return None

    # Get reference grid for shape
    grid_ref = grids_present[0]
    H, W = len(grid_ref), len(grid_ref[0])

    # Step 1: Use WO-06's lattice detection (full ACF → HNF → D8 pipeline)
    lattice = infer_lattice(grids_present)

    if lattice is None:
        # No periodic structure found
        return None

    # Step 2: Extract periods from lattice
    # Lattice from WO-06 is in HNF form: [[p_r, 0], [b, p_c]]
    # Periods are the diagonal elements
    p_r = abs(lattice.basis[0][0]) if lattice.basis[0][0] != 0 else 1
    p_c = abs(lattice.basis[1][1]) if lattice.basis[1][1] != 0 else 1

    # Validate periods
    if p_r <= 0 or p_c <= 0 or p_r > H or p_c > W:
        # Invalid periods
        return None

    # Step 3: Build phase table
    phase_table = build_phase_table(H, W, p_r, p_c)
    num_phases = p_r * p_c

    # Step 4: Build phase masks
    phase_masks = build_phase_masks(phase_table, num_phases)

    # Step 5: T_periodic closure
    # STUB FIX #1: WO-16 complete - use apply_lattice_closure for T_periodic constraint
    # Per engineering_spec.md §5.3: T_periodic removes PERIODIC/TILE with wrong phases
    closure = apply_lattice_closure

    # Step 6: Optional writers
    # STUB FIX #2: WO-15 complete - create PeriodicExpr instances for consistent phases
    # Per A1 (FY exactness): Only create writers if proven exact on training data
    train_pairs = theta.get("train_pairs", [])
    if not train_pairs:
        # Handle alternative format: trains = [{"input": X, "output": Y}, ...]
        trains = theta.get("trains", [])
        if trains:
            train_pairs = [(t["input"], t["output"]) for t in trains]

    writers = []
    if train_pairs:
        # Infer colors for consistent phases
        phase_colors = _infer_phase_colors(phase_table, num_phases, train_pairs)

        # Create PeriodicExpr for each consistent phase
        for phase_id, color in phase_colors.items():
            writers.append(PeriodicExpr(phase_id=phase_id, color=color))

    return PeriodicStructure(
        lattice=lattice,
        phase_table=phase_table,
        phase_masks=phase_masks,
        num_phases=num_phases,
        closure=closure,
        writers=writers
    )


# =============================================================================
# Helper: Verify Phase Consistency (for future use)
# =============================================================================


def verify_phase_consistency(
    phase_masks: List[np.ndarray],
    train_pairs: List[Tuple[Grid, Grid]]
) -> bool:
    """
    Verify that phase structure is consistent across training pairs.

    Per eng_spec.md line 99: "phase assignments exact on trains"
    Per eng_spec.md line 11: "A1 (FY exactness): every law must reproduce training outputs exactly"

    Algorithm:
        For each phase, collect output colors across all training pairs.
        If any phase maps to multiple different output colors → inconsistent → return False.

    Args:
        phase_masks: Boolean masks per phase (from build_phase_masks)
        train_pairs: Training input/output pairs [(X_i, Y_i), ...]

    Returns:
        True if all phases have consistent output colors across trains, False otherwise.

    Note:
        - Uses strict equality (grids already canonical from WO-05)
        - No palette import needed (Pattern A: arc_laws imports only from arc_core)
        - Palette orbit handled separately by WO-05's orbit_cprq()
    """
    # Empty inputs → trivially consistent
    if not phase_masks or not train_pairs:
        return True

    # For each phase, verify all pixels have consistent output colors
    for phase_id, mask in enumerate(phase_masks):
        # Find pixels belonging to this phase
        phase_pixels = np.argwhere(mask)

        if len(phase_pixels) == 0:
            continue

        # Collect output colors for this phase across all training pairs
        output_colors_seen = set()

        for X_i, Y_i in train_pairs:
            rows_y, cols_y = len(Y_i), len(Y_i[0])

            for r, c in phase_pixels:
                # Check if pixel is within training output bounds
                if r < rows_y and c < cols_y:
                    output_color = Y_i[r][c]
                    output_colors_seen.add(output_color)

        # A1 (FY exactness) check: all pixels in same phase must map to same output color
        if len(output_colors_seen) > 1:
            # Inconsistent phase → violates A1 → return False
            return False

    # All phases consistent → return True
    return True


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "PeriodicStructure",
    "build_periodic",
    "build_phase_table",
    "build_phase_masks",
]
