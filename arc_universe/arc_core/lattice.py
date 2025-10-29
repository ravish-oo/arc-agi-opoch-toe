"""
Lattice detection: FFT/HNF lattice + 2D KMP fallback (WO-06).

Per implementation_plan.md lines 228-238 and math_spec.md §6.3, §7.

Provides:
- infer_lattice(Xs): Deterministic periodic basis detection
- FFT autocorrelation → HNF → D8 canonical (primary path)
- 2D KMP fallback for degenerate/rank-1 cases
- All integer arithmetic, no floats, deterministic everywhere

Algorithm:
1. Integer ACF via color-indicator planes (circular/toroidal)
2. Peak extraction: max score → minimal norm → non-colinear
3. HNF reduction → D8 canonicalization
4. Multi-grid consensus (pooled ACF + verify)
5. Fallback to KMP for rank-1/degenerate cases
"""

from typing import List, Tuple, Optional
import numpy as np

from .order_hash import lex_min
from .types import Grid, Lattice


# ==============================================================================
# Integer Autocorrelation (Color-Indicator Planes)
# ==============================================================================

def compute_integer_acf(grid: Grid) -> np.ndarray:
    """
    Compute integer 2D autocorrelation using color-indicator planes.

    Per reference implementation: Use binary plane per color, circular shifts
    with np.roll, sum dot-products. Fully deterministic, integer, exact.

    Args:
        grid: Input grid

    Returns:
        2D integer autocorrelation array (same shape as grid)
        ACF[dr, dc] = count of matching color pairs at offset (dr, dc)

    Reference:
        ACF(Δr, Δc) = Σ_c Σ_r Σ_s B_c(r,s) * B_c((r+Δr) mod H, (s+Δc) mod W)
        where B_c is binary indicator plane for color c
    """
    # Convert to numpy array
    X = np.array(grid, dtype=np.int8)
    H, W = X.shape

    # Build indicator planes for all colors (0-9)
    num_colors = 10
    planes = [(X == c).astype(np.uint8) for c in range(num_colors)]

    # Compute ACF via circular shifts
    acf = np.zeros((H, W), dtype=np.int64)

    for dr in range(H):
        for dc in range(W):
            s = 0
            for B in planes:
                # Circular shift by (dr, dc) and compute dot product
                shifted = np.roll(np.roll(B, dr, axis=0), dc, axis=1)
                s += (B * shifted).sum(dtype=np.int64)
            acf[dr, dc] = s

    return acf


# ==============================================================================
# Degeneracy Detection (for KMP Fallback)
# ==============================================================================

def is_degenerate_acf(acf: np.ndarray) -> bool:
    """
    Check if ACF is degenerate (uniform/flat structure requiring KMP fallback).

    Per §6: Degenerate cases include:
    - Flat ACF: all nonzero offsets have identical score (no structure)
    - Near-uniform: minimal variation in scores (weak structure)

    Args:
        acf: 2D integer autocorrelation

    Returns:
        True if degenerate (should use KMP), False otherwise
    """
    H, W = acf.shape
    DC = int(acf[0, 0])

    # Collect all nonzero offset scores (exclude DC)
    nonzero_scores = []
    for dr in range(H):
        for dc in range(W):
            if dr == 0 and dc == 0:
                continue
            score = int(acf[dr, dc])
            if score > 0:
                nonzero_scores.append(score)

    if not nonzero_scores:
        # No structure at all → degenerate
        return True

    # Key insight: Perfect periodic grids have FEW DC-level offsets (the periods)
    # Uniform/degenerate grids have MANY DC-level offsets (everything matches)
    max_score = max(nonzero_scores)
    min_score = min(nonzero_scores)

    # Count how many offsets are at DC level
    dc_level_count = sum(1 for s in nonzero_scores if s == DC)

    if max_score == min_score and max_score == DC:
        # All nonzero offsets at DC level
        # This is degenerate ONLY if there are MANY such offsets
        # Perfect periodic has O(min(H,W)) offsets, uniform has O(H*W)
        if dc_level_count > min(H, W) * 2:
            # Too many DC-level offsets → uniform/degenerate
            return True
        # Few DC-level offsets → perfect periodic (NOT degenerate)
        return False

    if max_score == min_score:
        # All scores identical but not at DC level
        # This is flat if there are many offsets
        if len(nonzero_scores) > min(H, W) * 2:
            return True

    return False


# ==============================================================================
# Signed Offset Mapping & Two-Tier Peak Extraction
# ==============================================================================

def wrap_sign(delta: int, size: int) -> int:
    """
    Convert array index offset to signed representative.

    Per §4: Maps [0..size-1] to [-floor(size/2)..ceil(size/2)-1]
    for norm/tie-break decisions.

    Args:
        delta: Offset in [0, size-1]
        size: Grid dimension (H or W)

    Returns:
        Signed offset
    """
    if delta <= size // 2:
        return delta
    else:
        return delta - size


def infer_lattice_tier1(acf: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Tier-1 axis period check: DC-level matches on axes.

    Per §2: A true period satisfies ACF(Δr, 0) = ACF(0, 0) (DC level).
    This distinguishes perfect periods from partial matches.

    Args:
        acf: 2D integer autocorrelation

    Returns:
        (row_period, col_period) or None if either axis has no perfect period
        or if periods are trivial [1,1] (degenerate case)
    """
    H, W = acf.shape
    DC = int(acf[0, 0])

    # Find smallest nonzero row shift with DC-level match
    row_period = None
    for dr in range(1, H):
        if int(acf[dr, 0]) == DC:
            row_period = dr
            break

    # Find smallest nonzero col shift with DC-level match
    col_period = None
    for dc in range(1, W):
        if int(acf[0, dc]) == DC:
            col_period = dc
            break

    if row_period is not None and col_period is not None:
        # Reject trivial periods [1,1] as degenerate
        # Period [1,1] means every pixel matches every shift → uniform/meaningless
        if row_period == 1 and col_period == 1:
            return None  # Trigger KMP fallback
        return (row_period, col_period)
    else:
        return None


def extract_peaks_tier2(acf: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Tier-2 peak extraction: DC-level preferred, then maximal score.

    Per §3: Two-level peak selection:
    1. Prefer DC-level matches (perfect periods) if any exist
    2. Else use maximal score layer
    3. Always pick minimal-norm non-colinear pairs

    Args:
        acf: 2D integer autocorrelation

    Returns:
        ((v1_r, v1_c), (v2_r, v2_c)) or None if rank-1/degenerate
    """
    H, W = acf.shape
    DC = int(acf[0, 0])

    # Build candidate set S: all nonzero offsets (exclude DC at (0,0))
    candidates = []
    dc_level_candidates = []

    for dr in range(H):
        for dc in range(W):
            # Skip DC
            if dr == 0 and dc == 0:
                continue

            score = int(acf[dr, dc])
            if score <= 0:
                continue

            # Convert to signed offset
            offset_r = wrap_sign(dr, H)
            offset_c = wrap_sign(dc, W)

            norm_sq = offset_r**2 + offset_c**2

            candidates.append((score, norm_sq, offset_r, offset_c))

            # Track DC-level matches separately
            if score == DC:
                dc_level_candidates.append((norm_sq, offset_r, offset_c))

    if not candidates:
        # Flat ACF (no structure)
        return None

    # Two-level pick: prefer DC-level if any exist
    if dc_level_candidates:
        # Use DC-level matches (perfect periods)
        dc_level_candidates.sort()  # By (norm_sq, r, c)
        _, v1_r, v1_c = dc_level_candidates[0]

        # Find non-colinear at DC level
        def is_colinear(r, c):
            return v1_r * c - v1_c * r == 0

        non_colinear_dc = [(norm_sq, r, c) for norm_sq, r, c in dc_level_candidates
                           if not is_colinear(r, c)]

        if non_colinear_dc:
            _, v2_r, v2_c = non_colinear_dc[0]
            return ((v1_r, v1_c), (v2_r, v2_c))

    # Else use maximal score layer
    s_max = max(c[0] for c in candidates)

    # Choose v1: minimal norm among s_max, lex tie-break
    max_score_candidates = [(norm_sq, r, c) for score, norm_sq, r, c in candidates if score == s_max]
    max_score_candidates.sort()

    if not max_score_candidates:
        return None

    _, v1_r, v1_c = max_score_candidates[0]

    # Find v2: non-colinear
    def is_colinear(r, c):
        return v1_r * c - v1_c * r == 0

    non_colinear = [(norm_sq, r, c) for norm_sq, r, c in max_score_candidates
                    if not is_colinear(r, c)]

    if non_colinear:
        _, v2_r, v2_c = non_colinear[0]
        return ((v1_r, v1_c), (v2_r, v2_c))

    # Try next score level
    scores_below_max = sorted(set(c[0] for c in candidates if c[0] < s_max), reverse=True)

    for s in scores_below_max:
        level_candidates = [(norm_sq, r, c) for score, norm_sq, r, c in candidates
                           if score == s and not is_colinear(r, c)]
        if level_candidates:
            level_candidates.sort()
            _, v2_r, v2_c = level_candidates[0]
            return ((v1_r, v1_c), (v2_r, v2_c))

    # No non-colinear offset found → rank-1
    return None


# ==============================================================================
# Hermite Normal Form (HNF) for 2×2 Integer Matrices
# ==============================================================================

def hnf_2x2(basis: List[List[int]]) -> List[List[int]]:
    """
    Compute Hermite Normal Form for 2×2 integer basis.

    Per clarification §1: HNF is lower-triangular form:
        H = [[p_r, 0], [b, p_c]]
    with p_r, p_c > 0 and 0 <= b < p_c

    Standard column-reduction algorithm for integer lattices.

    Args:
        basis: 2×2 integer matrix [[v1_r, v1_c], [v2_r, v2_c]]
                where rows are basis vectors

    Returns:
        2×2 HNF matrix [[p_r, 0], [b, p_c]]
    """
    from math import gcd as math_gcd

    # Basis vectors as columns (standard HNF uses column form)
    # Transpose: columns = basis vectors
    col1 = [basis[0][0], basis[1][0]]  # [v1_r, v2_r]
    col2 = [basis[0][1], basis[1][1]]  # [v1_c, v2_c]

    # Check for degenerate (colinear) basis
    det = col1[0] * col2[1] - col1[1] * col2[0]
    if det == 0:
        return [[1, 0], [0, 1]]

    # Use extended Euclidean algorithm to reduce col1 to [g, 0]^T
    def extended_gcd(a, b):
        """Return (g, u, v) such that a*u + b*v = g = gcd(a,b)"""
        if b == 0:
            return (abs(a), 1 if a >= 0 else -1, 0)
        else:
            g, x1, y1 = extended_gcd(b, a % b)
            x = y1
            y = x1 - (a // b) * y1
            return (g, x, y)

    # Reduce first column
    g1, u1, v1 = extended_gcd(col1[0], col1[1])

    # Apply unimodular transformation to make col1 = [g1, 0]^T
    # [[u1, -col1[1]//g1], [v1, col1[0]//g1]]
    w1 = -col1[1] // g1 if g1 != 0 else 0
    w2 = col1[0] // g1 if g1 != 0 else 1

    new_col1 = [g1, 0]
    new_col2 = [u1 * col2[0] + v1 * col2[1], w1 * col2[0] + w2 * col2[1]]

    # Ensure first column entry is positive
    if new_col1[0] < 0:
        new_col1[0] = -new_col1[0]
        new_col2 = [-new_col2[0], -new_col2[1]]

    # Ensure second column's bottom entry is positive
    if new_col2[1] < 0:
        new_col2 = [new_col2[0], -new_col2[1]]

    # Reduce top entry of second column to range [0, bottom entry)
    if new_col2[1] > 0:
        q = new_col2[0] // new_col2[1]
        new_col2[0] = new_col2[0] - q * new_col2[1]
        # Equivalent to column op: col2 -= q * col1
        # But col1 is [p_r, 0], so only affects row 0

    # Normalize to [0, new_col2[1])
    if new_col2[1] > 0 and new_col2[0] < 0:
        new_col2[0] = new_col2[0] % new_col2[1]

    # Return as row-wise matrix (transpose back)
    p_r = abs(new_col1[0])
    p_c = abs(new_col2[1])
    b = new_col2[0] % p_c if p_c > 0 else 0

    if p_r == 0:
        p_r = 1
    if p_c == 0:
        p_c = 1

    return [[p_r, 0], [b, p_c]]


# ==============================================================================
# D8 Canonicalization for Basis Vectors
# ==============================================================================

def apply_d8_to_offset(offset: Tuple[int, int], transform: str) -> Tuple[int, int]:
    """
    Apply D8 transformation to an offset vector.

    Per clarification §4: Offsets are difference vectors (Δr, Δc).
    Apply same transform as used by ΠG on rasters.

    Transforms:
    - identity: (Δr, Δc)
    - rot90: (Δr, Δc) → (-Δc, Δr)
    - rot180: (Δr, Δc) → (-Δr, -Δc)
    - rot270: (Δr, Δc) → (Δc, -Δr)
    - flip_h: (Δr, Δc) → (Δr, -Δc)
    - flip_v: (Δr, Δc) → (-Δr, Δc)
    - flip_diag_main: (Δr, Δc) → (Δc, Δr)
    - flip_diag_anti: (Δr, Δc) → (-Δc, -Δr)

    Args:
        offset: (Δr, Δc) offset vector
        transform: D8 transform name

    Returns:
        Transformed offset
    """
    dr, dc = offset

    if transform == "identity":
        return (dr, dc)
    elif transform == "rot90":
        return (-dc, dr)
    elif transform == "rot180":
        return (-dr, -dc)
    elif transform == "rot270":
        return (dc, -dr)
    elif transform == "flip_h":
        return (dr, -dc)
    elif transform == "flip_v":
        return (-dr, dc)
    elif transform == "flip_diag_main":
        return (dc, dr)
    elif transform == "flip_diag_anti":
        return (-dc, -dr)
    else:
        raise ValueError(f"Unknown D8 transform: {transform}")


def d8_canonicalize_basis(v1: Tuple[int, int], v2: Tuple[int, int]) -> Tuple[List[List[int]], str]:
    """
    D8 canonicalization for basis vectors.

    Per clarification §4: Apply all 8 D8 transforms, convert to HNF,
    choose lex-min HNF as canonical form.

    Args:
        v1: First basis vector (Δr, Δc)
        v2: Second basis vector (Δr, Δc)

    Returns:
        (canonical_hnf_basis, transform_name)
    """
    D8_TRANSFORMS = [
        "identity",
        "rot90",
        "rot180",
        "rot270",
        "flip_h",
        "flip_v",
        "flip_diag_main",
        "flip_diag_anti",
    ]

    candidates = []

    for transform in D8_TRANSFORMS:
        # Apply transform to both basis vectors
        v1_t = apply_d8_to_offset(v1, transform)
        v2_t = apply_d8_to_offset(v2, transform)

        # Convert to HNF
        basis_t = [[v1_t[0], v1_t[1]], [v2_t[0], v2_t[1]]]
        hnf_t = hnf_2x2(basis_t)

        # Flatten for lex comparison
        flat = tuple(tuple(row) for row in hnf_t)

        candidates.append((flat, hnf_t, transform))

    # Choose lex-min
    _, canonical_hnf, transform_used = lex_min(candidates, key=lambda x: x[0])

    return canonical_hnf, transform_used


# ==============================================================================
# 2D KMP Period Detection (Fallback for Rank-1/Degenerate)
# ==============================================================================

def compute_kmp_lps(pattern: List[int]) -> List[int]:
    """
    Compute LPS (Longest Proper Prefix which is also Suffix) array for KMP.

    Args:
        pattern: 1D pattern as list of integers

    Returns:
        LPS array
    """
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

    return lps


def find_minimal_period_kmp(sequence: List[int]) -> int:
    """
    Find minimal period of a sequence using KMP LPS.

    Args:
        sequence: 1D sequence

    Returns:
        Minimal period (defaults to length if aperiodic)
    """
    n = len(sequence)
    if n == 0:
        return 1

    lps = compute_kmp_lps(sequence)

    # Minimal period is n - lps[n-1] if it divides n
    period = n - lps[n - 1]

    if n % period == 0:
        return period
    else:
        return n


def infer_lattice_kmp(grids: List[Grid]) -> Lattice:
    """
    Infer lattice using 2D KMP (fallback for rank-1/degenerate).

    Per clarification §6 (D2): Use KMP on rows and columns separately.

    Args:
        grids: List of grids

    Returns:
        Lattice with method="KMP"
    """
    # Pool all grids to find consensus periods
    row_periods = []
    col_periods = []

    for grid in grids:
        rows, cols = len(grid), len(grid[0])

        # Find row period (scan each row, take most frequent)
        grid_row_periods = []
        for r in range(rows):
            row_seq = grid[r]
            period = find_minimal_period_kmp(row_seq)
            grid_row_periods.append(period)

        # Find column period
        grid_col_periods = []
        for c in range(cols):
            col_seq = [grid[r][c] for r in range(rows)]
            period = find_minimal_period_kmp(col_seq)
            grid_col_periods.append(period)

        # Use minimal period (most frequent or lex-min)
        if grid_row_periods:
            row_periods.append(min(grid_row_periods))
        if grid_col_periods:
            col_periods.append(min(grid_col_periods))

    # Consensus: GCD of all detected periods
    from math import gcd as math_gcd
    from functools import reduce

    p_r = reduce(math_gcd, row_periods) if row_periods else 1
    p_c = reduce(math_gcd, col_periods) if col_periods else 1

    # Build HNF basis [[p_r, 0], [0, p_c]]
    basis = [[p_r, 0], [0, p_c]]

    return Lattice(basis=basis, periods=[p_r, p_c], method="KMP")


# ==============================================================================
# Main Lattice Inference (FFT + HNF + D8 Canonical)
# ==============================================================================

def infer_lattice(Xs: List[Grid]) -> Optional[Lattice]:
    """
    Infer periodic lattice from grids deterministically.

    Two-tier approach per final clarification:

    Tier-1 (axis check):
    - Check for DC-level matches on axes (perfect periods)
    - If both row and col periods found → diagonal HNF

    Tier-2 (general case):
    - DC-level preferred peak extraction
    - HNF reduction + D8 canonicalization
    - Multi-grid consensus (pooled ACF + verify)

    Fallback (KMP):
    - Triggered for rank-1 or flat ACF
    - Use 2D KMP on rows/columns

    Args:
        Xs: List of input grids (train ∪ test)

    Returns:
        Lattice with basis, periods, method

    Acceptance:
    - Deterministic (no randomness, no floats in comparisons)
    - Reproducible across runs
    - Method logged ("FFT" or "KMP")
    """
    if not Xs:
        # Empty input → no lattice to detect
        return None

    # Multi-grid consensus: pooled ACF
    # Per §5: Sum ACFs across all inputs
    acf_pooled = None

    for grid in Xs:
        acf = compute_integer_acf(grid)
        if acf_pooled is None:
            acf_pooled = acf
        else:
            # Ensure same shape
            if acf.shape != acf_pooled.shape:
                # Different grid sizes → use per-grid HNF + GCD consensus
                # Fall back to KMP
                return infer_lattice_kmp(Xs)
            acf_pooled += acf

    # Check for degenerate ACF (per §6 - triggers KMP fallback)
    if is_degenerate_acf(acf_pooled):
        # Flat/uniform ACF → use KMP fallback
        return infer_lattice_kmp(Xs)

    # Tier-1: Check for perfect axis periods (DC-level)
    axis_periods = infer_lattice_tier1(acf_pooled)

    if axis_periods is not None:
        # Both axes have perfect periods → diagonal HNF
        p_r, p_c = axis_periods
        basis = [[p_r, 0], [0, p_c]]

        # D8 canonicalization (for consistency with general case)
        canonical_basis, _ = d8_canonicalize_basis((p_r, 0), (0, p_c))

        p_r_canon = abs(canonical_basis[0][0])
        p_c_canon = abs(canonical_basis[1][1])

        return Lattice(basis=canonical_basis, periods=[p_r_canon, p_c_canon], method="FFT")

    # Tier-2: General peak extraction (DC-level preferred)
    peaks = extract_peaks_tier2(acf_pooled)

    if peaks is None:
        # Rank-1 or degenerate → KMP fallback
        return infer_lattice_kmp(Xs)

    v1, v2 = peaks

    # HNF reduction
    basis = [[v1[0], v1[1]], [v2[0], v2[1]]]
    hnf_basis = hnf_2x2(basis)

    # D8 canonicalization
    canonical_basis, _ = d8_canonicalize_basis(
        (hnf_basis[0][0], hnf_basis[0][1]),
        (hnf_basis[1][0], hnf_basis[1][1])
    )

    # Extract periods from HNF: [[p_r, 0], [b, p_c]]
    p_r = abs(canonical_basis[0][0])
    p_c = abs(canonical_basis[1][1])

    if p_r == 0:
        p_r = 1
    if p_c == 0:
        p_c = 1

    return Lattice(basis=canonical_basis, periods=[p_r, p_c], method="FFT")
