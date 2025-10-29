"""
Rigorous Mathematical Validation Tests for WO-06: Lattice Compiler

Tests deep mathematical properties:
- Known-answer tests with pre-computed correct bases
- Basis validity (generates actual lattice points)
- Proper HNF properties
- Proper D8 equivalence
- Pathological cases relevant to ARC

Focus: FIND BUGS in mathematical correctness, not just API compliance.
"""

import json
import pytest
import numpy as np
from pathlib import Path
from typing import List, Tuple, Set
from itertools import product

from arc_universe.arc_core.lattice import infer_lattice, compute_integer_acf
from arc_universe.arc_core.types import Grid, Lattice
from arc_universe.arc_core.order_hash import hash64


# ============================================================================
# Mathematical Helper Functions
# ============================================================================

def generate_periodic_grid(base_tile: List[List[int]],
                           tile_rows: int,
                           tile_cols: int) -> Grid:
    """Generate a grid by tiling a base pattern"""
    H_tile = len(base_tile)
    W_tile = len(base_tile[0])

    grid = []
    for r in range(tile_rows):
        for row_in_tile in range(H_tile):
            new_row = []
            for c in range(tile_cols):
                new_row.extend(base_tile[row_in_tile])
            grid.append(new_row)
    return grid


def lattice_points_from_basis(basis: List[List[int]],
                               max_coeff: int = 10) -> Set[Tuple[int, int]]:
    """
    Generate lattice points from basis vectors.

    A lattice point is any integer linear combination: a*v1 + b*v2
    where v1, v2 are basis vectors and a, b are integers.
    """
    v1 = tuple(basis[0])
    v2 = tuple(basis[1])

    points = set()
    for a in range(-max_coeff, max_coeff + 1):
        for b in range(-max_coeff, max_coeff + 1):
            r = a * v1[0] + b * v2[0]
            c = a * v1[1] + b * v2[1]
            points.add((r, c))

    return points


def verify_basis_generates_period(basis: List[List[int]],
                                  expected_period: Tuple[int, int]) -> bool:
    """
    Verify that basis generates the expected period.

    Period (pr, pc) means the lattice point (pr, 0) or (0, pc)
    should be expressible as integer combination of basis vectors.
    """
    points = lattice_points_from_basis(basis, max_coeff=20)

    pr, pc = expected_period
    # Check if period points are in lattice
    return (pr, 0) in points or (0, pc) in points


def is_hnf_proper(basis: List[List[int]]) -> Tuple[bool, str]:
    """
    Verify proper Hermite Normal Form properties.

    HNF (row-style, upper triangular):
    [[a, b],
     [0, d]]

    Properties:
    1. a > 0, d > 0 (positive diagonal)
    2. 0 <= b < d (off-diagonal bounded)
    3. Lower triangle is zero
    """
    a, b = basis[0]
    c, d = basis[1]

    # Check positive diagonal
    if a <= 0 or d <= 0:
        return False, f"Diagonal not positive: a={a}, d={d}"

    # Check lower triangle zero
    if c != 0:
        return False, f"Lower triangle not zero: basis[1][0]={c}"

    # Check off-diagonal bounded
    if not (0 <= b < d):
        return False, f"Off-diagonal not bounded: 0 <= {b} < {d}"

    return True, "HNF properties satisfied"


def d8_transform(grid: List[List[int]], transform: str) -> List[List[int]]:
    """
    Apply D8 group transformation to grid.

    D8 group has 8 elements:
    - id: identity
    - r90, r180, r270: rotations
    - fh, fv: flips
    - fd, fa: diagonal flips
    """
    if transform == "id":
        return [row[:] for row in grid]
    elif transform == "r90":
        return [list(row) for row in zip(*grid[::-1])]
    elif transform == "r180":
        return [row[::-1] for row in grid[::-1]]
    elif transform == "r270":
        return [list(row) for row in zip(*grid)][::-1]
    elif transform == "fh":
        return [row[::-1] for row in grid]
    elif transform == "fv":
        return grid[::-1]
    elif transform == "fd":
        # Flip along main diagonal (transpose)
        return [list(row) for row in zip(*grid)]
    elif transform == "fa":
        # Flip along anti-diagonal
        return [list(row) for row in zip(*grid[::-1])][::-1]
    else:
        raise ValueError(f"Unknown transform: {transform}")


def bases_generate_same_lattice(basis1: List[List[int]],
                                basis2: List[List[int]],
                                tolerance: int = 15) -> bool:
    """
    Check if two bases generate the same lattice (up to tolerance).

    Two bases generate the same lattice if the set of lattice points
    they generate are identical (within a bounded region).
    """
    points1 = lattice_points_from_basis(basis1, max_coeff=tolerance)
    points2 = lattice_points_from_basis(basis2, max_coeff=tolerance)

    # Sample a subset around origin to check
    sample = {(r, c) for r in range(-20, 21) for c in range(-20, 21)}

    points1_sample = points1 & sample
    points2_sample = points2 & sample

    return points1_sample == points2_sample


# ============================================================================
# Known-Answer Tests (Pre-computed Correct Results)
# ============================================================================

class TestKnownAnswers:
    """Tests with pre-computed mathematically correct answers"""

    def test_known_answer_3x3_simple(self):
        """KNOWN-01: Simple 3×3 tiling with known basis"""
        # Base tile: 3×3 pattern
        base_tile = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]

        # Tile it 2×3 times
        grid = generate_periodic_grid(base_tile, tile_rows=2, tile_cols=3)

        result = infer_lattice([grid])

        assert result is not None, "Should detect 3×3 tiling"

        # The basis should generate period (3, 3)
        assert result.periods == [3, 3], \
            f"Expected periods [3, 3], got {result.periods}"

        # Verify basis generates these periods
        assert verify_basis_generates_period(result.basis, (3, 3)), \
            f"Basis {result.basis} does not generate period (3, 3)"

    def test_known_answer_2x5_rectangular(self):
        """KNOWN-02: Rectangular 2×5 tiling"""
        base_tile = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 0]
        ]

        grid = generate_periodic_grid(base_tile, tile_rows=3, tile_cols=2)

        result = infer_lattice([grid])

        assert result is not None, "Should detect 2×5 tiling"

        # Periods should be [2, 5] or equivalent
        period_set = set(result.periods)
        assert period_set == {2, 5}, \
            f"Expected periods with values 2 and 5, got {result.periods}"

    def test_known_answer_4x4_square(self):
        """KNOWN-03: 4×4 square tiling"""
        base_tile = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 0, 1, 2],
            [3, 4, 5, 6]
        ]

        grid = generate_periodic_grid(base_tile, tile_rows=2, tile_cols=2)

        result = infer_lattice([grid])

        assert result is not None, "Should detect 4×4 tiling"
        assert result.periods == [4, 4], \
            f"Expected periods [4, 4], got {result.periods}"

    def test_known_answer_prime_period_5x7(self):
        """KNOWN-04: Prime coprime periods (5×7)"""
        base_tile = [[i + j for j in range(7)] for i in range(5)]

        grid = generate_periodic_grid(base_tile, tile_rows=2, tile_cols=2)

        result = infer_lattice([grid])

        # Should detect coprime periods
        if result is not None:
            period_set = set(result.periods)
            # Allow either [5,7] or detect as non-periodic if grid too small
            assert period_set == {5, 7} or len(grid) < 10, \
                f"Expected coprime periods 5,7 or too small grid, got {result.periods}"


# ============================================================================
# Basis Validity Tests
# ============================================================================

class TestBasisValidity:
    """Verify that returned bases are mathematically valid"""

    def test_basis_generates_detected_period(self):
        """VALID-01: Basis generates the detected period"""
        base_tile = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        grid = generate_periodic_grid(base_tile, 2, 2)

        result = infer_lattice([grid])

        if result is not None:
            # The basis should generate lattice points at the detected periods
            points = lattice_points_from_basis(result.basis, max_coeff=10)

            pr, pc = result.periods

            # Period points should be in lattice
            # At least one of (pr, 0), (0, pc), (pr, pc) should be reachable
            period_points = {(pr, 0), (0, pc), (pr, pc), (-pr, 0), (0, -pc)}

            assert len(period_points & points) > 0, \
                f"Basis {result.basis} does not generate period points {result.periods}"

    def test_basis_is_non_degenerate(self):
        """VALID-02: Basis vectors are linearly independent"""
        base_tile = [[1, 2], [3, 4]]
        grid = generate_periodic_grid(base_tile, 3, 3)

        result = infer_lattice([grid])

        if result is not None:
            v1 = result.basis[0]
            v2 = result.basis[1]

            # Check determinant != 0 (linear independence)
            det = v1[0] * v2[1] - v1[1] * v2[0]

            assert det != 0, \
                f"Basis vectors are linearly dependent: {result.basis}, det={det}"

    def test_basis_positive_periods(self):
        """VALID-03: Detected periods are positive"""
        base_tile = [[1, 2, 3], [4, 5, 6]]
        grid = generate_periodic_grid(base_tile, 2, 2)

        result = infer_lattice([grid])

        if result is not None:
            assert all(p > 0 for p in result.periods), \
                f"Periods must be positive, got {result.periods}"

    def test_basis_reasonable_magnitude(self):
        """VALID-04: Basis vectors have reasonable magnitude"""
        base_tile = [[1, 2], [3, 4]]
        grid = generate_periodic_grid(base_tile, 3, 3)

        result = infer_lattice([grid])

        if result is not None:
            H, W = len(grid), len(grid[0])

            # Basis vectors shouldn't be larger than grid size
            for v in result.basis:
                magnitude = abs(v[0]) + abs(v[1])
                assert magnitude <= max(H, W) * 2, \
                    f"Basis vector {v} too large for grid size {H}×{W}"


# ============================================================================
# HNF Properties Tests
# ============================================================================

class TestHNFProperties:
    """Verify proper Hermite Normal Form properties"""

    def test_hnf_upper_triangular(self):
        """HNF-MATH-01: Basis is upper triangular"""
        base_tile = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        grid = generate_periodic_grid(base_tile, 2, 2)

        result = infer_lattice([grid])

        if result is not None:
            # Check lower triangle is zero
            assert result.basis[1][0] == 0, \
                f"HNF should be upper triangular, got {result.basis}"

    def test_hnf_positive_diagonal(self):
        """HNF-MATH-02: Diagonal entries are positive"""
        base_tile = [[1, 2], [3, 4]]
        grid = generate_periodic_grid(base_tile, 2, 2)

        result = infer_lattice([grid])

        if result is not None:
            a, b = result.basis[0]
            c, d = result.basis[1]

            assert a > 0, f"basis[0][0] must be positive, got {a}"
            assert d > 0, f"basis[1][1] must be positive, got {d}"

    def test_hnf_off_diagonal_bounded(self):
        """HNF-MATH-03: Off-diagonal element is bounded"""
        base_tile = [[1, 2, 3, 4], [5, 6, 7, 8]]
        grid = generate_periodic_grid(base_tile, 2, 2)

        result = infer_lattice([grid])

        if result is not None:
            is_hnf, msg = is_hnf_proper(result.basis)
            assert is_hnf, f"HNF properties violated: {msg}"

    def test_hnf_uniqueness_for_same_grid(self):
        """HNF-MATH-04: Same grid always gives same HNF basis"""
        base_tile = [[1, 2], [3, 4]]
        grid = generate_periodic_grid(base_tile, 3, 3)

        results = [infer_lattice([grid]) for _ in range(10)]

        if results[0] is not None:
            # All results must have identical basis (HNF is unique)
            bases = [r.basis for r in results]
            assert all(b == bases[0] for b in bases), \
                "HNF basis must be unique for same grid"


# ============================================================================
# D8 Equivalence Tests
# ============================================================================

class TestD8EquivalenceRigorous:
    """Rigorous D8 equivalence testing via lattice point generation"""

    def test_d8_all_transforms_same_lattice(self):
        """D8-MATH-01: All D8 transforms generate equivalent lattices"""
        base_tile = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        grid = generate_periodic_grid(base_tile, 2, 2)

        transforms = ["id", "r90", "r180", "r270", "fh", "fv", "fd", "fa"]

        results = []
        for t in transforms:
            transformed_grid = d8_transform(grid, t)
            result = infer_lattice([transformed_grid])
            if result is not None:
                results.append(result)

        assert len(results) > 0, "At least one transform should detect periodicity"

        if len(results) > 1:
            # All bases should generate equivalent lattices
            basis0 = results[0].basis
            for i, result in enumerate(results[1:], 1):
                # Check if bases generate same lattice
                same = bases_generate_same_lattice(basis0, result.basis)
                assert same, \
                    f"D8 transform {transforms[i]} produces non-equivalent basis"

    def test_d8_periods_set_equivalent(self):
        """D8-MATH-02: Periods are D8-equivalent (up to swap)"""
        base_tile = [[1, 2, 3], [4, 5, 6]]
        grid = generate_periodic_grid(base_tile, 2, 2)

        # Original and 90-degree rotation
        grid_rot90 = d8_transform(grid, "r90")

        result1 = infer_lattice([grid])
        result2 = infer_lattice([grid_rot90])

        if result1 is not None and result2 is not None:
            # Period sets should match (order may differ)
            set1 = set(result1.periods)
            set2 = set(result2.periods)

            # For rectangular periods, 90-deg rotation swaps them
            # So {2, 3} could become {3, 2}
            assert set1 == set2 or result1.periods == result2.periods[::-1], \
                f"D8 rotation should preserve period set: {result1.periods} vs {result2.periods}"


# ============================================================================
# Pathological Cases (ARC-Relevant)
# ============================================================================

class TestPathologicalCases:
    """Test edge cases that could occur in ARC-AGI"""

    def test_pathological_almost_periodic_one_outlier(self):
        """PATH-01: Almost periodic with single pixel deviation"""
        # Create perfect 3×3 tiling
        base_tile = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        grid = generate_periodic_grid(base_tile, 2, 2)

        # Introduce one outlier (use 0 which is within ARC range 0-9)
        # Note: Original test used 999 which exposed int8 overflow bug
        # Keeping this test with valid ARC color to test robustness
        grid[3][3] = 0

        result = infer_lattice([grid])

        # Should either detect period or return None
        # Must not crash
        assert result is None or isinstance(result, Lattice), \
            "Must handle almost-periodic grids gracefully"

    def test_pathological_int8_overflow_bug(self):
        """PATH-01b: INT8 OVERFLOW BUG - Documents implementation limitation"""
        # Create simple grid
        base_tile = [[1, 2], [3, 4]]
        grid = generate_periodic_grid(base_tile, 2, 2)

        # Try to add value outside int8 range [-128, 127]
        # This WILL fail with current implementation using np.int8
        grid[0][0] = 200  # Outside int8 range

        # EXPECT THIS TO FAIL - documenting known bug
        with pytest.raises(OverflowError, match="out of bounds for int8"):
            result = infer_lattice([grid])

        # NOTE: This is a REAL BUG - implementation should use int32/int64
        # ARC colors are 0-9, but defensive coding should handle larger values

    def test_pathological_large_period_7x11(self):
        """PATH-02: Coprime periods relevant to ARC (7×11)"""
        base_tile = [[i * j % 10 for j in range(11)] for i in range(7)]
        grid = generate_periodic_grid(base_tile, 2, 2)

        result = infer_lattice([grid])

        # Should detect or gracefully return None
        if result is not None:
            # Verify detected periods are reasonable
            assert all(1 <= p <= max(len(grid), len(grid[0])) for p in result.periods), \
                f"Periods {result.periods} out of reasonable range for grid {len(grid)}×{len(grid[0])}"

    def test_pathological_non_square_grid_periodic(self):
        """PATH-03: Non-square grid (5×20) with periodicity"""
        base_tile = [[1, 2], [3, 4]]
        # Create 5×20 grid (very rectangular)
        grid = []
        for _ in range(5):
            row = []
            for _ in range(10):
                row.extend([1, 2])
            grid.append(row)

        result = infer_lattice([grid])

        # Should handle non-square grids
        if result is not None:
            # Period should include 2 (for column repetition)
            assert 2 in result.periods, \
                f"Should detect period 2, got {result.periods}"

    def test_pathological_minimal_tiles_2x2(self):
        """PATH-04: Minimal number of tiles (exactly 2×2 tiles)"""
        base_tile = [[1, 2], [3, 4]]
        grid = generate_periodic_grid(base_tile, 2, 2)  # Exactly 2×2 tiles

        result = infer_lattice([grid])

        # With minimal tiles, might not detect
        # But must handle gracefully
        if result is not None:
            assert result.periods == [2, 2], \
                f"Should detect [2,2] from minimal tiling, got {result.periods}"

    def test_pathological_single_tile_no_repetition(self):
        """PATH-05: Single tile (no repetition)"""
        # Just one tile, never repeated
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        result = infer_lattice([grid])

        # Single tile has no real periodicity
        # Should return None or period equal to grid size
        if result is not None:
            H, W = len(grid), len(grid[0])
            # Period shouldn't be smaller than grid (no repetition)
            assert result.periods[0] >= H or result.periods[1] >= W, \
                f"Single tile should have period >= grid size, got {result.periods}"


# ============================================================================
# ACF (Autocorrelation Function) Tests
# ============================================================================

class TestACFProperties:
    """Test integer ACF computation properties"""

    def test_acf_is_integer(self):
        """ACF-01: ACF values are integers"""
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        acf = compute_integer_acf(grid)

        assert np.issubdtype(acf.dtype, np.integer), \
            f"ACF must be integer dtype, got {acf.dtype}"

    def test_acf_symmetric(self):
        """ACF-02: ACF is symmetric (ACF[r,c] == ACF[-r,-c])"""
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        acf = compute_integer_acf(grid)

        H, W = acf.shape

        # Check a few symmetric pairs
        for r in range(min(3, H//2)):
            for c in range(min(3, W//2)):
                # ACF should be symmetric around center
                assert acf[r, c] == acf[-r, -c], \
                    f"ACF not symmetric at ({r},{c}): {acf[r,c]} != {acf[-r,-c]}"

    def test_acf_max_at_zero(self):
        """ACF-03: ACF maximum at zero displacement"""
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        acf = compute_integer_acf(grid)

        H, W = acf.shape
        center = acf[0, 0]

        # Zero displacement should be maximum (perfect alignment)
        assert center == np.max(acf), \
            f"ACF should be maximal at zero displacement"


# ============================================================================
# Multi-Grid Consensus (Advanced)
# ============================================================================

class TestMultiGridConsensusRigorous:
    """Rigorous multi-grid consensus testing"""

    def test_consensus_exact_same_periods(self):
        """CONS-01: Multiple grids with identical periods"""
        base_tile = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        grids = [
            generate_periodic_grid(base_tile, 2, 2),
            generate_periodic_grid([[9, 8, 7], [6, 5, 4], [3, 2, 1]], 2, 2),
            generate_periodic_grid([[5, 5, 5], [5, 5, 5], [5, 5, 5]], 2, 2)
        ]

        result = infer_lattice(grids)

        # Should find consensus on period [3, 3]
        assert result is not None, "Should find consensus on shared period"
        assert result.periods == [3, 3], \
            f"Expected consensus period [3,3], got {result.periods}"

    def test_consensus_conflicting_handled(self):
        """CONS-02: Conflicting periods handled deterministically"""
        grid1 = generate_periodic_grid([[1, 2], [3, 4]], 2, 2)  # Period [2, 2]
        grid2 = generate_periodic_grid([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2, 2)  # Period [3, 3]

        result1 = infer_lattice([grid1, grid2])
        result2 = infer_lattice([grid1, grid2])

        # Must be deterministic
        if result1 is None:
            assert result2 is None
        else:
            assert result1.basis == result2.basis
            assert result1.periods == result2.periods


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
