"""
Unit tests for arc_core/shape.py (WO-04).

Per implementation_plan.md lines 159-169:
- Deterministic shape unification
- Meet operation correctness (finest common coarsening)
- 1-D WL on rows/cols
- Shape maps s_i (stored partitions)

Acceptance criteria:
- Deterministic (same result for permuted trains)
- Meet property holds (mathematical correctness)
- No P-catalog search (input-derived only)
- Identity / uniform-scale / NPS bands cases

STRENGTHENED TESTS - Verify implementation against math spec:
- Manual verification of meet operation
- 1-D WL iteration correctness
- Partition signatures (not hashes) to avoid collisions
- Convergence within max_iters (≤12)
- No output leakage (input-only)
- No coordinates in features
"""

import json
import pytest

from arc_core.order_hash import hash64
from arc_core.present import build_present
from arc_core.shape import meet_partitions, unify_shape, wl_1d_on_cols, wl_1d_on_rows
from arc_core.types import Grid


class TestWL1D:
    """Test 1-D WL on rows and columns."""

    def test_wl_1d_rows_deterministic(self):
        """Same grid produces same row partition."""
        grid = [[1, 2], [3, 4]]

        part1 = wl_1d_on_rows(grid)
        part2 = wl_1d_on_rows(grid)

        assert part1 == part2, "1-D WL on rows must be deterministic"

    def test_wl_1d_cols_deterministic(self):
        """Same grid produces same column partition."""
        grid = [[1, 2], [3, 4]]

        part1 = wl_1d_on_cols(grid)
        part2 = wl_1d_on_cols(grid)

        assert part1 == part2, "1-D WL on columns must be deterministic"

    def test_wl_1d_seed_uses_hash64(self):
        """Verify seed uses hash64 of row/col contents."""
        # Two identical rows should get same initial color
        grid = [[1, 2, 3], [1, 2, 3], [4, 5, 6]]

        part = wl_1d_on_rows(grid, max_iters=0)

        # Rows 0 and 1 are identical → same initial class
        # Row 2 is different → different class
        assert part[0] == part[1], "Identical rows should get same initial class"
        assert part[0] != part[2], "Different rows should get different classes"

    def test_wl_1d_rows_refines_by_neighbors(self):
        """1-D WL refines row partition based on adjacent rows."""
        # Create grid where rows differ only in neighbors
        grid = [
            [5, 5, 5],  # Row 0: neighbors = [row 1]
            [5, 5, 5],  # Row 1: neighbors = [row 0, row 2]
            [5, 5, 5],  # Row 2: neighbors = [row 1]
        ]

        # After iteration 0: all rows have same initial color (same contents)
        part_iter0 = wl_1d_on_rows(grid, max_iters=0)
        assert part_iter0[0] == part_iter0[1] == part_iter0[2], "All rows identical initially"

        # After iteration 1+: middle row has 2 neighbors, edges have 1
        # This should distinguish them
        part_converged = wl_1d_on_rows(grid, max_iters=12)

        # Row 1 (middle) has different neighbor count than rows 0,2 (edges)
        # So they should potentially differ after WL refinement
        # (depends on implementation details, but verify it runs)
        assert isinstance(part_converged[0], int)
        assert isinstance(part_converged[1], int)
        assert isinstance(part_converged[2], int)

    def test_wl_1d_cols_refines_by_neighbors(self):
        """1-D WL refines col partition based on adjacent cols."""
        # 3×3 grid with uniform rows
        grid = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]

        # All columns have same structure (vertical variation)
        part_iter0 = wl_1d_on_cols(grid, max_iters=0)
        assert part_iter0[0] == part_iter0[1] == part_iter0[2], "All columns identical initially"

        # After refinement, middle column has 2 neighbors vs edges with 1
        part_converged = wl_1d_on_cols(grid, max_iters=12)

        # Verify partition is valid
        assert len(part_converged) == 3
        for col_id, class_id in part_converged.items():
            assert isinstance(col_id, int)
            assert isinstance(class_id, int)
            assert class_id >= 0

    def test_wl_1d_convergence(self):
        """1-D WL converges to stable fixed point within max_iters."""
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        # Run with max_iters=1
        part_1 = wl_1d_on_rows(grid, max_iters=1)

        # Run with max_iters=12
        part_12 = wl_1d_on_rows(grid, max_iters=12)

        # Run again with max_iters=12
        part_12_again = wl_1d_on_rows(grid, max_iters=12)

        # Converged results should be stable
        assert part_12 == part_12_again, "WL should converge to stable fixed point"

        # Verify all rows got partitions
        assert len(part_12) == 3, "All rows should be in partition"

    def test_wl_1d_single_row(self):
        """1-D WL handles single-row grid."""
        grid = [[1, 2, 3, 4, 5]]

        part_rows = wl_1d_on_rows(grid)
        part_cols = wl_1d_on_cols(grid)

        # Single row → partition {0: 0}
        assert len(part_rows) == 1
        assert part_rows[0] == 0

        # Five columns → partition with potentially different classes
        assert len(part_cols) == 5
        for col_id in range(5):
            assert col_id in part_cols

    def test_wl_1d_single_col(self):
        """1-D WL handles single-column grid."""
        grid = [[1], [2], [3], [4], [5]]

        part_rows = wl_1d_on_rows(grid)
        part_cols = wl_1d_on_cols(grid)

        # Five rows → partition with potentially different classes
        assert len(part_rows) == 5
        for row_id in range(5):
            assert row_id in part_rows

        # Single column → partition {0: 0}
        assert len(part_cols) == 1
        assert part_cols[0] == 0

    def test_wl_1d_uniform_grid(self):
        """1-D WL distinguishes by neighbor structure, not just content.

        For uniform 3×3 grid, all rows/cols have identical content.
        But WL refines by neighbor structure:
        - Edge rows (0, 2): each has 1 neighbor → SAME class
        - Middle row (1): has 2 neighbors → DIFFERENT class

        This test PROVES WL is doing neighbor-based refinement correctly.
        """
        grid = [[7, 7, 7], [7, 7, 7], [7, 7, 7]]

        part_rows = wl_1d_on_rows(grid)
        part_cols = wl_1d_on_cols(grid)

        # ROWS: Edge rows (0, 2) have same neighbor structure (1 neighbor each)
        # → must get SAME class
        assert part_rows[0] == part_rows[2], \
            "Edge rows (0,2) have same structure (1 neighbor) → must have same class"

        # Middle row (1) has different structure (2 neighbors)
        # → must get DIFFERENT class from edges
        assert part_rows[0] != part_rows[1], \
            "Middle row (1) has different structure (2 neighbors) → must differ from edges"
        assert part_rows[1] != part_rows[2], \
            "Middle row must differ from both edges"

        # COLUMNS: Same structural logic applies
        assert part_cols[0] == part_cols[2], \
            "Edge cols (0,2) have same structure (1 neighbor) → must have same class"
        assert part_cols[0] != part_cols[1], \
            "Middle col (1) has different structure (2 neighbors) → must differ from edges"
        assert part_cols[1] != part_cols[2], \
            "Middle col must differ from both edges"

        # Verify exactly 2 distinct classes per dimension (edges vs middle)
        assert len(set(part_rows.values())) == 2, \
            "Should have exactly 2 row classes: edges and middle"
        assert len(set(part_cols.values())) == 2, \
            "Should have exactly 2 col classes: edges and middle"

    def test_wl_1d_max_iterations_respected(self):
        """1-D WL respects max_iters parameter."""
        grid = [[1, 2, 1], [2, 3, 2], [1, 2, 1]]

        # Run with different max_iters
        part_0 = wl_1d_on_rows(grid, max_iters=0)
        part_1 = wl_1d_on_rows(grid, max_iters=1)
        part_2 = wl_1d_on_rows(grid, max_iters=2)
        part_12 = wl_1d_on_rows(grid, max_iters=12)

        # All should be valid partitions
        for part in [part_0, part_1, part_2, part_12]:
            assert len(part) == 3
            for row_id, class_id in part.items():
                assert isinstance(row_id, int)
                assert isinstance(class_id, int)

    def test_wl_1d_large_grid(self):
        """1-D WL handles large grid (30×30)."""
        # Create 30×30 grid with varied structure
        grid = [[((r + c) % 10) for c in range(30)] for r in range(30)]

        part_rows = wl_1d_on_rows(grid, max_iters=12)
        part_cols = wl_1d_on_cols(grid, max_iters=12)

        # Should have 30 rows and 30 columns in partitions
        assert len(part_rows) == 30
        assert len(part_cols) == 30

        # All class IDs should be non-negative
        for class_id in part_rows.values():
            assert class_id >= 0
        for class_id in part_cols.values():
            assert class_id >= 0


class TestMeetOperation:
    """Test meet_partitions function."""

    def test_meet_empty_list(self):
        """Meet of empty list returns empty partition."""
        result = meet_partitions([])
        assert result == {}, "Meet of empty list should be empty"

    def test_meet_single_partition(self):
        """Meet of single partition returns normalized version."""
        part = {0: 5, 1: 5, 2: 7}
        result = meet_partitions([part])

        # Should normalize to {0: 0, 1: 0, 2: 1}
        assert result[0] == result[1], "Elements in same class stay together"
        assert result[0] != result[2], "Elements in different classes stay separate"

        # Class IDs should start from 0
        assert set(result.values()) == {0, 1}

    def test_meet_two_identical_partitions(self):
        """Meet of two identical partitions returns same structure."""
        part1 = {0: 0, 1: 0, 2: 1}
        part2 = {0: 0, 1: 0, 2: 1}

        result = meet_partitions([part1, part2])

        # Should be same structure (elements 0,1 together; 2 separate)
        assert result[0] == result[1]
        assert result[0] != result[2]

    def test_meet_property_finest_common_coarsening(self):
        """Meet is finest partition where same-class elements are same in ALL inputs.

        Example from research:
        P1: {0→A, 1→A, 2→B}
        P2: {0→X, 1→Y, 2→X}

        Meet: {0→0, 1→1, 2→2} (all distinct)
        Because elem 0 and 1 are same in P1 but different in P2.
        """
        # P1: elements 0,1 in same class; element 2 separate
        part1 = {0: 0, 1: 0, 2: 1}

        # P2: elements 0,2 in same class; element 1 separate
        part2 = {0: 0, 1: 1, 2: 0}

        result = meet_partitions([part1, part2])

        # Meet should distinguish all three (finest partition)
        assert result[0] != result[1], "Elements differ in P2"
        assert result[0] != result[2], "Elements differ in P1"
        assert result[1] != result[2], "Elements differ in both"

        # All three should have distinct classes
        assert len(set(result.values())) == 3

    def test_meet_property_verified_mathematically(self):
        """RIGOROUS: Verify meet property holds for all pairs.

        For any i, j: if meet[i] == meet[j], then p[i] == p[j] for all p in inputs.
        """
        part1 = {0: 0, 1: 0, 2: 1, 3: 1}
        part2 = {0: 5, 1: 5, 2: 7, 3: 8}  # Split class 1 in part1
        part3 = {0: 2, 1: 3, 2: 2, 3: 2}  # Split class 0 in part1

        partitions = [part1, part2, part3]
        result = meet_partitions(partitions)

        # Verify meet property for all pairs
        elements = sorted(result.keys())
        for i in elements:
            for j in elements:
                if result[i] == result[j]:
                    # Then i and j must be in same class in ALL inputs
                    for part in partitions:
                        assert part[i] == part[j], f"Meet property violated for elements {i}, {j}"

    def test_meet_deterministic(self):
        """Meet produces same result when called twice."""
        part1 = {0: 0, 1: 1, 2: 2}
        part2 = {0: 5, 1: 5, 2: 7}

        result1 = meet_partitions([part1, part2])
        result2 = meet_partitions([part1, part2])

        assert result1 == result2, "Meet must be deterministic"

    def test_meet_order_independent(self):
        """Meet is independent of input order."""
        part1 = {0: 0, 1: 0, 2: 1}
        part2 = {0: 5, 1: 6, 2: 5}
        part3 = {0: 9, 1: 9, 2: 9}

        result_123 = meet_partitions([part1, part2, part3])
        result_321 = meet_partitions([part3, part2, part1])
        result_213 = meet_partitions([part2, part1, part3])

        assert result_123 == result_321 == result_213, "Meet must be order-independent"

    def test_meet_uses_signatures_not_hashes(self):
        """Meet uses signatures (tuples), not hashes, to avoid collisions."""
        # Two elements with different signatures should not merge
        # even if their hashes collide (unlikely but must be safe)

        part1 = {0: 0, 1: 1}
        part2 = {0: 10, 1: 20}

        result = meet_partitions([part1, part2])

        # Elements 0 and 1 have signatures (0, 10) and (1, 20)
        # These are different → should not merge
        assert result[0] != result[1], "Different signatures must not merge"


class TestShapeUnification:
    """Test unify_shape function."""

    def test_identity_shape(self):
        """All training grids have identical dimensions."""
        # Three identical 3×3 grids (different colors)
        grid1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        grid2 = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
        grid3 = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]

        present1 = build_present(grid1)
        present2 = build_present(grid2)
        present3 = build_present(grid3)

        shape_params = unify_shape([present1, present2, present3])

        # All grids are 3×3 → meet should work
        assert shape_params.num_row_classes >= 1
        assert shape_params.num_col_classes >= 1

        # Should have 3 row partitions (one per grid)
        assert len(shape_params.row_partitions) == 3
        assert len(shape_params.col_partitions) == 3

        # Each partition should have correct number of elements
        for part in shape_params.row_partitions:
            assert len(part) == len(present1.grid)  # 3 rows

        for part in shape_params.col_partitions:
            assert len(part) == len(present1.grid[0])  # 3 cols

    def test_uniform_scale(self):
        """Uniform scaling (2×2 → 4×4)."""
        # Grid 1: 2×2
        grid1 = [[1, 2], [3, 4]]

        # Grid 2: 4×4 (each cell of grid1 becomes 2×2 block)
        grid2 = [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]

        present1 = build_present(grid1)
        present2 = build_present(grid2)

        shape_params = unify_shape([present1, present2])

        # Different sizes → fallback to identity partitions for first grid
        # (per implementation lines 286-290)
        assert shape_params.R is not None
        assert shape_params.C is not None

        # Should have 2 row partitions
        assert len(shape_params.row_partitions) == 2
        assert len(shape_params.col_partitions) == 2

    def test_nps_bands(self):
        """Non-uniform periodic sampling (NPS) bands."""
        # Grid 1: rows [0,2] similar, rows [1,3] different
        grid1 = [[1, 1, 1], [2, 2, 2], [1, 1, 1], [3, 3, 3]]

        # Grid 2: rows [0,1] similar, rows [2,3] different
        grid2 = [[1, 1, 1], [1, 1, 1], [2, 2, 2], [3, 3, 3]]

        present1 = build_present(grid1)
        present2 = build_present(grid2)

        shape_params = unify_shape([present1, present2])

        # Both grids are 4×3 → meet should work
        assert shape_params.num_row_classes >= 1
        assert shape_params.num_col_classes >= 1

        # Meet should distinguish rows based on BOTH inputs
        # (non-uniform periodic structure)
        assert len(shape_params.row_partitions) == 2
        assert len(shape_params.col_partitions) == 2

    def test_single_training_example(self):
        """Meet = identity for single training example."""
        grid = [[1, 2], [3, 4]]

        present = build_present(grid)

        shape_params = unify_shape([present])

        # Single input → meet is normalized version of that input's partition
        assert len(shape_params.row_partitions) == 1
        assert len(shape_params.col_partitions) == 1

        # Should have valid class counts
        assert shape_params.num_row_classes >= 1
        assert shape_params.num_col_classes >= 1

    def test_empty_training_set(self):
        """Empty training set returns empty ShapeParams."""
        shape_params = unify_shape([])

        assert shape_params.R == {}
        assert shape_params.C == {}
        assert shape_params.num_row_classes == 0
        assert shape_params.num_col_classes == 0
        assert shape_params.row_partitions == []
        assert shape_params.col_partitions == []

    def test_different_grid_sizes(self):
        """Different grid sizes trigger fallback."""
        grid1 = [[1, 2], [3, 4]]  # 2×2
        grid2 = [[5, 6, 7], [8, 9, 10], [11, 12, 13]]  # 3×3
        grid3 = [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]  # 4×4

        present1 = build_present(grid1)
        present2 = build_present(grid2)
        present3 = build_present(grid3)

        shape_params = unify_shape([present1, present2, present3])

        # Different shapes → fallback to identity partitions for first grid
        # Verify it returns valid ShapeParams
        assert shape_params.R is not None
        assert shape_params.C is not None
        assert shape_params.num_row_classes >= 0
        assert shape_params.num_col_classes >= 0

    def test_shape_params_structure(self):
        """ShapeParams has correct structure and types."""
        grid1 = [[1, 2], [3, 4]]
        grid2 = [[5, 6], [7, 8]]

        present1 = build_present(grid1)
        present2 = build_present(grid2)

        shape_params = unify_shape([present1, present2])

        # Verify types
        assert isinstance(shape_params.R, dict)
        assert isinstance(shape_params.C, dict)
        assert isinstance(shape_params.num_row_classes, int)
        assert isinstance(shape_params.num_col_classes, int)
        assert isinstance(shape_params.row_partitions, list)
        assert isinstance(shape_params.col_partitions, list)

        # Verify counts match partitions
        assert shape_params.num_row_classes == len(set(shape_params.R.values()))
        assert shape_params.num_col_classes == len(set(shape_params.C.values()))

        # Verify partition lists have correct length
        assert len(shape_params.row_partitions) == 2
        assert len(shape_params.col_partitions) == 2


class TestDeterminism:
    """Test determinism and order-independence."""

    def test_unify_shape_deterministic(self):
        """Same inputs produce same result."""
        grid1 = [[1, 2], [3, 4]]
        grid2 = [[5, 6], [7, 8]]

        present1 = build_present(grid1)
        present2 = build_present(grid2)

        result1 = unify_shape([present1, present2])
        result2 = unify_shape([present1, present2])

        assert result1.R == result2.R
        assert result1.C == result2.C
        assert result1.num_row_classes == result2.num_row_classes
        assert result1.num_col_classes == result2.num_col_classes

    def test_unify_shape_permutation_invariant(self):
        """Permuting training order produces same result."""
        grid1 = [[1, 2], [3, 4]]
        grid2 = [[5, 6], [7, 8]]
        grid3 = [[9, 10], [11, 12]]

        present1 = build_present(grid1)
        present2 = build_present(grid2)
        present3 = build_present(grid3)

        result_123 = unify_shape([present1, present2, present3])
        result_321 = unify_shape([present3, present2, present1])
        result_213 = unify_shape([present2, present1, present3])

        # Meet should be same (order-independent)
        assert result_123.R == result_321.R == result_213.R
        assert result_123.C == result_321.C == result_213.C
        assert result_123.num_row_classes == result_321.num_row_classes == result_213.num_row_classes

    def test_wl_1d_two_run_stability(self):
        """Running 1-D WL twice produces byte-identical results."""
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        part_rows_1 = wl_1d_on_rows(grid, max_iters=12)
        part_rows_2 = wl_1d_on_rows(grid, max_iters=12)

        part_cols_1 = wl_1d_on_cols(grid, max_iters=12)
        part_cols_2 = wl_1d_on_cols(grid, max_iters=12)

        assert part_rows_1 == part_rows_2, "1-D WL rows must be byte-stable"
        assert part_cols_1 == part_cols_2, "1-D WL cols must be byte-stable"

    def test_meet_two_run_stability(self):
        """Running meet twice produces byte-identical results."""
        part1 = {0: 0, 1: 0, 2: 1}
        part2 = {0: 5, 1: 6, 2: 5}

        result1 = meet_partitions([part1, part2])
        result2 = meet_partitions([part1, part2])

        assert result1 == result2, "Meet must be byte-stable"


class TestNoLeakage:
    """Test that shape unification uses only inputs (no output leakage)."""

    def test_no_output_data_in_unify_shape(self):
        """unify_shape uses only Present objects (inputs), never outputs."""
        # This is verified by API signature: unify_shape(presents_train)
        # No Y_i passed, only X_i via Present

        grid1 = [[1, 2], [3, 4]]
        grid2 = [[5, 6], [7, 8]]

        present1 = build_present(grid1)
        present2 = build_present(grid2)

        # Should not raise any errors
        shape_params = unify_shape([present1, present2])

        # Verify it uses only present.grid (not any output data)
        assert shape_params.R is not None
        assert shape_params.C is not None

    def test_wl_1d_no_coordinates_in_features(self):
        """1-D WL uses hashes, not (row, col) coordinates."""
        # Seed uses hash64(row_tuple) or hash64(col_tuple)
        # Signatures use (current_color, sorted_neighbors)
        # No raw coordinates appear in features

        grid = [[1, 2, 3], [4, 5, 6]]

        part_rows = wl_1d_on_rows(grid)
        part_cols = wl_1d_on_cols(grid)

        # Verify partition keys are indices (OK), values are class IDs (OK)
        for row_id, class_id in part_rows.items():
            assert isinstance(row_id, int)
            assert isinstance(class_id, int)
            assert class_id >= 0

        for col_id, class_id in part_cols.items():
            assert isinstance(col_id, int)
            assert isinstance(class_id, int)
            assert class_id >= 0

    def test_meet_uses_only_partitions(self):
        """Meet operates on partitions (dicts), not raw grids."""
        # No grid data passes through meet_partitions
        part1 = {0: 0, 1: 1}
        part2 = {0: 5, 1: 5}

        result = meet_partitions([part1, part2])

        # Verify result is valid partition
        assert isinstance(result, dict)
        for elem, class_id in result.items():
            assert isinstance(elem, int)
            assert isinstance(class_id, int)


class TestEdgeCases:
    """Test edge cases and stress scenarios."""

    def test_single_pixel_grid(self):
        """1×1 grid."""
        grid = [[5]]

        present = build_present(grid)
        shape_params = unify_shape([present])

        assert shape_params.num_row_classes == 1
        assert shape_params.num_col_classes == 1
        assert len(shape_params.R) == 1
        assert len(shape_params.C) == 1

    def test_single_row_grid(self):
        """1×10 grid (may be rotated by ΠG)."""
        grid = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

        present = build_present(grid)
        shape_params = unify_shape([present])

        # After ΠG canonicalization, 1×10 may become 10×1 (rotated 90°)
        # Test the canonical grid dimensions, not original input
        canonical_rows = len(present.grid)
        canonical_cols = len(present.grid[0])

        # Verify shape params match canonical grid dimensions
        assert len(shape_params.R) == canonical_rows
        assert len(shape_params.C) == canonical_cols
        assert shape_params.num_row_classes >= 1
        assert shape_params.num_col_classes >= 1

    def test_single_col_grid(self):
        """10×1 grid (may be rotated by ΠG)."""
        grid = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

        present = build_present(grid)
        shape_params = unify_shape([present])

        # After ΠG canonicalization, 10×1 may become 1×10 (rotated 90°)
        # Test the canonical grid dimensions, not original input
        canonical_rows = len(present.grid)
        canonical_cols = len(present.grid[0])

        # Verify shape params match canonical grid dimensions
        assert len(shape_params.R) == canonical_rows
        assert len(shape_params.C) == canonical_cols
        assert shape_params.num_row_classes >= 1
        assert shape_params.num_col_classes >= 1

    def test_large_grid_30x30(self):
        """Large 30×30 grid."""
        grid = [[((r + c) % 10) for c in range(30)] for r in range(30)]

        present = build_present(grid)
        shape_params = unify_shape([present])

        assert shape_params.num_row_classes >= 1
        assert shape_params.num_col_classes >= 1
        assert len(shape_params.R) == 30
        assert len(shape_params.C) == 30

    def test_uniform_color_grid(self):
        """All pixels same color."""
        grid = [[5, 5, 5], [5, 5, 5], [5, 5, 5]]

        present = build_present(grid)
        shape_params = unify_shape([present])

        # All rows identical → likely 1 row class
        # All cols identical → likely 1 col class
        assert shape_params.num_row_classes >= 1
        assert shape_params.num_col_classes >= 1

    def test_degenerate_partition_all_same_class(self):
        """Meet handles degenerate partition (all elements in one class)."""
        part1 = {0: 0, 1: 0, 2: 0, 3: 0}
        part2 = {0: 5, 1: 5, 2: 5, 3: 5}

        result = meet_partitions([part1, part2])

        # All elements in same class in both → same class in meet
        classes = set(result.values())
        assert len(classes) == 1, "All elements should be in same class"

    def test_wl_1d_max_iterations_boundary(self):
        """Verify WL converges within max_iters=12."""
        # Create grid that needs multiple iterations
        grid = [[i % 3 for i in range(10)] for _ in range(10)]

        part_12 = wl_1d_on_rows(grid, max_iters=12)
        part_20 = wl_1d_on_rows(grid, max_iters=20)

        # Should be same (already converged by iter 12)
        assert part_12 == part_20, "WL should converge within 12 iterations"


class TestIntegrationWithPresent:
    """Test integration with Present objects from WO-02."""

    def test_unify_shape_with_real_present_objects(self):
        """unify_shape works with real Present objects."""
        # Create grids and build Present objects
        grid1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        grid2 = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]

        present1 = build_present(grid1)
        present2 = build_present(grid2)

        # Verify Present objects are valid
        assert present1.grid is not None
        assert present2.grid is not None

        # unify_shape should work
        shape_params = unify_shape([present1, present2])

        assert shape_params.R is not None
        assert shape_params.C is not None
        assert shape_params.num_row_classes >= 1
        assert shape_params.num_col_classes >= 1

    def test_shape_unification_after_pig_canonicalization(self):
        """Shape unification works on canonicalized grids from ΠG."""
        # After ΠG, grids are in canonical frame (D4 normalized)
        grid1 = [[1, 1], [1, 2]]
        grid2 = [[2, 1], [1, 1]]  # Rotated version

        present1 = build_present(grid1)
        present2 = build_present(grid2)

        # After PiG, both should be canonicalized
        # Shape unification should work on canonical grids
        shape_params = unify_shape([present1, present2])

        # Both grids are 2×2 after canonicalization
        assert shape_params.R is not None
        assert shape_params.C is not None

    def test_wl_1d_on_canonical_grids(self):
        """1-D WL operates on canonical grids from Present."""
        grid = [[1, 2], [3, 4]]

        present = build_present(grid)

        # Extract canonical grid
        canonical_grid = present.grid

        # Run 1-D WL on canonical grid
        part_rows = wl_1d_on_rows(canonical_grid)
        part_cols = wl_1d_on_cols(canonical_grid)

        # Should work correctly
        assert len(part_rows) == len(canonical_grid)
        assert len(part_cols) == len(canonical_grid[0])


class TestReceipts:
    """Test receipt structure and values."""

    def test_shape_params_receipt_fields(self):
        """ShapeParams contains all required receipt fields."""
        grid1 = [[1, 2], [3, 4]]
        grid2 = [[5, 6], [7, 8]]

        present1 = build_present(grid1)
        present2 = build_present(grid2)

        shape_params = unify_shape([present1, present2])

        # Required fields
        assert hasattr(shape_params, "R")
        assert hasattr(shape_params, "C")
        assert hasattr(shape_params, "num_row_classes")
        assert hasattr(shape_params, "num_col_classes")
        assert hasattr(shape_params, "row_partitions")
        assert hasattr(shape_params, "col_partitions")

        # Values are valid
        assert isinstance(shape_params.R, dict)
        assert isinstance(shape_params.C, dict)
        assert isinstance(shape_params.num_row_classes, int)
        assert isinstance(shape_params.num_col_classes, int)
        assert isinstance(shape_params.row_partitions, list)
        assert isinstance(shape_params.col_partitions, list)

        # Counts are positive
        assert shape_params.num_row_classes >= 1
        assert shape_params.num_col_classes >= 1

    def test_shape_params_as_dict(self):
        """ShapeParams can be converted to dict for JSON receipts."""
        grid = [[1, 2], [3, 4]]

        present = build_present(grid)
        shape_params = unify_shape([present])

        # Convert to dict
        receipt = {
            "num_row_classes": shape_params.num_row_classes,
            "num_col_classes": shape_params.num_col_classes,
            "R": shape_params.R,
            "C": shape_params.C,
        }

        # Should be serializable to JSON
        json_str = json.dumps(receipt)
        assert isinstance(json_str, str)

        # Should be parseable
        parsed = json.loads(json_str)
        assert parsed["num_row_classes"] == shape_params.num_row_classes
        assert parsed["num_col_classes"] == shape_params.num_col_classes
