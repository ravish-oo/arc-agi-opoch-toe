"""
Unit tests for arc_core/order_hash.py (WO-01).

Per implementation_plan.md lines 123-126:
- Determinism across repeated calls
- Nested tuples; dict order irrelevance
- boundary_hash on plus-shape component (verify E4 boundary)

Acceptance criteria:
- hash64 stable across runs (deterministic)
- lex_min invariant under permutation
- boundary_hash uses E4 (4-connected) neighbors
"""

import pytest

from arc_core.order_hash import boundary_hash, hash64, lex_min
from arc_core.types import Pixel


class TestHash64:
    """Test deterministic hashing with SHA-256."""

    def test_determinism_simple(self):
        """Same input produces same hash across multiple calls."""
        obj = [1, 2, 3, 4, 5]
        hash1 = hash64(obj)
        hash2 = hash64(obj)
        assert hash1 == hash2, "hash64 must be deterministic"

    def test_determinism_complex(self):
        """Nested structures produce stable hashes."""
        obj = {"a": [1, 2, {"b": 3}], "c": (4, 5)}
        hash1 = hash64(obj)
        hash2 = hash64(obj)
        assert hash1 == hash2

    def test_dict_order_irrelevance(self):
        """Dict key order doesn't affect hash (canonical JSON)."""
        dict1 = {"a": 1, "b": 2, "c": 3}
        dict2 = {"c": 3, "a": 1, "b": 2}
        dict3 = {"b": 2, "c": 3, "a": 1}

        assert hash64(dict1) == hash64(dict2) == hash64(dict3)

    def test_different_inputs_different_hashes(self):
        """Different inputs should produce different hashes (collision unlikely)."""
        hash1 = hash64([1, 2, 3])
        hash2 = hash64([1, 2, 4])
        assert hash1 != hash2

    def test_nested_dict_canonical(self):
        """Nested dicts with different key orders hash identically."""
        obj1 = {"outer": {"a": 1, "b": 2}, "c": 3}
        obj2 = {"c": 3, "outer": {"b": 2, "a": 1}}
        assert hash64(obj1) == hash64(obj2)

    def test_returns_64bit_int(self):
        """Hash fits in 64 bits (0 to 2^64-1)."""
        h = hash64("test")
        assert 0 <= h < 2**64


class TestLexMin:
    """Test lexicographic minimum by global order."""

    def test_lex_min_pixels_row_major(self):
        """Pixels ordered by row-major (row, col)."""
        pixels = [Pixel(1, 2), Pixel(0, 5), Pixel(1, 0), Pixel(0, 3)]
        result = lex_min(pixels)
        # Lex-min: (0, 3) comes before (0, 5), (1, 0), (1, 2)
        assert result == Pixel(0, 3)

    def test_lex_min_integers(self):
        """Simple integer comparison."""
        items = [5, 2, 8, 1, 3]
        assert lex_min(items) == 1

    def test_lex_min_with_key(self):
        """Use key function to extract comparison tuple."""
        items = [(2, "b"), (1, "a"), (2, "a")]
        result = lex_min(items, key=lambda x: x)
        assert result == (1, "a")

    def test_lex_min_tuples(self):
        """Tuples compared lexicographically."""
        tuples = [(1, 3, 5), (1, 2, 9), (1, 3, 4), (0, 9, 9)]
        assert lex_min(tuples) == (0, 9, 9)

    def test_lex_min_permutation_invariant(self):
        """Result same regardless of input order."""
        items1 = [3, 1, 4, 1, 5, 9, 2, 6]
        items2 = [9, 5, 6, 2, 4, 3, 1, 1]
        assert lex_min(items1) == lex_min(items2) == 1

    def test_lex_min_empty_raises(self):
        """Empty iterable raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            lex_min([])

    def test_lex_min_single_element(self):
        """Single element is trivially the minimum."""
        assert lex_min([42]) == 42


class TestBoundaryHash:
    """Test E4 (4-connected) boundary hashing."""

    def test_boundary_hash_plus_shape(self):
        """
        Plus-shape component (4 edge pixels are boundary with E4).

        Grid:
            0 1 0
            1 1 1
            0 1 0

        Component (color 1): {(0,1), (1,0), (1,1), (1,2), (2,1)}
        Boundary (E4): {(0,1), (1,0), (1,2), (2,1)}
        Center (1,1) is NOT boundary - all its 4-neighbors are also color 1.
        """
        grid = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
        component = {Pixel(0, 1), Pixel(1, 0), Pixel(1, 1), Pixel(1, 2), Pixel(2, 1)}

        # Get hash (should be deterministic)
        h1 = boundary_hash(component, grid)
        h2 = boundary_hash(component, grid)
        assert h1 == h2, "boundary_hash must be deterministic"

        # Only 4 edge pixels are boundary (center has all same-color 4-neighbors)
        expected_boundary = [(0, 1), (1, 0), (1, 2), (2, 1)]
        expected_hash = hash64(expected_boundary)
        assert h1 == expected_hash

    def test_boundary_hash_single_pixel(self):
        """Single pixel surrounded by different color is all boundary."""
        grid = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
        component = {Pixel(1, 1)}

        h = boundary_hash(component, grid)
        expected = hash64([(1, 1)])
        assert h == expected

    def test_boundary_hash_solid_block_interior(self):
        """
        3x3 solid block: edge pixels are boundary, center is not (E4).

        Grid:
            1 1 1
            1 1 1
            1 1 1

        Edge pixels have out-of-bounds neighbors → boundary.
        Center (1,1) has all 4-neighbors in-bounds with same color → NOT boundary.
        """
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        component = {
            Pixel(0, 0),
            Pixel(0, 1),
            Pixel(0, 2),
            Pixel(1, 0),
            Pixel(1, 1),
            Pixel(1, 2),
            Pixel(2, 0),
            Pixel(2, 1),
            Pixel(2, 2),
        }

        h = boundary_hash(component, grid)
        # 8 edge pixels are boundary (center (1,1) is not)
        expected_boundary = [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            # (1, 1) NOT included - interior pixel
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ]
        expected = hash64(expected_boundary)
        assert h == expected

    def test_boundary_hash_2x2_in_larger_grid(self):
        """
        2x2 component inside larger grid (E4 boundary = all 4 pixels).

        Grid:
            0 0 0 0
            0 2 2 0
            0 2 2 0
            0 0 0 0

        All 4 pixels of color 2 have 4-neighbors with color 0 → all boundary.
        """
        grid = [
            [0, 0, 0, 0],
            [0, 2, 2, 0],
            [0, 2, 2, 0],
            [0, 0, 0, 0],
        ]
        component = {Pixel(1, 1), Pixel(1, 2), Pixel(2, 1), Pixel(2, 2)}

        h = boundary_hash(component, grid)
        expected_boundary = [(1, 1), (1, 2), (2, 1), (2, 2)]
        expected = hash64(expected_boundary)
        assert h == expected

    def test_boundary_hash_empty_component(self):
        """Empty component hashes to empty list."""
        grid = [[0, 0], [0, 0]]
        component = set()

        h = boundary_hash(component, grid)
        expected = hash64([])
        assert h == expected

    def test_boundary_hash_deterministic_sorting(self):
        """Boundary pixels sorted before hashing (row-major order)."""
        grid = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
        # Add pixels in non-sorted order
        component = {Pixel(2, 1), Pixel(0, 1), Pixel(1, 2), Pixel(1, 0), Pixel(1, 1)}

        h = boundary_hash(component, grid)
        # Boundary (E4): (0,1), (1,0), (1,2), (2,1) - excludes center (1,1)
        expected = hash64([(0, 1), (1, 0), (1, 2), (2, 1)])
        assert h == expected
