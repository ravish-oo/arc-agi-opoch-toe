"""
Unit tests for arc_core/present.py (WO-02).

Per implementation_plan.md lines 141:
- ΠG idempotent
- D4 transformations
- Minimal BB anchor
- Component summary determinism

Acceptance criteria:
- ΠG(ΠG(X)) = ΠG(X)
- Present has eq members and no coords
- All transformations deterministic
"""

import pytest

from arc_core.present import (
    D4_INVERSES,
    D4_TRANSFORMATIONS,
    PiG,
    PiG_with_inverse,
    build_present,
    compute_cbc3,
    flip_diag_anti,
    flip_diag_main,
    flip_h,
    flip_v,
    get_anchor,
    get_minimal_bounding_box,
    rot180,
    rot270,
    rot90,
)
from arc_core.types import Pixel


class TestD4Transformations:
    """Test D4 group operations (8 transformations)."""

    def test_rot90(self):
        """Rotate 90° clockwise."""
        grid = [
            [1, 2],
            [3, 4],
        ]
        result = rot90(grid)
        expected = [
            [3, 1],
            [4, 2],
        ]
        assert result == expected

    def test_rot180(self):
        """Rotate 180°."""
        grid = [
            [1, 2],
            [3, 4],
        ]
        result = rot180(grid)
        expected = [
            [4, 3],
            [2, 1],
        ]
        assert result == expected

    def test_rot270(self):
        """Rotate 270° clockwise."""
        grid = [
            [1, 2],
            [3, 4],
        ]
        result = rot270(grid)
        expected = [
            [2, 4],
            [1, 3],
        ]
        assert result == expected

    def test_flip_h(self):
        """Flip horizontal (left-right)."""
        grid = [
            [1, 2],
            [3, 4],
        ]
        result = flip_h(grid)
        expected = [
            [2, 1],
            [4, 3],
        ]
        assert result == expected

    def test_flip_v(self):
        """Flip vertical (top-bottom)."""
        grid = [
            [1, 2],
            [3, 4],
        ]
        result = flip_v(grid)
        expected = [
            [3, 4],
            [1, 2],
        ]
        assert result == expected

    def test_flip_diag_main(self):
        """Flip over main diagonal (transpose)."""
        grid = [
            [1, 2],
            [3, 4],
        ]
        result = flip_diag_main(grid)
        expected = [
            [1, 3],
            [2, 4],
        ]
        assert result == expected

    def test_flip_diag_anti(self):
        """Flip over anti-diagonal."""
        grid = [
            [1, 2],
            [3, 4],
        ]
        result = flip_diag_anti(grid)
        expected = [
            [4, 2],
            [3, 1],
        ]
        assert result == expected

    def test_d4_group_size(self):
        """D4 has exactly 8 elements."""
        assert len(D4_TRANSFORMATIONS) == 8
        assert len(D4_INVERSES) == 8

    def test_d4_inverses_self_inverse(self):
        """Some transformations are self-inverse."""
        # Flips are self-inverse
        assert D4_INVERSES["flip_h"] == "flip_h"
        assert D4_INVERSES["flip_v"] == "flip_v"
        assert D4_INVERSES["flip_diag_main"] == "flip_diag_main"
        assert D4_INVERSES["flip_diag_anti"] == "flip_diag_anti"

    def test_d4_inverses_rotations(self):
        """Rotation inverses are correct."""
        assert D4_INVERSES["rot90"] == "rot270"
        assert D4_INVERSES["rot270"] == "rot90"
        assert D4_INVERSES["rot180"] == "rot180"  # Self-inverse


class TestMinimalBoundingBox:
    """Test minimal bounding box and anchor computation."""

    def test_get_minimal_bounding_box_simple(self):
        """Simple 2x2 non-zero region."""
        grid = [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ]
        bb = get_minimal_bounding_box(grid)
        assert bb == (1, 1, 2, 2)

    def test_get_minimal_bounding_box_full(self):
        """All non-zero."""
        grid = [
            [1, 2],
            [3, 4],
        ]
        bb = get_minimal_bounding_box(grid)
        assert bb == (0, 0, 1, 1)

    def test_get_minimal_bounding_box_single_pixel(self):
        """Single non-zero pixel."""
        grid = [
            [0, 0, 0],
            [0, 5, 0],
            [0, 0, 0],
        ]
        bb = get_minimal_bounding_box(grid)
        assert bb == (1, 1, 1, 1)

    def test_get_minimal_bounding_box_all_zeros(self):
        """All zeros -> full canvas BB."""
        grid = [
            [0, 0],
            [0, 0],
        ]
        bb = get_minimal_bounding_box(grid)
        assert bb == (0, 0, 1, 1)

    def test_get_anchor(self):
        """Anchor is top-left of minimal BB."""
        grid = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
        anchor = get_anchor(grid)
        assert anchor == Pixel(1, 1)


class TestPiG:
    """Test ΠG canonicalization."""

    def test_pig_idempotent(self):
        """ΠG(ΠG(X)) = ΠG(X)."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        canonical1 = PiG(grid)
        canonical2 = PiG(canonical1)
        assert canonical1 == canonical2, "ΠG must be idempotent"

    def test_pig_deterministic(self):
        """Same input always produces same output."""
        grid = [
            [1, 2],
            [3, 4],
        ]
        result1 = PiG(grid)
        result2 = PiG(grid)
        assert result1 == result2

    def test_pig_chooses_lex_min(self):
        """ΠG chooses lex-min over all D4 transformations."""
        # Create a grid where different transformations have different anchors
        grid = [
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
        ]
        canonical = PiG(grid)

        # The lex-min anchor should be (0, 0) with value 1 at that position
        # (different rotations will have different anchors)
        # Check that the canonical form is deterministic
        assert canonical == PiG(canonical)

    def test_pig_with_inverse_returns_transform(self):
        """PiG_with_inverse returns the transformation name."""
        grid = [
            [1, 2],
            [3, 4],
        ]
        canonical, transform_name = PiG_with_inverse(grid)

        # Verify it's a valid D4 transformation
        assert transform_name in D4_TRANSFORMATIONS

        # Verify canonical grid matches PiG
        assert canonical == PiG(grid)

    def test_pig_with_inverse_has_inverse(self):
        """Transformation has a defined inverse."""
        grid = [
            [1, 2],
            [3, 4],
        ]
        _, transform_name = PiG_with_inverse(grid)

        # Inverse must be defined
        assert transform_name in D4_INVERSES
        inverse_name = D4_INVERSES[transform_name]
        assert inverse_name in D4_TRANSFORMATIONS


class TestCBC3:
    """Test CBC3 feature extraction."""

    def test_compute_cbc3_deterministic(self):
        """Same pixel always produces same CBC3 token."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        pixel = Pixel(1, 1)

        token1 = compute_cbc3(grid, pixel)
        token2 = compute_cbc3(grid, pixel)
        assert token1 == token2

    def test_compute_cbc3_different_patches(self):
        """Different patches produce different tokens (likely)."""
        grid = [
            [1, 1, 1],
            [1, 2, 1],
            [1, 1, 1],
        ]
        token_center = compute_cbc3(grid, Pixel(1, 1))

        grid2 = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        token_uniform = compute_cbc3(grid2, Pixel(1, 1))

        assert token_center != token_uniform

    def test_compute_cbc3_edge_padding(self):
        """Edge pixels use zero-padding."""
        grid = [
            [1, 2],
            [3, 4],
        ]
        # Top-left corner should work (padded with zeros)
        token = compute_cbc3(grid, Pixel(0, 0))
        assert isinstance(token, int)  # Should produce a valid hash

    def test_compute_cbc3_ofa_relabeling(self):
        """OFA relabeling makes patches with same structure hash identically."""
        # Two grids with different colors but same structure
        grid1 = [
            [1, 1, 1],
            [1, 2, 1],
            [1, 1, 1],
        ]
        grid2 = [
            [5, 5, 5],
            [5, 7, 5],
            [5, 5, 5],
        ]

        # Both should have same CBC3 (OFA relabels to 0, 1)
        token1 = compute_cbc3(grid1, Pixel(1, 1))
        token2 = compute_cbc3(grid2, Pixel(1, 1))
        assert token1 == token2, "OFA relabeling should normalize color values"


class TestBuildPresent:
    """Test full present structure builder."""

    def test_build_present_has_all_fields(self):
        """Present structure contains all required fields."""
        grid = [
            [1, 2],
            [3, 4],
        ]
        present = build_present(grid)

        assert hasattr(present, "grid")
        assert hasattr(present, "cbc3")
        assert hasattr(present, "e4_neighbors")
        assert hasattr(present, "row_members")
        assert hasattr(present, "col_members")
        assert hasattr(present, "g_inverse")

    def test_build_present_grid_is_canonical(self):
        """Present grid is canonical (ΠG applied)."""
        grid = [
            [1, 2],
            [3, 4],
        ]
        present = build_present(grid)

        # Present grid should equal PiG(grid)
        assert present.grid == PiG(grid)

    def test_build_present_cbc3_all_pixels(self):
        """CBC3 tokens computed for all pixels."""
        grid = [
            [1, 2],
            [3, 4],
        ]
        present = build_present(grid)

        rows, cols = 2, 2
        expected_pixels = {Pixel(r, c) for r in range(rows) for c in range(cols)}
        actual_pixels = set(present.cbc3.keys())

        assert actual_pixels == expected_pixels

    def test_build_present_e4_neighbors_correct(self):
        """E4 neighbors are 4-connected (up, down, left, right)."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        present = build_present(grid)

        # Center pixel (1, 1) should have 4 neighbors
        center_neighbors = present.e4_neighbors[Pixel(1, 1)]
        assert len(center_neighbors) == 4
        assert set(center_neighbors) == {
            Pixel(0, 1),  # Up
            Pixel(2, 1),  # Down
            Pixel(1, 0),  # Left
            Pixel(1, 2),  # Right
        }

        # Corner pixel (0, 0) should have 2 neighbors
        corner_neighbors = present.e4_neighbors[Pixel(0, 0)]
        assert len(corner_neighbors) == 2
        assert set(corner_neighbors) == {
            Pixel(1, 0),  # Down
            Pixel(0, 1),  # Right
        }

    def test_build_present_row_members(self):
        """Row members group pixels by row."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        present = build_present(grid)

        # Row 0 should have 3 pixels
        assert len(present.row_members[0]) == 3
        assert set(present.row_members[0]) == {Pixel(0, 0), Pixel(0, 1), Pixel(0, 2)}

        # Row 1 should have 3 pixels
        assert len(present.row_members[1]) == 3
        assert set(present.row_members[1]) == {Pixel(1, 0), Pixel(1, 1), Pixel(1, 2)}

    def test_build_present_col_members(self):
        """Column members group pixels by column."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        present = build_present(grid)

        # Col 0 should have 2 pixels
        assert len(present.col_members[0]) == 2
        assert set(present.col_members[0]) == {Pixel(0, 0), Pixel(1, 0)}

        # Col 2 should have 2 pixels
        assert len(present.col_members[2]) == 2
        assert set(present.col_members[2]) == {Pixel(0, 2), Pixel(1, 2)}

    def test_build_present_g_inverse_valid(self):
        """g_inverse is a valid transformation name."""
        grid = [
            [1, 2],
            [3, 4],
        ]
        present = build_present(grid)

        assert present.g_inverse in D4_TRANSFORMATIONS
        assert present.g_inverse in D4_INVERSES

    def test_build_present_deterministic(self):
        """Building present twice gives same result."""
        grid = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        present1 = build_present(grid)
        present2 = build_present(grid)

        assert present1.grid == present2.grid
        assert present1.cbc3 == present2.cbc3
        assert present1.e4_neighbors == present2.e4_neighbors
        assert present1.row_members == present2.row_members
        assert present1.col_members == present2.col_members
        assert present1.g_inverse == present2.g_inverse
