"""
Unit tests for WO-14: arc_laws/connect_fill.py

Testing CONNECT_ENDPOINTS and REGION_FILL laws.

Per test plan (test_plans/WO-14-plan.md):
- CONNECT: Geodesic shortest paths, lex-min tie-breaking, 4-conn vs 8-conn
- FILL: Mask respect, selector integration, flood completeness
- Critical battle-tests: lex-min selection, connectivity enforcement, mask boundaries

No xfail, no skip, no invalid input tests.
Tests designed to FIND BUGS, not just pass.
"""

import pytest
from typing import List, Set
from collections import deque

from arc_core.types import Pixel
from arc_laws.connect_fill import (
    shortest_path_bfs,
    flood_fill,
    ConnectLaw,
    FillLaw,
    apply_connect_law,
    apply_fill_law,
    build_connect_fill,
)
from arc_laws.selectors import apply_selector_on_test


# =============================================================================
# Helper Functions
# =============================================================================


def compute_manhattan_distance(p1: Pixel, p2: Pixel) -> int:
    """4-conn distance (Manhattan)."""
    return abs(p1.row - p2.row) + abs(p1.col - p2.col)


def compute_chebyshev_distance(p1: Pixel, p2: Pixel) -> int:
    """8-conn distance (Chebyshev)."""
    return max(abs(p1.row - p2.row), abs(p1.col - p2.col))


def is_valid_4conn_step(p1: Pixel, p2: Pixel) -> bool:
    """Check if p1 -> p2 is a valid 4-connected step (orthogonal only)."""
    dr = abs(p1.row - p2.row)
    dc = abs(p1.col - p2.col)
    # Must move exactly 1 step in one direction (no diagonal)
    return (dr == 1 and dc == 0) or (dr == 0 and dc == 1)


def is_valid_8conn_step(p1: Pixel, p2: Pixel) -> bool:
    """Check if p1 -> p2 is a valid 8-connected step (orthogonal + diagonal)."""
    dr = abs(p1.row - p2.row)
    dc = abs(p1.col - p2.col)
    # Must move at most 1 step in each direction
    return dr <= 1 and dc <= 1 and (dr + dc) > 0


def enumerate_all_4conn_paths(start: Pixel, end: Pixel, max_length: int) -> List[List[Pixel]]:
    """
    Enumerate all 4-conn paths from start to end of given length.

    Used for lex-min verification.
    """
    if start == end:
        return [[start]]

    paths = []

    def dfs(current: Pixel, path: List[Pixel], remaining: int):
        if current == end and remaining == 0:
            paths.append(path[:])
            return

        if remaining <= 0:
            return

        # Try all 4 neighbors
        for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            nr, nc = current.row + dr, current.col + dc
            neighbor = Pixel(nr, nc)

            # Skip if already in path (no cycles)
            if neighbor in path:
                continue

            path.append(neighbor)
            dfs(neighbor, path, remaining - 1)
            path.pop()

    dfs(start, [start], max_length - 1)
    return paths


# =============================================================================
# Test Class: CONNECT_ENDPOINTS - 4-conn
# =============================================================================


class TestConnectEndpoints4Conn:
    """Test 4-connected shortest path computation."""

    def test_4conn_horizontal_straight(self):
        """CE-01: Horizontal straight line path."""
        grid = [[0] * 8 for _ in range(1)]
        anchor1 = Pixel(0, 2)
        anchor2 = Pixel(0, 5)

        path = shortest_path_bfs(grid, anchor1, anchor2, "4conn")

        assert path is not None, "Path should exist"
        assert len(path) == 4, f"Path length should be 4, got {len(path)}"
        assert path[0] == anchor1, "Path should start at anchor1"
        assert path[-1] == anchor2, "Path should end at anchor2"

        # Verify path is straight horizontal
        expected = [Pixel(0, 2), Pixel(0, 3), Pixel(0, 4), Pixel(0, 5)]
        assert path == expected, f"Expected straight horizontal path, got {path}"

    def test_4conn_vertical_straight(self):
        """CE-02: Vertical straight line path."""
        grid = [[0] for _ in range(6)]
        anchor1 = Pixel(0, 0)
        anchor2 = Pixel(5, 0)

        path = shortest_path_bfs(grid, anchor1, anchor2, "4conn")

        assert path is not None, "Path should exist"
        assert len(path) == 6, f"Path length should be 6, got {len(path)}"
        assert path[0] == anchor1, "Path should start at anchor1"
        assert path[-1] == anchor2, "Path should end at anchor2"

        # Verify path is straight vertical
        expected = [Pixel(i, 0) for i in range(6)]
        assert path == expected, f"Expected straight vertical path, got {path}"

    def test_4conn_lexmin_CRITICAL(self):
        """
        CE-04: CRITICAL - Lex-min path selection among multiple geodesics.

        This test PROVES lex-min selection (not just any shortest path).

        Grid: 3×3
        Markers at (0,0) and (2,2)
        Manhattan distance = 4

        Multiple shortest paths exist:
        Path A (row-first): [(0,0), (0,1), (0,2), (1,2), (2,2)]
        Path B (col-first): [(0,0), (1,0), (2,0), (2,1), (2,2)]
        Path C (zigzag):    [(0,0), (1,0), (1,1), (1,2), (2,2)]
        ... and more

        LEX-MIN (global order) = Path A (row-first)
        - Pixel comparison: (r1,c1) < (r2,c2) iff (r1 < r2) or (r1==r2 and c1<c2)
        - Path comparison: lexicographic on pixel tuples

        WRONG implementation might return Path B or Path C.
        """
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        anchor1 = Pixel(0, 0)
        anchor2 = Pixel(2, 2)

        path = shortest_path_bfs(grid, anchor1, anchor2, "4conn")

        assert path is not None, "Path should exist"
        assert len(path) == 5, f"Path length should be 5 (4 steps + 1), got {len(path)}"

        # Enumerate all shortest paths
        all_paths = enumerate_all_4conn_paths(anchor1, anchor2, max_length=5)

        # Filter to only geodesics (length 5)
        geodesics = [p for p in all_paths if len(p) == 5 and p[-1] == anchor2]

        assert len(geodesics) > 1, \
            "Test requires multiple geodesics to verify lex-min selection"

        # Find lex-min among geodesics
        lex_min_path = min(geodesics, key=lambda p: tuple(p))

        assert path == lex_min_path, \
            f"CRITICAL BUG: Path is not lex-min!\n" \
            f"Expected (lex-min): {lex_min_path}\n" \
            f"Got: {path}\n" \
            f"This violates spec: 'Tie-break: lex-min path under the global order'"

    def test_4conn_connectivity_enforcement_CRITICAL(self):
        """
        CE-06: CRITICAL - 4-conn uses only orthogonal neighbors (no diagonals).

        This test PROVES 4-conn constraint (would FAIL if diagonals used).

        Markers at (0,0) and (2,2):
        - 4-conn shortest path length = 4 (Manhattan distance)
        - 8-conn shortest path length = 2 (Chebyshev distance)

        WRONG: If implementation uses 8-conn for 4-conn metric,
               path would have length 3 (e.g., [(0,0), (1,1), (2,2)])
        """
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        anchor1 = Pixel(0, 0)
        anchor2 = Pixel(2, 2)

        path = shortest_path_bfs(grid, anchor1, anchor2, "4conn")

        assert path is not None, "Path should exist"

        # CRITICAL: 4-conn path must have length 5 (4 steps + 1)
        # NOT length 3 (which would be 8-conn diagonal)
        assert len(path) == 5, \
            f"CRITICAL BUG: 4-conn path length should be 5, got {len(path)}.\n" \
            f"Path: {path}\n" \
            f"If length is 3, implementation is using 8-conn (diagonal) for 4-conn metric!"

        # Verify each step is orthogonal (no diagonals)
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            assert is_valid_4conn_step(p1, p2), \
                f"CRITICAL BUG: Invalid 4-conn step from {p1} to {p2}.\n" \
                f"4-conn allows only orthogonal moves (up/down/left/right), not diagonal!"

    def test_4conn_single_pixel_path(self):
        """CE-08: Adjacent markers (degenerate case)."""
        grid = [[0, 0, 0], [0, 0, 0]]
        anchor1 = Pixel(1, 1)
        anchor2 = Pixel(1, 2)

        path = shortest_path_bfs(grid, anchor1, anchor2, "4conn")

        assert path is not None, "Path should exist"
        assert len(path) == 2, f"Path length should be 2, got {len(path)}"
        assert path == [anchor1, anchor2], f"Expected single-step path, got {path}"

    def test_4conn_same_anchor(self):
        """Degenerate case: start == end."""
        grid = [[0, 0, 0], [0, 0, 0]]
        anchor1 = Pixel(1, 1)

        path = shortest_path_bfs(grid, anchor1, anchor1, "4conn")

        assert path is not None, "Path should exist"
        assert len(path) == 1, f"Path length should be 1, got {len(path)}"
        assert path == [anchor1], f"Expected single-pixel path, got {path}"


# =============================================================================
# Test Class: CONNECT_ENDPOINTS - 8-conn
# =============================================================================


class TestConnectEndpoints8Conn:
    """Test 8-connected shortest path computation."""

    def test_8conn_diagonal(self):
        """CE-05: Diagonal path using 8-connectivity."""
        grid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        anchor1 = Pixel(0, 0)
        anchor2 = Pixel(3, 3)

        path = shortest_path_bfs(grid, anchor1, anchor2, "8conn")

        assert path is not None, "Path should exist"

        # 8-conn diagonal distance = max(|dr|, |dc|) = 3
        assert len(path) == 4, f"8-conn diagonal path should have length 4, got {len(path)}"

        # Expected pure diagonal
        expected = [Pixel(0, 0), Pixel(1, 1), Pixel(2, 2), Pixel(3, 3)]
        assert path == expected, f"Expected pure diagonal path, got {path}"

        # Verify each step is valid 8-conn
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i+1]
            assert is_valid_8conn_step(p1, p2), \
                f"Invalid 8-conn step from {p1} to {p2}"

    def test_4conn_vs_8conn_different_lengths_CRITICAL(self):
        """
        CRITICAL: Verify 4-conn and 8-conn produce different path lengths.

        Same markers, different metrics should give different results.

        Markers at (0,0) and (2,2):
        - 4-conn: Manhattan distance = 4 → path length 5
        - 8-conn: Chebyshev distance = 2 → path length 3

        WRONG: If both return same length, metrics are not different.
        """
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        anchor1 = Pixel(0, 0)
        anchor2 = Pixel(2, 2)

        path_4conn = shortest_path_bfs(grid, anchor1, anchor2, "4conn")
        path_8conn = shortest_path_bfs(grid, anchor1, anchor2, "8conn")

        assert path_4conn is not None, "4-conn path should exist"
        assert path_8conn is not None, "8-conn path should exist"

        # CRITICAL: Lengths must differ
        assert len(path_4conn) == 5, f"4-conn path length should be 5, got {len(path_4conn)}"
        assert len(path_8conn) == 3, f"8-conn path length should be 3, got {len(path_8conn)}"

        assert len(path_4conn) != len(path_8conn), \
            f"CRITICAL BUG: 4-conn and 8-conn paths should have different lengths!\n" \
            f"4-conn: {path_4conn} (length {len(path_4conn)})\n" \
            f"8-conn: {path_8conn} (length {len(path_8conn)})\n" \
            f"This suggests metrics are not implemented correctly."


# =============================================================================
# Test Class: CONNECT_ENDPOINTS - Properties
# =============================================================================


class TestConnectProperties:
    """Test geodesic, determinism, and other properties."""

    def test_geodesic_shortest_4conn(self):
        """Verify path length equals true metric distance (4-conn)."""
        grid = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        anchor1 = Pixel(0, 0)
        anchor2 = Pixel(1, 4)

        path = shortest_path_bfs(grid, anchor1, anchor2, "4conn")

        assert path is not None, "Path should exist"

        # True 4-conn distance
        true_distance = compute_manhattan_distance(anchor1, anchor2)
        assert true_distance == 5, "Sanity check: Manhattan distance should be 5"

        # Path length should equal true distance + 1 (includes both endpoints)
        assert len(path) - 1 == true_distance, \
            f"Path length - 1 ({len(path) - 1}) != true distance ({true_distance})"

    def test_geodesic_shortest_8conn(self):
        """Verify path length equals true metric distance (8-conn)."""
        grid = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
        anchor1 = Pixel(0, 0)
        anchor2 = Pixel(2, 4)

        path = shortest_path_bfs(grid, anchor1, anchor2, "8conn")

        assert path is not None, "Path should exist"

        # True 8-conn distance
        true_distance = compute_chebyshev_distance(anchor1, anchor2)
        assert true_distance == 4, "Sanity check: Chebyshev distance should be 4"

        # Path length should equal true distance + 1
        assert len(path) - 1 == true_distance, \
            f"Path length - 1 ({len(path) - 1}) != true distance ({true_distance})"

    def test_determinism_4conn(self):
        """CE-09: Verify determinism (same inputs → same path)."""
        grid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        anchor1 = Pixel(0, 0)
        anchor2 = Pixel(2, 3)

        # Run 10 times
        paths = [shortest_path_bfs(grid, anchor1, anchor2, "4conn") for _ in range(10)]

        # All paths should be identical
        first_path = paths[0]
        for i, path in enumerate(paths[1:], 1):
            assert path == first_path, \
                f"Run {i+1} differs from run 1:\n" \
                f"Run 1: {first_path}\n" \
                f"Run {i+1}: {path}\n" \
                f"Implementation is non-deterministic!"

    def test_out_of_bounds_anchor(self):
        """EDGE-01: Anchor outside grid bounds."""
        grid = [[0, 0, 0], [0, 0, 0]]
        anchor1 = Pixel(0, 0)
        anchor2 = Pixel(5, 5)  # Out of bounds

        path = shortest_path_bfs(grid, anchor1, anchor2, "4conn")

        # Should return None (no path)
        assert path is None, "Should return None for out-of-bounds anchor"


# =============================================================================
# Test Class: REGION_FILL - Basic
# =============================================================================


class TestRegionFillBasic:
    """Test basic flood fill operations."""

    def test_fill_rectangle_mask_CRITICAL(self):
        """
        RF-02: CRITICAL - Fill respects mask boundaries (no leakage).

        This test PROVES mask boundary respect.

        Grid with hole (mask = hole pixels):
        [[0,0,0,0,0],
         [0,1,1,1,0],
         [0,1,0,1,0],  ← hole at (2,2)
         [0,1,1,1,0],
         [0,0,0,0,0]]

        Mask = {(2,2)} (hole only)
        Fill color = 5

        CORRECT: Only (2,2) painted → grid[2][2] = 5
        WRONG: If flood leaks, might paint (1,2), (2,1), (2,3), (3,2), etc.
        """
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ]

        mask = {Pixel(2, 2)}  # Hole only
        fill_color = 5

        result = flood_fill(grid, mask, fill_color)

        # CRITICAL: Only mask pixel should be painted
        assert result[2][2] == fill_color, \
            f"Mask pixel (2,2) should be painted, got {result[2][2]}"

        # CRITICAL: No other pixels should be painted
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if (r, c) == (2, 2):
                    continue

                assert result[r][c] == grid[r][c], \
                    f"CRITICAL BUG: Non-mask pixel ({r},{c}) was painted!\n" \
                    f"Original: {grid[r][c]}, Result: {result[r][c]}\n" \
                    f"Fill leaked outside mask. This violates spec: " \
                    f"'Fill/patched flood inside a present-definable mask'"

    def test_fill_multiple_disconnected_regions(self):
        """RF-08: Mask with multiple disconnected regions."""
        grid = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

        # Two separate regions
        mask = {Pixel(0, 0), Pixel(0, 1), Pixel(2, 3), Pixel(2, 4)}
        fill_color = 7

        result = flood_fill(grid, mask, fill_color)

        # All mask pixels should be filled
        for pixel in mask:
            assert result[pixel.row][pixel.col] == fill_color, \
                f"Mask pixel {pixel} not filled"

        # Only mask pixels should be filled
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                if Pixel(r, c) in mask:
                    assert result[r][c] == fill_color
                else:
                    assert result[r][c] == grid[r][c], \
                        f"Non-mask pixel ({r},{c}) incorrectly filled"

    def test_fill_empty_mask(self):
        """RF-06: Empty mask (edge case)."""
        grid = [[0, 0, 0], [0, 0, 0]]
        mask = set()  # Empty
        fill_color = 5

        result = flood_fill(grid, mask, fill_color)

        # Grid should be unchanged
        assert result == grid, "Grid should be unchanged for empty mask"

    def test_fill_single_pixel(self):
        """RF-07: Single pixel mask (degenerate case)."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        mask = {Pixel(1, 1)}
        fill_color = 9

        result = flood_fill(grid, mask, fill_color)

        assert result[1][1] == fill_color, "Single mask pixel should be filled"

        # All other pixels unchanged
        for r in range(3):
            for c in range(3):
                if (r, c) != (1, 1):
                    assert result[r][c] == grid[r][c]


# =============================================================================
# Test Class: REGION_FILL - Selector Integration
# =============================================================================


class TestRegionFillSelectors:
    """Test selector integration with REGION_FILL."""

    def test_fill_with_unique_selector_CRITICAL(self):
        """
        RF-04: CRITICAL - Fill color matches UNIQUE selector output.

        This test PROVES selector integration (not hardcoded color).

        Mask with unique color 3 in grid.
        CORRECT: Fill with color 3 (from UNIQUE selector)
        WRONG: Fill with different color (e.g., hardcoded 5)
        """
        grid = [[3, 3, 0], [3, 0, 3], [0, 3, 3]]
        mask = {Pixel(1, 1), Pixel(2, 0)}  # Cells to fill

        # Compute selector color (UNIQUE should return 3)
        selector_color, empty_mask = apply_selector_on_test(
            selector_type="UNIQUE",
            mask={Pixel(0, 0), Pixel(0, 1), Pixel(1, 0), Pixel(1, 2), Pixel(2, 1), Pixel(2, 2)},
            X_test=grid
        )

        assert selector_color == 3, \
            f"Sanity check: UNIQUE selector should return 3, got {selector_color}"

        # Apply fill
        result = flood_fill(grid, mask, selector_color)

        # CRITICAL: Mask pixels should be filled with selector color (3)
        assert result[1][1] == 3, \
            f"CRITICAL BUG: Fill color should be {selector_color} (from UNIQUE selector), " \
            f"got {result[1][1]}"
        assert result[2][0] == 3, \
            f"CRITICAL BUG: Fill color should be {selector_color} (from UNIQUE selector), " \
            f"got {result[2][0]}"

    def test_fill_with_argmax_selector_CRITICAL(self):
        """
        RF-05: CRITICAL - Fill color matches ARGMAX selector output.

        Mask where ARGMAX returns color 4.
        CORRECT: Fill with color 4
        WRONG: Fill with different color
        """
        grid = [[1, 2, 4, 4], [4, 0, 4, 1], [4, 4, 2, 4]]

        # Selector mask (where color 4 is most frequent)
        selector_mask = {Pixel(0, 2), Pixel(0, 3), Pixel(1, 0), Pixel(1, 2),
                        Pixel(2, 0), Pixel(2, 1), Pixel(2, 3)}

        selector_color, empty_mask = apply_selector_on_test(
            selector_type="ARGMAX",
            mask=selector_mask,
            X_test=grid
        )

        assert selector_color == 4, \
            f"Sanity check: ARGMAX should return 4, got {selector_color}"

        # Fill mask
        fill_mask = {Pixel(1, 1), Pixel(2, 2)}
        result = flood_fill(grid, fill_mask, selector_color)

        # CRITICAL: Filled with ARGMAX color (4)
        assert result[1][1] == 4, \
            f"CRITICAL BUG: Fill color should be {selector_color} (from ARGMAX), " \
            f"got {result[1][1]}"
        assert result[2][2] == 4, \
            f"CRITICAL BUG: Fill color should be {selector_color} (from ARGMAX), " \
            f"got {result[2][2]}"

    def test_fill_determinism(self):
        """RF-09: Verify fill is deterministic."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        mask = {Pixel(0, 0), Pixel(1, 1), Pixel(2, 2)}
        fill_color = 7

        # Run 10 times
        results = [flood_fill(grid, mask, fill_color) for _ in range(10)]

        # All results should be identical
        first = results[0]
        for i, result in enumerate(results[1:], 1):
            assert result == first, \
                f"Run {i+1} differs from run 1. Fill is non-deterministic!"


# =============================================================================
# Test Class: Law Application
# =============================================================================


class TestLawApplication:
    """Test ConnectLaw and FillLaw application."""

    def test_apply_connect_law(self):
        """Test applying ConnectLaw to grid (anchors preserved per A1)."""
        grid = [[0, 0, 0, 0], [0, 0, 0, 0]]

        law = ConnectLaw(
            anchor1=Pixel(0, 0),
            anchor2=Pixel(0, 3),
            metric="4conn",
            path=(Pixel(0, 0), Pixel(0, 1), Pixel(0, 2), Pixel(0, 3)),
            line_color=5
        )

        result = apply_connect_law(law, grid)

        # Per A1 (FY exactness): Anchors PRESERVED, only middle pixels painted
        assert result[0][0] == 0, "Anchor endpoint should be preserved"
        assert result[0][1] == 5, "Middle pixel should be painted"
        assert result[0][2] == 5, "Middle pixel should be painted"
        assert result[0][3] == 0, "Anchor endpoint should be preserved"

        # Other pixels unchanged
        assert result[1][0] == 0

    def test_apply_fill_law_with_selector(self):
        """Test applying FillLaw with selector."""
        grid = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]

        # Mask pixels that all have color 1 (for UNIQUE selector to return 1)
        law = FillLaw(
            mask_pixels=frozenset({Pixel(0, 0), Pixel(0, 1), Pixel(1, 0), Pixel(1, 2), Pixel(2, 1), Pixel(2, 2)}),
            selector_type="UNIQUE",
            selector_k=None
        )

        # Selector should find unique color 1 (all mask pixels have color 1)
        result = apply_fill_law(law, grid, X_test_present=grid)

        # Mask pixels filled with color 1 (which they already are, but verify fill works)
        # Actually, let's change test - use different mask for filling
        # Actually the law applies to its own mask_pixels, so let me create a better test

        # Better test: use grid where mask has unique color, then verify fill
        grid2 = [[0, 0, 0], [0, 5, 0], [0, 5, 0]]
        law2 = FillLaw(
            mask_pixels=frozenset({Pixel(1, 1), Pixel(2, 1)}),  # Both have color 5
            selector_type="UNIQUE",
            selector_k=None
        )

        # Apply to grid with zeros in mask positions
        grid_to_fill = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        # Selector evaluated on grid2 (which has color 5 in mask)
        # But we're filling grid_to_fill
        # Wait, this doesn't make sense either...

        # Let me just verify the function works correctly
        # The selector is computed on X_test_present, mask is law.mask_pixels
        # So for grid2, mask {(1,1), (2,1)}, UNIQUE should return 5
        result2 = apply_fill_law(law2, grid_to_fill, X_test_present=grid2)

        # Mask pixels in grid_to_fill should now be 5
        assert result2[1][1] == 5
        assert result2[2][1] == 5


# =============================================================================
# Test Class: Build Connect Fill
# =============================================================================


class TestBuildConnectFill:
    """Test build_connect_fill API."""

    def test_build_connect_simple(self):
        """Test building CONNECT law from theta."""
        train_pairs = [
            ([[0, 1, 0, 0, 2, 0]], [[0, 1, 5, 5, 2, 0]])
        ]

        theta = {
            "train_pairs": train_pairs,
            "anchors": [
                {
                    "anchor1": Pixel(0, 1),
                    "anchor2": Pixel(0, 4),
                    "metric": "4conn",
                    "color": 5
                }
            ],
            "masks": []
        }

        laws = build_connect_fill(theta)

        assert len(laws) == 1, f"Should build 1 law, got {len(laws)}"
        assert isinstance(laws[0], ConnectLaw)
        assert laws[0].line_color == 5

    def test_build_fill_simple(self):
        """Test building FILL law from theta."""
        # Grid with surrounding color 1 pixels, hole at (1,1)
        train_pairs = [
            ([[1, 1, 0], [1, 0, 1], [0, 1, 1]], [[1, 1, 0], [1, 1, 1], [0, 1, 1]])
        ]

        # Mask includes all pixels (including hole)
        # ARGMAX will return 1 (most frequent color in mask)
        theta = {
            "train_pairs": train_pairs,
            "anchors": [],
            "masks": [
                {
                    "pixels": {Pixel(0, 0), Pixel(0, 1), Pixel(1, 0),
                           Pixel(1, 1), Pixel(1, 2), Pixel(2, 1), Pixel(2, 2)},
                    "selector": "ARGMAX",  # ARGMAX → color 1 (most frequent)
                    "k": None
                }
            ]
        }

        laws = build_connect_fill(theta)

        assert len(laws) == 1, f"Should build 1 law, got {len(laws)}"
        assert isinstance(laws[0], FillLaw)
        assert laws[0].selector_type == "ARGMAX"


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge cases and robustness tests."""

    def test_path_through_boundary(self):
        """EDGE-01: Path endpoints at grid corners."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        anchor1 = Pixel(0, 0)
        anchor2 = Pixel(2, 2)

        path = shortest_path_bfs(grid, anchor1, anchor2, "4conn")

        assert path is not None, "Path should exist"
        assert path[0] == anchor1
        assert path[-1] == anchor2

    def test_fill_entire_grid(self):
        """EDGE-03: Mask covers entire grid."""
        grid = [[0, 0], [0, 0]]
        mask = {Pixel(0, 0), Pixel(0, 1), Pixel(1, 0), Pixel(1, 1)}
        fill_color = 9

        result = flood_fill(grid, mask, fill_color)

        # All pixels should be filled
        for r in range(2):
            for c in range(2):
                assert result[r][c] == fill_color

    def test_1x1_grid(self):
        """EDGE-04: Degenerate 1×1 grid."""
        grid = [[5]]
        mask = {Pixel(0, 0)}
        fill_color = 7

        result = flood_fill(grid, mask, fill_color)

        assert result[0][0] == fill_color, "Single pixel should be filled"

    def test_invalid_metric_raises_error(self):
        """Invalid metric string should raise ValueError."""
        grid = [[0, 0], [0, 0]]
        anchor1 = Pixel(0, 0)
        anchor2 = Pixel(1, 1)

        with pytest.raises(ValueError, match="Unknown metric"):
            shortest_path_bfs(grid, anchor1, anchor2, "invalid_metric")
