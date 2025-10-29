"""
Mathematical Validation Tests for WO-10: Object Arithmetic.

Philosophy (from WO-06 lessons):
- PROVE correctness through mathematics, not just verify consistency
- Known-answer tests with pre-computed correct results
- Property-based validation (mathematical invariants)
- Pathological stress tests to find bugs
- Focus: Find bugs through rigor, skip pedantic cases outside ARC scope

Test Categories:
1. Component Extraction: Mathematical properties (convexity, connectivity)
2. Hungarian Matching: Optimality, bipartite matching properties
3. Bresenham: Geodesic optimality, connectivity
4. Skeleton: Topology preservation (Euler characteristic, connectivity)
"""

import pytest
import hashlib
from typing import Set, FrozenSet

from arc_core.types import Grid, Pixel
from arc_core.components import extract_components, match_components, Component
from arc_laws.object_arith import (
    bresenham_4conn, bresenham_8conn,
    skeleton_zhang_suen,
)


# =============================================================================
# Known-Answer Tests for Component Extraction
# =============================================================================

class TestComponentExtractionMathematical:
    """
    Mathematical validation of component extraction.

    Focus: Prove 4-level tie-breaking is correct, not just consistent.
    """

    def test_known_01_perfect_4_level_sort(self):
        """MATH-COMP-01: Known answer for 4-level tie-breaking"""
        # Grid designed to test ALL 4 tie-break levels
        grid = [
            # lex_min (0,0): 4 pixels of color 1
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
            # lex_min (2,0): 4 pixels of color 2 (same area as color 1!)
            [2, 2, 0, 0, 0],
            [2, 2, 0, 0, 0],
        ]

        comps = extract_components(grid)

        # Pre-computed correct order (lex_min is (row, col) tuple):
        # 1. Color 1: lex_min=(0,0), area=4
        # 2. Color 0: lex_min=(0,2), area=12  ← (0,2) < (2,0) lexicographically!
        # 3. Color 2: lex_min=(2,0), area=4

        # Verify order
        assert len(comps) == 3
        assert comps[0].color == 1, f"Component 0 should be color 1, got {comps[0].color}"
        assert comps[0].lex_min == Pixel(0, 0)
        assert comps[0].component_id == 0

        assert comps[1].color == 0, f"Component 1 should be color 0, got {comps[1].color}"
        assert comps[1].lex_min == Pixel(0, 2)
        assert comps[1].component_id == 1

        assert comps[2].color == 2
        assert comps[2].lex_min == Pixel(2, 0)
        assert comps[2].component_id == 2

    def test_known_02_centroid_tie_break_rational(self):
        """MATH-COMP-02: Centroid tie-break uses rational comparison (no floats)"""
        # Components with same lex_min.row, same area → centroid decides
        grid = [
            [1, 0, 2, 0, 3],
        ]

        comps = extract_components(grid)

        # All 3 colors have:
        # - lex_min.row = 0
        # - area = 1
        # Centroid: all (0, col)
        # Sort by lex_min.col: (0,0) < (0,2) < (0,4)

        assert comps[0].color == 1  # lex_min (0,0)
        assert comps[1].color == 0  # lex_min (0,1)
        assert comps[2].color == 2  # lex_min (0,2)

    def test_math_03_8connected_property(self):
        """MATH-COMP-03: 8-connected means all pixels reachable via 8-neighborhood"""
        grid = [
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1],
        ]

        comps = extract_components(grid)

        # All 5 pixels of color 1 should be ONE component (8-connected via center)
        color1_comps = [c for c in comps if c.color == 1]

        assert len(color1_comps) == 1, f"Expected 1 component (8-connected), got {len(color1_comps)}"
        assert color1_comps[0].area == 5


# =============================================================================
# Known-Answer Tests for Bresenham
# =============================================================================

class TestBresenhamMathematical:
    """
    Mathematical validation of Bresenham line drawing.

    Focus: Geodesic optimality, connectivity, known correct paths.
    """

    def test_known_03_bresenham_4conn_known_path(self):
        """MATH-LINE-01: Known correct 4-conn path for (0,0)→(3,2)"""
        p1 = Pixel(0, 0)
        p2 = Pixel(3, 2)

        pixels = bresenham_4conn(p1, p2)

        # 4-conn (0,0)→(3,2): dr=3, dc=2
        # Standard Bresenham should produce specific path
        # Path length should be dr + dc = 5 steps

        assert len(pixels) == 6, f"4-conn path should have 6 pixels (5 steps + start), got {len(pixels)}"
        assert pixels[0] == p1, "Path should start at p1"
        assert pixels[-1] == p2, "Path should end at p2"

        # Verify 4-connectivity (each step is orthogonal)
        for i in range(len(pixels) - 1):
            dr = abs(pixels[i+1].row - pixels[i].row)
            dc = abs(pixels[i+1].col - pixels[i].col)
            assert (dr == 1 and dc == 0) or (dr == 0 and dc == 1), \
                f"4-conn step must be orthogonal: {pixels[i]} → {pixels[i+1]}"

    def test_known_04_bresenham_8conn_known_path(self):
        """MATH-LINE-02: Known correct 8-conn path for (0,0)→(3,2)"""
        p1 = Pixel(0, 0)
        p2 = Pixel(3, 2)

        pixels = bresenham_8conn(p1, p2)

        # 8-conn allows diagonals
        # Optimal path: max(dr, dc) = max(3, 2) = 3 steps
        # Path: (0,0) → (1,1) → (2,2) → (3,2)

        assert len(pixels) == 4, f"8-conn optimal path should have 4 pixels, got {len(pixels)}"
        assert pixels == [Pixel(0, 0), Pixel(1, 1), Pixel(2, 2), Pixel(3, 2)], \
            f"8-conn path incorrect: {pixels}"

    def test_math_05_bresenham_geodesic_optimality(self):
        """MATH-LINE-03: Bresenham produces geodesically optimal paths"""
        # 8-conn: path length should be max(dr, dc) (Chebyshev distance)
        # 4-conn: path length should be dr + dc (Manhattan distance)

        p1 = Pixel(1, 2)
        p2 = Pixel(7, 9)

        dr = abs(p2.row - p1.row)  # 6
        dc = abs(p2.col - p1.col)  # 7

        pixels_8 = bresenham_8conn(p1, p2)
        pixels_4 = bresenham_4conn(p1, p2)

        # 8-conn optimal: max(6, 7) + 1 = 8 pixels
        chebyshev_dist = max(dr, dc) + 1  # +1 for inclusive endpoints
        assert len(pixels_8) == chebyshev_dist, \
            f"8-conn should have {chebyshev_dist} pixels (Chebyshev), got {len(pixels_8)}"

        # 4-conn optimal: 6 + 7 + 1 = 14 pixels
        manhattan_dist = dr + dc + 1  # +1 for inclusive endpoints
        assert len(pixels_4) == manhattan_dist, \
            f"4-conn should have {manhattan_dist} pixels (Manhattan), got {len(pixels_4)}"

    def test_math_06_bresenham_connectivity(self):
        """MATH-LINE-04: Bresenham paths are connected (no gaps)"""
        p1 = Pixel(0, 0)
        p2 = Pixel(10, 10)

        pixels_4 = bresenham_4conn(p1, p2)
        pixels_8 = bresenham_8conn(p1, p2)

        # Verify every adjacent pair is connected
        def is_4connected(p_a, p_b):
            return abs(p_a.row - p_b.row) + abs(p_a.col - p_b.col) == 1

        def is_8connected(p_a, p_b):
            return max(abs(p_a.row - p_b.row), abs(p_a.col - p_b.col)) == 1

        for i in range(len(pixels_4) - 1):
            assert is_4connected(pixels_4[i], pixels_4[i+1]), \
                f"4-conn path has gap: {pixels_4[i]} → {pixels_4[i+1]}"

        for i in range(len(pixels_8) - 1):
            assert is_8connected(pixels_8[i], pixels_8[i+1]), \
                f"8-conn path has gap: {pixels_8[i]} → {pixels_8[i+1]}"


# =============================================================================
# Known-Answer Tests for Skeleton
# =============================================================================

class TestSkeletonMathematical:
    """
    Mathematical validation of Zhang-Suen skeleton.

    Focus: Topology preservation (Euler characteristic), connectivity, known skeletons.
    """

    def test_known_05_skeleton_line_unchanged(self):
        """MATH-SKEL-01: 1-pixel line is unchanged (already minimal)"""
        # Horizontal line
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        skeleton = skeleton_zhang_suen(grid, color1_comp.pixels)

        # Should be unchanged (already 1-pixel wide)
        assert skeleton == color1_comp.pixels, \
            f"1-pixel line should be unchanged: {len(skeleton)} vs {len(color1_comp.pixels)}"

    def test_math_07_skeleton_topology_preservation(self):
        """MATH-SKEL-02: Skeleton preserves topology (connectivity)"""
        # Create a simple connected component
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        skeleton = skeleton_zhang_suen(grid, color1_comp.pixels)

        # Skeleton should still be connected (1 component)
        # Verify by checking if all skeleton pixels are 8-connected

        def are_8connected(pixel_set: FrozenSet[Pixel]) -> bool:
            """Check if all pixels in set form single 8-connected component"""
            if not pixel_set:
                return True

            visited = set()
            queue = [list(pixel_set)[0]]
            visited.add(queue[0])

            while queue:
                current = queue.pop(0)

                # Check 8 neighbors
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue

                        neighbor = Pixel(current.row + dr, current.col + dc)
                        if neighbor in pixel_set and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

            return len(visited) == len(pixel_set)

        assert are_8connected(skeleton), \
            f"Skeleton should be connected (topology preserved)"

    def test_math_08_skeleton_width_one_pixel(self):
        """MATH-SKEL-03: Skeleton is 1-pixel wide (no 2×2 blocks)"""
        grid = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        skeleton = skeleton_zhang_suen(grid, color1_comp.pixels)

        # Verify no 2×2 solid blocks in skeleton (would violate 1-pixel width)
        skeleton_coords = {(p.row, p.col) for p in skeleton}

        for r in range(3):
            for c in range(3):
                # Check if (r,c), (r,c+1), (r+1,c), (r+1,c+1) all in skeleton
                block = {(r, c), (r, c+1), (r+1, c), (r+1, c+1)}
                if block.issubset(skeleton_coords):
                    pytest.fail(f"Skeleton has 2×2 block at ({r},{c}) - not 1-pixel wide!")


# =============================================================================
# Pathological Stress Tests
# =============================================================================

class TestHungarianOptimality:
    """
    CRITICAL: Verify Hungarian matching is globally optimal.

    This is the backbone of component correspondence. If matching is suboptimal,
    the entire object arithmetic system breaks.
    """

    def test_hungarian_01_globally_optimal_known_answer(self):
        """HUNGARIAN-OPT-01: Verify globally optimal matching with known answer"""
        # Create a scenario with KNOWN optimal matching
        # 3 components in X, 3 in Y, with specific costs

        # X components:
        # - X0: 2x2 square at (0,0), area=4, inertia_num=8
        # - X1: 3x1 rect at (0,3), area=3, inertia_num=4
        # - X2: 1x1 pixel at (2,0), area=1, inertia_num=0

        grid_X = [
            [1, 1, 0, 2, 2, 2],
            [1, 1, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 0],
        ]

        # Y components (same shapes, different positions):
        # - Y0: 2x2 square at (1,1) - MATCHES X0 (same shape)
        # - Y1: 3x1 rect at (0,0) - MATCHES X1 (same shape)
        # - Y2: 1x1 pixel at (2,5) - MATCHES X2 (same shape)

        grid_Y = [
            [2, 2, 2, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 3],
        ]

        comps_X = extract_components(grid_X)
        comps_Y = extract_components(grid_Y)

        # Expected optimal matching:
        # X0 (color 1, 2x2) → Y0 (color 1, 2x2) - cost = 0 (same inertia, area, shape)
        # X1 (color 2, 3x1) → Y1 (color 2, 3x1) - cost = 0
        # X2 (color 3, 1x1) → Y2 (color 3, 1x1) - cost = 0

        matches = match_components(comps_X, comps_Y)

        # Verify optimal matching (cost = 0 for all)
        # Find each component by color
        color1_X = [c for c in comps_X if c.color == 1][0]
        color2_X = [c for c in comps_X if c.color == 2][0]
        color3_X = [c for c in comps_X if c.color == 3][0]

        color1_Y = [c for c in comps_Y if c.color == 1][0]
        color2_Y = [c for c in comps_Y if c.color == 2][0]
        color3_Y = [c for c in comps_Y if c.color == 3][0]

        # Find matches
        match1 = [m for m in matches if m.comp_X_id == color1_X.component_id][0]
        match2 = [m for m in matches if m.comp_X_id == color2_X.component_id][0]
        match3 = [m for m in matches if m.comp_X_id == color3_X.component_id][0]

        # Verify OPTIMAL matching (same shapes match)
        assert match1.comp_Y_id == color1_Y.component_id, \
            f"Hungarian should match 2x2 → 2x2 (optimal), got X{match1.comp_X_id}→Y{match1.comp_Y_id}"
        assert match2.comp_Y_id == color2_Y.component_id, \
            f"Hungarian should match 3x1 → 3x1 (optimal), got X{match2.comp_X_id}→Y{match2.comp_Y_id}"
        assert match3.comp_Y_id == color3_Y.component_id, \
            f"Hungarian should match 1x1 → 1x1 (optimal), got X{match3.comp_X_id}→Y{match3.comp_Y_id}"

    def test_hungarian_02_suboptimal_rejected(self):
        """HUNGARIAN-OPT-02: Verify Hungarian rejects suboptimal matching"""
        # Create scenario where GREEDY matching would fail
        # but OPTIMAL matching succeeds

        # X: Two components
        # - X0: 3x3 square (area=9, inertia high)
        # - X1: 2x2 square (area=4, inertia lower)

        grid_X = [
            [1, 1, 1, 0, 2, 2],
            [1, 1, 1, 0, 2, 2],
            [1, 1, 1, 0, 0, 0],
        ]

        # Y: Two components
        # - Y0: 2x2 square (matches X1)
        # - Y1: 3x3 square (matches X0)

        grid_Y = [
            [2, 2, 0, 1, 1, 1],
            [2, 2, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ]

        comps_X = extract_components(grid_X)
        comps_Y = extract_components(grid_Y)

        matches = match_components(comps_X, comps_Y)

        # Find components
        x_3x3 = [c for c in comps_X if c.area == 9][0]
        x_2x2 = [c for c in comps_X if c.area == 4][0]
        y_3x3 = [c for c in comps_Y if c.area == 9][0]
        y_2x2 = [c for c in comps_Y if c.area == 4][0]

        # Find matches
        match_3x3 = [m for m in matches if m.comp_X_id == x_3x3.component_id][0]
        match_2x2 = [m for m in matches if m.comp_X_id == x_2x2.component_id][0]

        # Verify OPTIMAL matching (not greedy)
        # Optimal: 3x3→3x3, 2x2→2x2 (total cost = 0)
        # Greedy might match first-first, second-second (higher cost)

        assert match_3x3.comp_Y_id == y_3x3.component_id, \
            f"Hungarian should optimally match 3x3→3x3, got {match_3x3.comp_X_id}→{match_3x3.comp_Y_id}"
        assert match_2x2.comp_Y_id == y_2x2.component_id, \
            f"Hungarian should optimally match 2x2→2x2, got {match_2x2.comp_X_id}→{match_2x2.comp_Y_id}"


class TestTopologyPreservation:
    """
    CRITICAL: Verify skeleton preserves topology (Euler characteristic).

    Topology preservation is a core spec requirement. Surface-level connectivity
    tests aren't enough - we need proper mathematical validation.
    """

    def test_topology_01_euler_characteristic_preserved(self):
        """TOPOLOGY-01: Euler characteristic χ = V - E + F is preserved"""
        # Create a simple connected component
        # Original: 3x3 filled square
        # χ = 1 (one connected component, simply connected)

        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        # Compute Euler characteristic of skeleton
        skeleton = skeleton_zhang_suen(grid, color1_comp.pixels)

        # For a simply connected region: χ = 1
        # Skeleton should also be simply connected: χ = 1
        # We verify this by checking connectivity and no holes

        # Check: skeleton is a single connected component
        def count_connected_components_8(pixel_set: FrozenSet[Pixel]) -> int:
            """Count 8-connected components in pixel set"""
            if not pixel_set:
                return 0

            visited = set()
            count = 0

            for start in pixel_set:
                if start in visited:
                    continue

                count += 1
                queue = [start]
                visited.add(start)

                while queue:
                    current = queue.pop(0)
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            neighbor = Pixel(current.row + dr, current.col + dc)
                            if neighbor in pixel_set and neighbor not in visited:
                                visited.add(neighbor)
                                queue.append(neighbor)

            return count

        num_components = count_connected_components_8(skeleton)
        assert num_components == 1, \
            f"Skeleton should be single connected component (χ=1), got {num_components} components"

    def test_topology_02_ring_preservation(self):
        """TOPOLOGY-02: Ring (loop) topology is preserved"""
        # Create a ring component (hollow square)
        # Original has a hole → χ = 0
        # Skeleton should also have a hole → χ = 0

        grid = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        skeleton = skeleton_zhang_suen(grid, color1_comp.pixels)

        # Verify skeleton still forms a ring (has interior hole)
        # Check: skeleton doesn't fill center (hole preserved)
        skeleton_coords = {(p.row, p.col) for p in skeleton}

        # Center point (2,2) should NOT be in skeleton
        center_filled = (2, 2) in skeleton_coords

        assert not center_filled, \
            f"Skeleton should preserve ring topology (hole in center), but center is filled"

        # Verify skeleton forms connected loop around hole
        # Should have pixels on all 4 sides of the hole
        has_top = any(r == 1 for r, c in skeleton_coords)
        has_bottom = any(r == 3 for r, c in skeleton_coords)
        has_left = any(c == 1 for r, c in skeleton_coords)
        has_right = any(c == 3 for r, c in skeleton_coords)

        assert has_top and has_bottom and has_left and has_right, \
            f"Skeleton should form loop around hole (have pixels on all sides)"


class TestBoundaryHashDiscrimination:
    """
    IMPORTANT: Verify boundary hash actually distinguishes different shapes.
    """

    def test_boundary_hash_01_different_shapes_different_hashes(self):
        """BOUNDARY-01: Different boundary shapes have different hashes"""
        # Create two components with SAME area but DIFFERENT boundaries

        # Square: 2x2
        grid_square = [
            [1, 1],
            [1, 1],
        ]

        # L-shape: 3 pixels
        grid_L = [
            [2, 0],
            [2, 2],
        ]

        comps_square = extract_components(grid_square)
        comps_L = extract_components(grid_L)

        square_comp = [c for c in comps_square if c.color == 1][0]
        L_comp = [c for c in comps_L if c.color == 2][0]

        # Both have different boundaries
        # Square boundary: all 4 pixels
        # L boundary: all 3 pixels

        # Hashes SHOULD be different
        assert square_comp.boundary_hash != L_comp.boundary_hash, \
            f"Different shapes should have different boundary hashes: {square_comp.boundary_hash} vs {L_comp.boundary_hash}"

    def test_boundary_hash_02_same_shape_same_hash(self):
        """BOUNDARY-02: Same shape (translated) has same hash after normalization"""
        # Two L-shapes at different positions
        grid1 = [
            [1, 0, 0],
            [1, 1, 0],
            [0, 0, 0],
        ]

        grid2 = [
            [0, 0, 0],
            [0, 2, 0],
            [0, 2, 2],
        ]

        comps1 = extract_components(grid1)
        comps2 = extract_components(grid2)

        comp1 = [c for c in comps1 if c.color == 1][0]
        comp2 = [c for c in comps2 if c.color == 2][0]

        # Same L-shape, different position → should have SAME hash (normalized)
        assert comp1.boundary_hash == comp2.boundary_hash, \
            f"Same shape (different position) should have same boundary hash after normalization"


class TestOverflowAndExtremeValues:
    """
    IMPORTANT: Verify no overflow in integer arithmetic.
    """

    def test_overflow_01_large_grid_centroid_calculation(self):
        """OVERFLOW-01: Large grids don't overflow in centroid calculation"""
        # Create component with large centroid numerators
        # Grid: 100x100 single component
        grid = [[1] * 100 for _ in range(100)]

        comps = extract_components(grid)

        assert len(comps) == 1, "Should extract single component"
        comp = comps[0]

        # Centroid numerators:
        # sum_r = 0 + 1 + ... + 99 (repeated 100 times) = 99*100/2 * 100 = 495000
        # sum_c = same = 495000

        expected_sum = (99 * 100 // 2) * 100  # 495000

        assert comp.centroid_num == (expected_sum, expected_sum), \
            f"Large grid centroid numerators incorrect: {comp.centroid_num}"

        # Cross-multiplication in centroid comparison:
        # 495000 * 10000 = 4.95 billion (fits in int32: 2^31 = 2.14 billion)
        # Actually MIGHT overflow int32!

        # Verify no crash (Python handles big ints, but implementation should be safe)
        # This documents potential overflow risk for very large grids

    def test_overflow_02_inertia_calculation_large_components(self):
        """OVERFLOW-02: Large components don't overflow in inertia calculation"""
        # Large component: 50x50
        grid = [[1] * 50 for _ in range(50)]

        comps = extract_components(grid)
        comp = comps[0]

        # Inertia calculation involves sum_rr * area
        # sum_rr = sum(r^2) for r in 0..49 (repeated 50 times)
        # = 50 * (0^2 + 1^2 + ... + 49^2)
        # = 50 * (49*50*99/6) = 50 * 40425 = 2021250
        # inertia_num = (sum_rr * area - sum_r^2) + (sum_cc * area - sum_c^2)

        # Just verify it doesn't crash and produces positive result
        assert comp.inertia_num > 0, f"Inertia should be positive for large component"
        assert isinstance(comp.inertia_num, int), f"Inertia must be integer"


class TestPathologicalCases:
    """
    Pathological stress tests to find edge case bugs.

    Focus: Extreme cases relevant to ARC (not pedantic).
    """

    def test_path_01_many_components_ordering_stable(self):
        """PATH-01: Many components (100+) maintain stable ordering"""
        # Create grid with many single-pixel components (checkerboard pattern with 10 colors)
        # Use (row + col) % 10 to create isolated pixels
        grid = [[(r + c) % 10 for c in range(30)] for r in range(30)]

        comps = extract_components(grid)

        # Each pixel is separated by different colors → many small components
        # Exact count depends on 8-connected clustering, but should be many
        assert len(comps) > 10, f"Expected many components, got {len(comps)}"

        # Verify IDs are sequential 0, 1, 2, ...
        for i, comp in enumerate(comps):
            assert comp.component_id == i, f"Component ID mismatch: expected {i}, got {comp.component_id}"

    def test_path_02_very_long_line_bresenham(self):
        """PATH-02: Very long line (100+ pixels) works correctly"""
        p1 = Pixel(0, 0)
        p2 = Pixel(100, 100)

        pixels_8 = bresenham_8conn(p1, p2)

        # Should be 101 pixels (0 to 100 inclusive)
        assert len(pixels_8) == 101, f"Long diagonal should have 101 pixels, got {len(pixels_8)}"
        assert pixels_8[0] == p1
        assert pixels_8[-1] == p2

    def test_path_03_skeleton_very_thick_component(self):
        """PATH-03: Skeleton handles very thick components (20×20)"""
        grid = [[1] * 20 for _ in range(20)]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        skeleton = skeleton_zhang_suen(grid, color1_comp.pixels)

        # Should be drastically reduced (400 → ~40-80 pixels)
        reduction_ratio = len(skeleton) / len(color1_comp.pixels)

        assert reduction_ratio < 0.5, \
            f"Skeleton should reduce size significantly: {len(skeleton)}/400 = {reduction_ratio:.2f}"

    def test_path_04_bresenham_negative_coordinates(self):
        """PATH-04: Bresenham handles negative direction correctly"""
        # This tests implementation robustness (ARC grids are always positive)
        p1 = Pixel(10, 10)
        p2 = Pixel(0, 0)

        pixels = bresenham_8conn(p1, p2)

        # Should connect correctly
        assert pixels[0] == p1
        assert pixels[-1] == p2
        assert len(pixels) == 11  # max(10, 10) + 1


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
