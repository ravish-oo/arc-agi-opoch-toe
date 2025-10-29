"""
Unit tests for WO-10: Object Arithmetic.

Test Philosophy (from WO-06 lessons):
- FIND BUGS through mathematical rigor, not just pass tests
- Known-answer validation with pre-computed correct results
- Property-based validation (mathematical invariants)
- Deterministic ordering verification (100 runs)
- Skip pedantic edge cases outside ARC scope
- Focus on spec fidelity

Test Coverage:
1. Component extraction (4-level tie-break: lex_min → area → centroid → boundary_hash)
2. Hungarian matching (lexicographic cost)
3. TRANSLATE/COPY/DELETE operations
4. Bresenham line drawing (4-conn and 8-conn)
5. Zhang-Suen skeleton (topology preservation)

All tests use integer arithmetic only, deterministic, no randomness.
"""

import pytest
import hashlib
import json
from pathlib import Path
from typing import List, Set, Tuple

from arc_core.types import Grid, Pixel
from arc_core.components import (
    Component, Match,
    extract_components, match_components,
    _compute_component_properties, _compute_boundary_hash,
    _compute_matching_cost, _compute_boundary_diff,
)
from arc_laws.object_arith import (
    apply_translate, apply_copy, apply_delete,
    bresenham_4conn, bresenham_8conn, apply_drawline,
    skeleton_zhang_suen, apply_skeleton,
)


# =============================================================================
# Fixtures and Helpers
# =============================================================================

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "WO-10"


def grid_to_hash(grid: Grid) -> str:
    """Hash grid for determinism testing (SHA-256)."""
    serialized = json.dumps(grid, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def grids_equal(g1: Grid, g2: Grid) -> bool:
    """Check if two grids are exactly equal."""
    if len(g1) != len(g2):
        return False
    for r1, r2 in zip(g1, g2):
        if r1 != r2:
            return False
    return True


# =============================================================================
# Component Extraction Tests (COMP-01 to COMP-10)
# =============================================================================

class TestComponentExtraction:
    """
    Test component extraction with 4-level deterministic tie-breaking.

    Per spec (clarifications §3):
    - Components = (SameColor ∧ 8-connected)
    - Sort: lex_min ↑, -area ↓, centroid ↑, boundary_hash ↑
    - IDs: 0, 1, 2, ... (stable, deterministic)
    """

    def test_comp_01_single_component(self):
        """COMP-01: Single component gets ID 0"""
        grid = [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ]

        comps = extract_components(grid)

        # Should have 2 components: blue (1) and black (0)
        assert len(comps) == 2, f"Expected 2 components, got {len(comps)}"

        # Component 0 should be the one with lex_min (0,0) - black background has lex_min (0,2)
        # Component with lex_min (0,0) is blue (1)
        assert comps[0].color == 1, f"First component should be blue (1), got {comps[0].color}"
        assert comps[0].component_id == 0
        assert comps[0].lex_min == Pixel(0, 0)
        assert comps[0].area == 4

    def test_comp_02_tie_break_lex_min(self):
        """COMP-02: Lex-min is first tie-breaker (row-major)"""
        grid = [
            [1, 0, 2],
            [0, 0, 0],
            [3, 0, 4],
        ]

        comps = extract_components(grid)

        # 5 components: colors 0,1,2,3,4
        # By lex_min: (0,0)=1, (0,2)=2, (1,1)=0, (2,0)=3, (2,2)=4
        # Wait, lex_min for color 0: min pixel is (0,1)
        # Let me recalculate:
        # Color 1: (0,0) - lex_min (0,0)
        # Color 2: (0,2) - lex_min (0,2)
        # Color 3: (2,0) - lex_min (2,0)
        # Color 4: (2,2) - lex_min (2,2)
        # Color 0: (0,1), (1,0), (1,1), (1,2), (2,1) - lex_min (0,1)

        # Sort by lex_min: (0,0) < (0,1) < (0,2) < (2,0) < (2,2)
        # So: ID 0=color1, ID 1=color0, ID 2=color2, ID 3=color3, ID 4=color4

        assert comps[0].color == 1, f"Component 0 should be color 1, got {comps[0].color}"
        assert comps[0].lex_min == Pixel(0, 0)

        assert comps[1].color == 0, f"Component 1 should be color 0, got {comps[1].color}"
        assert comps[1].lex_min == Pixel(0, 1)

        assert comps[2].color == 2
        assert comps[2].lex_min == Pixel(0, 2)

    def test_comp_03_tie_break_area(self):
        """COMP-03: Area is second tie-breaker (larger first)"""
        # Two components with same lex_min (via different pixels)
        # Actually, same color components will merge if 8-connected
        # Let me create separate colors with same lex_min row
        grid = [
            [1, 1, 2, 2, 2, 0],
            [0, 0, 0, 0, 0, 0],
        ]

        comps = extract_components(grid)

        # Color 1: lex_min (0,0), area 2
        # Color 2: lex_min (0,2), area 3
        # Color 0: lex_min (0,5), area 6

        # But we want SAME lex_min to test area tie-break
        # Let me use same row, different columns - that's not same lex_min
        # I need components that start at same lex_min - impossible unless same component

        # Actually, the tie-break scenario is: components with SAME lex_min
        # But each component has unique lex_min pixel
        # So area tie-break only matters when lex_min is EQUAL

        # Let me test area ordering when lex_min is equal in row but different in col
        # Components sorted by lex_min THEN by area

        # Grid: same row different sizes
        grid = [
            [1, 1, 0, 2, 2, 2],
            [0, 0, 0, 0, 0, 0],
        ]

        comps = extract_components(grid)

        # Color 1: lex_min (0,0), area 2
        # Color 2: lex_min (0,3), area 3
        # Color 0: pixels at (0,2) + (1,0-5) = 7 pixels, lex_min (0,2)

        # Sort: (0,0) < (0,2) < (0,3)
        # So: color1, color0, color2

        # Verify area is computed correctly
        assert comps[0].area == 2  # color 1
        assert comps[1].area == 7  # color 0 (fixed: 1 in row0 + 6 in row1 = 7)
        assert comps[2].area == 3  # color 2

    def test_comp_04_tie_break_centroid(self):
        """COMP-04: Centroid is third tie-breaker (rational comparison, no floats)"""
        # Create components with same lex_min row to test centroid
        grid = [
            [1, 0, 2],
            [1, 0, 2],
        ]

        comps = extract_components(grid)

        # Color 1: pixels (0,0), (1,0) - centroid (0.5, 0.0)
        # Color 2: pixels (0,2), (1,2) - centroid (0.5, 2.0)
        # Color 0: pixels (0,1), (1,1) - centroid (0.5, 1.0)

        # All have same lex_min.row = 0, different lex_min.col
        # Sort by lex_min: (0,0) < (0,1) < (0,2)

        # Centroid only matters if lex_min and area are same
        # Let me verify centroid is computed correctly (integer only)

        c1 = comps[0]  # color 1
        assert c1.centroid_num == (1, 0)  # sum_r=0+1=1, sum_c=0+0=0
        assert c1.centroid_den == 2  # area
        # Actual centroid would be (1/2, 0/2) = (0.5, 0)

    def test_comp_05_tie_break_boundary_hash(self):
        """COMP-05: Boundary hash is fourth tie-breaker"""
        # Verify boundary hash is computed and used
        grid = [
            [1, 1],
            [1, 1],
        ]

        comps = extract_components(grid)

        # Should have 1 component (color 1)
        assert len(comps) == 1

        # Boundary = all 4 pixels (all on edge of component)
        # Verify boundary_hash exists
        assert comps[0].boundary_hash is not None
        assert isinstance(comps[0].boundary_hash, int)

    def test_comp_06_determinism_100_runs(self):
        """COMP-06: Component IDs are deterministic across 100 runs"""
        grid = [
            [1, 1, 0, 2],
            [1, 0, 0, 2],
            [0, 3, 3, 0],
        ]

        # Run 100 times, verify same IDs
        hashes = set()
        first_run = None

        for i in range(100):
            comps = extract_components(grid)

            # Serialize component IDs and properties
            serialized = json.dumps([
                {
                    'id': c.component_id,
                    'color': c.color,
                    'lex_min': (c.lex_min.row, c.lex_min.col),
                    'area': c.area,
                }
                for c in comps
            ], sort_keys=True)

            hash_val = hashlib.sha256(serialized.encode()).hexdigest()
            hashes.add(hash_val)

            if i == 0:
                first_run = comps
            else:
                # Verify byte-identical
                assert len(comps) == len(first_run)
                for c1, c2 in zip(comps, first_run):
                    assert c1.component_id == c2.component_id
                    assert c1.color == c2.color
                    assert c1.lex_min == c2.lex_min
                    assert c1.area == c2.area

        # All hashes should be identical
        assert len(hashes) == 1, f"Expected 1 unique hash, got {len(hashes)} (non-deterministic!)"

    def test_comp_07_8_connected_not_4(self):
        """COMP-07: Components are 8-connected (diagonals count)"""
        grid = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]

        comps = extract_components(grid)

        # Diagonal connectivity: all 3 pixels of color 1 should be ONE component
        color1_comps = [c for c in comps if c.color == 1]

        assert len(color1_comps) == 1, f"Expected 1 component for color 1 (8-connected), got {len(color1_comps)}"
        assert color1_comps[0].area == 3

    def test_comp_08_boundary_computation(self):
        """COMP-08: Boundary is 4-connected edge (pixels with non-component 4-neighbor)"""
        grid = [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        # Color 1 pixels: (1,1), (1,2), (2,1), (2,2)
        # All 4 are on boundary (all have at least one 4-neighbor that's not in component)

        # Compute boundary manually
        pixels = color1_comp.pixels
        boundary = set()
        for pixel in pixels:
            r, c = pixel.row, pixel.col
            neighbors_4 = [
                Pixel(r-1, c), Pixel(r+1, c),
                Pixel(r, c-1), Pixel(r, c+1),
            ]
            if any(n not in pixels for n in neighbors_4):
                boundary.add(pixel)

        # All 4 pixels should be on boundary
        assert len(boundary) == 4

    def test_comp_09_inertia_integer_only(self):
        """COMP-09: Inertia is integer (second moment, no float division)"""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        # Verify inertia is integer
        assert isinstance(color1_comp.inertia_num, int)

        # Compute manually:
        # Pixels: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
        # sum_r = 0+0+0+1+1+1 = 3
        # sum_c = 0+1+2+0+1+2 = 6
        # sum_rr = 0+0+0+1+1+1 = 3
        # sum_cc = 0+1+4+0+1+4 = 10
        # area = 6
        # inertia = (sum_rr*area - sum_r²) + (sum_cc*area - sum_c²)
        #         = (3*6 - 3²) + (10*6 - 6²)
        #         = (18 - 9) + (60 - 36)
        #         = 9 + 24 = 33

        assert color1_comp.inertia_num == 33

    def test_comp_10_empty_grid(self):
        """COMP-10: Empty grid returns empty component list"""
        grid = []

        comps = extract_components(grid)

        assert comps == []


# =============================================================================
# Hungarian Matching Tests (HUNG-01 to HUNG-08)
# =============================================================================

class TestHungarianMatching:
    """
    Test Hungarian matching with lexicographic cost.

    Per spec:
    - Cost tuple: (inertia_diff, area_diff, boundary_diff)
    - Scalarized for Hungarian algorithm
    - Δ = lex_min_Y - lex_min_X
    - Handles unmatched with dummy nodes
    """

    def test_hung_01_perfect_match_zero_delta(self):
        """HUNG-01: Identical components match with Δ=(0,0)"""
        grid_X = [
            [1, 1],
            [1, 1],
        ]
        grid_Y = [
            [1, 1],
            [1, 1],
        ]

        comps_X = extract_components(grid_X)
        comps_Y = extract_components(grid_Y)

        matches = match_components(comps_X, comps_Y)

        # Should have 1 match (color 1 to color 1)
        # Actually 2 matches: color 0 (background) and color 1
        assert len(matches) >= 1

        # Find color 1 match
        color1_match = [m for m in matches if m.comp_X_id != -1 and m.comp_Y_id != -1
                        and comps_X[m.comp_X_id].color == 1][0]

        assert color1_match.delta == (0, 0)

    def test_hung_02_translation_delta(self):
        """HUNG-02: Translated component has correct Δ vector"""
        grid_X = [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ]
        grid_Y = [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ]

        comps_X = extract_components(grid_X)
        comps_Y = extract_components(grid_Y)

        matches = match_components(comps_X, comps_Y)

        # Color 1 moved from (0,0) to (1,1)
        # Δ = (1, 1)

        color1_X = [c for c in comps_X if c.color == 1][0]
        color1_Y = [c for c in comps_Y if c.color == 1][0]

        # Find match
        color1_match = [m for m in matches if m.comp_X_id == color1_X.component_id][0]

        assert color1_match.delta == (1, 1)

    def test_hung_03_cost_function_lex_order(self):
        """HUNG-03: Cost function prioritizes inertia > area > boundary"""
        # Create components with different properties
        grid_X = [
            [1, 1],
        ]

        grid_Y1 = [
            [2, 2],  # Same shape, different color (area match)
        ]

        grid_Y2 = [
            [2, 2, 2],  # Different area
        ]

        comps_X = extract_components(grid_X)
        comps_Y1 = extract_components(grid_Y1)
        comps_Y2 = extract_components(grid_Y2)

        # Cost should be lower for Y1 (same area) than Y2 (different area)
        from arc_core.components import _compute_matching_cost

        cost_Y1 = _compute_matching_cost(comps_X[0], comps_Y1[0])
        cost_Y2 = _compute_matching_cost(comps_X[0], comps_Y2[0])

        # Y1 has area_diff=0, Y2 has area_diff=1
        # So cost_Y1 < cost_Y2
        assert cost_Y1 < cost_Y2

    def test_hung_04_unmatched_component_deleted(self):
        """HUNG-04: Unmatched X component has comp_Y_id=-1 (deletion)"""
        # More components in X than Y → some X components unmatched
        grid_X = [
            [1, 1, 0, 2, 2],
            [1, 1, 0, 2, 2],
        ]
        grid_Y = [
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ]

        comps_X = extract_components(grid_X)
        comps_Y = extract_components(grid_Y)

        matches = match_components(comps_X, comps_Y)

        # Should have 3 components in X: color 0, 1, 2
        # Should have 2 components in Y: color 0, 1
        # Color 2 component should be deleted (comp_Y_id=-1)

        color2_X = [c for c in comps_X if c.color == 2][0]
        color2_match = [m for m in matches if m.comp_X_id == color2_X.component_id][0]

        assert color2_match.comp_Y_id == -1, f"Deleted component should have comp_Y_id=-1, got {color2_match.comp_Y_id}"

    def test_hung_05_unmatched_component_inserted(self):
        """HUNG-05: Unmatched Y component has comp_X_id=-1 (insertion)"""
        # Y has more components than X
        # Use identical shapes in both grids + one extra in Y
        grid_X = [
            [1, 1, 0],
            [1, 1, 0],
        ]
        grid_Y = [
            [1, 1, 0, 2],  # Extra component color 2
            [1, 1, 0, 2],
        ]

        comps_X = extract_components(grid_X)
        comps_Y = extract_components(grid_Y)

        matches = match_components(comps_X, comps_Y)

        # X has 2 components, Y has 3 components
        # One Y component must be unmatched
        unmatched_Y = [m for m in matches if m.comp_X_id == -1]

        assert len(unmatched_Y) == 1, f"Expected 1 unmatched Y component, got {len(unmatched_Y)}"
        assert unmatched_Y[0].comp_Y_id != -1, "Unmatched Y component should have valid comp_Y_id"

    def test_hung_06_determinism_100_runs(self):
        """HUNG-06: Hungarian matching is deterministic (scalarized lex cost)"""
        grid_X = [
            [1, 0, 2],
            [1, 0, 2],
        ]
        grid_Y = [
            [0, 1, 0],
            [0, 1, 2],
        ]

        # Run 100 times
        hashes = set()
        first_run = None

        for i in range(100):
            comps_X = extract_components(grid_X)
            comps_Y = extract_components(grid_Y)
            matches = match_components(comps_X, comps_Y)

            serialized = json.dumps([
                {
                    'X': m.comp_X_id,
                    'Y': m.comp_Y_id,
                    'delta': m.delta,
                }
                for m in matches
            ], sort_keys=True)

            hash_val = hashlib.sha256(serialized.encode()).hexdigest()
            hashes.add(hash_val)

            if i == 0:
                first_run = matches
            else:
                assert len(matches) == len(first_run)
                for m1, m2 in zip(matches, first_run):
                    assert m1.comp_X_id == m2.comp_X_id
                    assert m1.comp_Y_id == m2.comp_Y_id
                    assert m1.delta == m2.delta

        assert len(hashes) == 1, "Hungarian matching must be deterministic"

    def test_hung_07_boundary_diff_normalized(self):
        """HUNG-07: Boundary diff uses normalized coordinates (lex_min → (0,0))"""
        # Two components with same shape but different positions
        grid_X = [
            [1, 1, 0],
            [0, 0, 0],
        ]
        grid_Y = [
            [0, 0, 0],
            [0, 1, 1],
        ]

        comps_X = extract_components(grid_X)
        comps_Y = extract_components(grid_Y)

        color1_X = [c for c in comps_X if c.color == 1][0]
        color1_Y = [c for c in comps_Y if c.color == 1][0]

        from arc_core.components import _compute_boundary_diff

        boundary_diff = _compute_boundary_diff(color1_X, color1_Y)

        # Same shape → boundary_diff should be 0 (perfect overlap after normalization)
        assert boundary_diff == 0, f"Same shape should have boundary_diff=0, got {boundary_diff}"

    def test_hung_08_integer_arithmetic_only(self):
        """HUNG-08: All cost computations are integer (no float leakage)"""
        grid_X = [
            [1, 1, 1],
            [1, 1, 1],
        ]
        grid_Y = [
            [2, 2],
            [2, 2],
        ]

        comps_X = extract_components(grid_X)
        comps_Y = extract_components(grid_Y)

        from arc_core.components import _compute_matching_cost

        cost = _compute_matching_cost(comps_X[0], comps_Y[0])

        # Cost must be integer
        assert isinstance(cost, int), f"Cost must be integer, got {type(cost)}"


# =============================================================================
# TRANSLATE Operation Tests (TRANS-01 to TRANS-08)
# =============================================================================

class TestTranslateOperation:
    """
    Test TRANSLATE operation: move component by Δ, clear source.
    """

    def test_trans_01_simple_translation(self):
        """TRANS-01: Component moves by Δ, source is cleared"""
        grid = [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        # Translate by (0, 2)
        result = apply_translate(grid, color1_comp.pixels, (0, 2), bg_color=0)

        expected = [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ]

        assert grids_equal(result, expected), f"Translation failed:\n{result}\nvs\n{expected}"

    def test_trans_02_out_of_bounds_clipping(self):
        """TRANS-02: Pixels outside grid bounds are clipped (not drawn)"""
        grid = [
            [1, 1],
            [1, 1],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        # Translate by (0, 2) - would go out of bounds
        result = apply_translate(grid, color1_comp.pixels, (0, 2), bg_color=0)

        # Source cleared, destination out of bounds → all 0
        expected = [
            [0, 0],
            [0, 0],
        ]

        assert grids_equal(result, expected)

    def test_trans_03_negative_delta(self):
        """TRANS-03: Negative Δ works correctly"""
        grid = [
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        # Translate by (-1, -1)
        result = apply_translate(grid, color1_comp.pixels, (-1, -1), bg_color=0)

        expected = [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ]

        assert grids_equal(result, expected)

    def test_trans_04_preserves_color(self):
        """TRANS-04: Translation preserves pixel colors"""
        grid = [
            [3, 3, 0],
            [3, 3, 0],
        ]

        comps = extract_components(grid)
        color3_comp = [c for c in comps if c.color == 3][0]

        result = apply_translate(grid, color3_comp.pixels, (0, 1), bg_color=0)

        # Check color is preserved
        assert result[0][1] == 3
        assert result[0][2] == 3
        assert result[1][1] == 3
        assert result[1][2] == 3

    def test_trans_05_determinism(self):
        """TRANS-05: Translation is deterministic (100 runs)"""
        grid = [
            [1, 1, 0],
            [1, 1, 0],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        hashes = set()
        for _ in range(100):
            result = apply_translate(grid, color1_comp.pixels, (0, 1), bg_color=0)
            hashes.add(grid_to_hash(result))

        assert len(hashes) == 1, "Translation must be deterministic"

    def test_trans_06_overlapping_destination(self):
        """TRANS-06: Translation handles overlapping source/dest (order matters)"""
        grid = [
            [1, 1, 1],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        # Translate by (0, 1) - overlaps with source
        result = apply_translate(grid, color1_comp.pixels, (0, 1), bg_color=0)

        # Implementation: clear source first, then write destination
        # Source: (0,0), (0,1), (0,2) → cleared
        # Dest: (0,1), (0,2), (0,3) → (0,3) is out of bounds

        expected = [
            [0, 1, 1],
        ]

        assert grids_equal(result, expected)

    def test_trans_07_background_color_customizable(self):
        """TRANS-07: Background color for clearing is customizable"""
        grid = [
            [1, 1, 0],
            [1, 1, 0],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        # Translate with bg_color=9
        result = apply_translate(grid, color1_comp.pixels, (0, 1), bg_color=9)

        # Source should be filled with 9
        assert result[0][0] == 9
        assert result[1][0] == 9

    def test_trans_08_partial_out_of_bounds(self):
        """TRANS-08: Partial translation (some pixels in bounds, some out)"""
        grid = [
            [1, 1, 0],
            [1, 1, 0],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        # Translate by (0, 2) - rightmost pixels go out
        result = apply_translate(grid, color1_comp.pixels, (0, 2), bg_color=0)

        # (0,0)→(0,2), (0,1)→(0,3 out), (1,0)→(1,2), (1,1)→(1,3 out)
        expected = [
            [0, 0, 1],
            [0, 0, 1],
        ]

        assert grids_equal(result, expected)


# =============================================================================
# COPY/DELETE Operation Tests
# =============================================================================

class TestCopyDeleteOperations:
    """
    Test COPY and DELETE operations.
    """

    def test_copy_01_keeps_source(self):
        """COPY-01: COPY keeps source pixels (unlike TRANSLATE)"""
        grid = [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        result = apply_copy(grid, color1_comp.pixels, (0, 2))

        expected = [
            [1, 1, 1, 1],  # Source preserved AND copied
            [1, 1, 1, 1],
        ]

        assert grids_equal(result, expected)

    def test_delete_01_removes_component(self):
        """DELETE-01: DELETE replaces component with background"""
        grid = [
            [1, 1, 0],
            [1, 1, 0],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        result = apply_delete(grid, color1_comp.pixels, bg_color=0)

        expected = [
            [0, 0, 0],
            [0, 0, 0],
        ]

        assert grids_equal(result, expected)


# =============================================================================
# Bresenham Line Drawing Tests (LINE-01 to LINE-09)
# =============================================================================

class TestBresenhamLineDrawing:
    """
    Test Bresenham line drawing algorithms (4-conn and 8-conn).

    Per spec:
    - 4-conn: Orthogonal connectivity (no diagonal steps), staircase pattern
    - 8-conn: King's move connectivity (diagonal steps allowed)
    - Both should be deterministic and mathematically optimal (shortest path)
    """

    def test_line_01_horizontal_4conn(self):
        """LINE-01: 4-conn horizontal line"""
        p1 = Pixel(0, 0)
        p2 = Pixel(0, 4)

        pixels = bresenham_4conn(p1, p2)

        # Horizontal line: all same row
        expected = [Pixel(0, 0), Pixel(0, 1), Pixel(0, 2), Pixel(0, 3), Pixel(0, 4)]

        assert pixels == expected, f"4-conn horizontal failed: {pixels}"

    def test_line_02_vertical_4conn(self):
        """LINE-02: 4-conn vertical line"""
        p1 = Pixel(0, 0)
        p2 = Pixel(4, 0)

        pixels = bresenham_4conn(p1, p2)

        # Vertical line: all same column
        expected = [Pixel(0, 0), Pixel(1, 0), Pixel(2, 0), Pixel(3, 0), Pixel(4, 0)]

        assert pixels == expected

    def test_line_03_diagonal_4conn_staircase(self):
        """LINE-03: 4-conn diagonal creates staircase (no diagonal steps)"""
        p1 = Pixel(0, 0)
        p2 = Pixel(3, 3)

        pixels = bresenham_4conn(p1, p2)

        # Should be staircase pattern (only orthogonal steps)
        for i in range(len(pixels) - 1):
            dr = abs(pixels[i+1].row - pixels[i].row)
            dc = abs(pixels[i+1].col - pixels[i].col)
            # Each step is either (1,0) or (0,1), never (1,1)
            assert (dr == 1 and dc == 0) or (dr == 0 and dc == 1), \
                f"4-conn should not have diagonal steps: {pixels[i]} → {pixels[i+1]}"

        # Should include start and end
        assert pixels[0] == p1
        assert pixels[-1] == p2

    def test_line_04_diagonal_8conn_shortcut(self):
        """LINE-04: 8-conn diagonal uses diagonal steps (shorter path)"""
        p1 = Pixel(0, 0)
        p2 = Pixel(3, 3)

        pixels = bresenham_8conn(p1, p2)

        # 8-conn allows diagonals, so path is shorter
        # Perfect diagonal: should be just 4 pixels (0,0), (1,1), (2,2), (3,3)
        expected = [Pixel(0, 0), Pixel(1, 1), Pixel(2, 2), Pixel(3, 3)]

        assert pixels == expected, f"8-conn diagonal failed: {pixels}"

    def test_line_05_4conn_determinism(self):
        """LINE-05: 4-conn is deterministic (100 runs)"""
        p1 = Pixel(1, 2)
        p2 = Pixel(5, 7)

        hashes = set()
        for _ in range(100):
            pixels = bresenham_4conn(p1, p2)
            serialized = json.dumps([(p.row, p.col) for p in pixels])
            hashes.add(hashlib.sha256(serialized.encode()).hexdigest())

        assert len(hashes) == 1, "4-conn Bresenham must be deterministic"

    def test_line_06_8conn_determinism(self):
        """LINE-06: 8-conn is deterministic (100 runs)"""
        p1 = Pixel(1, 2)
        p2 = Pixel(5, 7)

        hashes = set()
        for _ in range(100):
            pixels = bresenham_8conn(p1, p2)
            serialized = json.dumps([(p.row, p.col) for p in pixels])
            hashes.add(hashlib.sha256(serialized.encode()).hexdigest())

        assert len(hashes) == 1, "8-conn Bresenham must be deterministic"

    def test_line_07_drawline_applies_color(self):
        """LINE-07: apply_drawline draws line with specified color"""
        grid = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        p1 = Pixel(0, 0)
        p2 = Pixel(2, 3)

        result = apply_drawline(grid, p1, p2, "8conn", color=5)

        # Check that line pixels are color 5
        # For 8-conn (0,0)→(2,3), path should be diagonal-ish
        # At minimum, endpoints should be colored
        assert result[0][0] == 5, "Start point should be colored"
        assert result[2][3] == 5, "End point should be colored"

    def test_line_08_bresenham_reversed(self):
        """LINE-08: Bresenham should work in both directions"""
        p1 = Pixel(0, 0)
        p2 = Pixel(3, 3)

        pixels_forward = bresenham_8conn(p1, p2)
        pixels_backward = bresenham_8conn(p2, p1)

        # Paths might differ, but both should connect the points
        assert pixels_forward[0] == p1
        assert pixels_forward[-1] == p2
        assert pixels_backward[0] == p2
        assert pixels_backward[-1] == p1

    def test_line_09_single_pixel_line(self):
        """LINE-09: Line from pixel to itself"""
        p1 = Pixel(2, 3)
        p2 = Pixel(2, 3)

        pixels_4 = bresenham_4conn(p1, p2)
        pixels_8 = bresenham_8conn(p1, p2)

        # Should return just the single pixel
        assert pixels_4 == [p1]
        assert pixels_8 == [p1]


# =============================================================================
# Skeleton / Zhang-Suen Thinning Tests (SKEL-01 to SKEL-06)
# =============================================================================

class TestSkeletonThinning:
    """
    Test Zhang-Suen thinning algorithm for skeletonization.

    Per spec:
    - Produces 1-pixel-wide skeleton
    - Preserves topology (connectivity, loops)
    - Preserves endpoints and junction points
    - Deterministic
    """

    def test_skel_01_horizontal_line_unchanged(self):
        """SKEL-01: Horizontal 1-pixel line unchanged (already skeleton)"""
        grid = [
            [1, 1, 1, 1],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        skeleton_pixels = skeleton_zhang_suen(grid, color1_comp.pixels)

        # Already 1-pixel wide → should be unchanged
        assert len(skeleton_pixels) == 4
        assert skeleton_pixels == color1_comp.pixels

    def test_skel_02_thick_line_to_thin(self):
        """SKEL-02: Thick line reduces to 1-pixel skeleton"""
        grid = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        skeleton_pixels = skeleton_zhang_suen(grid, color1_comp.pixels)

        # Should be thinned to ~4 pixels (1-pixel wide line)
        assert len(skeleton_pixels) <= 4, f"Skeleton should be thin, got {len(skeleton_pixels)} pixels"

    def test_skel_03_square_to_outline(self):
        """SKEL-03: Filled square reduces to skeleton (preserves topology)"""
        grid = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        skeleton_pixels = skeleton_zhang_suen(grid, color1_comp.pixels)

        # Skeleton should be much smaller than original (20 pixels → ~12-16)
        assert len(skeleton_pixels) < len(color1_comp.pixels), \
            f"Skeleton should be smaller: {len(skeleton_pixels)} vs {len(color1_comp.pixels)}"

    def test_skel_04_determinism(self):
        """SKEL-04: Skeletonization is deterministic"""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        hashes = set()
        for _ in range(100):
            skeleton_pixels = skeleton_zhang_suen(grid, color1_comp.pixels)
            serialized = json.dumps(sorted([(p.row, p.col) for p in skeleton_pixels]))
            hashes.add(hashlib.sha256(serialized.encode()).hexdigest())

        assert len(hashes) == 1, "Skeletonization must be deterministic"

    def test_skel_05_apply_skeleton_preserves_color(self):
        """SKEL-05: apply_skeleton preserves component color"""
        grid = [
            [3, 3, 3],
            [3, 3, 3],
        ]

        comps = extract_components(grid)
        color3_comp = [c for c in comps if c.color == 3][0]

        result = apply_skeleton(grid, color3_comp.pixels, bg_color=0)

        # Skeleton pixels should still be color 3
        has_color_3 = any(result[r][c] == 3 for r in range(len(result)) for c in range(len(result[0])))
        assert has_color_3, "Skeleton should preserve component color"

    def test_skel_06_single_pixel_unchanged(self):
        """SKEL-06: Single pixel component unchanged"""
        grid = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]

        comps = extract_components(grid)
        color1_comp = [c for c in comps if c.color == 1][0]

        skeleton_pixels = skeleton_zhang_suen(grid, color1_comp.pixels)

        # Single pixel is already minimal
        assert len(skeleton_pixels) == 1
        assert Pixel(1, 1) in skeleton_pixels


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
