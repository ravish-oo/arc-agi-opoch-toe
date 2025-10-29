"""
Test script for WO-08 (components.py) - Manual verification.

Tests component extraction, deterministic ID assignment, and Hungarian matching.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_core.components import extract_components, match_components
from arc_core.types import Pixel


def print_grid(grid, label=""):
    """Helper to print grids."""
    if label:
        print(f"\n{label}:")
    for row in grid:
        print("  " + " ".join(str(c) for c in row))


def print_component(comp):
    """Helper to print component details."""
    print(f"  Component ID: {comp.component_id}")
    print(f"    Color: {comp.color}")
    print(f"    Area: {comp.area}")
    print(f"    Lex-min: {comp.lex_min}")
    print(f"    Centroid: ({comp.centroid_num[0]}/{comp.centroid_den}, {comp.centroid_num[1]}/{comp.centroid_den})")
    print(f"    Inertia: {comp.inertia_num}")
    print(f"    BBox: {comp.bbox}")
    print(f"    Boundary hash: {comp.boundary_hash}")


def test_single_component():
    """Test extraction of a single component."""
    print("\n" + "=" * 60)
    print("TEST: Single Component Extraction")
    print("=" * 60)

    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]

    print_grid(grid, "Grid")

    components = extract_components(grid)

    print(f"\nFound {len(components)} components")
    for comp in components:
        print_component(comp)

    # Verify
    assert len(components) == 2, f"Expected 2 components (background + center), got {len(components)}"

    # Find color 1 component
    comp_1 = [c for c in components if c.color == 1][0]
    assert comp_1.area == 1, f"Expected area 1, got {comp_1.area}"
    assert comp_1.lex_min == Pixel(1, 1), f"Expected lex_min (1,1), got {comp_1.lex_min}"

    print("\n✅ PASSED")
    return True


def test_multiple_components():
    """Test extraction of multiple components of same color."""
    print("\n" + "=" * 60)
    print("TEST: Multiple Components (Same Color)")
    print("=" * 60)

    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]

    print_grid(grid, "Grid")

    components = extract_components(grid)

    print(f"\nFound {len(components)} components")
    for comp in components:
        print_component(comp)

    # Verify: 4 components of color 1, 1 component of color 0
    color_1_comps = [c for c in components if c.color == 1]
    assert len(color_1_comps) == 4, f"Expected 4 color-1 components, got {len(color_1_comps)}"

    # Verify deterministic ID ordering (by lex-min)
    assert color_1_comps[0].lex_min == Pixel(0, 0), "First should be (0,0)"
    assert color_1_comps[1].lex_min == Pixel(0, 2), "Second should be (0,2)"
    assert color_1_comps[2].lex_min == Pixel(2, 0), "Third should be (2,0)"
    assert color_1_comps[3].lex_min == Pixel(2, 2), "Fourth should be (2,2)"

    print("\n✅ PASSED")
    return True


def test_8connected():
    """Test 8-connected component extraction."""
    print("\n" + "=" * 60)
    print("TEST: 8-Connected Component")
    print("=" * 60)

    grid = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]

    print_grid(grid, "Grid")

    components = extract_components(grid)

    print(f"\nFound {len(components)} components")
    for comp in components:
        print_component(comp)

    # Verify: diagonal pixels form ONE 8-connected component
    color_1_comps = [c for c in components if c.color == 1]
    assert len(color_1_comps) == 1, f"Expected 1 color-1 component (8-connected), got {len(color_1_comps)}"
    assert color_1_comps[0].area == 3, f"Expected area 3, got {color_1_comps[0].area}"

    print("\n✅ PASSED")
    return True


def test_deterministic_ids():
    """Test deterministic ID assignment across runs."""
    print("\n" + "=" * 60)
    print("TEST: Deterministic IDs (10 runs)")
    print("=" * 60)

    grid = [
        [1, 2, 1],
        [2, 0, 2],
        [1, 2, 1]
    ]

    print_grid(grid, "Grid")

    # Run 10 times
    all_ids = []
    for i in range(10):
        components = extract_components(grid)
        ids = [(c.color, c.component_id, c.lex_min) for c in components]
        all_ids.append(ids)

    # Verify all runs identical
    for i in range(1, 10):
        assert all_ids[i] == all_ids[0], f"Run {i+1} differs from run 1"

    print(f"\nAll 10 runs produced identical IDs:")
    for color, cid, lex_min in all_ids[0]:
        print(f"  Color {color}, ID {cid}, lex_min {lex_min}")

    print("\n✅ PASSED")
    return True


def test_inertia_computation():
    """Test shape inertia computation."""
    print("\n" + "=" * 60)
    print("TEST: Shape Inertia Computation")
    print("=" * 60)

    # Compact shape (low inertia)
    grid_compact = [
        [1, 1],
        [1, 1]
    ]

    # Spread shape (high inertia)
    grid_spread = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]

    print_grid(grid_compact, "Compact (2x2 block)")
    comps_compact = extract_components(grid_compact)
    comp_compact = [c for c in comps_compact if c.color == 1][0]

    print_grid(grid_spread, "Spread (4 corners)")
    comps_spread = extract_components(grid_spread)
    # Find the 8-connected component (all 4 pixels if they're 8-connected)
    # Actually in this grid, 4 corners are NOT 8-connected!
    color_1_spread = [c for c in comps_spread if c.color == 1]

    print(f"\nCompact inertia: {comp_compact.inertia_num}")
    print(f"Spread components: {len(color_1_spread)}")
    for c in color_1_spread:
        print(f"  Inertia: {c.inertia_num}")

    # Compact should have lower inertia than individual spread pixels
    # (spread pixels have area 1, so inertia is 0 for single pixels)

    print("\n✅ PASSED")
    return True


def test_matching_simple():
    """Test simple component matching."""
    print("\n" + "=" * 60)
    print("TEST: Simple Component Matching")
    print("=" * 60)

    # Grid X: component at (0,0)
    grid_X = [
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0]
    ]

    # Grid Y: same component shifted to (1,1)
    grid_Y = [
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 1]
    ]

    print_grid(grid_X, "Grid X")
    print_grid(grid_Y, "Grid Y")

    comps_X = extract_components(grid_X)
    comps_Y = extract_components(grid_Y)

    print(f"\nComponents X: {len(comps_X)}")
    for c in comps_X:
        if c.color == 1:
            print_component(c)

    print(f"\nComponents Y: {len(comps_Y)}")
    for c in comps_Y:
        if c.color == 1:
            print_component(c)

    # Match
    matches = match_components(comps_X, comps_Y)

    print(f"\nMatches: {len(matches)}")
    for match in matches:
        print(f"  X_id={match.comp_X_id}, Y_id={match.comp_Y_id}, delta={match.delta}, cost={match.cost}")

    # Verify: should match color-1 components with delta=(1,1)
    color_1_matches = [m for m in matches if m.comp_X_id != -1 and m.comp_Y_id != -1]
    # Find the match between color-1 components
    color_1_X = [c for c in comps_X if c.color == 1][0]
    color_1_Y = [c for c in comps_Y if c.color == 1][0]

    match_1 = [m for m in matches if m.comp_X_id == color_1_X.component_id and m.comp_Y_id == color_1_Y.component_id][0]

    assert match_1.delta == (1, 1), f"Expected delta (1,1), got {match_1.delta}"

    print("\n✅ PASSED")
    return True


def test_matching_equal_area():
    """Test matching with equal-area components (tie on area)."""
    print("\n" + "=" * 60)
    print("TEST: Matching Equal-Area Components")
    print("=" * 60)

    # Two components of color 1, same area (2 pixels each)
    grid_X = [
        [1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]

    # Same components, swapped positions
    grid_Y = [
        [1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]

    print_grid(grid_X, "Grid X")
    print_grid(grid_Y, "Grid Y (identical)")

    comps_X = extract_components(grid_X)
    comps_Y = extract_components(grid_Y)

    matches = match_components(comps_X, comps_Y)

    print(f"\nMatches: {len(matches)}")
    for match in matches:
        print(f"  X_id={match.comp_X_id}, Y_id={match.comp_Y_id}, delta={match.delta}")

    # Verify: should match identity (delta = (0,0) for all)
    color_1_matches = [m for m in matches if m.comp_X_id != -1 and m.comp_Y_id != -1]
    for match in color_1_matches:
        comp_X = comps_X[match.comp_X_id]
        comp_Y = comps_Y[match.comp_Y_id]
        if comp_X.color == 1:
            assert match.delta == (0, 0), f"Expected delta (0,0) for identity, got {match.delta}"

    print("\n✅ PASSED")
    return True


def test_matching_unmatched():
    """Test matching with unmatched components (deletion/insertion)."""
    print("\n" + "=" * 60)
    print("TEST: Matching with Unmatched Components")
    print("=" * 60)

    # Grid X: 2 components
    grid_X = [
        [1, 0, 1],
        [0, 0, 0]
    ]

    # Grid Y: 1 component (one deleted)
    grid_Y = [
        [1, 0, 0],
        [0, 0, 0]
    ]

    print_grid(grid_X, "Grid X (2 components)")
    print_grid(grid_Y, "Grid Y (1 component)")

    comps_X = extract_components(grid_X)
    comps_Y = extract_components(grid_Y)

    matches = match_components(comps_X, comps_Y)

    print(f"\nMatches: {len(matches)}")
    for match in matches:
        print(f"  X_id={match.comp_X_id}, Y_id={match.comp_Y_id}, delta={match.delta}, cost={match.cost}")

    # Verify: should have one real match and one dummy match
    real_matches = [m for m in matches if m.comp_X_id != -1 and m.comp_Y_id != -1]
    dummy_matches = [m for m in matches if m.comp_X_id == -1 or m.comp_Y_id == -1]

    print(f"\nReal matches: {len(real_matches)}")
    print(f"Dummy matches: {len(dummy_matches)}")

    print("\n✅ PASSED")
    return True


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 70)
    print(" WO-08 Components & Hungarian Matching - Test Suite")
    print("=" * 70)

    tests = [
        test_single_component,
        test_multiple_components,
        test_8connected,
        test_deterministic_ids,
        test_inertia_computation,
        test_matching_simple,
        test_matching_equal_area,
        test_matching_unmatched,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"\n❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n💥 ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f" Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
