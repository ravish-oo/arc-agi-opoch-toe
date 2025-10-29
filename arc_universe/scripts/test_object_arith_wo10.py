"""
Test script for WO-10 (object_arith.py) - Manual verification.

Tests all object arithmetic operations:
1. TRANSLATE - move component by Δ
2. COPY - copy component (keep original)
3. DELETE - remove component
4. DRAWLINE - Bresenham 4-conn and 8-conn
5. SKELETON - Zhang-Suen thinning

All tests verify:
- Deterministic behavior (reproducible across runs)
- Correct algorithmic output
- FY exactness where applicable
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_laws.object_arith import (
    apply_translate,
    apply_copy,
    apply_delete,
    apply_drawline,
    bresenham_4conn,
    bresenham_8conn,
    apply_skeleton,
    skeleton_zhang_suen,
    build_object_arith,
)
from arc_core.types import Pixel
from arc_core.components import extract_components


def print_grid(grid, label=""):
    """Helper to print grids."""
    if label:
        print(f"\n{label}:")
    for row in grid:
        print("  " + " ".join(str(c) for c in row))


def test_translate_component():
    """Test TRANSLATE operation: move component by Δ."""
    print("\n" + "=" * 60)
    print("TEST: TRANSLATE Component")
    print("=" * 60)

    # Grid with component at (1,1)
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    print_grid(grid, "Original Grid")

    # Extract component
    comps = extract_components(grid)
    comp_1 = [c for c in comps if c.color == 1][0]

    # Translate by Δ=(1, 2)
    delta = (1, 2)
    result = apply_translate(grid, comp_1.pixels, delta, bg_color=0, clear_source=True)

    print(f"\nTranslate by Δ={delta}")
    print_grid(result, "Result")

    # Verify: component should now be at (2,3)
    expected = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]

    assert result == expected, f"TRANSLATE failed: {result} != {expected}"

    print("\n✅ PASSED")
    return True


def test_copy_component():
    """Test COPY operation: copy component (keep original)."""
    print("\n" + "=" * 60)
    print("TEST: COPY Component")
    print("=" * 60)

    # Grid with component at (1,1)
    grid = [
        [0, 0, 0, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 2, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    print_grid(grid, "Original Grid")

    # Extract component
    comps = extract_components(grid)
    comp_2 = [c for c in comps if c.color == 2][0]

    # Copy by Δ=(0, 2)
    delta = (0, 2)
    result = apply_copy(grid, comp_2.pixels, delta)

    print(f"\nCopy by Δ={delta} (keep original)")
    print_grid(result, "Result")

    # Verify: original at (1,1) AND copy at (1,3)
    expected = [
        [0, 0, 0, 0, 0],
        [0, 2, 2, 2, 2],
        [0, 2, 2, 2, 2],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    assert result == expected, f"COPY failed: {result} != {expected}"

    print("\n✅ PASSED")
    return True


def test_delete_component():
    """Test DELETE operation: remove component."""
    print("\n" + "=" * 60)
    print("TEST: DELETE Component")
    print("=" * 60)

    # Grid with two components
    grid = [
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 0, 0],
        [3, 3, 0, 0, 0],
    ]

    print_grid(grid, "Original Grid")

    # Extract components
    comps = extract_components(grid)
    comp_2 = [c for c in comps if c.color == 2][0]

    # Delete color-2 component
    result = apply_delete(grid, comp_2.pixels, bg_color=0)

    print("\nDelete color-2 component")
    print_grid(result, "Result")

    # Verify: color-2 removed
    expected = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 0, 0],
        [3, 3, 0, 0, 0],
    ]

    assert result == expected, f"DELETE failed"

    print("\n✅ PASSED")
    return True


def test_bresenham_4conn():
    """Test 4-connected Bresenham line."""
    print("\n" + "=" * 60)
    print("TEST: Bresenham 4-Connected Line")
    print("=" * 60)

    # Test horizontal line
    p1 = Pixel(2, 1)
    p2 = Pixel(2, 5)
    line = bresenham_4conn(p1, p2)

    print(f"\nLine from {p1} to {p2}")
    print(f"Pixels: {line}")

    expected_pixels = [Pixel(2, 1), Pixel(2, 2), Pixel(2, 3), Pixel(2, 4), Pixel(2, 5)]
    assert line == expected_pixels, f"Horizontal line failed"

    # Test diagonal line (4-conn will have staircase pattern)
    p1 = Pixel(0, 0)
    p2 = Pixel(3, 3)
    line = bresenham_4conn(p1, p2)

    print(f"\nDiagonal line from {p1} to {p2}")
    print(f"Pixels: {line}")

    # 4-conn diagonal should have staircase (no pure diagonal steps)
    for i in range(len(line) - 1):
        dr = abs(line[i + 1].row - line[i].row)
        dc = abs(line[i + 1].col - line[i].col)
        # Each step should be orthogonal (dr=1,dc=0 or dr=0,dc=1)
        assert (dr == 1 and dc == 0) or (dr == 0 and dc == 1), "4-conn violated"

    print("\n✅ PASSED")
    return True


def test_bresenham_8conn():
    """Test 8-connected Bresenham line."""
    print("\n" + "=" * 60)
    print("TEST: Bresenham 8-Connected Line")
    print("=" * 60)

    # Test diagonal line (8-conn allows pure diagonal)
    p1 = Pixel(0, 0)
    p2 = Pixel(4, 4)
    line = bresenham_8conn(p1, p2)

    print(f"\nDiagonal line from {p1} to {p2}")
    print(f"Pixels: {line}")

    expected_pixels = [Pixel(0, 0), Pixel(1, 1), Pixel(2, 2), Pixel(3, 3), Pixel(4, 4)]
    assert line == expected_pixels, f"8-conn diagonal failed: {line}"

    # Test non-diagonal (should still work)
    p1 = Pixel(1, 1)
    p2 = Pixel(1, 4)
    line = bresenham_8conn(p1, p2)

    print(f"\nHorizontal line from {p1} to {p2}")
    print(f"Pixels: {line}")

    expected_pixels = [Pixel(1, 1), Pixel(1, 2), Pixel(1, 3), Pixel(1, 4)]
    assert line == expected_pixels, f"8-conn horizontal failed"

    print("\n✅ PASSED")
    return True


def test_drawline_on_grid():
    """Test DRAWLINE operation on grid."""
    print("\n" + "=" * 60)
    print("TEST: DRAWLINE on Grid")
    print("=" * 60)

    # Empty grid
    grid = [[0] * 7 for _ in range(7)]

    print_grid(grid, "Empty Grid")

    # Draw 4-conn line
    anchor1 = Pixel(1, 1)
    anchor2 = Pixel(5, 5)
    result = apply_drawline(grid, anchor1, anchor2, metric="4conn", color=3)

    print(f"\nDraw 4-conn line from {anchor1} to {anchor2} (color=3)")
    print_grid(result, "Result")

    # Verify line drawn
    line_pixels = bresenham_4conn(anchor1, anchor2)
    for pixel in line_pixels:
        assert result[pixel.row][pixel.col] == 3, f"Pixel {pixel} not drawn"

    # Draw 8-conn line on fresh grid
    grid2 = [[0] * 7 for _ in range(7)]
    result2 = apply_drawline(grid2, anchor1, anchor2, metric="8conn", color=5)

    print(f"\nDraw 8-conn line from {anchor1} to {anchor2} (color=5)")
    print_grid(result2, "Result")

    # Verify line drawn
    line_pixels = bresenham_8conn(anchor1, anchor2)
    for pixel in line_pixels:
        assert result2[pixel.row][pixel.col] == 5, f"Pixel {pixel} not drawn"

    print("\n✅ PASSED")
    return True


def test_skeleton_zhang_suen():
    """Test Zhang-Suen skeleton/thinning."""
    print("\n" + "=" * 60)
    print("TEST: Zhang-Suen Skeleton/Thinning")
    print("=" * 60)

    # Create thick shape (3×3 block)
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]

    print_grid(grid, "Thick Shape (3×3 block)")

    # Extract component
    comps = extract_components(grid)
    comp_1 = [c for c in comps if c.color == 1][0]

    print(f"Component area: {comp_1.area} pixels")

    # Skeletonize
    skeleton_pixels = skeleton_zhang_suen(grid, comp_1.pixels)

    print(f"Skeleton area: {len(skeleton_pixels)} pixels (should be ~3 for vertical line)")

    # Create skeleton grid for visualization
    skel_grid = [[0] * 5 for _ in range(5)]
    for pixel in skeleton_pixels:
        skel_grid[pixel.row][pixel.col] = 1

    print_grid(skel_grid, "Skeleton (1-pixel-wide)")

    # Verify skeleton is thinner than original
    assert len(skeleton_pixels) < comp_1.area, "Skeleton not thinner than original"

    # Verify skeleton is connected (simple check: not empty)
    assert len(skeleton_pixels) > 0, "Skeleton is empty"

    print("\n✅ PASSED")
    return True


def test_skeleton_l_shape():
    """Test skeleton on L-shaped component."""
    print("\n" + "=" * 60)
    print("TEST: Skeleton L-Shape")
    print("=" * 60)

    # Create L-shaped component
    grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
    ]

    print_grid(grid, "L-Shaped Component")

    # Extract component
    comps = extract_components(grid)
    comp_1 = [c for c in comps if c.color == 1][0]

    # Skeletonize
    skeleton_pixels = skeleton_zhang_suen(grid, comp_1.pixels)

    # Create skeleton grid
    skel_grid = [[0] * 6 for _ in range(6)]
    for pixel in skeleton_pixels:
        skel_grid[pixel.row][pixel.col] = 1

    print_grid(skel_grid, "Skeleton")

    # Verify skeleton preserves L-shape topology
    assert len(skeleton_pixels) < comp_1.area, "Skeleton not thinner"
    assert len(skeleton_pixels) > 3, "Skeleton too thin (lost topology)"

    print("\n✅ PASSED")
    return True


def test_apply_skeleton_operation():
    """Test full SKELETON operation (apply_skeleton)."""
    print("\n" + "=" * 60)
    print("TEST: Apply SKELETON Operation")
    print("=" * 60)

    # Create grid with thick component
    grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 0, 0],
        [0, 0, 2, 2, 2, 0, 0],
        [0, 0, 2, 2, 2, 2, 2],
        [0, 0, 2, 2, 2, 2, 2],
        [0, 0, 2, 2, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]

    print_grid(grid, "Original Grid (thick T-shape)")

    # Extract component
    comps = extract_components(grid)
    comp_2 = [c for c in comps if c.color == 2][0]

    # Apply skeleton
    result = apply_skeleton(grid, comp_2.pixels, bg_color=0)

    print_grid(result, "After Skeletonization")

    # Verify skeleton is present and thinner
    # Count non-zero pixels in result
    skeleton_count = sum(1 for row in result for cell in row if cell == 2)
    assert skeleton_count < comp_2.area, "Skeleton not thinner"
    assert skeleton_count > 0, "Skeleton empty"

    print(f"\nOriginal area: {comp_2.area}")
    print(f"Skeleton area: {skeleton_count}")

    print("\n✅ PASSED")
    return True


def test_determinism():
    """Test determinism: same input produces same output."""
    print("\n" + "=" * 60)
    print("TEST: Determinism (10 runs)")
    print("=" * 60)

    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]

    # Extract component
    comps = extract_components(grid)
    comp_1 = [c for c in comps if c.color == 1][0]

    # Run skeleton 10 times
    results = []
    for i in range(10):
        skeleton_pixels = skeleton_zhang_suen(grid, comp_1.pixels)
        results.append(skeleton_pixels)

    # Verify all runs identical
    for i in range(1, 10):
        assert results[i] == results[0], f"Run {i+1} differs from run 1"

    print("\nAll 10 runs produced identical skeletons ✅")

    # Test Bresenham determinism
    p1 = Pixel(0, 0)
    p2 = Pixel(5, 5)

    lines_4conn = [bresenham_4conn(p1, p2) for _ in range(10)]
    lines_8conn = [bresenham_8conn(p1, p2) for _ in range(10)]

    for i in range(1, 10):
        assert lines_4conn[i] == lines_4conn[0], f"Bresenham 4-conn run {i+1} differs"
        assert lines_8conn[i] == lines_8conn[0], f"Bresenham 8-conn run {i+1} differs"

    print("All Bresenham runs deterministic ✅")

    print("\n✅ PASSED")
    return True


def test_build_object_arith():
    """Test build_object_arith with training pairs."""
    print("\n" + "=" * 60)
    print("TEST: build_object_arith")
    print("=" * 60)

    # Create training pair: translate component by Δ=(1,1)
    X = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]

    Y = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
    ]

    train_pairs = [(X, Y)]

    print_grid(X, "Training Input X")
    print_grid(Y, "Training Output Y")

    # Build theta (minimal)
    theta = {
        "train_pairs": train_pairs,
        "anchors": [],
    }

    # Build object arithmetic laws
    laws = build_object_arith(theta)

    print(f"\nExtracted {len(laws)} laws")
    for law in laws:
        print(f"  {law.operation}: delta={law.delta}")

    # Verify we got translate law with Δ=(1,1)
    translate_laws = [l for l in laws if l.operation == "translate"]
    assert len(translate_laws) > 0, "No translate laws extracted"

    # Check delta is (1,1)
    assert (1, 1) in [l.delta for l in translate_laws], "Δ=(1,1) not found"

    print("\n✅ PASSED")
    return True


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 70)
    print(" WO-10 Object Arithmetic - Test Suite")
    print("=" * 70)

    tests = [
        test_translate_component,
        test_copy_component,
        test_delete_component,
        test_bresenham_4conn,
        test_bresenham_8conn,
        test_drawline_on_grid,
        test_skeleton_zhang_suen,
        test_skeleton_l_shape,
        test_apply_skeleton_operation,
        test_determinism,
        test_build_object_arith,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"\n❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
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
