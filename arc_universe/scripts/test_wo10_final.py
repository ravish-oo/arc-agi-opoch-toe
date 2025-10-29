"""
Test script for WO-10 (object_arith.py) - Final verification.

Tests all core object arithmetic operations with deterministic verification.
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
)
from arc_core.types import Pixel


def print_grid(grid, label=""):
    """Helper to print grids."""
    if label:
        print(f"\n{label}:")
    for row in grid:
        print("  " + " ".join(str(c) for c in row))


def test_translate():
    """Test TRANSLATE operation."""
    print("\n" + "=" * 60)
    print("TEST 1: TRANSLATE")
    print("=" * 60)

    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    comp_pixels = frozenset([
        Pixel(1, 1), Pixel(1, 2),
        Pixel(2, 1), Pixel(2, 2),
    ])

    print_grid(grid, "Original")
    result = apply_translate(grid, comp_pixels, (1, 2), clear_source=True)
    print_grid(result, "After TRANSLATE by Δ=(1,2)")

    expected = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1],
    ]

    assert result == expected, "TRANSLATE failed"
    print("✅ PASSED")
    return True


def test_copy():
    """Test COPY operation."""
    print("\n" + "=" * 60)
    print("TEST 2: COPY")
    print("=" * 60)

    grid = [[0] * 6 for _ in range(4)]
    grid[1][1] = grid[1][2] = 2
    grid[2][1] = grid[2][2] = 2

    comp_pixels = frozenset([Pixel(1, 1), Pixel(1, 2), Pixel(2, 1), Pixel(2, 2)])

    print_grid(grid, "Original")
    result = apply_copy(grid, comp_pixels, (0, 3))
    print_grid(result, "After COPY by Δ=(0,3)")

    # Both original and copy should be present
    assert result[1][1] == 2 and result[1][2] == 2, "Original missing"
    assert result[1][4] == 2 and result[1][5] == 2, "Copy missing"

    print("✅ PASSED")
    return True


def test_delete():
    """Test DELETE operation."""
    print("\n" + "=" * 60)
    print("TEST 3: DELETE")
    print("=" * 60)

    grid = [
        [1, 1, 0, 2, 2],
        [1, 1, 0, 2, 2],
    ]

    comp_pixels = frozenset([Pixel(0, 3), Pixel(0, 4), Pixel(1, 3), Pixel(1, 4)])

    print_grid(grid, "Original")
    result = apply_delete(grid, comp_pixels, bg_color=0)
    print_grid(result, "After DELETE")

    expected = [
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
    ]

    assert result == expected, "DELETE failed"
    print("✅ PASSED")
    return True


def test_bresenham_4conn():
    """Test 4-connected Bresenham."""
    print("\n" + "=" * 60)
    print("TEST 4: Bresenham 4-Connected")
    print("=" * 60)

    # Horizontal line
    line = bresenham_4conn(Pixel(2, 1), Pixel(2, 5))
    print(f"Horizontal: {line}")
    assert len(line) == 5, "Wrong length"

    # Vertical line
    line = bresenham_4conn(Pixel(1, 3), Pixel(4, 3))
    print(f"Vertical: {line}")
    assert len(line) == 4, "Wrong length"

    # Diagonal (should create staircase)
    line = bresenham_4conn(Pixel(0, 0), Pixel(3, 3))
    print(f"Diagonal (staircase): {line}")

    # Verify 4-connectivity (no pure diagonals)
    for i in range(len(line) - 1):
        dr = abs(line[i+1].row - line[i].row)
        dc = abs(line[i+1].col - line[i].col)
        assert (dr == 1 and dc == 0) or (dr == 0 and dc == 1), f"Not 4-conn at step {i}"

    print("✅ PASSED")
    return True


def test_bresenham_8conn():
    """Test 8-connected Bresenham."""
    print("\n" + "=" * 60)
    print("TEST 5: Bresenham 8-Connected")
    print("=" * 60)

    # Perfect diagonal
    line = bresenham_8conn(Pixel(0, 0), Pixel(4, 4))
    print(f"Diagonal: {line}")
    expected = [Pixel(0, 0), Pixel(1, 1), Pixel(2, 2), Pixel(3, 3), Pixel(4, 4)]
    assert line == expected, f"Wrong diagonal: {line}"

    # Horizontal (should still work)
    line = bresenham_8conn(Pixel(2, 1), Pixel(2, 4))
    print(f"Horizontal: {line}")
    assert len(line) == 4, "Wrong length"

    print("✅ PASSED")
    return True


def test_drawline():
    """Test DRAWLINE operation."""
    print("\n" + "=" * 60)
    print("TEST 6: DRAWLINE")
    print("=" * 60)

    grid = [[0] * 6 for _ in range(6)]

    result = apply_drawline(grid, Pixel(1, 1), Pixel(4, 4), "8conn", color=3)
    print_grid(result, "8-conn line from (1,1) to (4,4)")

    # Verify line drawn
    assert result[1][1] == 3, "Start not drawn"
    assert result[2][2] == 3, "Middle not drawn"
    assert result[4][4] == 3, "End not drawn"

    print("✅ PASSED")
    return True


def test_skeleton():
    """Test SKELETON operation."""
    print("\n" + "=" * 60)
    print("TEST 7: SKELETON (Zhang-Suen)")
    print("=" * 60)

    # 3x3 block
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]

    comp_pixels = frozenset([
        Pixel(1, 1), Pixel(1, 2), Pixel(1, 3),
        Pixel(2, 1), Pixel(2, 2), Pixel(2, 3),
        Pixel(3, 1), Pixel(3, 2), Pixel(3, 3),
    ])

    print_grid(grid, "Original 3x3 block")
    skeleton = skeleton_zhang_suen(grid, comp_pixels, max_iterations=10)

    skel_grid = [[0] * 5 for _ in range(5)]
    for p in skeleton:
        skel_grid[p.row][p.col] = 1

    print_grid(skel_grid, "Skeleton")

    assert len(skeleton) < len(comp_pixels), "Skeleton not thinner"
    assert len(skeleton) > 0, "Skeleton empty"

    print(f"Reduced from {len(comp_pixels)} to {len(skeleton)} pixels")
    print("✅ PASSED")
    return True


def test_determinism():
    """Test determinism across multiple runs."""
    print("\n" + "=" * 60)
    print("TEST 8: Determinism (10 runs)")
    print("=" * 60)

    # Test Bresenham determinism
    p1, p2 = Pixel(0, 0), Pixel(5, 5)
    lines_4 = [bresenham_4conn(p1, p2) for _ in range(10)]
    lines_8 = [bresenham_8conn(p1, p2) for _ in range(10)]

    for i in range(1, 10):
        assert lines_4[i] == lines_4[0], f"4-conn run {i+1} differs"
        assert lines_8[i] == lines_8[0], f"8-conn run {i+1} differs"

    print("Bresenham deterministic ✅")

    # Test skeleton determinism
    grid = [[0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]]

    comp_pixels = frozenset([Pixel(1, 1), Pixel(1, 2), Pixel(1, 3),
                             Pixel(2, 1), Pixel(2, 2), Pixel(2, 3),
                             Pixel(3, 1), Pixel(3, 2), Pixel(3, 3)])

    skeletons = [skeleton_zhang_suen(grid, comp_pixels, max_iterations=10) for _ in range(10)]

    for i in range(1, 10):
        assert skeletons[i] == skeletons[0], f"Skeleton run {i+1} differs"

    print("Skeleton deterministic ✅")
    print("✅ PASSED")
    return True


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 70)
    print(" WO-10 Object Arithmetic - Final Test Suite")
    print("=" * 70)

    tests = [
        test_translate,
        test_copy,
        test_delete,
        test_bresenham_4conn,
        test_bresenham_8conn,
        test_drawline,
        test_skeleton,
        test_determinism,
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
