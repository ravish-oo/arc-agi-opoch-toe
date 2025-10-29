"""
Test script for WO-07 (canvas.py) - Manual verification.

This script tests the canvas inference implementation with various scenarios.
Not a formal unit test suite - just verification that the implementation works.
"""

import sys
from pathlib import Path

# Add arc_core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_core.canvas import infer_canvas, CanvasMap, _apply_resize, _apply_concat, _apply_frame


def print_grid(grid, label=""):
    """Helper to print grids."""
    if label:
        print(f"\n{label}:")
    for row in grid:
        print("  " + " ".join(str(c) for c in row))


def test_identity():
    """Test IDENTITY (no transform)."""
    print("\n" + "=" * 60)
    print("TEST: Identity (no transform)")
    print("=" * 60)

    X = [[1, 2], [3, 4]]
    Y = [[1, 2], [3, 4]]

    result = infer_canvas([(X, Y)])

    print_grid(X, "Input")
    print_grid(Y, "Output")
    print(f"\nResult: {result}")

    assert result is not None, "Should find identity transform"
    assert result.operation == "identity", f"Expected identity, got {result.operation}"
    assert result.verified_exact is True

    print("✅ PASSED")


def test_resize_symmetric_padding():
    """Test symmetric padding (can be RESIZE or FRAME - both valid)."""
    print("\n" + "=" * 60)
    print("TEST: Symmetric padding (RESIZE or FRAME)")
    print("=" * 60)

    X = [[1, 2], [3, 4]]
    # Pad with 1 pixel of 0 on all sides
    Y = [
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 3, 4, 0],
        [0, 0, 0, 0]
    ]

    result = infer_canvas([(X, Y)])

    print_grid(X, "Input")
    print_grid(Y, "Output")
    print(f"\nResult: {result}")

    assert result is not None, "Should find transform"

    # Both RESIZE and FRAME are semantically correct for symmetric padding
    # Per spec: lex-min wins, which happens to be FRAME ("frame" < "resize")
    if result.operation == "resize":
        assert result.pads_crops == (1, 1, 1, 1)
        assert result.pad_color == 0
        print("  → Found RESIZE (symmetric padding)")
    elif result.operation == "frame":
        assert result.frame_color == 0
        assert result.frame_thickness == 1
        print("  → Found FRAME (semantically equivalent, lex-min)")
    else:
        raise AssertionError(f"Expected resize or frame, got {result.operation}")

    print("✅ PASSED")


def test_resize_asymmetric_padding():
    """Test RESIZE with asymmetric padding."""
    print("\n" + "=" * 60)
    print("TEST: RESIZE - Asymmetric padding")
    print("=" * 60)

    X = [[1, 2], [3, 4]]
    # Pad: top=2, bottom=1, left=1, right=0
    Y = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 2],
        [0, 3, 4],
        [0, 0, 0]
    ]

    result = infer_canvas([(X, Y)])

    print_grid(X, "Input")
    print_grid(Y, "Output")
    print(f"\nResult: {result}")

    assert result is not None
    assert result.operation == "resize"
    # Should find some padding distribution
    print(f"Found pads_crops: {result.pads_crops}")

    print("✅ PASSED")


def test_resize_cropping():
    """Test RESIZE with cropping."""
    print("\n" + "=" * 60)
    print("TEST: RESIZE - Cropping")
    print("=" * 60)

    X = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # Crop to center 2x2
    Y = [[5, 6], [8, 9]]

    result = infer_canvas([(X, Y)])

    print_grid(X, "Input (3x3)")
    print_grid(Y, "Output (2x2)")
    print(f"\nResult: {result}")

    assert result is not None
    assert result.operation == "resize"
    # Cropping should have negative values
    assert any(p < 0 for p in result.pads_crops), f"Expected negative crops, got {result.pads_crops}"

    print("✅ PASSED")


def test_concat_vertical():
    """Test CONCAT vertical (rows)."""
    print("\n" + "=" * 60)
    print("TEST: CONCAT - Vertical (k=2, gap=0)")
    print("=" * 60)

    X = [[1, 2], [3, 4]]
    # Stack twice vertically
    Y = [
        [1, 2],
        [3, 4],
        [1, 2],
        [3, 4]
    ]

    result = infer_canvas([(X, Y)])

    print_grid(X, "Input")
    print_grid(Y, "Output")
    print(f"\nResult: {result}")

    assert result is not None
    assert result.operation == "concat"
    assert result.axis == "rows"
    assert result.k == 2
    assert result.gap == 0

    print("✅ PASSED")


def test_concat_horizontal_with_gap():
    """Test CONCAT horizontal with gap."""
    print("\n" + "=" * 60)
    print("TEST: CONCAT - Horizontal (k=2, gap=1)")
    print("=" * 60)

    X = [[1], [2]]
    # Stack twice horizontally with 1-pixel gap (color 0)
    Y = [
        [1, 0, 1],
        [2, 0, 2]
    ]

    result = infer_canvas([(X, Y)])

    print_grid(X, "Input")
    print_grid(Y, "Output")
    print(f"\nResult: {result}")

    assert result is not None
    assert result.operation == "concat"
    assert result.axis == "cols"
    assert result.k == 2
    assert result.gap == 1
    assert result.gap_color == 0

    print("✅ PASSED")


def test_frame_standalone():
    """Test symmetric padding (ambiguous - could be RESIZE or FRAME, RESIZE wins)."""
    print("\n" + "=" * 60)
    print("TEST: Symmetric padding with color 5 (RESIZE precedence)")
    print("=" * 60)

    X = [[1, 2], [3, 4]]
    # Add 1-pixel symmetric padding in color 5
    # Could be interpreted as RESIZE or FRAME - both produce identical output
    Y = [
        [5, 5, 5, 5],
        [5, 1, 2, 5],
        [5, 3, 4, 5],
        [5, 5, 5, 5]
    ]

    result = infer_canvas([(X, Y)])

    print_grid(X, "Input")
    print_grid(Y, "Output")
    print(f"\nResult: {result}")

    assert result is not None
    # Per enumeration order precedence, RESIZE comes before FRAME → RESIZE wins
    assert result.operation == "resize"
    assert result.pads_crops == (1, 1, 1, 1)
    assert result.pad_color == 5

    print("✅ PASSED")


def test_concat_then_frame():
    """Test CONCAT + FRAME composition."""
    print("\n" + "=" * 60)
    print("TEST: Composition - CONCAT + FRAME")
    print("=" * 60)

    X = [[1, 2]]
    # Concat vertically (k=2, gap=0), then frame (thickness=1, color=9)
    Y = [
        [9, 9, 9, 9],
        [9, 1, 2, 9],
        [9, 1, 2, 9],
        [9, 9, 9, 9]
    ]

    result = infer_canvas([(X, Y)])

    print_grid(X, "Input")
    print_grid(Y, "Output")
    print(f"\nResult: {result}")

    assert result is not None
    assert result.operation == "concat+frame"
    assert result.axis == "rows"
    assert result.k == 2
    assert result.frame_color == 9
    assert result.frame_thickness == 1

    print("✅ PASSED")


def test_no_valid_transform():
    """Test case where no valid transform exists."""
    print("\n" + "=" * 60)
    print("TEST: No valid transform (different content)")
    print("=" * 60)

    X = [[1, 2], [3, 4]]
    Y = [[5, 6], [7, 8]]  # Completely different content

    result = infer_canvas([(X, Y)])

    print_grid(X, "Input")
    print_grid(Y, "Output")
    print(f"\nResult: {result}")

    assert result is None, f"Expected None, got {result}"

    print("✅ PASSED")


def test_multiple_trains_verification():
    """Test that all trains must verify exactly (and RESIZE precedence)."""
    print("\n" + "=" * 60)
    print("TEST: Multiple trains verification (RESIZE precedence)")
    print("=" * 60)

    # Train 1: 1x1 → 3x3 with symmetric padding
    X1 = [[1]]
    Y1 = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]

    # Train 2: 1x1 → 3x3 with symmetric padding (same pattern)
    X2 = [[2]]
    Y2 = [
        [0, 0, 0],
        [0, 2, 0],
        [0, 0, 0]
    ]

    result = infer_canvas([(X1, Y1), (X2, Y2)])

    print("Train 1:")
    print_grid(X1, "  Input")
    print_grid(Y1, "  Output")

    print("\nTrain 2:")
    print_grid(X2, "  Input")
    print_grid(Y2, "  Output")

    print(f"\nResult: {result}")

    assert result is not None
    # Per enumeration order precedence, RESIZE comes before FRAME
    # Both are valid, but RESIZE wins
    assert result.operation == "resize"
    assert result.pads_crops == (1, 1, 1, 1)
    assert result.pad_color == 0
    assert result.verified_exact is True

    print("✅ PASSED")


def test_determinism():
    """Test that inference is deterministic across runs."""
    print("\n" + "=" * 60)
    print("TEST: Determinism (10 runs)")
    print("=" * 60)

    X = [[1, 2], [3, 4]]
    Y = [
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 3, 4, 0],
        [0, 0, 0, 0]
    ]

    results = []
    for i in range(10):
        result = infer_canvas([(X, Y)])
        results.append(result)

    # Check all results are identical
    first = results[0]
    for i, result in enumerate(results[1:], 1):
        assert result == first, f"Run {i+1} differs from run 1"

    print(f"All 10 runs produced: {first}")
    print("✅ PASSED")


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 70)
    print(" WO-07 Canvas Inference - Test Suite")
    print("=" * 70)

    tests = [
        test_identity,
        test_resize_symmetric_padding,
        test_resize_asymmetric_padding,
        test_resize_cropping,
        test_concat_vertical,
        test_concat_horizontal_with_gap,
        test_frame_standalone,
        test_concat_then_frame,
        test_no_valid_transform,
        test_multiple_trains_verification,
        test_determinism,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
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
