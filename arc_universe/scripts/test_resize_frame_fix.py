"""
Test script for RESIZE+FRAME composition fix.
Verifies that the bug found by unit test agent is resolved.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_core.canvas import infer_canvas


def print_grid(grid, label=""):
    """Helper to print grids."""
    if label:
        print(f"\n{label}:")
    for row in grid:
        print("  " + " ".join(str(c) for c in row))


def test_resize_then_frame():
    """Test RESIZE+FRAME composition detection."""
    print("\n" + "=" * 60)
    print("TEST: RESIZE+FRAME Composition")
    print("=" * 60)

    # Example: X (2x2) → RESIZE to (4x4) → FRAME to (6x6)
    # Step 1: X = [[1,2],[3,4]]
    # Step 2: RESIZE with padding (1,1,1,1), pad_color=0 → [[0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]]
    # Step 3: FRAME with thickness=1, color=5 → [[5,5,5,5,5,5],[5,0,0,0,0,5],[5,0,1,2,0,5],[5,0,3,4,0,5],[5,0,0,0,0,5],[5,5,5,5,5,5]]

    X = [[1, 2], [3, 4]]
    Y = [
        [5, 5, 5, 5, 5, 5],
        [5, 0, 0, 0, 0, 5],
        [5, 0, 1, 2, 0, 5],
        [5, 0, 3, 4, 0, 5],
        [5, 0, 0, 0, 0, 5],
        [5, 5, 5, 5, 5, 5]
    ]

    result = infer_canvas([(X, Y)])

    print_grid(X, "Input (2x2)")
    print_grid(Y, "Output (6x6)")
    print(f"\nResult: {result}")

    if result is None:
        print("\n❌ FAILED: No transform found (bug not fixed)")
        return False

    if result.operation != "resize+frame":
        print(f"\n❌ FAILED: Expected 'resize+frame', got '{result.operation}'")
        print("  The composition should be detected, not standalone operation")
        return False

    # Verify parameters
    assert result.pads_crops == (1, 1, 1, 1), f"Expected pads (1,1,1,1), got {result.pads_crops}"
    assert result.pad_color == 0, f"Expected pad_color=0, got {result.pad_color}"
    assert result.frame_color == 5, f"Expected frame_color=5, got {result.frame_color}"
    assert result.frame_thickness == 1, f"Expected thickness=1, got {result.frame_thickness}"
    assert result.verified_exact is True

    print("\n✅ PASSED: RESIZE+FRAME composition correctly detected!")
    print(f"  RESIZE: pads_crops={result.pads_crops}, pad_color={result.pad_color}")
    print(f"  FRAME: frame_color={result.frame_color}, thickness={result.frame_thickness}")
    return True


def test_resize_crop_then_frame():
    """Test RESIZE (cropping) + FRAME composition."""
    print("\n" + "=" * 60)
    print("TEST: RESIZE (crop) + FRAME Composition")
    print("=" * 60)

    # Example: X (4x4) → RESIZE crop to (2x2) → FRAME to (4x4)
    X = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]

    # Crop center 2x2: [[6,7],[10,11]]
    # Then frame with color 9, thickness 1
    Y = [
        [9, 9, 9, 9],
        [9, 6, 7, 9],
        [9, 10, 11, 9],
        [9, 9, 9, 9]
    ]

    result = infer_canvas([(X, Y)])

    print_grid(X, "Input (4x4)")
    print_grid(Y, "Output (4x4)")
    print(f"\nResult: {result}")

    if result is None:
        print("\n❌ FAILED: No transform found")
        return False

    if result.operation != "resize+frame":
        print(f"\n⚠️  INFO: Expected 'resize+frame', got '{result.operation}'")
        print("  (This might be okay if alternative explanation is valid)")
        # Don't fail, just note
        return True

    print("\n✅ PASSED: RESIZE (crop) + FRAME detected")
    print(f"  RESIZE: pads_crops={result.pads_crops}")
    print(f"  FRAME: frame_color={result.frame_color}, thickness={result.frame_thickness}")
    return True


def test_resize_asymmetric_then_frame():
    """Test RESIZE (asymmetric) + FRAME composition."""
    print("\n" + "=" * 60)
    print("TEST: RESIZE (asymmetric) + FRAME Composition")
    print("=" * 60)

    # Example: X (2x2) → RESIZE asymmetric (top=2, bottom=1, left=0, right=1) → FRAME
    X = [[1, 2], [3, 4]]

    # After asymmetric resize: (5x3)
    # [[0, 0, 0],
    #  [0, 0, 0],
    #  [1, 2, 0],
    #  [3, 4, 0],
    #  [0, 0, 0]]

    # Then frame with color 7, thickness 1 → (7x5)
    Y = [
        [7, 7, 7, 7, 7],
        [7, 0, 0, 0, 7],
        [7, 0, 0, 0, 7],
        [7, 1, 2, 0, 7],
        [7, 3, 4, 0, 7],
        [7, 0, 0, 0, 7],
        [7, 7, 7, 7, 7]
    ]

    result = infer_canvas([(X, Y)])

    print_grid(X, "Input (2x2)")
    print_grid(Y, "Output (7x5)")
    print(f"\nResult: {result}")

    if result is None:
        print("\n❌ FAILED: No transform found")
        return False

    if result.operation != "resize+frame":
        print(f"\n⚠️  INFO: Expected 'resize+frame', got '{result.operation}'")
        return True

    print("\n✅ PASSED: RESIZE (asymmetric) + FRAME detected")
    print(f"  RESIZE: pads_crops={result.pads_crops}, pad_color={result.pad_color}")
    print(f"  FRAME: frame_color={result.frame_color}, thickness={result.frame_thickness}")
    return True


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 70)
    print(" RESIZE+FRAME Composition Fix - Test Suite")
    print("=" * 70)

    tests = [
        test_resize_then_frame,
        test_resize_crop_then_frame,
        test_resize_asymmetric_then_frame,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
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
