"""
Unit tests for selectors.py (WO-13).

Per implementation_plan.md lines 300-318:
- Goal: ARGMAX_COLOR, ARGMIN_NONZERO, UNIQUE_COLOR, MODE_k×k, PARITY_COLOR
- API: apply_selector_on_test(selector_type, mask, X_test, histogram) -> (color, empty_flag)
- Tests: empty mask handling, histogram recomputation, tie-breaks, receipts

Acceptance criteria:
- Recompute histogram on ΠG(X_test) for non-empty masks
- When empty_mask=True, return (None, True)
- Tie-breaks use global order (smallest canonical color)
- Deterministic (no randomness)
"""

import pytest

from arc_core.types import Pixel
from arc_laws.selectors import apply_selector_on_test


# ==============================================================================
# Test Empty Mask Handling (CRITICAL per spec lines 306-307)
# ==============================================================================


class TestEmptyMaskHandling:
    """CRITICAL: Empty mask must return (None, True) to signal T_select deletion."""

    def test_empty_mask_argmax(self):
        """Empty mask with ARGMAX returns (None, True)."""
        grid = [[1, 2], [3, 4]]
        mask = set()  # Empty mask

        color, empty_flag = apply_selector_on_test("ARGMAX", mask, grid)

        assert color is None, "Empty mask must return None for color"
        assert empty_flag is True, "Empty mask must return True for empty_flag"

    def test_empty_mask_unique(self):
        """Empty mask with UNIQUE returns (None, True)."""
        grid = [[1, 2], [3, 4]]
        mask = set()

        color, empty_flag = apply_selector_on_test("UNIQUE", mask, grid)

        assert color is None
        assert empty_flag is True

    def test_empty_mask_argmin(self):
        """Empty mask with ARGMIN_NONZERO returns (None, True)."""
        grid = [[1, 2], [3, 4]]
        mask = set()

        color, empty_flag = apply_selector_on_test("ARGMIN_NONZERO", mask, grid)

        assert color is None
        assert empty_flag is True

    def test_empty_mask_mode(self):
        """Empty mask with MODE_kxk returns (None, True)."""
        grid = [[1, 2], [3, 4]]
        mask = set()

        color, empty_flag = apply_selector_on_test("MODE_kxk", mask, grid, k=3)

        assert color is None
        assert empty_flag is True

    def test_empty_mask_parity(self):
        """Empty mask with PARITY returns (None, True)."""
        grid = [[1, 2], [3, 4]]
        mask = set()

        color, empty_flag = apply_selector_on_test("PARITY", mask, grid)

        assert color is None
        assert empty_flag is True


# ==============================================================================
# Test ARGMAX_COLOR
# ==============================================================================


class TestArgmaxColor:
    """Test ARGMAX_COLOR selector."""

    def test_argmax_simple(self):
        """ARGMAX with clear winner."""
        grid = [[1, 1], [1, 2]]
        mask = {Pixel(0, 0), Pixel(0, 1), Pixel(1, 0), Pixel(1, 1)}

        color, empty_flag = apply_selector_on_test("ARGMAX", mask, grid)

        assert color == 1, "Color 1 appears 3 times (max)"
        assert empty_flag is False

    def test_argmax_tie_smallest(self):
        """ARGMAX tie-break: smallest color (global order)."""
        grid = [[1, 2], [1, 2]]
        mask = {Pixel(0, 0), Pixel(0, 1), Pixel(1, 0), Pixel(1, 1)}

        # Color 1: 2 times, Color 2: 2 times (tie)
        color, empty_flag = apply_selector_on_test("ARGMAX", mask, grid)

        assert color == 1, "Tie-break: smallest color (1 < 2)"
        assert empty_flag is False

    def test_argmax_single_color(self):
        """ARGMAX with single color in mask."""
        grid = [[5, 5], [5, 5]]
        mask = {Pixel(0, 0), Pixel(0, 1), Pixel(1, 0)}

        color, empty_flag = apply_selector_on_test("ARGMAX", mask, grid)

        assert color == 5
        assert empty_flag is False

    def test_argmax_partial_mask(self):
        """ARGMAX with mask covering only part of grid."""
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        mask = {Pixel(0, 0), Pixel(1, 1), Pixel(2, 2)}  # Diagonal: 1, 5, 9

        color, empty_flag = apply_selector_on_test("ARGMAX", mask, grid)

        # All different, tie-break to smallest
        assert color == 1
        assert empty_flag is False

    def test_argmax_with_zeros(self):
        """ARGMAX includes color 0 (background)."""
        grid = [[0, 0], [0, 1]]
        mask = {Pixel(0, 0), Pixel(0, 1), Pixel(1, 0), Pixel(1, 1)}

        # Color 0: 3 times, Color 1: 1 time
        color, empty_flag = apply_selector_on_test("ARGMAX", mask, grid)

        assert color == 0, "Color 0 is valid for ARGMAX"
        assert empty_flag is False

    def test_argmax_out_of_bounds_ignored(self):
        """ARGMAX ignores out-of-bounds pixels in mask."""
        grid = [[1, 2]]
        mask = {Pixel(0, 0), Pixel(0, 1), Pixel(5, 5)}  # (5,5) out of bounds

        color, empty_flag = apply_selector_on_test("ARGMAX", mask, grid)

        # Only count (0,0)=1 and (0,1)=2, tie-break to 1
        assert color == 1
        assert empty_flag is False

    def test_argmax_histogram_from_mask_only_CRITICAL(self):
        """CRITICAL: Histogram computed from MASK pixels only, NOT entire grid.

        This test would FAIL if implementation uses entire grid histogram.
        """
        # Grid where entire-grid histogram gives DIFFERENT answer than mask-only histogram
        grid = [[1, 1, 1, 1, 1],  # Row 0: five 1's
                [2, 2, 2, 2, 2],  # Row 1: five 2's
                [3, 9, 9, 9, 9]]  # Row 2: one 3, four 9's

        # Mask selects ONLY row 2
        mask = {Pixel(2, 0), Pixel(2, 1), Pixel(2, 2), Pixel(2, 3), Pixel(2, 4)}

        # CORRECT (mask-only histogram): {3:1, 9:4} → argmax is 9
        # WRONG (entire-grid histogram): {1:5, 2:5, 3:1, 9:4} → argmax is 1 or 2 (tie)

        color, empty_flag = apply_selector_on_test("ARGMAX", mask, grid)

        assert color == 9, \
            f"CRITICAL BUG: Expected 9 (argmax from mask {3:1, 9:4}), got {color}. " \
            f"Implementation is using ENTIRE GRID histogram instead of MASK-ONLY histogram! " \
            f"If entire grid used, would get 1 or 2 (count=5 each). " \
            f"This violates spec: histogram must be from mask pixels only!"
        assert empty_flag is False, "Mask is non-empty"


# ==============================================================================
# Test ARGMIN_NONZERO
# ==============================================================================


class TestArgminNonzero:
    """Test ARGMIN_NONZERO selector."""

    def test_argmin_simple(self):
        """ARGMIN_NONZERO with clear winner."""
        grid = [[1, 2], [2, 2]]
        mask = {Pixel(0, 0), Pixel(0, 1), Pixel(1, 0), Pixel(1, 1)}

        # Color 1: 1 time, Color 2: 3 times
        color, empty_flag = apply_selector_on_test("ARGMIN_NONZERO", mask, grid)

        assert color == 1, "Color 1 has lowest count"
        assert empty_flag is False

    def test_argmin_tie_smallest(self):
        """ARGMIN_NONZERO tie-break: smallest color."""
        grid = [[1, 2]]
        mask = {Pixel(0, 0), Pixel(0, 1)}

        # Color 1: 1 time, Color 2: 1 time (tie)
        color, empty_flag = apply_selector_on_test("ARGMIN_NONZERO", mask, grid)

        assert color == 1, "Tie-break: smallest color"
        assert empty_flag is False

    def test_argmin_ignores_zero(self):
        """ARGMIN_NONZERO ignores color 0."""
        grid = [[0, 0], [0, 5]]
        mask = {Pixel(0, 0), Pixel(0, 1), Pixel(1, 0), Pixel(1, 1)}

        # Color 0: 3 times (ignored), Color 5: 1 time
        color, empty_flag = apply_selector_on_test("ARGMIN_NONZERO", mask, grid)

        assert color == 5, "Color 0 is ignored"
        assert empty_flag is False

    def test_argmin_only_zeros(self):
        """ARGMIN_NONZERO with only zeros returns None."""
        grid = [[0, 0], [0, 0]]
        mask = {Pixel(0, 0), Pixel(0, 1)}

        color, empty_flag = apply_selector_on_test("ARGMIN_NONZERO", mask, grid)

        assert color is None, "All zeros → no non-zero color"
        assert empty_flag is False  # Mask is non-empty, just no non-zero colors

    def test_argmin_multiple_nonzero(self):
        """ARGMIN_NONZERO with multiple non-zero colors."""
        grid = [[1, 2, 3], [1, 2, 3]]
        mask = {Pixel(0, 0), Pixel(0, 1), Pixel(0, 2), Pixel(1, 0), Pixel(1, 1), Pixel(1, 2)}

        # Color 1: 2, Color 2: 2, Color 3: 2 (all tied)
        color, empty_flag = apply_selector_on_test("ARGMIN_NONZERO", mask, grid)

        assert color == 1, "Three-way tie, smallest is 1"
        assert empty_flag is False


# ==============================================================================
# Test UNIQUE_COLOR
# ==============================================================================


class TestUniqueColor:
    """Test UNIQUE_COLOR selector."""

    def test_unique_single_color(self):
        """UNIQUE with single color returns that color."""
        grid = [[3, 3], [3, 3]]
        mask = {Pixel(0, 0), Pixel(0, 1), Pixel(1, 0)}

        color, empty_flag = apply_selector_on_test("UNIQUE", mask, grid)

        assert color == 3, "Single color 3"
        assert empty_flag is False

    def test_unique_multiple_colors(self):
        """UNIQUE with multiple colors returns None."""
        grid = [[1, 2], [3, 4]]
        mask = {Pixel(0, 0), Pixel(0, 1)}

        # Colors 1 and 2 in mask
        color, empty_flag = apply_selector_on_test("UNIQUE", mask, grid)

        assert color is None, "Multiple colors → not unique"
        assert empty_flag is False

    def test_unique_single_pixel(self):
        """UNIQUE with single pixel."""
        grid = [[7, 8]]
        mask = {Pixel(0, 0)}

        color, empty_flag = apply_selector_on_test("UNIQUE", mask, grid)

        assert color == 7
        assert empty_flag is False

    def test_unique_all_same(self):
        """UNIQUE with all pixels same color."""
        grid = [[9, 9, 9], [9, 9, 9]]
        mask = {Pixel(0, 0), Pixel(0, 1), Pixel(0, 2), Pixel(1, 0), Pixel(1, 1), Pixel(1, 2)}

        color, empty_flag = apply_selector_on_test("UNIQUE", mask, grid)

        assert color == 9
        assert empty_flag is False

    def test_unique_two_colors(self):
        """UNIQUE with exactly two colors returns None."""
        grid = [[5, 6]]
        mask = {Pixel(0, 0), Pixel(0, 1)}

        color, empty_flag = apply_selector_on_test("UNIQUE", mask, grid)

        assert color is None
        assert empty_flag is False


# ==============================================================================
# Test MODE_k×k
# ==============================================================================


class TestModeKxK:
    """Test MODE_k×k selector."""

    def test_mode_3x3_center(self):
        """MODE_3×3 centered on pixel."""
        grid = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
        mask = {Pixel(1, 1)}  # Center pixel

        # 3×3 window around (1,1): all 9 pixels
        # Color 1: 8 times, Color 2: 1 time
        color, empty_flag = apply_selector_on_test("MODE_kxk", mask, grid, k=3)

        assert color == 1, "Mode in 3×3 window is 1"
        assert empty_flag is False

    def test_mode_3x3_corner(self):
        """MODE_3×3 at corner (partial window)."""
        grid = [[1, 2], [3, 4]]
        mask = {Pixel(0, 0)}  # Top-left corner

        # 3×3 window around (0,0): covers (-1,-1) to (1,1)
        # Only (0,0), (0,1), (1,0), (1,1) are in bounds
        # Colors: 1, 2, 3, 4 (all different, tie-break to 1)
        color, empty_flag = apply_selector_on_test("MODE_kxk", mask, grid, k=3)

        assert color == 1, "Tie-break to smallest"
        assert empty_flag is False

    def test_mode_1x1(self):
        """MODE_1×1 returns pixel's own color."""
        grid = [[5, 6], [7, 8]]
        mask = {Pixel(1, 1)}

        # 1×1 window is just the pixel itself
        color, empty_flag = apply_selector_on_test("MODE_kxk", mask, grid, k=1)

        assert color == 8
        assert empty_flag is False

    def test_mode_5x5(self):
        """MODE_5×5 on larger grid."""
        grid = [
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1],
            [1, 2, 3, 2, 1],
            [1, 2, 2, 2, 1],
            [1, 1, 1, 1, 1],
        ]
        mask = {Pixel(2, 2)}  # Center

        # 5×5 window covers entire grid
        # Color 1: 16, Color 2: 8, Color 3: 1
        color, empty_flag = apply_selector_on_test("MODE_kxk", mask, grid, k=5)

        assert color == 1, "Color 1 is mode"
        assert empty_flag is False

    def test_mode_multiple_pixels_in_mask(self):
        """MODE with multiple pixels in mask aggregates histograms."""
        grid = [[1, 2], [3, 4]]
        mask = {Pixel(0, 0), Pixel(1, 1)}

        # Aggregate 3×3 windows around both (0,0) and (1,1)
        # Both windows overlap, so counts accumulate
        color, empty_flag = apply_selector_on_test("MODE_kxk", mask, grid, k=3)

        # Should return a color (aggregated mode)
        assert color is not None
        assert empty_flag is False

    def test_mode_out_of_bounds(self):
        """MODE with pixel out of bounds."""
        grid = [[1, 2]]
        mask = {Pixel(5, 5)}  # Out of bounds

        color, empty_flag = apply_selector_on_test("MODE_kxk", mask, grid, k=3)

        # No valid pixels → no histogram → None
        assert color is None
        assert empty_flag is False

    def test_mode_kxk_window_only_CRITICAL(self):
        """CRITICAL: MODE uses k×k window ONLY, NOT entire grid.

        This test would FAIL if implementation uses entire grid instead of k×k window.
        """
        # Grid 7×7 where entire-grid mode is DIFFERENT from 3×3 window mode
        grid = [
            [1, 1, 1, 1, 1, 1, 1],  # Row 0: seven 1's
            [1, 1, 1, 1, 1, 1, 1],  # Row 1: seven 1's
            [1, 1, 1, 1, 1, 1, 1],  # Row 2: seven 1's
            [1, 1, 1, 9, 1, 1, 1],  # Row 3: six 1's, one 9
            [1, 1, 1, 9, 1, 1, 1],  # Row 4: six 1's, one 9
            [1, 1, 1, 9, 1, 1, 1],  # Row 5: six 1's, one 9
            [1, 1, 1, 9, 1, 1, 1],  # Row 6: six 1's, one 9
        ]

        # Mask at center of 9's column
        mask = {Pixel(4, 3)}  # Row 4, Col 3

        # CORRECT (3×3 window around (4,3)): covers (3,2) to (5,4)
        # Window pixels: row 3: [1,9,1], row 4: [1,9,1], row 5: [1,9,1]
        # Window histogram: {1:6, 9:3} → mode is 1
        #
        # WRONG (entire 7×7 grid): {1:45, 9:4} → mode is 1
        # BUT: if we had more 9's in grid but not in window, this would catch it

        # Better test: Let's make grid have MORE 9's overall, but window has different mode
        grid = [
            [9, 9, 9, 9, 9, 9, 9],  # Row 0: seven 9's
            [9, 9, 9, 9, 9, 9, 9],  # Row 1: seven 9's
            [9, 9, 9, 9, 9, 9, 9],  # Row 2: seven 9's
            [9, 9, 1, 1, 1, 9, 9],  # Row 3: three 1's in middle
            [9, 9, 1, 1, 1, 9, 9],  # Row 4: three 1's in middle
            [9, 9, 1, 1, 1, 9, 9],  # Row 5: three 1's in middle
            [9, 9, 9, 9, 9, 9, 9],  # Row 6: seven 9's
        ]

        # Mask at center
        mask = {Pixel(4, 3)}  # Row 4, Col 3

        # CORRECT (3×3 window around (4,3)): covers (3,2) to (5,4)
        # Window: row 3: [1,1,1], row 4: [1,1,1], row 5: [1,1,1]
        # Window histogram: {1:9} → mode is 1
        #
        # WRONG (entire 7×7 grid): {1:9, 9:40} → mode is 9

        color, empty_flag = apply_selector_on_test("MODE_kxk", mask, grid, k=3)

        assert color == 1, \
            f"CRITICAL BUG: Expected 1 (mode in 3×3 window), got {color}. " \
            f"3×3 window around (4,3) has histogram {{1:9}} → mode is 1. " \
            f"If ENTIRE GRID used, histogram would be {{1:9, 9:40}} → mode is 9. " \
            f"This violates spec: MODE must use k×k window only, not entire grid!"
        assert empty_flag is False, "Mask is non-empty"


# ==============================================================================
# Test PARITY_COLOR
# ==============================================================================


class TestParityColor:
    """Test PARITY_COLOR selector."""

    def test_parity_even(self):
        """PARITY with even count returns 0."""
        grid = [[1, 2], [3, 4]]
        mask = {Pixel(0, 0), Pixel(0, 1)}  # 2 pixels (even)

        color, empty_flag = apply_selector_on_test("PARITY", mask, grid)

        assert color == 0, "Even count → 0"
        assert empty_flag is False

    def test_parity_odd(self):
        """PARITY with odd count returns 1."""
        grid = [[1, 2, 3]]
        mask = {Pixel(0, 0), Pixel(0, 1), Pixel(0, 2)}  # 3 pixels (odd)

        color, empty_flag = apply_selector_on_test("PARITY", mask, grid)

        assert color == 1, "Odd count → 1"
        assert empty_flag is False

    def test_parity_single_pixel(self):
        """PARITY with single pixel (odd)."""
        grid = [[5]]
        mask = {Pixel(0, 0)}

        color, empty_flag = apply_selector_on_test("PARITY", mask, grid)

        assert color == 1, "1 pixel (odd) → 1"
        assert empty_flag is False

    def test_parity_large_even(self):
        """PARITY with large even count."""
        grid = [[1] * 10 for _ in range(10)]  # 10×10 grid
        mask = {Pixel(r, c) for r in range(10) for c in range(10)}  # All 100 pixels

        color, empty_flag = apply_selector_on_test("PARITY", mask, grid)

        assert color == 0, "100 pixels (even) → 0"
        assert empty_flag is False

    def test_parity_large_odd(self):
        """PARITY with large odd count."""
        grid = [[1] * 10 for _ in range(10)]
        mask = {Pixel(r, c) for r in range(10) for c in range(9)}  # 90 pixels
        mask.add(Pixel(9, 9))  # Add 1 more → 91 pixels (odd)

        color, empty_flag = apply_selector_on_test("PARITY", mask, grid)

        assert color == 1, "91 pixels (odd) → 1"
        assert empty_flag is False


# ==============================================================================
# Test Determinism
# ==============================================================================


class TestDeterminism:
    """Test deterministic behavior."""

    def test_argmax_deterministic(self):
        """ARGMAX produces same result on repeated calls."""
        grid = [[1, 2, 3], [1, 2, 3]]
        mask = {Pixel(0, 0), Pixel(0, 1), Pixel(0, 2), Pixel(1, 0), Pixel(1, 1), Pixel(1, 2)}

        result1 = apply_selector_on_test("ARGMAX", mask, grid)
        result2 = apply_selector_on_test("ARGMAX", mask, grid)
        result3 = apply_selector_on_test("ARGMAX", mask, grid)

        assert result1 == result2 == result3, "Must be deterministic"

    def test_unique_deterministic(self):
        """UNIQUE produces same result on repeated calls."""
        grid = [[5, 6]]
        mask = {Pixel(0, 0), Pixel(0, 1)}

        result1 = apply_selector_on_test("UNIQUE", mask, grid)
        result2 = apply_selector_on_test("UNIQUE", mask, grid)

        assert result1 == result2

    def test_mode_deterministic(self):
        """MODE produces same result on repeated calls."""
        grid = [[1, 2], [3, 4]]
        mask = {Pixel(1, 1)}

        result1 = apply_selector_on_test("MODE_kxk", mask, grid, k=3)
        result2 = apply_selector_on_test("MODE_kxk", mask, grid, k=3)

        assert result1 == result2


# ==============================================================================
# Test Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_selector_type(self):
        """Invalid selector type raises ValueError."""
        grid = [[1, 2]]
        mask = {Pixel(0, 0)}

        with pytest.raises(ValueError, match="Invalid selector_type"):
            apply_selector_on_test("INVALID", mask, grid)

    def test_mode_missing_k(self):
        """MODE_kxk without k parameter raises ValueError."""
        grid = [[1, 2]]
        mask = {Pixel(0, 0)}

        with pytest.raises(ValueError, match="MODE_kxk requires k parameter"):
            apply_selector_on_test("MODE_kxk", mask, grid)

    def test_single_pixel_grid(self):
        """Selectors work on 1×1 grid."""
        grid = [[7]]
        mask = {Pixel(0, 0)}

        # ARGMAX
        color, _ = apply_selector_on_test("ARGMAX", mask, grid)
        assert color == 7

        # UNIQUE
        color, _ = apply_selector_on_test("UNIQUE", mask, grid)
        assert color == 7

        # PARITY
        color, _ = apply_selector_on_test("PARITY", mask, grid)
        assert color == 1  # 1 pixel (odd)

    def test_mask_entirely_out_of_bounds(self):
        """Mask with all pixels out of bounds."""
        grid = [[1, 2]]
        mask = {Pixel(10, 10), Pixel(20, 20)}  # All out of bounds

        # Should behave like empty mask (no valid pixels)
        color, empty_flag = apply_selector_on_test("ARGMAX", mask, grid)

        # Mask is non-empty (has pixels), but no pixels in bounds → no histogram → None
        assert color is None
        assert empty_flag is False  # Mask itself is not empty, just no valid pixels

    def test_histogram_parameter_ignored(self):
        """Histogram parameter is ignored (recomputed on test)."""
        grid = [[1, 2]]
        mask = {Pixel(0, 0), Pixel(0, 1)}

        # Provide misleading histogram (should be ignored)
        fake_histogram = {5: 100, 6: 200}

        color, _ = apply_selector_on_test("ARGMAX", mask, grid, histogram=fake_histogram)

        # Should recompute and get 1 (tie-break over 1 and 2)
        assert color == 1, "Histogram parameter should be ignored"


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_band_mask_argmax(self):
        """Per spec line 316: 'band-mask argmax with non-empty mask'."""
        # Simulate a band (e.g., middle row)
        grid = [[1, 1, 1], [2, 2, 3], [4, 4, 4]]
        mask = {Pixel(1, 0), Pixel(1, 1), Pixel(1, 2)}  # Middle row

        # Middle row has: 2, 2, 3 → ARGMAX is 2
        color, empty_flag = apply_selector_on_test("ARGMAX", mask, grid)

        assert color == 2
        assert empty_flag is False

    def test_component_unique(self):
        """Per spec line 318: 'unique in component'."""
        # Simulate a component with uniform color
        grid = [[0, 0, 0], [0, 5, 5], [0, 5, 5]]
        mask = {Pixel(1, 1), Pixel(1, 2), Pixel(2, 1), Pixel(2, 2)}  # 2×2 block of 5's

        color, empty_flag = apply_selector_on_test("UNIQUE", mask, grid)

        assert color == 5, "Component has unique color 5"
        assert empty_flag is False

    def test_empty_on_test_deletion_signal(self):
        """Per spec lines 306-307: 'empty mask signals T_select closure to remove'."""
        # This is the CRITICAL case: mask empty on test
        grid = [[1, 2]]
        mask = set()  # Empty on test (e.g., component doesn't appear)

        color, empty_flag = apply_selector_on_test("ARGMAX", mask, grid)

        # T_select closure will check empty_flag and DELETE this expression
        assert empty_flag is True, "CRITICAL: Must signal deletion"
        assert color is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
