"""
Battle-tests for WO-18: arc_fixedpoint/eval_unpresent.py

Philosophy: Small module, focused tests. Test evaluation and unpresent operations.

Test Plan: test_plans/WO-18-plan.md
Target: evaluate_and_unpresent(U_star, Xhat, theta, g_test) -> Y_star
"""

import pytest
import numpy as np

from arc_core.types import Pixel
from arc_core.present import D4_TRANSFORMATIONS
from arc_fixedpoint.eval_unpresent import evaluate_and_unpresent
from arc_fixedpoint.expressions import LocalPaintExpr


# =============================================================================
# Phase 1: CRITICAL - Core Functionality (6 tests)
# =============================================================================


class TestSingletonsRequirement:
    """Phase 1: CRITICAL - Singleton enforcement."""

    def test_non_singleton_raises_assertion_CRITICAL(self):
        """
        SINGLETON-01: |U[q]| > 1 → AssertionError.

        Bug to catch: Missing singleton check.
        """
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")

        # U with non-singleton (2 expressions at same pixel)
        U_non_singleton = {
            Pixel(0, 0): {expr1, expr2},  # MULTIPLE expressions
        }

        Xhat = [[0]]
        theta = {}
        g_test = "identity"

        # CRITICAL: Should raise AssertionError
        with pytest.raises(AssertionError, match="Non-singletons remain"):
            evaluate_and_unpresent(U_non_singleton, Xhat, theta, g_test)

    def test_all_singletons_passes(self):
        """
        SINGLETON-02: All singletons → no error.

        Bug to catch: Wrong singleton check (e.g., checking wrong condition).
        """
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=0, color=2, mask_type="role")

        # U with all singletons
        U_singletons = {
            Pixel(0, 0): {expr1},  # Singleton
            Pixel(0, 1): {expr2},  # Singleton
        }

        Xhat = [[0, 0]]
        theta = {}
        g_test = "identity"

        # Should not raise error
        Y_star = evaluate_and_unpresent(U_singletons, Xhat, theta, g_test)

        assert Y_star is not None, "Should return grid"
        assert len(Y_star) == 1, "Should have 1 row"
        assert len(Y_star[0]) == 2, "Should have 2 cols"


class TestEvaluationCorrectness:
    """Phase 1: CRITICAL - Exact evaluation."""

    def test_exact_evaluation_with_local_paint(self):
        """
        EVAL-01: LocalPaintExpr → Y^ has correct colors.

        Bug to catch: Wrong eval call, wrong pixel coordinates.
        """
        # Create expressions that return specific colors
        expr_color1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr_color2 = LocalPaintExpr(role_id=0, color=2, mask_type="role")
        expr_color3 = LocalPaintExpr(role_id=0, color=3, mask_type="role")
        expr_color4 = LocalPaintExpr(role_id=0, color=4, mask_type="role")

        # 2×2 grid with specific colors per pixel
        U_star = {
            Pixel(0, 0): {expr_color1},  # Color 1
            Pixel(0, 1): {expr_color2},  # Color 2
            Pixel(1, 0): {expr_color3},  # Color 3
            Pixel(1, 1): {expr_color4},  # Color 4
        }

        Xhat = [[0, 0], [0, 0]]  # Canonical grid (not used by LocalPaint)
        theta = {}
        g_test = "identity"

        Y_star = evaluate_and_unpresent(U_star, Xhat, theta, g_test)

        # CRITICAL: Verify exact colors
        assert Y_star[0][0] == 1, f"BUG: Expected color 1 at (0,0), got {Y_star[0][0]}"
        assert Y_star[0][1] == 2, f"BUG: Expected color 2 at (0,1), got {Y_star[0][1]}"
        assert Y_star[1][0] == 3, f"BUG: Expected color 3 at (1,0), got {Y_star[1][0]}"
        assert Y_star[1][1] == 4, f"BUG: Expected color 4 at (1,1), got {Y_star[1][1]}"

    def test_all_pixels_evaluated(self):
        """
        EVAL-02: Every pixel in U_star gets evaluated.

        Bug to catch: Missing pixels in evaluation loop.
        """
        # Create 3×3 grid with unique colors
        U_star = {}
        expected_colors = {}

        for r in range(3):
            for c in range(3):
                pixel = Pixel(r, c)
                color = r * 3 + c  # Unique color per pixel: 0,1,2,3,4,5,6,7,8
                expr = LocalPaintExpr(role_id=0, color=color, mask_type="role")
                U_star[pixel] = {expr}
                expected_colors[pixel] = color

        Xhat = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        theta = {}
        g_test = "identity"

        Y_star = evaluate_and_unpresent(U_star, Xhat, theta, g_test)

        # Verify every pixel was evaluated correctly
        for r in range(3):
            for c in range(3):
                expected = expected_colors[Pixel(r, c)]
                actual = Y_star[r][c]
                assert actual == expected, \
                    f"BUG: Pixel ({r},{c}) expected {expected}, got {actual}"


class TestUnpresentCorrectness:
    """Phase 1: CRITICAL - Unpresent operation."""

    def test_identity_transform_no_change(self):
        """
        UNPRESENT-01: g_test="identity" → Y* == Y^.

        Bug to catch: Wrong transformation applied.
        """
        expr = LocalPaintExpr(role_id=0, color=5, mask_type="role")

        U_star = {
            Pixel(0, 0): {expr},
            Pixel(0, 1): {expr},
        }

        Xhat = [[0, 0]]
        theta = {}
        g_test = "identity"  # No transformation

        Y_star = evaluate_and_unpresent(U_star, Xhat, theta, g_test)

        # With identity, Y* should equal Y^ (no change)
        assert Y_star[0][0] == 5, "BUG: Identity transform changed grid"
        assert Y_star[0][1] == 5, "BUG: Identity transform changed grid"

    def test_determinism_byte_identical(self):
        """
        UNPRESENT-02: Run twice → byte-identical outputs.

        Bug to catch: Non-deterministic dict iteration, random behavior.
        """
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=0, color=2, mask_type="role")

        U_star = {
            Pixel(0, 0): {expr1},
            Pixel(0, 1): {expr2},
        }

        Xhat = [[0, 0]]
        theta = {}
        g_test = "rot90"

        # Run twice
        Y1 = evaluate_and_unpresent(U_star, Xhat, theta, g_test)
        Y2 = evaluate_and_unpresent(U_star, Xhat, theta, g_test)

        # CRITICAL: Verify byte-identical
        assert Y1 == Y2, \
            f"DETERMINISM BUG: Y1 != Y2\nY1: {Y1}\nY2: {Y2}"


# =============================================================================
# Phase 2: MEDIUM - D4 Transformations (4 tests)
# =============================================================================


class TestD4Transformations:
    """Phase 2: MEDIUM - D4 transforms."""

    def test_rot90_inverse_dimensions(self):
        """
        D4-01: rot270 (inverse of rot90) produces correct dimensions.

        Bug to catch: Dimension mismatch after unpresent.
        """
        # Create 2×3 grid (non-square)
        U_star = {}
        for r in range(2):
            for c in range(3):
                expr = LocalPaintExpr(role_id=0, color=r * 3 + c, mask_type="role")
                U_star[Pixel(r, c)] = {expr}

        Xhat = [[0, 0, 0], [0, 0, 0]]  # 2 rows × 3 cols
        theta = {}
        g_test = "rot270"  # Inverse of rot90

        Y_star = evaluate_and_unpresent(U_star, Xhat, theta, g_test)

        # rot270 on 2×3 should produce 3×2
        assert len(Y_star) == 3, \
            f"BUG: rot270 on 2×3 should produce 3 rows, got {len(Y_star)}"
        assert len(Y_star[0]) == 2, \
            f"BUG: rot270 on 2×3 should produce 2 cols, got {len(Y_star[0])}"

    def test_flip_h_self_inverse(self):
        """
        D4-02: flip_h (self-inverse) unpresent correct.

        Bug to catch: Wrong flip applied.
        """
        # Create 2×2 grid with distinct values
        U_star = {
            Pixel(0, 0): {LocalPaintExpr(role_id=0, color=1, mask_type="role")},
            Pixel(0, 1): {LocalPaintExpr(role_id=0, color=2, mask_type="role")},
            Pixel(1, 0): {LocalPaintExpr(role_id=0, color=3, mask_type="role")},
            Pixel(1, 1): {LocalPaintExpr(role_id=0, color=4, mask_type="role")},
        }
        # Y^ before flip_h:
        # [[1, 2],
        #  [3, 4]]

        Xhat = [[0, 0], [0, 0]]
        theta = {}
        g_test = "flip_h"

        Y_star = evaluate_and_unpresent(U_star, Xhat, theta, g_test)

        # flip_h (horizontal flip) swaps columns:
        # [[2, 1],
        #  [4, 3]]
        assert Y_star[0][0] == 2, f"BUG: flip_h wrong at (0,0)"
        assert Y_star[0][1] == 1, f"BUG: flip_h wrong at (0,1)"
        assert Y_star[1][0] == 4, f"BUG: flip_h wrong at (1,0)"
        assert Y_star[1][1] == 3, f"BUG: flip_h wrong at (1,1)"

    def test_rot180_self_inverse(self):
        """
        D4-03: rot180 (self-inverse) unpresent correct.

        Bug to catch: Wrong rotation applied.
        """
        U_star = {
            Pixel(0, 0): {LocalPaintExpr(role_id=0, color=1, mask_type="role")},
            Pixel(0, 1): {LocalPaintExpr(role_id=0, color=2, mask_type="role")},
            Pixel(1, 0): {LocalPaintExpr(role_id=0, color=3, mask_type="role")},
            Pixel(1, 1): {LocalPaintExpr(role_id=0, color=4, mask_type="role")},
        }
        # Y^ before rot180:
        # [[1, 2],
        #  [3, 4]]

        Xhat = [[0, 0], [0, 0]]
        theta = {}
        g_test = "rot180"

        Y_star = evaluate_and_unpresent(U_star, Xhat, theta, g_test)

        # rot180 rotates 180 degrees:
        # [[4, 3],
        #  [2, 1]]
        assert Y_star[0][0] == 4, f"BUG: rot180 wrong at (0,0)"
        assert Y_star[0][1] == 3, f"BUG: rot180 wrong at (0,1)"
        assert Y_star[1][0] == 2, f"BUG: rot180 wrong at (1,0)"
        assert Y_star[1][1] == 1, f"BUG: rot180 wrong at (1,1)"

    def test_all_eight_transforms_no_crash(self):
        """
        D4-04: All 8 g_test values work without crashing.

        Bug to catch: Missing or misspelled transformation name.
        """
        U_star = {
            Pixel(0, 0): {LocalPaintExpr(role_id=0, color=1, mask_type="role")},
            Pixel(0, 1): {LocalPaintExpr(role_id=0, color=2, mask_type="role")},
        }

        Xhat = [[0, 0]]
        theta = {}

        # All 8 D4 transformation names
        all_g_test = [
            "identity",
            "rot90",
            "rot180",
            "rot270",
            "flip_h",
            "flip_v",
            "flip_diag_main",
            "flip_diag_anti",
        ]

        for g_test in all_g_test:
            try:
                Y_star = evaluate_and_unpresent(U_star, Xhat, theta, g_test)
                assert Y_star is not None, \
                    f"BUG: g_test='{g_test}' returned None"
            except KeyError as e:
                pytest.fail(f"BUG: g_test='{g_test}' not found in D4_TRANSFORMATIONS: {e}")
            except Exception as e:
                pytest.fail(f"BUG: g_test='{g_test}' raised exception: {e}")


# =============================================================================
# Phase 3: INTEGRATION (Optional) - 2 tests
# =============================================================================


class TestIntegration:
    """Phase 3: INTEGRATION - With real expressions."""

    def test_real_local_paint_expr(self):
        """
        INT-01: Real LocalPaintExpr evaluation produces ARC colors (0-9).

        Bug to catch: eval() returns invalid colors.
        """
        # Test all ARC colors (0-9)
        for color in range(10):
            expr = LocalPaintExpr(role_id=0, color=color, mask_type="role")
            U_star = {Pixel(0, 0): {expr}}

            Xhat = [[0]]
            theta = {}
            g_test = "identity"

            Y_star = evaluate_and_unpresent(U_star, Xhat, theta, g_test)

            assert Y_star[0][0] == color, \
                f"BUG: LocalPaintExpr(color={color}) returned {Y_star[0][0]}"

            # Verify color in valid ARC range
            assert 0 <= Y_star[0][0] <= 9, \
                f"BUG: Invalid ARC color {Y_star[0][0]} (must be 0-9)"

    def test_round_trip_consistency(self):
        """
        INT-02: Unpresent should be inverse of present transformation.

        This tests the mathematical property: g^{-1}(g(Y)) ≈ Y

        Note: We test the unpresent part (g^{-1}) assuming present (g) is correct.
        """
        # Create a simple 2×2 grid
        Y_original = [[1, 2], [3, 4]]

        # Test round-trip for each D4 transform and its inverse
        transform_pairs = [
            ("identity", "identity"),      # Self-inverse
            ("rot90", "rot270"),           # Inverse pair
            ("rot180", "rot180"),          # Self-inverse
            ("rot270", "rot90"),           # Inverse pair
            ("flip_h", "flip_h"),          # Self-inverse
            ("flip_v", "flip_v"),          # Self-inverse
            ("flip_diag_main", "flip_diag_main"),  # Self-inverse
            ("flip_diag_anti", "flip_diag_anti"),  # Self-inverse
        ]

        for forward_name, inverse_name in transform_pairs:
            # Apply forward transform
            Y_transformed = D4_TRANSFORMATIONS[forward_name](Y_original)

            # Apply inverse transform (simulating unpresent)
            Y_recovered = D4_TRANSFORMATIONS[inverse_name](Y_transformed)

            # Should recover original
            assert Y_recovered == Y_original, \
                f"ROUND-TRIP BUG: {forward_name} → {inverse_name} doesn't recover original\n" \
                f"Original: {Y_original}\n" \
                f"Recovered: {Y_recovered}"
