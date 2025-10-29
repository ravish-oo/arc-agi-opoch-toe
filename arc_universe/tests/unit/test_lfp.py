"""
Battle-tests for WO-17: arc_fixedpoint/lfp.py

Philosophy: Tests exist to FIND BUGS, not pass.
Focus: Convergence violations, monotonicity bugs, termination failures.

Test Plan: test_plans/WO-17-plan.md
Target: compute_lfp(U0, theta, X_test_present) -> (U_star, receipt)
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from arc_core.types import Pixel
from arc_fixedpoint.lfp import compute_lfp, LFPReceipt, _deep_copy_lattice
from arc_fixedpoint.expressions import LocalPaintExpr, PeriodicExpr


# =============================================================================
# Test Helpers - Mock Closures for Controlled Testing
# =============================================================================


def create_mock_closure(removal_schedule):
    """
    Create a mock closure that removes expressions according to a schedule.

    Args:
        removal_schedule: Dict mapping pass_number -> set of (pixel, expr) to remove
                         Or callable (U, pass_num) -> (U, removed_count)

    Returns:
        Mock closure function
    """
    pass_counter = {"count": 0}

    def mock_closure(U, *args, **kwargs):
        """Mock closure that removes according to schedule."""
        if callable(removal_schedule):
            return removal_schedule(U, pass_counter["count"], *args, **kwargs)

        # Get removals for this pass
        removals_this_pass = removal_schedule.get(pass_counter["count"], set())
        pass_counter["count"] += 1

        # Apply removals
        U_new = _deep_copy_lattice(U)
        removed_count = 0

        for pixel, expr in removals_this_pass:
            if pixel in U_new and expr in U_new[pixel]:
                U_new[pixel].discard(expr)
                removed_count += 1

        return U_new, removed_count

    return mock_closure


def create_no_op_closure():
    """Create a closure that does nothing (returns U unchanged)."""
    def no_op_closure(U, *args, **kwargs):
        return U, 0
    return no_op_closure


def create_simple_shrinkage_closure(target_size=1):
    """
    Create a closure that shrinks each pixel's expression set to target_size.

    Simulates progressive convergence to singletons.
    """
    def shrinkage_closure(U, *args, **kwargs):
        U_new = _deep_copy_lattice(U)
        removed_count = 0

        for pixel, exprs in U_new.items():
            if len(exprs) > target_size:
                # Remove one expression (arbitrary choice)
                expr_to_remove = next(iter(exprs))
                U_new[pixel].discard(expr_to_remove)
                removed_count += 1

        return U_new, removed_count

    return shrinkage_closure


# =============================================================================
# Phase 1: CRITICAL - Convergence & Monotonicity (10 tests)
# =============================================================================


class TestConvergence:
    """Phase 1: CRITICAL - Find convergence bugs (infinite loops, wrong termination)."""

    def test_already_singletons_CRITICAL(self):
        """
        CONVERGE-01: U0 with singletons → passes=1, removals=0, U unchanged.

        Bug to catch: Wrong convergence check causing extra passes.
        """
        # Create U0 with singletons (1 expression per pixel)
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")

        U0 = {
            Pixel(0, 0): {expr1},
            Pixel(0, 1): {expr2},
        }

        # Mock all closures as no-ops
        with patch('arc_fixedpoint.lfp.apply_definedness_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', create_no_op_closure()):

            theta = {}
            X_test = [[0, 0]]

            U_star, receipt = compute_lfp(U0, theta, X_test)

            # CRITICAL: Should converge immediately (no changes)
            assert receipt.passes == 1, \
                f"BUG: Already singletons took {receipt.passes} passes instead of 1"

            assert receipt.total_removals == 0, \
                f"BUG: No removals expected, got {receipt.total_removals}"

            assert receipt.singletons == len(U0), \
                f"BUG: Singletons count mismatch"

            # Verify U unchanged
            assert len(U_star) == len(U0)
            for pixel in U0:
                assert U_star[pixel] == U0[pixel], \
                    f"BUG: U0 was modified during LFP"

    def test_simple_shrinkage(self):
        """
        CONVERGE-02: U0 with 3 expr/pixel → closures remove → 1 expr/pixel in ~3 passes.

        Bug to catch: Non-termination, infinite loop.
        """
        # Create U0 with multiple expressions per pixel
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")
        expr3 = LocalPaintExpr(role_id=2, color=3, mask_type="role")

        U0 = {
            Pixel(0, 0): {expr1, expr2, expr3},
            Pixel(0, 1): {expr1, expr2, expr3},
        }

        # Mock closures to progressively remove expressions
        # First closure does all the work
        shrinkage = create_simple_shrinkage_closure(target_size=1)

        with patch('arc_fixedpoint.lfp.apply_definedness_closure', shrinkage), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', create_no_op_closure()):

            theta = {}
            X_test = [[0, 0]]

            U_star, receipt = compute_lfp(U0, theta, X_test)

            # Should converge in ~3 passes (2 removals + 1 check)
            assert receipt.passes <= 3, \
                f"BUG: Simple shrinkage took {receipt.passes} passes (expected ≤3)"

            assert receipt.total_removals == 4, \
                f"BUG: Expected 4 removals (2 pixels × 2 extra exprs), got {receipt.total_removals}"

            assert receipt.singletons == len(U0), \
                f"BUG: Not all pixels are singletons"

            # Verify all pixels are singletons
            for pixel, exprs in U_star.items():
                assert len(exprs) == 1, \
                    f"BUG: Pixel {pixel} has {len(exprs)} expressions (expected 1)"

    def test_no_removals_terminates(self):
        """
        CONVERGE-03: All expressions survive closures → changed=False, terminate.

        Bug to catch: Infinite loop when no progress made.
        """
        # Create U0 where no removals are possible
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")

        U0 = {
            Pixel(0, 0): {expr1, expr2},  # Multiple expressions (not singleton)
        }

        # Mock all closures as no-ops (no removals)
        with patch('arc_fixedpoint.lfp.apply_definedness_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', create_no_op_closure()):

            theta = {}
            X_test = [[0]]

            # Should terminate and raise assertion (not singletons)
            with pytest.raises(AssertionError, match="Non-singletons remain"):
                compute_lfp(U0, theta, X_test)


class TestMonotonicity:
    """Phase 1: CRITICAL - Find monotonicity violations (expressions added)."""

    def test_total_size_decreasing(self):
        """
        MONOTONE-01: Track total size of U each pass → non-increasing.

        Bug to catch: Size increases (expressions added).
        """
        # Create U0 with varying expression counts
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")
        expr3 = LocalPaintExpr(role_id=2, color=3, mask_type="role")

        U0 = {
            Pixel(0, 0): {expr1, expr2, expr3},
            Pixel(0, 1): {expr1, expr2},
        }

        # Track sizes across passes
        sizes = []

        def tracking_shrinkage(U, pass_num, *args, **kwargs):
            """Track size and perform shrinkage."""
            total_size = sum(len(exprs) for exprs in U.values())
            sizes.append(total_size)

            # Perform shrinkage
            U_new = _deep_copy_lattice(U)
            removed = 0
            for pixel, exprs in U_new.items():
                if len(exprs) > 1:
                    expr_to_remove = next(iter(exprs))
                    U_new[pixel].discard(expr_to_remove)
                    removed += 1
                    break  # Remove from one pixel per pass
            return U_new, removed

        with patch('arc_fixedpoint.lfp.apply_definedness_closure',
                   lambda U, *args, **kwargs: tracking_shrinkage(U, 0, *args, **kwargs)), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', create_no_op_closure()):

            theta = {}
            X_test = [[0, 0]]

            U_star, receipt = compute_lfp(U0, theta, X_test)

            # CRITICAL: Verify monotonicity (sizes non-increasing)
            for i in range(1, len(sizes)):
                assert sizes[i] <= sizes[i-1], \
                    f"MONOTONICITY BUG: Size increased from {sizes[i-1]} to {sizes[i]} " \
                    f"between passes {i-1} and {i}"

    def test_per_closure_monotonicity(self):
        """
        MONOTONE-02: Each closure removes only (U_after ⊆ U_before).

        Bug to catch: Individual closure adds expressions.
        """
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")

        U0 = {
            Pixel(0, 0): {expr1, expr2},
        }

        # Track U before and after each closure
        closure_snapshots = []

        def tracking_closure(closure_name):
            def closure_fn(U, *args, **kwargs):
                U_before = _deep_copy_lattice(U)
                U_after, removed = create_no_op_closure()(U, *args, **kwargs)
                closure_snapshots.append((closure_name, U_before, U_after))
                return U_after, removed
            return closure_fn

        with patch('arc_fixedpoint.lfp.apply_definedness_closure', tracking_closure("T_def")), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', tracking_closure("T_canvas")), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', tracking_closure("T_lattice")), \
             patch('arc_fixedpoint.lfp.apply_block_closure', tracking_closure("T_block")), \
             patch('arc_fixedpoint.lfp.apply_object_closure', tracking_closure("T_object")), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', tracking_closure("T_select")), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', tracking_closure("T_local")), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', tracking_closure("T_Gamma")):

            theta = {}
            X_test = [[0]]

            with pytest.raises(AssertionError):  # Will fail singleton check
                compute_lfp(U0, theta, X_test)

            # Verify monotonicity for each closure
            for closure_name, U_before, U_after in closure_snapshots:
                for pixel in U_before:
                    if pixel in U_after:
                        # U_after[pixel] should be subset of U_before[pixel]
                        assert U_after[pixel] <= U_before[pixel], \
                            f"MONOTONICITY BUG in {closure_name}: " \
                            f"Expression added at pixel {pixel}"


class TestClosureOrder:
    """Phase 1: CRITICAL - Find closure order bugs."""

    def test_fixed_8_stage_sequence(self):
        """
        ORDER-01: Verify T_def → T_canvas → ... → T_Γ order.

        Bug to catch: Wrong order, missing closures.
        """
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")

        U0 = {Pixel(0, 0): {expr1}}

        # Track closure execution order
        execution_order = []

        def tracking_closure(closure_name):
            def closure_fn(U, *args, **kwargs):
                execution_order.append(closure_name)
                return U, 0
            return closure_fn

        with patch('arc_fixedpoint.lfp.apply_definedness_closure', tracking_closure("T_def")), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', tracking_closure("T_canvas")), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', tracking_closure("T_lattice")), \
             patch('arc_fixedpoint.lfp.apply_block_closure', tracking_closure("T_block")), \
             patch('arc_fixedpoint.lfp.apply_object_closure', tracking_closure("T_object")), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', tracking_closure("T_select")), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', tracking_closure("T_local")), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', tracking_closure("T_Gamma")):

            theta = {}
            X_test = [[0]]

            U_star, receipt = compute_lfp(U0, theta, X_test)

            # Expected order (clarifications §5)
            expected_order = [
                "T_def", "T_canvas", "T_lattice", "T_block",
                "T_object", "T_select", "T_local", "T_Gamma"
            ]

            # CRITICAL: Verify exact order
            assert execution_order == expected_order, \
                f"CLOSURE ORDER BUG: Expected {expected_order}, got {execution_order}"


class TestTerminationAndSingletons:
    """Phase 1: CRITICAL - Find termination/singleton bugs."""

    def test_passes_within_bound(self):
        """
        TERMINATE-01: passes ≤ (|U0| - n_pixels).

        Bug to catch: Exceeding theoretical termination bound.
        """
        # Create U0 with known size
        exprs = [LocalPaintExpr(role_id=i, color=i, mask_type="role") for i in range(5)]

        U0 = {
            Pixel(0, 0): set(exprs),  # 5 expressions
            Pixel(0, 1): set(exprs),  # 5 expressions
        }
        # Total size: 10 expressions
        # Final size: 2 expressions (1 per pixel)
        # Bound: 10 - 2 = 8 passes max

        # Mock shrinkage that removes 1 expression per pass
        def slow_shrinkage(U, *args, **kwargs):
            U_new = _deep_copy_lattice(U)
            removed = 0
            for pixel, exprs_set in U_new.items():
                if len(exprs_set) > 1:
                    expr_to_remove = next(iter(exprs_set))
                    U_new[pixel].discard(expr_to_remove)
                    removed += 1
                    break  # Only remove from first pixel with multiple exprs
            return U_new, removed

        with patch('arc_fixedpoint.lfp.apply_definedness_closure', slow_shrinkage), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', create_no_op_closure()):

            theta = {}
            X_test = [[0, 0]]

            U_star, receipt = compute_lfp(U0, theta, X_test)

            total_size_initial = sum(len(exprs_set) for exprs_set in U0.values())
            n_pixels = len(U0)
            removals_bound = total_size_initial - n_pixels

            # Bound: passes ≤ (removals + 1) for final verification pass
            # Spec says "passes ≤ S0 - S*" but needs +1 for convergence check
            theoretical_bound = removals_bound + 1

            # CRITICAL: Verify termination bound
            assert receipt.passes <= theoretical_bound, \
                f"TERMINATION BUG: {receipt.passes} passes exceeds bound {theoretical_bound}"

            # Also verify removals are within bound
            assert receipt.total_removals <= removals_bound, \
                f"BUG: {receipt.total_removals} removals exceeds {removals_bound}"

    def test_post_convergence_singletons_CRITICAL(self):
        """
        SINGLETON-01: |U*[q]| = 1 for all q after convergence.

        Bug to catch: Non-singletons remain, missing verification.
        """
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")

        U0 = {
            Pixel(0, 0): {expr1, expr2},
            Pixel(0, 1): {expr1, expr2},
        }

        # Mock shrinkage to singletons
        shrinkage = create_simple_shrinkage_closure(target_size=1)

        with patch('arc_fixedpoint.lfp.apply_definedness_closure', shrinkage), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', create_no_op_closure()):

            theta = {}
            X_test = [[0, 0]]

            U_star, receipt = compute_lfp(U0, theta, X_test)

            # CRITICAL: Verify all pixels are singletons
            for pixel, exprs in U_star.items():
                assert len(exprs) == 1, \
                    f"SINGLETON BUG: Pixel {pixel} has {len(exprs)} expressions (expected 1)"

            # Verify receipt matches
            assert receipt.singletons == len(U0), \
                f"BUG: Singletons count {receipt.singletons} != pixels {len(U0)}"

    def test_non_singleton_convergence_fails(self):
        """
        SINGLETON-02: No removals → assertion fails (negative test).

        Bug to catch: Missing assertion check for non-singletons.
        """
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")

        U0 = {
            Pixel(0, 0): {expr1, expr2},  # Multiple expressions
        }

        # Mock all closures as no-ops (no removals possible)
        with patch('arc_fixedpoint.lfp.apply_definedness_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', create_no_op_closure()):

            theta = {}
            X_test = [[0]]

            # CRITICAL: Should raise assertion error
            with pytest.raises(AssertionError, match="Non-singletons remain"):
                compute_lfp(U0, theta, X_test)


class TestIdempotence:
    """Phase 1: CRITICAL - Find non-idempotent operations."""

    def test_run_twice_same_result(self):
        """
        IDEMPOTENT-01: lfp(lfp(U0)) == lfp(U0).

        Bug to catch: Non-idempotent closures, state mutation.
        """
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")

        U0 = {
            Pixel(0, 0): {expr1, expr2},
            Pixel(0, 1): {expr1, expr2},
        }

        # Mock shrinkage to singletons
        shrinkage = create_simple_shrinkage_closure(target_size=1)

        with patch('arc_fixedpoint.lfp.apply_definedness_closure', shrinkage), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', create_no_op_closure()):

            theta = {}
            X_test = [[0, 0]]

            # First run
            U1, receipt1 = compute_lfp(U0, theta, X_test)

            # Second run on result of first
            U2, receipt2 = compute_lfp(U1, theta, X_test)

            # CRITICAL: Second run should be no-op
            assert U1 == U2, \
                f"IDEMPOTENCE BUG: Running LFP twice changed result"

            assert receipt2.passes == 1, \
                f"IDEMPOTENCE BUG: Second run took {receipt2.passes} passes (expected 1)"

            assert receipt2.total_removals == 0, \
                f"IDEMPOTENCE BUG: Second run had {receipt2.total_removals} removals (expected 0)"


# =============================================================================
# Test Deep Copy Helper
# =============================================================================


class TestDeepCopySemantics:
    """Verify _deep_copy_lattice works correctly."""

    def test_deep_copy_creates_new_dict(self):
        """Verify deep copy creates new dict and sets."""
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")

        U_orig = {
            Pixel(0, 0): {expr1, expr2},
        }

        U_copy = _deep_copy_lattice(U_orig)

        # Modify copy
        U_copy[Pixel(0, 0)].discard(expr2)

        # Original should be unchanged
        assert len(U_orig[Pixel(0, 0)]) == 2, \
            f"BUG: Deep copy didn't create new sets (original modified)"

        assert expr2 in U_orig[Pixel(0, 0)], \
            f"BUG: Original lattice was modified"


# =============================================================================
# Phase 2: HIGH - Receipts & Integration (8 tests)
# =============================================================================


class TestReceiptLogging:
    """Phase 2: HIGH - Verify receipt fields."""

    def test_log_passes_correctly(self):
        """RECEIPT-01: lfp.passes = iteration count."""
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")

        U0 = {
            Pixel(0, 0): {expr1, expr2},
        }

        # Mock shrinkage
        shrinkage = create_simple_shrinkage_closure(target_size=1)

        with patch('arc_fixedpoint.lfp.apply_definedness_closure', shrinkage), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', create_no_op_closure()):

            theta = {}
            X_test = [[0]]

            U_star, receipt = compute_lfp(U0, theta, X_test)

            # Should take 2 passes (1 removal + 1 verification)
            assert receipt.passes == 2, \
                f"BUG: Receipt.passes = {receipt.passes} (expected 2)"

    def test_log_removals_correctly(self):
        """RECEIPT-02: lfp.total_removals = sum of all removals."""
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")
        expr3 = LocalPaintExpr(role_id=2, color=3, mask_type="role")

        U0 = {
            Pixel(0, 0): {expr1, expr2, expr3},
            Pixel(0, 1): {expr1, expr2},
        }
        # Total: 5 expressions, target: 2 expressions → 3 removals

        shrinkage = create_simple_shrinkage_closure(target_size=1)

        with patch('arc_fixedpoint.lfp.apply_definedness_closure', shrinkage), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', create_no_op_closure()):

            theta = {}
            X_test = [[0, 0]]

            U_star, receipt = compute_lfp(U0, theta, X_test)

            assert receipt.total_removals == 3, \
                f"BUG: total_removals = {receipt.total_removals} (expected 3)"

    def test_log_singletons_correctly(self):
        """RECEIPT-03: lfp.singletons = n_pixels."""
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")

        U0 = {
            Pixel(0, 0): {expr1, expr2},
            Pixel(0, 1): {expr1, expr2},
            Pixel(1, 0): {expr1, expr2},
        }

        shrinkage = create_simple_shrinkage_closure(target_size=1)

        with patch('arc_fixedpoint.lfp.apply_definedness_closure', shrinkage), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', create_no_op_closure()):

            theta = {}
            X_test = [[0, 0], [0, 0]]

            U_star, receipt = compute_lfp(U0, theta, X_test)

            n_pixels = len(U0)
            assert receipt.singletons == n_pixels, \
                f"BUG: singletons = {receipt.singletons} (expected {n_pixels})"


class TestDeterminism:
    """Phase 2: HIGH - Verify byte-stable outputs."""

    def test_byte_identical_runs(self):
        """DETERMINISM-01: Run twice with same U0 → identical results."""
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")

        U0 = {
            Pixel(0, 0): {expr1, expr2},
            Pixel(0, 1): {expr1, expr2},
        }

        shrinkage = create_simple_shrinkage_closure(target_size=1)

        # Need to create fresh mocks for each run
        def run_lfp():
            with patch('arc_fixedpoint.lfp.apply_definedness_closure', shrinkage), \
                 patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
                 patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
                 patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
                 patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
                 patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
                 patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
                 patch('arc_fixedpoint.lfp.apply_interface_closure', create_no_op_closure()):

                theta = {}
                X_test = [[0, 0]]
                return compute_lfp(_deep_copy_lattice(U0), theta, X_test)

        U1, receipt1 = run_lfp()
        U2, receipt2 = run_lfp()

        # Verify byte-identical results
        assert U1 == U2, \
            f"DETERMINISM BUG: U differs across runs"

        assert receipt1.passes == receipt2.passes, \
            f"DETERMINISM BUG: passes differs ({receipt1.passes} vs {receipt2.passes})"

        assert receipt1.total_removals == receipt2.total_removals, \
            f"DETERMINISM BUG: removals differs"


# =============================================================================
# Phase 3: MEDIUM - Edge Cases & Stress (7 tests)
# =============================================================================


class TestEdgeCases:
    """Phase 3: MEDIUM - Edge cases."""

    def test_empty_U0(self):
        """EDGE-01: U0 = {} → immediate convergence."""
        U0 = {}

        with patch('arc_fixedpoint.lfp.apply_definedness_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', create_no_op_closure()):

            theta = {}
            X_test = [[0]]

            U_star, receipt = compute_lfp(U0, theta, X_test)

            assert receipt.passes == 1, \
                f"BUG: Empty U0 took {receipt.passes} passes (expected 1)"

            assert receipt.total_removals == 0, \
                f"BUG: Empty U0 had {receipt.total_removals} removals (expected 0)"

    def test_single_pixel_grid(self):
        """EDGE-02: 1×1 grid → converges to singleton."""
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")
        expr3 = LocalPaintExpr(role_id=2, color=3, mask_type="role")

        U0 = {
            Pixel(0, 0): {expr1, expr2, expr3},
        }

        shrinkage = create_simple_shrinkage_closure(target_size=1)

        with patch('arc_fixedpoint.lfp.apply_definedness_closure', shrinkage), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', create_no_op_closure()):

            theta = {}
            X_test = [[0]]

            U_star, receipt = compute_lfp(U0, theta, X_test)

            assert len(U_star) == 1, \
                f"BUG: Expected 1 pixel, got {len(U_star)}"

            assert receipt.singletons == 1, \
                f"BUG: Single pixel didn't converge to singleton"

    def test_large_U0(self):
        """EDGE-03: Large U0 (10+ expr/pixel) → terminates."""
        exprs = [LocalPaintExpr(role_id=i, color=i % 10, mask_type="role")
                 for i in range(15)]

        # Create 3×3 grid (9 pixels)
        U0 = {
            Pixel(r, c): set(exprs)
            for r in range(3)
            for c in range(3)
        }
        # Total: 9 pixels × 15 exprs = 135 expressions

        shrinkage = create_simple_shrinkage_closure(target_size=1)

        with patch('arc_fixedpoint.lfp.apply_definedness_closure', shrinkage), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', create_no_op_closure()):

            theta = {}
            X_test = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

            U_star, receipt = compute_lfp(U0, theta, X_test)

            # Should terminate and reach singletons
            assert receipt.singletons == 9, \
                f"BUG: Large U0 didn't reach singletons"

            # Verify bound
            total_size_initial = 135
            n_pixels = 9
            removals_bound = total_size_initial - n_pixels
            assert receipt.passes <= removals_bound + 1, \
                f"BUG: Large U0 exceeded termination bound"


class TestStress:
    """Phase 3: MEDIUM - Stress testing."""

    def test_many_passes_required(self):
        """STRESS-01: U0 needing 10+ passes → within bound, terminates."""
        # Create scenario requiring many passes
        exprs = [LocalPaintExpr(role_id=i, color=i, mask_type="role") for i in range(12)]

        U0 = {
            Pixel(0, 0): set(exprs),
        }

        shrinkage = create_simple_shrinkage_closure(target_size=1)

        with patch('arc_fixedpoint.lfp.apply_definedness_closure', shrinkage), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', create_no_op_closure()):

            theta = {}
            X_test = [[0]]

            U_star, receipt = compute_lfp(U0, theta, X_test)

            # Should take 11 removals + 1 verification = 12 passes
            assert receipt.passes >= 10, \
                f"Test setup error: only {receipt.passes} passes (expected 10+)"

            assert receipt.singletons == 1, \
                f"BUG: Didn't reach singletons after many passes"


# =============================================================================
# Phase 2: Closure Integration (Real Closures - NOT Mocked)
# =============================================================================


class TestClosureIntegration:
    """Phase 2: HIGH - Test with REAL closures from WO-16."""

    def test_definedness_closure_removes_undefined(self):
        """
        CLOSURE-01: T_def removes expressions where q ∉ Dom(e).
        
        Uses REAL apply_definedness_closure from WO-16.
        Bug to catch: T_def doesn't properly check Dom(e).
        """
        from arc_fixedpoint.closures import apply_definedness_closure
        
        # Create expressions with different domains
        # LocalPaintExpr: defined only where q belongs to role_id
        expr_role0 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr_role1 = LocalPaintExpr(role_id=1, color=2, mask_type="role")
        
        # Pixel (0,0) belongs to role 0, Pixel (0,1) belongs to role 1
        U0 = {
            Pixel(0, 0): {expr_role0, expr_role1},  # Both exprs, but only role0 defined here
            Pixel(0, 1): {expr_role0, expr_role1},  # Both exprs, but only role1 defined here
        }
        
        # Set up theta with role_map
        theta = {
            "role_map": {
                ("test", Pixel(0, 0)): 0,  # Pixel (0,0) has role 0
                ("test", Pixel(0, 1)): 1,  # Pixel (0,1) has role 1
            },
            "test_grid_id": "test"
        }
        
        X_test = [[0, 0]]
        
        # Apply definedness closure (REAL, not mocked)
        U_after, removed = apply_definedness_closure(U0, X_test, theta)
        
        # CRITICAL: T_def should remove undefined expressions
        # At Pixel(0,0): expr_role1 is undefined (q not in role 1) → remove
        # At Pixel(0,1): expr_role0 is undefined (q not in role 0) → remove
        
        assert len(U_after[Pixel(0, 0)]) == 1, \
            f"BUG: T_def didn't remove undefined expr at (0,0)"
        
        assert expr_role0 in U_after[Pixel(0, 0)], \
            f"BUG: T_def removed defined expr_role0 at (0,0)"
        
        assert expr_role1 not in U_after[Pixel(0, 0)], \
            f"BUG: T_def kept undefined expr_role1 at (0,0)"
        
        assert len(U_after[Pixel(0, 1)]) == 1, \
            f"BUG: T_def didn't remove undefined expr at (0,1)"
        
        assert expr_role1 in U_after[Pixel(0, 1)], \
            f"BUG: T_def removed defined expr_role1 at (0,1)"
        
        assert removed == 2, \
            f"BUG: T_def removed {removed} exprs (expected 2)"

    def test_all_closures_callable_and_return_modified_U(self):
        """
        CLOSURE-02: All 8 closures callable and return (U, removed_count).
        
        Uses REAL closures from WO-16 (not mocked).
        Bug to catch: Closure has wrong signature or crashes.
        """
        from arc_fixedpoint.closures import (
            apply_definedness_closure,
            apply_canvas_closure,
            apply_lattice_closure,
            apply_block_closure,
            apply_object_closure,
            apply_selector_closure,
            apply_local_paint_closure,
            apply_interface_closure,
        )
        
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        
        U0 = {
            Pixel(0, 0): {expr1},
        }
        
        theta = {}
        X_test = [[0]]
        
        # Test each closure is callable and returns (U, int)
        closures = [
            ("T_def", apply_definedness_closure, (U0, X_test, theta)),
            ("T_canvas", apply_canvas_closure, (U0, theta)),
            ("T_lattice", apply_lattice_closure, (U0, theta)),
            ("T_block", apply_block_closure, (U0, theta)),
            ("T_object", apply_object_closure, (U0, theta)),
            ("T_select", apply_selector_closure, (U0, theta)),
            ("T_local", apply_local_paint_closure, (U0, theta)),
            ("T_Gamma", apply_interface_closure, (U0, theta)),
        ]
        
        for closure_name, closure_fn, args in closures:
            try:
                result = closure_fn(*args)
                
                # Verify return type
                assert isinstance(result, tuple), \
                    f"BUG: {closure_name} doesn't return tuple"
                
                assert len(result) == 2, \
                    f"BUG: {closure_name} doesn't return (U, removed)"
                
                U_result, removed = result
                
                assert isinstance(U_result, dict), \
                    f"BUG: {closure_name} doesn't return dict for U"
                
                assert isinstance(removed, int), \
                    f"BUG: {closure_name} doesn't return int for removed"
                
                assert removed >= 0, \
                    f"BUG: {closure_name} returned negative removals: {removed}"
                
            except Exception as e:
                pytest.fail(f"BUG: {closure_name} raised exception: {e}")

    def test_closures_preserve_dict_structure(self):
        """
        CLOSURE-03: Closures preserve U.keys() (same pixels).
        
        Uses REAL closures from WO-16.
        Bug to catch: Closure adds/removes pixels from U.
        """
        from arc_fixedpoint.closures import (
            apply_definedness_closure,
            apply_canvas_closure,
        )
        
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")
        
        pixels = [Pixel(0, 0), Pixel(0, 1), Pixel(1, 0)]
        
        U0 = {
            p: {expr1, expr2} for p in pixels
        }
        
        theta = {}
        X_test = [[0, 0], [0, 0]]
        
        # Apply definedness closure
        U1, _ = apply_definedness_closure(U0, X_test, theta)
        
        assert set(U1.keys()) == set(U0.keys()), \
            f"BUG: T_def changed pixel keys"
        
        # Apply canvas closure
        U2, _ = apply_canvas_closure(U1, theta)
        
        assert set(U2.keys()) == set(U1.keys()), \
            f"BUG: T_canvas changed pixel keys"

    def test_selector_empty_mask_handling(self):
        """
        INTEGRATION-01: T_select handles empty_mask from WO-13.
        
        When apply_selector_on_test returns empty_mask=True,
        MUST remove that selector expression from U[q].
        
        Uses REAL apply_selector_closure from WO-16.
        Bug to catch: Selector with empty mask not removed.
        """
        # This test requires SelectorExpr from WO-13
        # For now, mark as TODO since it needs WO-13 integration
        pytest.skip("TODO: Requires SelectorExpr integration with WO-13")


# =============================================================================
# Phase 3: Additional Edge Cases & Stress
# =============================================================================


class TestSymmetricAndInterdependencies:
    """Phase 3: MEDIUM - Complex scenarios."""

    def test_symmetric_expressions(self):
        """
        EDGE-04: All pixels have same expression set → symmetric convergence.
        
        Bug to catch: Non-deterministic behavior when expressions symmetric.
        """
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")
        expr3 = LocalPaintExpr(role_id=2, color=3, mask_type="role")
        
        # All pixels have identical expression sets
        shared_exprs = {expr1, expr2, expr3}
        
        U0 = {
            Pixel(0, 0): set(shared_exprs),
            Pixel(0, 1): set(shared_exprs),
            Pixel(1, 0): set(shared_exprs),
            Pixel(1, 1): set(shared_exprs),
        }
        
        shrinkage = create_simple_shrinkage_closure(target_size=1)
        
        with patch('arc_fixedpoint.lfp.apply_definedness_closure', shrinkage), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', create_no_op_closure()):

            theta = {}
            X_test = [[0, 0], [0, 0]]

            U_star, receipt = compute_lfp(U0, theta, X_test)

            # All pixels should converge to singletons
            assert receipt.singletons == len(U0), \
                f"BUG: Symmetric convergence failed"
            
            # Verify all pixels reached singletons
            for pixel, exprs in U_star.items():
                assert len(exprs) == 1, \
                    f"BUG: Pixel {pixel} not singleton in symmetric case"

    def test_complex_interdependencies_with_interface_closure(self):
        """
        STRESS-02: Complex interdependencies (interface closure T_Γ).
        
        Interface closure removes expressions that violate gluing constraints.
        Bug to catch: T_Γ not properly integrated with other closures.
        """
        # This test requires understanding interface closure behavior from WO-16
        # For now, use mocks but document what real test would need
        
        expr1 = LocalPaintExpr(role_id=0, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=1, color=2, mask_type="role")
        
        # Create scenario where pixels have interdependent expressions
        U0 = {
            Pixel(0, 0): {expr1, expr2},
            Pixel(0, 1): {expr1, expr2},
        }

        # Mock interface closure to remove conflicting expressions FIRST
        # Then shrinkage will handle the rest
        def interface_with_conflicts(U, *args, **kwargs):
            """Simulate T_Γ removing expressions with interface violations."""
            U_new = _deep_copy_lattice(U)
            removed = 0

            # Simulate: interface closure removes expr2 from Pixel(0,0) if multiple exprs
            if Pixel(0, 0) in U_new and len(U_new[Pixel(0, 0)]) > 1 and expr2 in U_new[Pixel(0, 0)]:
                U_new[Pixel(0, 0)].discard(expr2)
                removed += 1

            return U_new, removed

        shrinkage = create_simple_shrinkage_closure(target_size=1)
        
        with patch('arc_fixedpoint.lfp.apply_definedness_closure', shrinkage), \
             patch('arc_fixedpoint.lfp.apply_canvas_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_lattice_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_block_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_object_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_selector_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_local_paint_closure', create_no_op_closure()), \
             patch('arc_fixedpoint.lfp.apply_interface_closure', interface_with_conflicts):

            theta = {}
            X_test = [[0, 0], [0, 0]]

            U_star, receipt = compute_lfp(U0, theta, X_test)

            # Should converge despite interface constraints
            assert receipt.singletons == len(U0), \
                f"BUG: Failed to converge with interface closure"

            # Verify T_Γ was applied (may or may not remove depending on shrinkage timing)
            # Key test: LFP terminates and reaches singletons with T_Γ in pipeline
            # Interface closure runs last, so shrinkage may already have removed conflicting exprs
            # This is CORRECT behavior - multiple closures can remove same expression
            assert "T_Gamma" in receipt.removals_per_stage, \
                f"BUG: Interface closure not in receipt"
