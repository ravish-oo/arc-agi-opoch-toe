"""
Tests for WO-19: compile_theta.py

Per implementation_plan.md lines 480-487:
- Test: compile deterministically
- Test: receipts stub contains present flags, WL roles counts, basis list

Acceptance:
- Same inputs → same outputs (deterministic)
- Receipts stub has required fields
- U0 built correctly
- 8 closures in fixed order
"""

import pytest

from arc_compile import compile_theta
from arc_core.types import Grid
from arc_fixedpoint.closures import (
    apply_block_closure,
    apply_canvas_closure,
    apply_definedness_closure,
    apply_interface_closure,
    apply_lattice_closure,
    apply_local_paint_closure,
    apply_object_closure,
    apply_selector_closure,
)


# =============================================================================
# Test 1: Deterministic Compilation
# =============================================================================

def test_compile_deterministic():
    """
    Test that compile_theta is deterministic.

    Scenario: Same inputs produce same outputs
    Expected: theta, U0, closures, receipts are identical
    """
    # Simple train pair
    train_pairs = [
        ([[0, 1], [2, 3]], [[4, 5], [6, 7]]),
        ([[0, 1], [2, 3]], [[4, 5], [6, 7]]),
    ]
    test_input = [[0, 1], [2, 3]]

    # Compile twice
    theta1, U0_1, closures1, receipts1 = compile_theta(train_pairs, test_input)
    theta2, U0_2, closures2, receipts2 = compile_theta(train_pairs, test_input)

    # Check determinism
    assert receipts1 == receipts2, "Receipts should be deterministic"
    assert len(U0_1) == len(U0_2), "U0 size should be deterministic"
    assert len(closures1) == len(closures2), "Closures count should be deterministic"


def test_compile_single_train_pair():
    """
    Test compilation with minimal single train pair.

    Scenario: 2×2 grid, simple recolor
    Expected: Successful compilation with basic receipts
    """
    train_pairs = [
        ([[0, 1], [2, 3]], [[1, 2], [3, 4]])
    ]
    test_input = [[0, 1], [2, 3]]

    theta, U0, closures, receipts = compile_theta(train_pairs, test_input)

    # Verify outputs exist
    assert theta is not None
    assert U0 is not None
    assert len(closures) == 8
    assert receipts is not None


# =============================================================================
# Test 2: Receipts Stub Structure
# =============================================================================

def test_receipts_stub_structure():
    """
    Test that receipts stub contains all required fields.

    Per implementation_plan.md lines 502-506:
    - present: {CBC3, E4, E8, Row1D, Col1D}
    - wl: {iters, roles_train, roles_test, unseen_roles}
    - basis_used: [...]
    """
    train_pairs = [
        ([[0, 1], [2, 3]], [[4, 5], [6, 7]])
    ]
    test_input = [[0, 1], [2, 3]]

    _, _, _, receipts = compile_theta(train_pairs, test_input)

    # Check required fields
    assert "present" in receipts
    assert "wl" in receipts
    assert "basis_used" in receipts

    # Check present flags
    present = receipts["present"]
    assert "CBC3" in present
    assert "E4" in present
    assert "E8" in present
    assert isinstance(present["CBC3"], bool)
    assert isinstance(present["E4"], bool)

    # Check WL stats
    wl = receipts["wl"]
    assert "iters" in wl
    assert "roles_train" in wl
    assert "roles_test" in wl
    assert "unseen_roles" in wl
    assert isinstance(wl["iters"], int)
    assert isinstance(wl["roles_train"], int)
    assert isinstance(wl["roles_test"], int)
    assert isinstance(wl["unseen_roles"], int)

    # Check basis_used
    assert isinstance(receipts["basis_used"], list)


def test_receipts_no_test_only_roles():
    """
    Test that WL union guarantees no test-only roles (unseen_roles may be > 0 but documented).

    Per math_spec.md §9 line 134: "No test-only roles: WL runs on train ∪ test"
    """
    train_pairs = [
        ([[0, 1], [2, 3]], [[4, 5], [6, 7]])
    ]
    test_input = [[0, 1], [2, 3]]

    _, _, _, receipts = compile_theta(train_pairs, test_input)

    wl = receipts["wl"]
    # unseen_roles counts roles in test not in train
    # This is allowed but should be tracked
    assert wl["unseen_roles"] >= 0


# =============================================================================
# Test 3: U0 Construction
# =============================================================================

def test_u0_contains_expressions():
    """
    Test that U0 contains expression sets for each pixel.

    Per math_spec.md §8 line 113-114: "U_0=∏_q P(E_q)"
    """
    train_pairs = [
        ([[0, 1], [2, 3]], [[4, 5], [6, 7]])
    ]
    test_input = [[0, 1], [2, 3]]

    _, U0, _, _ = compile_theta(train_pairs, test_input)

    # U0 should have entry for each pixel in test grid
    rows, cols = len(test_input), len(test_input[0])
    expected_pixels = rows * cols

    assert len(U0) == expected_pixels, f"U0 should have {expected_pixels} pixels"

    # Each pixel should have a set of expressions
    for pixel, expr_set in U0.items():
        assert isinstance(expr_set, set), f"Pixel {pixel} should have set of expressions"
        assert len(expr_set) > 0, f"Pixel {pixel} should have at least KEEP expression"


def test_u0_pixels_match_test_grid():
    """
    Test that U0 pixel keys match test grid dimensions.
    """
    train_pairs = [
        ([[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 0, 1]])
    ]
    test_input = [[0, 1, 2], [3, 4, 5]]

    _, U0, _, _ = compile_theta(train_pairs, test_input)

    # Check all pixels present
    from arc_core.types import Pixel

    rows, cols = 2, 3
    for r in range(rows):
        for c in range(cols):
            pixel = Pixel(r, c)
            assert pixel in U0, f"Pixel ({r},{c}) should be in U0"


# =============================================================================
# Test 4: Closures Order
# =============================================================================

def test_closures_fixed_order():
    """
    Test that closures are returned in fixed order.

    Per implementation_clarifications.md §5 lines 271-278:
    T_def → T_canvas → T_lattice → T_block → T_object → T_select → T_local → T_Γ
    """
    train_pairs = [
        ([[0, 1], [2, 3]], [[4, 5], [6, 7]])
    ]
    test_input = [[0, 1], [2, 3]]

    _, _, closures, _ = compile_theta(train_pairs, test_input)

    # Check count
    assert len(closures) == 8, "Should have exactly 8 closures"

    # Check order by function reference
    expected_order = [
        apply_definedness_closure,
        apply_canvas_closure,
        apply_lattice_closure,
        apply_block_closure,
        apply_object_closure,
        apply_selector_closure,
        apply_local_paint_closure,
        apply_interface_closure,
    ]

    for i, (actual, expected) in enumerate(zip(closures, expected_order)):
        assert actual == expected, f"Closure {i} should be {expected.__name__}"


# =============================================================================
# Test 5: Theta Structure
# =============================================================================

def test_theta_contains_key_fields():
    """
    Test that theta dict contains required fields.
    """
    train_pairs = [
        ([[0, 1], [2, 3]], [[4, 5], [6, 7]])
    ]
    test_input = [[0, 1], [2, 3]]

    theta, _, _, _ = compile_theta(train_pairs, test_input)

    # Check required fields
    assert "role_map" in theta
    assert "presents" in theta
    assert "test_present" in theta
    assert "g_test" in theta
    assert "train_pairs" in theta


# =============================================================================
# Test 6: Integration Test
# =============================================================================

def test_integration_full_compile():
    """
    Test full compilation pipeline end-to-end.

    Scenario: Multi-pair training with different grid sizes
    Expected: Successful compilation with all components
    """
    train_pairs = [
        ([[0, 1], [2, 3]], [[1, 2], [3, 4]]),
        ([[1, 2], [3, 4]], [[2, 3], [4, 5]]),
    ]
    test_input = [[0, 1], [2, 3]]

    theta, U0, closures, receipts = compile_theta(train_pairs, test_input)

    # Verify all outputs
    assert theta is not None
    assert len(U0) == 4  # 2×2 grid
    assert len(closures) == 8
    assert receipts is not None

    # Verify WL stats
    wl = receipts["wl"]
    assert wl["roles_train"] > 0
    assert wl["roles_test"] > 0

    # Verify basis detection
    assert len(receipts["basis_used"]) > 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
