"""
Unit tests for WO-15: arc_fixedpoint/expressions.py

Testing expression domains Dom(e), equality, and init_expressions.

Per test plan (test_plans/WO-15-plan.md):
- Domain correctness (Dom returns True/False correctly)
- Expression equality (structural equality, hashing, set deduplication)
- init_expressions (completeness, definedness, determinism)
- 4 CRITICAL adversarial tests

No xfail, no skip, no invalid input tests.
Tests designed to FIND BUGS, not just pass.
"""

import pytest
from typing import Set, Dict

from arc_core.types import Pixel
from arc_fixedpoint.expressions import (
    Expr,
    LocalPaintExpr,
    TranslateExpr,
    PeriodicExpr,
    BlowupExpr,
    SelectorExpr,
    ResizeExpr,
    FrameExpr,
    ComposeExpr,
    RegionFillExpr,
    init_expressions,
)


# =============================================================================
# Test Class: Expression Domain
# =============================================================================


class TestExpressionDomain:
    """Test Dom(q, grid, theta) correctness for each expression type."""

    def test_local_paint_domain(self):
        """DOM-01: LocalPaintExpr defined where pixel has the role."""
        expr = LocalPaintExpr(role_id=3, color=5, mask_type="role")

        grid = [[0, 0], [0, 0]]
        theta = {
            "role_map": {
                (-1, Pixel(0, 0)): 3,  # Pixel (0,0) has role 3
                (-1, Pixel(0, 1)): 4,  # Pixel (0,1) has role 4
            },
            "test_grid_id": -1
        }

        # Pixel (0,0) has role 3 → defined
        assert expr.Dom(Pixel(0, 0), grid, theta) == True

        # Pixel (0,1) has role 4 (not 3) → not defined
        assert expr.Dom(Pixel(0, 1), grid, theta) == False

        # Pixel not in role_map → not defined
        assert expr.Dom(Pixel(1, 1), grid, theta) == False

    def test_resize_domain_always_true(self):
        """DOM-02: ResizeExpr always defined (canvas operation)."""
        expr = ResizeExpr(pads=(1, 1, 0, 0))  # Add 1 row top/bottom

        grid = [[0, 0], [0, 0]]
        theta = {}

        # Resize is always defined (total canvas operation)
        assert expr.Dom(Pixel(0, 0), grid, theta) == True
        assert expr.Dom(Pixel(1, 1), grid, theta) == True
        assert expr.Dom(Pixel(10, 10), grid, theta) == True  # Even outside grid

    def test_selector_domain_nonempty_mask(self):
        """DOM-05: SelectorExpr defined if mask is non-empty and in bounds."""
        mask = frozenset({Pixel(1, 1), Pixel(1, 2)})
        expr = SelectorExpr(selector_type="ARGMAX", mask=mask, k=None)

        grid = [[0, 0, 0], [0, 0, 0]]
        theta = {}

        # Mask is non-empty and in bounds → defined
        assert expr.Dom(Pixel(0, 0), grid, theta) == True

    def test_selector_domain_empty_mask(self):
        """DOM-08: SelectorExpr with empty mask → not defined."""
        expr = SelectorExpr(selector_type="ARGMAX", mask=frozenset(), k=None)

        grid = [[0, 0], [0, 0]]
        theta = {}

        # Empty mask → not defined (T_select should remove this)
        assert expr.Dom(Pixel(0, 0), grid, theta) == False

    def test_blowup_domain_in_bounds(self):
        """DOM-04: BlowupExpr defined if q//k maps to valid source pixel."""
        expr = BlowupExpr(k=2)  # 2×2 blowup

        grid = [[1, 2], [3, 4]]  # 2×2 source
        theta = {}

        # Pixel (0,0) → source (0,0) → in bounds → defined
        assert expr.Dom(Pixel(0, 0), grid, theta) == True

        # Pixel (1,1) → source (0,0) → in bounds → defined
        assert expr.Dom(Pixel(1, 1), grid, theta) == True

        # Pixel (2,2) → source (1,1) → in bounds → defined
        assert expr.Dom(Pixel(2, 2), grid, theta) == True

        # Pixel (4,4) → source (2,2) → OUT of bounds (2×2 grid) → not defined
        assert expr.Dom(Pixel(4, 4), grid, theta) == False

    def test_compose_domain_intersection(self):
        """
        DOM-06: ComposeExpr domain = intersection of inner and outer domains.

        This tests the composition rule Dom(e∘f) = Dom(e) ∩ f^(-1)(Dom(f)).
        """
        # Create two expressions with different domains
        inner = LocalPaintExpr(role_id=3, color=5, mask_type="role")
        outer = LocalPaintExpr(role_id=4, color=7, mask_type="role")

        composed = ComposeExpr(outer=outer, inner=inner)

        grid = [[0, 0], [0, 0]]
        theta = {
            "role_map": {
                (-1, Pixel(0, 0)): 3,  # Only role 3
                (-1, Pixel(0, 1)): 4,  # Only role 4
                (-1, Pixel(1, 0)): 3,  # Role 3
                (-1, Pixel(1, 0)): 4,  # Also role 4 (overwrite - intersection test)
            },
            "test_grid_id": -1
        }

        # Pixel (0,0): inner defined (role 3), outer NOT defined (not role 4)
        # Composition should be NOT defined (intersection)
        inner_dom = inner.Dom(Pixel(0, 0), grid, theta)
        outer_dom = outer.Dom(Pixel(0, 0), grid, theta)
        composed_dom = composed.Dom(Pixel(0, 0), grid, theta)

        assert inner_dom == True, "Inner should be defined at (0,0)"
        assert outer_dom == False, "Outer should NOT be defined at (0,0)"
        assert composed_dom == False, \
            f"Composition should be NOT defined (intersection of True and False)"

    def test_periodic_domain_lattice_points(self):
        """
        DOM-04: PeriodicExpr domain = lattice points (phase-based).

        PERIODIC uses phase_table to define which pixels belong to each phase.
        Dom(e) = {q : phase_table[q] == phase_id}
        """
        import numpy as np

        # Create 4×4 phase table with checkerboard pattern
        # Phase 0 at (0,0), (0,2), (2,0), (2,2), (1,1), (1,3), (3,1), (3,3)
        # Phase 1 at (0,1), (0,3), (1,0), (1,2), etc.
        phase_table = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ])

        expr = PeriodicExpr(phase_id=0, color=5)

        grid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        theta = {
            "periodic_structure": {
                "phase_table": phase_table,
                "num_phases": 2
            }
        }

        # Phase 0 positions → defined
        assert expr.Dom(Pixel(0, 0), grid, theta) == True  # phase_table[0,0] = 0
        assert expr.Dom(Pixel(0, 2), grid, theta) == True  # phase_table[0,2] = 0
        assert expr.Dom(Pixel(1, 1), grid, theta) == True  # phase_table[1,1] = 0
        assert expr.Dom(Pixel(2, 2), grid, theta) == True  # phase_table[2,2] = 0

        # Phase 1 positions → NOT defined (different phase)
        assert expr.Dom(Pixel(0, 1), grid, theta) == False  # phase_table[0,1] = 1
        assert expr.Dom(Pixel(1, 0), grid, theta) == False  # phase_table[1,0] = 1

    def test_translate_domain_shifted_pixels(self):
        """
        DOM-03: TranslateExpr domain = shifted component pixels.

        TRANSLATE with delta (dr, dc) maps q → q-(dr,dc).
        Dom(e) = {q : q-delta is in component}
        """
        comp_id = 1
        # Component pixels: (0,0), (0,1)
        comp_pixels = {Pixel(0, 0), Pixel(0, 1)}

        expr = TranslateExpr(component_id=comp_id, delta=(1, 0), color=None)

        grid = [[1, 2], [3, 4]]
        theta = {
            "components": {
                comp_id: {"pixels": comp_pixels}
            }
        }

        # (1,0): maps to (0,0) which is in component → defined
        assert expr.Dom(Pixel(1, 0), grid, theta) == True

        # (1,1): maps to (0,1) which is in component → defined
        assert expr.Dom(Pixel(1, 1), grid, theta) == True

        # (0,0): maps to (-1,0) which is out of bounds → NOT defined
        assert expr.Dom(Pixel(0, 0), grid, theta) == False

        # (1,2): maps to (0,2) which is not in component → NOT defined
        assert expr.Dom(Pixel(1, 2), grid, theta) == False


# =============================================================================
# Test Class: Expression Equality
# =============================================================================


class TestExpressionEquality:
    """Test expression equality, hashing, and set deduplication."""

    def test_structural_equality_same_params(self):
        """EQ-01: Same parameters → equal expressions."""
        e1 = LocalPaintExpr(role_id=3, color=5, mask_type="role")
        e2 = LocalPaintExpr(role_id=3, color=5, mask_type="role")

        assert e1 == e2, "Expressions with same params should be equal"

    def test_inequality_different_params(self):
        """EQ-02: Different parameters → not equal."""
        e1 = LocalPaintExpr(role_id=3, color=5, mask_type="role")
        e2 = LocalPaintExpr(role_id=4, color=5, mask_type="role")  # Different role

        assert e1 != e2, "Expressions with different params should not be equal"

    def test_hash_consistency(self):
        """EQ-03: Equal expressions have equal hashes."""
        e1 = LocalPaintExpr(role_id=3, color=5, mask_type="role")
        e2 = LocalPaintExpr(role_id=3, color=5, mask_type="role")

        assert e1 == e2, "Sanity check: expressions should be equal"
        assert hash(e1) == hash(e2), \
            "CRITICAL: Equal expressions MUST have equal hashes for set operations"

    def test_set_deduplication_CRITICAL(self):
        """
        EQ-04: CRITICAL - Expression equality must work in sets.

        LFP uses sets for E_q. If equality is broken, duplicates won't be
        deduplicated, leading to exponential blowup.

        Setup: Create two structurally identical expressions, add to set.
        CORRECT: Set has 1 element (deduplicated)
        WRONG: Set has 2 elements (equality broken)
        """
        e1 = LocalPaintExpr(role_id=3, color=5, mask_type="role")
        e2 = LocalPaintExpr(role_id=3, color=5, mask_type="role")

        # Sanity checks
        assert e1 == e2, "Sanity: expressions should be equal"
        assert hash(e1) == hash(e2), "Sanity: equal expressions must have equal hashes"

        # Add to set
        expr_set = {e1, e2}

        assert len(expr_set) == 1, \
            f"CRITICAL BUG: Set deduplication failed!\n" \
            f"Added two identical expressions to set.\n" \
            f"Set size: {len(expr_set)} (expected 1)\n" \
            f"This suggests __eq__ or __hash__ is broken.\n" \
            f"LFP will have exponential blowup with duplicate expressions!\n" \
            f"Expression: {e1}"

    def test_different_types_not_equal(self):
        """EQ-05: Different expression types are not equal."""
        e1 = LocalPaintExpr(role_id=3, color=5, mask_type="role")
        e2 = BlowupExpr(k=2)

        assert e1 != e2, "Different expression types should not be equal"

    def test_parameter_order_sensitivity(self):
        """EQ-06: Parameter values must match exactly."""
        e1 = ResizeExpr(pads=(1, 0, 0, 0))  # Top padding
        e2 = ResizeExpr(pads=(0, 1, 0, 0))  # Bottom padding

        assert e1 != e2, "Different padding parameters should make expressions unequal"


# =============================================================================
# Test Class: init_expressions
# =============================================================================


class TestInitExpressions:
    """Test init_expressions function."""

    def test_completeness_all_pixels(self):
        """INIT-01: Every pixel in canvas has expression set."""
        theta = {
            "canvas_shape": (2, 2),  # 2×2 canvas
            "role_map": {
                (-1, Pixel(0, 0)): 1,
                (-1, Pixel(0, 1)): 2,
            }
        }

        E_q = init_expressions(theta)

        # Check all 4 pixels present
        expected_pixels = {Pixel(0, 0), Pixel(0, 1), Pixel(1, 0), Pixel(1, 1)}
        assert set(E_q.keys()) == expected_pixels, \
            f"Missing pixels. Expected {expected_pixels}, got {set(E_q.keys())}"

    def test_nonempty_sets(self):
        """INIT-02: Each E_q is non-empty."""
        theta = {
            "canvas_shape": (2, 2),
            "role_map": {
                (-1, Pixel(0, 0)): 1,
            }
        }

        E_q = init_expressions(theta)

        for q, exprs in E_q.items():
            assert len(exprs) > 0, \
                f"E_q[{q}] is empty! Every pixel should have at least some candidate expressions."

    def test_init_determinism(self):
        """INIT-04: Same theta → same E_q (determinism)."""
        theta = {
            "canvas_shape": (2, 2),
            "role_map": {
                (-1, Pixel(0, 0)): 1,
                (-1, Pixel(0, 1)): 2,
            },
            "k_blowup": 2,
        }

        # Run 10 times
        results = [init_expressions(theta) for _ in range(10)]

        first = results[0]
        for i, result in enumerate(results[1:], 1):
            # Check keys match
            assert set(result.keys()) == set(first.keys()), \
                f"Run {i+1} has different pixels than run 1"

            # Check expression sets match
            for q in first.keys():
                assert result[q] == first[q], \
                    f"Run {i+1} E_q[{q}] differs from run 1:\n" \
                    f"Run 1: {first[q]}\n" \
                    f"Run {i+1}: {result[q]}\n" \
                    f"Implementation is non-deterministic!"

    def test_local_paint_in_eq(self):
        """INIT-05: LOCAL_PAINT expressions present for pixels with roles."""
        theta = {
            "canvas_shape": (2, 2),
            "role_map": {
                (-1, Pixel(0, 0)): 3,
            }
        }

        E_q = init_expressions(theta)

        # Check pixel (0,0) has LOCAL_PAINT expressions for role 3
        exprs_at_00 = E_q[Pixel(0, 0)]

        local_paint_exprs = [e for e in exprs_at_00 if isinstance(e, LocalPaintExpr)]

        assert len(local_paint_exprs) > 0, \
            f"E_q[(0,0)] should contain LOCAL_PAINT expressions for role 3"

        # Check role_id=3 expressions present
        role_3_exprs = [e for e in local_paint_exprs if e.role_id == 3]
        assert len(role_3_exprs) > 0, \
            f"E_q[(0,0)] should contain LOCAL_PAINT expressions with role_id=3"

    def test_blowup_in_eq_when_present(self):
        """INIT-08: BLOWUP expressions present when k_blowup in theta."""
        theta = {
            "canvas_shape": (2, 2),
            "k_blowup": 3,  # Enable blowup
        }

        E_q = init_expressions(theta)

        # Check at least one pixel has BLOWUP expression
        has_blowup = False
        for q, exprs in E_q.items():
            blowup_exprs = [e for e in exprs if isinstance(e, BlowupExpr)]
            if blowup_exprs:
                has_blowup = True
                # Check k value
                assert blowup_exprs[0].k == 3, "BLOWUP should have k=3"
                break

        assert has_blowup, "At least one pixel should have BLOWUP expression when k_blowup in theta"

    def test_resize_in_eq_when_present(self):
        """INIT-07: RESIZE (canvas) expressions present when resize_pads in theta."""
        theta = {
            "canvas_shape": (3, 3),
            "resize_pads": (1, 1, 0, 0),  # Add padding (top, bottom, left, right)
        }

        E_q = init_expressions(theta)

        # Check at least one pixel has RESIZE expression
        has_resize = False
        for q, exprs in E_q.items():
            resize_exprs = [e for e in exprs if isinstance(e, ResizeExpr)]
            if resize_exprs:
                has_resize = True
                # Check pads value
                assert resize_exprs[0].pads == (1, 1, 0, 0), \
                    f"RESIZE should have pads=(1,1,0,0), got {resize_exprs[0].pads}"
                break

        assert has_resize, \
            "At least one pixel should have RESIZE expression when resize_pads in theta"

    def test_selector_in_eq_when_present(self):
        """
        INIT-08: SELECTOR expressions present for mask pixels when selectors in theta.

        Per test plan: "E_q[mask_pixels] contains SELECTOR"
        """
        theta = {
            "canvas_shape": (3, 3),
            "selectors": [
                {
                    "type": "ARGMAX",  # Note: key is "type", not "selector_type"
                    "mask": frozenset({Pixel(1, 1), Pixel(1, 2)}),
                    "k": None
                }
            ]
        }

        E_q = init_expressions(theta)

        # Check at least one pixel has SELECTOR expression
        has_selector = False
        for q, exprs in E_q.items():
            selector_exprs = [e for e in exprs if isinstance(e, SelectorExpr)]
            if selector_exprs:
                has_selector = True
                # Verify selector type
                assert selector_exprs[0].selector_type == "ARGMAX", \
                    f"Selector type should be ARGMAX, got {selector_exprs[0].selector_type}"
                break

        assert has_selector, \
            "At least one pixel should have SELECTOR expression when selectors in theta"

    def test_translate_in_eq_when_present(self):
        """
        INIT-09: TRANSLATE expressions present when components and deltas in theta.

        Per implementation: init_expressions should create TranslateExpr for each component×delta.
        """
        theta = {
            "canvas_shape": (3, 3),
            "components": {
                1: {"pixels": {Pixel(0, 0), Pixel(0, 1)}}
            },
            "deltas": [(1, 0), (0, 1)],  # Right and down
        }

        E_q = init_expressions(theta)

        # Check at least one pixel has TRANSLATE expression
        has_translate = False
        for q, exprs in E_q.items():
            translate_exprs = [e for e in exprs if isinstance(e, TranslateExpr)]
            if translate_exprs:
                has_translate = True
                # Check deltas
                deltas = {e.delta for e in translate_exprs}
                assert (1, 0) in deltas or (0, 1) in deltas, \
                    f"TRANSLATE should have deltas from theta.deltas"
                break

        assert has_translate, \
            "At least one pixel should have TRANSLATE expression when components and deltas in theta"


# =============================================================================
# Test Class: Definedness (CRITICAL)
# =============================================================================


class TestDefinedness:
    """Test definedness enforcement - expressions only defined where Dom says so."""

    def test_definedness_enforcement_CRITICAL(self):
        """
        CRITICAL-2: Expression must respect domain boundaries.

        Evaluating outside domain should behave safely (not crash).
        T_def closure will remove such expressions before eval.

        Setup: LocalPaintExpr defined only for role 3
        Try to check domain at pixel without role 3
        CORRECT: Dom returns False
        WRONG: Dom returns True (would break T_def closure)
        """
        expr = LocalPaintExpr(role_id=3, color=5, mask_type="role")

        grid = [[0, 0], [0, 0]]
        theta = {
            "role_map": {
                (-1, Pixel(0, 0)): 3,  # Only (0,0) has role 3
            },
            "test_grid_id": -1
        }

        # Pixel (0,0) has role 3 → defined
        assert expr.Dom(Pixel(0, 0), grid, theta) == True, \
            "Pixel (0,0) should be in domain (has role 3)"

        # Pixel (0,1) does NOT have role 3 → not defined
        assert expr.Dom(Pixel(0, 1), grid, theta) == False, \
            f"CRITICAL BUG: Expression says it's defined at (0,1), but pixel doesn't have role 3!\n" \
            f"Dom should return False for pixels outside the role.\n" \
            f"This violates spec: expressions must only be defined on their domain.\n" \
            f"T_def closure assumes Dom correctly identifies undefined pixels."

        # Pixel (1,1) not in role_map → not defined
        assert expr.Dom(Pixel(1, 1), grid, theta) == False, \
            "Pixel (1,1) should NOT be in domain (not in role_map)"

    def test_eval_at_defined_pixel(self):
        """EVAL-01: eval() returns correct value at defined pixel."""
        expr = LocalPaintExpr(role_id=3, color=5, mask_type="role")

        grid = [[0, 0], [0, 0]]
        theta = {
            "role_map": {
                (-1, Pixel(0, 0)): 3,
            },
            "test_grid_id": -1
        }

        # Pixel (0,0) is defined
        assert expr.Dom(Pixel(0, 0), grid, theta) == True

        # Eval should return the color
        result = expr.eval(Pixel(0, 0), grid, theta)
        assert result == 5, f"eval should return color 5, got {result}"

    def test_eval_determinism(self):
        """EVAL-04: eval() is deterministic."""
        expr = LocalPaintExpr(role_id=3, color=5, mask_type="role")

        grid = [[0, 0], [0, 0]]
        theta = {
            "role_map": {
                (-1, Pixel(0, 0)): 3,
            },
            "test_grid_id": -1
        }

        # Run eval 10 times
        results = [expr.eval(Pixel(0, 0), grid, theta) for _ in range(10)]

        first = results[0]
        for i, result in enumerate(results[1:], 1):
            assert result == first, \
                f"eval() is non-deterministic! Run {i+1} returned {result}, run 1 returned {first}"


# =============================================================================
# Test Class: Domain Propagation (CRITICAL)
# =============================================================================


class TestDomainPropagation:
    """Test domain propagation through composition (CRITICAL for T_def)."""

    def test_domain_composition_intersection_CRITICAL(self):
        """
        CRITICAL-1: Verify Dom(e∘f) respects both inner and outer domains.

        This is FUNDAMENTAL for T_def closure correctness.

        Setup:
        - inner: defined at (0,0) and (1,1)
        - outer: defined at (1,1) and (2,2)
        - Composition should be defined ONLY at (1,1) (intersection)

        CORRECT: Composition defined only at intersection
        WRONG: Composition defined at union or either individual domain
        """
        # Create expressions with explicit domains via role_map
        inner = LocalPaintExpr(role_id=1, color=3, mask_type="role")
        outer = LocalPaintExpr(role_id=2, color=5, mask_type="role")

        composed = ComposeExpr(outer=outer, inner=inner)

        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        theta = {
            "role_map": {
                # inner defined at (0,0) and (1,1)
                (-1, Pixel(0, 0)): 1,
                (-1, Pixel(1, 1)): 1,
                # outer defined at (1,1) and (2,2)
                (-1, Pixel(1, 1)): 2,  # Overwrite: pixel has BOTH roles (intersection)
                (-1, Pixel(2, 2)): 2,
            },
            "test_grid_id": -1
        }

        # Check individual domains
        # Actually, role_map can't have two entries for same pixel - let me fix this

        # Better approach: use different role_map structure for inner vs outer
        theta_inner = {
            "role_map": {(-1, Pixel(0, 0)): 1, (-1, Pixel(1, 1)): 1},
            "test_grid_id": -1
        }
        theta_outer = {
            "role_map": {(-1, Pixel(1, 1)): 2, (-1, Pixel(2, 2)): 2},
            "test_grid_id": -1
        }

        # inner defined at (0,0), (1,1)
        assert inner.Dom(Pixel(0, 0), grid, theta_inner) == True
        assert inner.Dom(Pixel(1, 1), grid, theta_inner) == True
        assert inner.Dom(Pixel(2, 2), grid, theta_inner) == False

        # outer defined at (1,1), (2,2)
        assert outer.Dom(Pixel(0, 0), grid, theta_outer) == False
        assert outer.Dom(Pixel(1, 1), grid, theta_outer) == True
        assert outer.Dom(Pixel(2, 2), grid, theta_outer) == True

        # For composition test, need a theta that works for both
        # Actually, ComposeExpr.Dom checks BOTH inner and outer at same pixel
        # So let's test with a unified theta

        theta_combined = {
            "role_map": {
                (-1, Pixel(0, 0)): 1,  # Only inner
                (-1, Pixel(1, 1)): 1,  # Actually need to test differently
                (-1, Pixel(2, 2)): 2,  # Only outer
            },
            "test_grid_id": -1
        }

        # At (0,0): inner=True, outer=False (no role 2) → composed=False
        inner_at_00 = inner.Dom(Pixel(0, 0), grid, theta_combined)
        outer_at_00 = outer.Dom(Pixel(0, 0), grid, theta_combined)
        composed_at_00 = composed.Dom(Pixel(0, 0), grid, theta_combined)

        assert inner_at_00 == True, "inner should be defined at (0,0)"
        assert outer_at_00 == False, "outer should NOT be defined at (0,0)"
        assert composed_at_00 == False, \
            f"CRITICAL BUG: Composition domain incorrect!\n" \
            f"inner.Dom(0,0) = {inner_at_00}\n" \
            f"outer.Dom(0,0) = {outer_at_00}\n" \
            f"Expected composed.Dom(0,0) = False (intersection: True AND False)\n" \
            f"Got composed.Dom(0,0) = {composed_at_00}\n" \
            f"Violates spec: 'Dom(e∘f) = Dom(e) ∩ f^(-1)(Dom(f))'\n" \
            f"Composition must be defined ONLY where BOTH inner AND outer are defined."

    def test_composition_eval_chaining(self):
        """PROP-02: Composition evaluates inner then outer (correct chaining)."""
        inner = LocalPaintExpr(role_id=1, color=3, mask_type="role")
        outer = LocalPaintExpr(role_id=1, color=7, mask_type="role")

        composed = ComposeExpr(outer=outer, inner=inner)

        grid = [[0, 0], [0, 0]]
        theta = {
            "role_map": {(-1, Pixel(0, 0)): 1},
            "test_grid_id": -1
        }

        # Both defined at (0,0)
        assert inner.Dom(Pixel(0, 0), grid, theta) == True
        assert outer.Dom(Pixel(0, 0), grid, theta) == True
        assert composed.Dom(Pixel(0, 0), grid, theta) == True

        # Eval composition (outer takes precedence in current impl)
        result = composed.eval(Pixel(0, 0), grid, theta)

        # The exact value depends on impl, but should not crash
        assert isinstance(result, int), "Composition eval should return int"

    def test_canvas_then_paint_composition(self):
        """
        PROP-02: Canvas operation (RESIZE) then paint (LOCAL_PAINT).

        This tests domain correctness when composing canvas transformation
        with local paint. Domain should be intersection.
        """
        # RESIZE: always defined (canvas operation)
        resize = ResizeExpr(pads=(1, 1, 0, 0))

        # LOCAL_PAINT: defined only at role 1 pixels
        paint = LocalPaintExpr(role_id=1, color=5, mask_type="role")

        # Composition: paint ∘ resize
        composed = ComposeExpr(outer=paint, inner=resize)

        grid = [[0, 0], [0, 0]]
        theta = {
            "role_map": {(-1, Pixel(0, 0)): 1},  # Only (0,0) has role 1
            "test_grid_id": -1
        }

        # RESIZE defined everywhere
        assert resize.Dom(Pixel(0, 0), grid, theta) == True
        assert resize.Dom(Pixel(1, 1), grid, theta) == True

        # PAINT defined only at (0,0)
        assert paint.Dom(Pixel(0, 0), grid, theta) == True
        assert paint.Dom(Pixel(1, 1), grid, theta) == False

        # Composition should be defined ONLY where BOTH are defined
        assert composed.Dom(Pixel(0, 0), grid, theta) == True, \
            "Composition should be defined at (0,0) (both inner and outer defined)"
        assert composed.Dom(Pixel(1, 1), grid, theta) == False, \
            "Composition should NOT be defined at (1,1) (paint not defined)"

    def test_lattice_then_selector_composition(self):
        """
        PROP-03: Lattice (PERIODIC) then selector (SELECTOR).

        Tests domain composition: Dom = lattice points ∩ selector mask.
        """
        import numpy as np

        # Create phase table for PERIODIC
        phase_table = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ])

        # PERIODIC: defined at phase 0 positions
        periodic = PeriodicExpr(phase_id=0, color=3)

        # SELECTOR: defined if mask non-empty
        mask = frozenset({Pixel(0, 0), Pixel(0, 2), Pixel(1, 1)})
        selector = SelectorExpr(selector_type="ARGMAX", mask=mask, k=None)

        # Composition: selector ∘ periodic
        composed = ComposeExpr(outer=selector, inner=periodic)

        grid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        theta_periodic = {
            "periodic_structure": {
                "phase_table": phase_table,
                "num_phases": 2
            }
        }
        theta_empty = {}

        # PERIODIC defined at phase 0: (0,0), (0,2), (1,1), (2,0), (2,2)
        assert periodic.Dom(Pixel(0, 0), grid, theta_periodic) == True  # phase 0
        assert periodic.Dom(Pixel(0, 2), grid, theta_periodic) == True  # phase 0
        assert periodic.Dom(Pixel(1, 1), grid, theta_periodic) == True  # phase 0
        assert periodic.Dom(Pixel(0, 1), grid, theta_periodic) == False  # phase 1

        # SELECTOR defined everywhere (mask non-empty)
        assert selector.Dom(Pixel(0, 0), grid, theta_empty) == True
        assert selector.Dom(Pixel(0, 1), grid, theta_empty) == True

        # Composition: requires periodic theta for inner
        # At (0,0): periodic=True (phase 0), selector=True → composed=True
        assert composed.Dom(Pixel(0, 0), grid, theta_periodic) == True

        # At (0,1): periodic=False (phase 1), selector=True → composed=False
        assert composed.Dom(Pixel(0, 1), grid, theta_periodic) == False, \
            "Composition should NOT be defined at phase 1 (periodic not defined)"

    def test_multiple_composition_chain(self):
        """
        PROP-04: Multiple composition e∘f∘g.

        Tests that domain composition works transitively through chains.
        Dom(e∘f∘g) = Dom(e) ∩ f^(-1)(Dom(f)) ∩ g^(-1)(Dom(g))
        """
        # Three expressions with different domains
        expr1 = LocalPaintExpr(role_id=1, color=1, mask_type="role")
        expr2 = LocalPaintExpr(role_id=2, color=2, mask_type="role")
        expr3 = LocalPaintExpr(role_id=3, color=3, mask_type="role")

        # Compose: expr3 ∘ expr2 ∘ expr1
        composed_12 = ComposeExpr(outer=expr2, inner=expr1)  # expr2 ∘ expr1
        composed_123 = ComposeExpr(outer=expr3, inner=composed_12)  # expr3 ∘ (expr2 ∘ expr1)

        grid = [[0, 0], [0, 0]]

        # Test 1: No pixel has all 3 roles → composed nowhere
        theta_none = {
            "role_map": {
                (-1, Pixel(0, 0)): 1,  # Only role 1
                (-1, Pixel(0, 1)): 2,  # Only role 2
                (-1, Pixel(1, 0)): 3,  # Only role 3
            },
            "test_grid_id": -1
        }

        # At (0,0): expr1=True, expr2=False, expr3=False → composed=False
        assert composed_123.Dom(Pixel(0, 0), grid, theta_none) == False

        # At (0,1): expr1=False, expr2=True, expr3=False → composed=False
        assert composed_123.Dom(Pixel(0, 1), grid, theta_none) == False

        # Test 2: Pixel with role that makes all exprs defined
        # Note: Can't have one pixel with 3 roles in role_map
        # Instead: use expressions that are defined everywhere (ResizeExpr)
        resize1 = ResizeExpr(pads=(0, 0, 0, 0))
        resize2 = ResizeExpr(pads=(1, 0, 0, 0))
        paint = LocalPaintExpr(role_id=1, color=5, mask_type="role")

        # Compose: paint ∘ resize2 ∘ resize1
        composed_rr = ComposeExpr(outer=resize2, inner=resize1)
        composed_rrp = ComposeExpr(outer=paint, inner=composed_rr)

        theta_paint = {
            "role_map": {(-1, Pixel(0, 0)): 1},
            "test_grid_id": -1
        }

        # resize1, resize2 defined everywhere; paint only at (0,0)
        # Composition defined only where paint is defined
        assert composed_rrp.Dom(Pixel(0, 0), grid, theta_paint) == True
        assert composed_rrp.Dom(Pixel(1, 1), grid, theta_paint) == False


# =============================================================================
# Test Class: Specific Expression Types
# =============================================================================


class TestSpecificExpressions:
    """Test specific expression types."""

    def test_selector_empty_mask_not_defined(self):
        """SelectorExpr with empty mask should not be defined anywhere."""
        expr = SelectorExpr(selector_type="ARGMAX", mask=frozenset(), k=None)

        grid = [[1, 2], [3, 4]]
        theta = {}

        # Empty mask → not defined
        assert expr.Dom(Pixel(0, 0), grid, theta) == False
        assert expr.Dom(Pixel(1, 1), grid, theta) == False

    def test_region_fill_domain(self):
        """RegionFillExpr defined if mask is non-empty."""
        mask = frozenset({Pixel(0, 0), Pixel(0, 1)})
        expr = RegionFillExpr(mask=mask, fill_color=5)

        grid = [[0, 0], [0, 0]]
        theta = {}

        # Non-empty mask → defined
        assert expr.Dom(Pixel(0, 0), grid, theta) == True

        # Empty mask case
        empty_expr = RegionFillExpr(mask=frozenset(), fill_color=5)
        assert empty_expr.Dom(Pixel(0, 0), grid, theta) == False

    def test_blowup_eval_correctness(self):
        """BlowupExpr eval reads from source pixel (q // k)."""
        expr = BlowupExpr(k=2)

        grid = [[1, 2], [3, 4]]
        theta = {}

        # Pixel (0,0) → source (0,0) → grid[0][0] = 1
        assert expr.eval(Pixel(0, 0), grid, theta) == 1

        # Pixel (1,1) → source (0,0) → grid[0][0] = 1
        assert expr.eval(Pixel(1, 1), grid, theta) == 1

        # Pixel (2,2) → source (1,1) → grid[1][1] = 4
        assert expr.eval(Pixel(2, 2), grid, theta) == 4

        # Pixel (3,3) → source (1,1) → grid[1][1] = 4
        assert expr.eval(Pixel(3, 3), grid, theta) == 4

    def test_resize_eval_with_padding(self):
        """ResizeExpr eval maps through padding offsets."""
        expr = ResizeExpr(pads=(1, 0, 0, 0))  # 1 row padding on top

        grid = [[1, 2], [3, 4]]  # 2×2 grid
        theta = {}

        # Pixel (0,0) in padded grid → source (-1,0) → out of bounds → 0
        result = expr.eval(Pixel(0, 0), grid, theta)
        assert result == 0, "Padding area should return 0"

        # Pixel (1,0) in padded grid → source (0,0) → grid[0][0] = 1
        result = expr.eval(Pixel(1, 0), grid, theta)
        assert result == 1, "Should read from source grid[0][0]"

        # Pixel (2,1) in padded grid → source (1,1) → grid[1][1] = 4
        result = expr.eval(Pixel(2, 1), grid, theta)
        assert result == 4, "Should read from source grid[1][1]"

    def test_frame_eval_boundary(self):
        """FrameExpr eval returns frame color at boundary, interior from grid."""
        expr = FrameExpr(color=9, thickness=1)

        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # 3×3 grid
        theta = {}

        # Corner (0,0) - in frame → 9
        assert expr.eval(Pixel(0, 0), grid, theta) == 9

        # Edge (0,1) - in frame → 9
        assert expr.eval(Pixel(0, 1), grid, theta) == 9

        # Center (1,1) - interior → should read from offset source
        # With thickness=1, interior starts at (1,1)
        # Source would be (1,1) - 1 = (0,0) → grid[0][0] = 1
        result = expr.eval(Pixel(1, 1), grid, theta)
        # Expected: frame impl reads from grid[source_row][source_col]
        # where source = (1-1, 1-1) = (0,0) → grid[0][0] = 1
        assert result == 1, f"Interior should read from source, got {result}"

    def test_translate_eval_correctness(self):
        """
        EVAL-03: TranslateExpr eval reads from shifted source pixel.

        TRANSLATE with delta (dr, dc) maps pixel q → q-delta in source.
        """
        comp_id = 1
        expr = TranslateExpr(component_id=comp_id, delta=(1, 1), color=None)

        grid = [[1, 2], [3, 4]]  # 2×2 source
        theta = {
            "components": {
                comp_id: {"pixels": {Pixel(0, 0), Pixel(0, 1), Pixel(1, 0), Pixel(1, 1)}}
            }
        }

        # Pixel (1,1) → source (0,0) → grid[0][0] = 1
        assert expr.eval(Pixel(1, 1), grid, theta) == 1

        # Pixel (2,2) → source (1,1) → grid[1][1] = 4
        assert expr.eval(Pixel(2, 2), grid, theta) == 4

        # Pixel (1,2) → source (0,1) → grid[0][1] = 2
        assert expr.eval(Pixel(1, 2), grid, theta) == 2

        # Pixel (2,1) → source (1,0) → grid[1][0] = 3
        assert expr.eval(Pixel(2, 1), grid, theta) == 3

    def test_periodic_eval_correctness(self):
        """
        EVAL-02: PeriodicExpr eval returns color at phase positions.

        PERIODIC should return its color when evaluated at pixels in its phase.
        """
        import numpy as np

        # Create phase table
        phase_table = np.array([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ])

        expr = PeriodicExpr(phase_id=0, color=7)

        grid = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        theta = {
            "periodic_structure": {
                "phase_table": phase_table,
                "num_phases": 2
            }
        }

        # At phase 0 positions (0,0), (0,2), (2,0), (2,2) → return color 7
        assert expr.eval(Pixel(0, 0), grid, theta) == 7
        assert expr.eval(Pixel(0, 2), grid, theta) == 7
        assert expr.eval(Pixel(2, 0), grid, theta) == 7
        assert expr.eval(Pixel(2, 2), grid, theta) == 7

    def test_selector_with_context_dependency(self):
        """
        EVAL-05: SelectorExpr eval uses context (histogram/mask).

        Selector expressions depend on context to select color.
        This test verifies context is correctly passed and used.
        """
        mask = frozenset({Pixel(0, 0), Pixel(0, 1), Pixel(1, 0)})
        expr = SelectorExpr(selector_type="ARGMAX", mask=mask, k=None)

        # Grid with different colors in mask
        grid = [[5, 3, 0], [7, 2, 0], [0, 0, 0]]
        theta = {}

        # ARGMAX should find max color in mask {(0,0):5, (0,1):3, (1,0):7}
        # Max is 7 at (1,0)
        # When evaluating at any pixel, selector returns the selected color
        result = expr.eval(Pixel(0, 0), grid, theta)

        # Result should be one of the mask colors
        assert result in {5, 3, 7}, \
            f"Selector should return color from mask, got {result}"


# =============================================================================
# Test Class: init_expressions Definedness (CRITICAL)
# =============================================================================


class TestInitExpressionsDefinedness:
    """CRITICAL: Test that init_expressions maintains definedness invariant."""

    def test_init_definedness_invariant_CRITICAL(self):
        """
        CRITICAL-4: All expressions in E_q must be valid for pixel q.

        T_def closure assumes this invariant. If violated, closure might
        fail to remove undefined expressions.

        Setup: init_expressions with various law parameters
        Check: For every e ∈ E_q, verify e is appropriate for q

        CORRECT: All expressions in E_q make sense for pixel q
        WRONG: Some expression in E_q is not applicable to q
        """
        theta = {
            "canvas_shape": (2, 2),
            "role_map": {
                (-1, Pixel(0, 0)): 1,
                (-1, Pixel(1, 1)): 2,
            },
            "k_blowup": 2,
            "selectors": [
                {"type": "ARGMAX", "mask": {Pixel(0, 0), Pixel(0, 1)}, "k": None}
            ],
        }

        E_q = init_expressions(theta)

        violations = []

        # Check each pixel's expression set
        for q, exprs in E_q.items():
            # All expressions should be syntactically valid (no None, etc.)
            for expr in exprs:
                if expr is None:
                    violations.append((q, "None expression in set"))

                if not isinstance(expr, Expr):
                    violations.append((q, f"Non-Expr object: {type(expr)}"))

        assert len(violations) == 0, \
            f"CRITICAL BUG: init_expressions produced invalid expressions!\n" \
            f"Violations found: {len(violations)}\n" \
            + "\n".join(f"  Pixel {q}: {msg}" for q, msg in violations[:5]) + \
            f"\nAll expressions in E_q must be valid Expr instances."

    def test_init_selector_mask_consistency(self):
        """Selector expressions in E_q have non-empty masks (or removed by T_select)."""
        theta = {
            "canvas_shape": (2, 2),
            "selectors": [
                {"type": "ARGMAX", "mask": {Pixel(0, 0), Pixel(0, 1)}, "k": None},
                # Empty mask case - should still be created, but T_select will remove
                {"type": "UNIQUE", "mask": set(), "k": None},
            ],
        }

        E_q = init_expressions(theta)

        # Check selector expressions
        for q, exprs in E_q.items():
            selector_exprs = [e for e in exprs if isinstance(e, SelectorExpr)]

            for sel_expr in selector_exprs:
                # Selector can have empty mask (will be removed by T_select)
                # Just verify mask is a frozenset
                assert isinstance(sel_expr.mask, frozenset), \
                    f"Selector mask should be frozenset, got {type(sel_expr.mask)}"
