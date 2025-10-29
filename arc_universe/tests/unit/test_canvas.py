"""
Unit tests for WO-07: arc_core/canvas.py

Per test plan test_plans/WO-07-plan.md:
- Test all 6 operation types: IDENTITY, RESIZE, CONCAT, FRAME, RESIZE+FRAME, CONCAT+FRAME
- Verify EXACT forward mapping (pixel-perfect on ALL trains)
- Verify lex-min selection when multiple valid candidates
- Verify bounded enumeration (k ≤ 10, gap ≤ 10)
- Verify Y observation only (no parameter leakage)

CRITICAL: All assertions verify EXACT values, not just existence.
Per WO-05 lessons: Battle-test the implementation, don't just tick checkboxes.
"""

import json
import pytest
from pathlib import Path

from arc_core.canvas import (
    infer_canvas,
    CanvasMap,
    _apply_resize,
    _apply_concat,
    _apply_frame,
    _grids_equal
)


# =============================================================================
# Fixture Loading
# =============================================================================

def load_fixture(filename: str) -> dict:
    """Load test fixture from WO-07 directory."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "WO-07" / filename
    with open(fixture_path, "r") as f:
        return json.load(f)


# =============================================================================
# Test Class 1: Identity Detection
# =============================================================================

class TestIdentity:
    """Test IDENTITY operation (X == Y)."""

    def test_identity_exact_match(self):
        """
        Verify identity detection when X == Y.

        Spec: arc_core/canvas.py lines 269-278
        """
        fixture = load_fixture("identity_2x2.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]
        expected = fixture["expected"]

        result = infer_canvas(train_pairs)

        # PROVE identity detected
        assert result is not None, \
            "Identity transform should return CanvasMap, not None"

        assert result.operation == expected["operation"], \
            f"Operation must be EXACTLY '{expected['operation']}', got '{result.operation}'"

        # PROVE verification
        assert result.verified_exact == expected["verified_exact"], \
            "Identity must be verified exact"

        # PROVE no other params set
        assert result.pads_crops is None, \
            "Identity has no pads_crops, must be None"
        assert result.axis is None, \
            "Identity has no axis, must be None"
        assert result.frame_color is None, \
            "Identity has no frame_color, must be None"
        assert result.k is None, \
            "Identity has no k, must be None"

    def test_identity_multiple_trains(self):
        """Verify identity with multiple train pairs."""
        # 3 trains, all identity
        train_pairs = [
            ([[1, 2], [3, 4]], [[1, 2], [3, 4]]),
            ([[5, 6], [7, 8]], [[5, 6], [7, 8]]),
            ([[9, 0], [1, 2]], [[9, 0], [1, 2]]),
        ]

        result = infer_canvas(train_pairs)

        # PROVE identity on all trains
        assert result is not None, "Identity should be detected"
        assert result.operation == "identity", \
            f"Operation must be 'identity', got '{result.operation}'"
        assert result.verified_exact == True, \
            "All trains match → verified_exact must be True"

    def test_not_identity_different_grids(self):
        """Verify non-identity when X != Y."""
        train_pairs = [
            ([[1, 2], [3, 4]], [[5, 6], [7, 8]])  # Different grids
        ]

        result = infer_canvas(train_pairs)

        # PROVE not identity (could be None or different operation)
        if result is not None:
            assert result.operation != "identity", \
                "X != Y should not produce identity operation"


# =============================================================================
# Test Class 2: RESIZE Operations
# =============================================================================

class TestResize:
    """Test RESIZE operations (padding/cropping)."""

    def test_resize_pad_symmetric(self):
        """
        Verify symmetric padding detection and exact parameters.

        Spec: arc_core/canvas.py lines 281-387
        """
        fixture = load_fixture("resize_pad_symmetric.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]
        expected = fixture["expected"]

        result = infer_canvas(train_pairs)

        # PROVE resize detected
        assert result is not None, \
            "Symmetric padding should be detected"

        assert result.operation == expected["operation"], \
            f"Operation must be EXACTLY '{expected['operation']}', got '{result.operation}'"

        # PROVE EXACT pads_crops (not just "is tuple")
        assert result.pads_crops == tuple(expected["pads_crops"]), \
            f"pads_crops must be EXACTLY {tuple(expected['pads_crops'])}, got {result.pads_crops}"

        # PROVE pad_color observed from Y0
        assert result.pad_color == expected["pad_color"], \
            f"pad_color must be {expected['pad_color']} (observed from Y0), got {result.pad_color}"

        # PROVE exact verification
        assert result.verified_exact == expected["verified_exact"], \
            "Transform must be verified exact on all trains"

        # PROVE forward mapping correct (pixel-perfect)
        for input_grid, output_grid in train_pairs:
            computed = _apply_resize(input_grid, result.pads_crops, result.pad_color)
            assert _grids_equal(computed, output_grid), \
                "Forward mapping X → Y must be pixel-perfect"

    def test_resize_pad_asymmetric_top_left(self):
        """Verify asymmetric padding (top-left heavy)."""
        fixture = load_fixture("pad_asymmetric_top_left.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]
        expected = fixture["expected"]

        result = infer_canvas(train_pairs)

        # PROVE exact asymmetric pads_crops
        assert result is not None, "Asymmetric padding should be detected"
        assert result.pads_crops == tuple(expected["pads_crops"]), \
            f"pads_crops must be EXACTLY {tuple(expected['pads_crops'])} (top-left heavy), got {result.pads_crops}"

        assert result.pad_color == expected["pad_color"], \
            f"pad_color must be {expected['pad_color']}, got {result.pad_color}"

        # PROVE forward mapping
        for input_grid, output_grid in train_pairs:
            computed = _apply_resize(input_grid, result.pads_crops, result.pad_color)
            assert _grids_equal(computed, output_grid), \
                "Asymmetric padding forward mapping must be exact"

    def test_resize_crop_symmetric(self):
        """Verify symmetric cropping (negative pads_crops)."""
        fixture = load_fixture("resize_crop.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]
        expected = fixture["expected"]

        result = infer_canvas(train_pairs)

        # PROVE crop detected with EXACT negative values
        assert result is not None, "Cropping should be detected"
        assert result.operation == "resize", \
            f"Operation must be 'resize', got '{result.operation}'"

        assert result.pads_crops == tuple(expected["pads_crops"]), \
            f"pads_crops must be EXACTLY {tuple(expected['pads_crops'])} (negative for crop), got {result.pads_crops}"

        # PROVE pad_color is None for cropping
        assert result.pad_color is None or result.pad_color == expected["pad_color"], \
            f"pad_color must be None for cropping, got {result.pad_color}"

        # PROVE forward mapping
        for input_grid, output_grid in train_pairs:
            computed = _apply_resize(input_grid, result.pads_crops, result.pad_color or 0)
            assert _grids_equal(computed, output_grid), \
                "Cropping forward mapping must be pixel-perfect"

    def test_resize_crop_to_empty(self):
        """Verify over-cropping produces empty grid (degenerate case)."""
        fixture = load_fixture("crop_to_empty.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]

        result = infer_canvas(train_pairs)

        # PROVE degenerate case handled
        # Over-cropping should either:
        # 1. Return None (no valid transform)
        # 2. Return resize with extreme negative pads_crops
        if result is not None:
            assert result.operation == "resize", \
                f"Over-cropping operation must be 'resize', got '{result.operation}'"

            # Verify produces empty grid
            for input_grid, output_grid in train_pairs:
                computed = _apply_resize(input_grid, result.pads_crops, result.pad_color or 0)
                assert _grids_equal(computed, output_grid), \
                    "Over-cropping must produce [[]] as expected"

    def test_resize_pad_color_observed(self):
        """Verify pad_color is observed from Y0, not invented."""
        # Custom fixture: padding with color 7
        train_pairs = [
            ([[1]], [[7, 7, 7], [7, 1, 7], [7, 7, 7]]),
            ([[2]], [[7, 7, 7], [7, 2, 7], [7, 7, 7]])
        ]

        result = infer_canvas(train_pairs)

        # PROVE pad_color observed (not default 0)
        assert result is not None, "Padding should be detected"
        assert result.operation == "resize", \
            f"Operation must be 'resize', got '{result.operation}'"
        assert result.pad_color == 7, \
            f"pad_color must be 7 (observed from Y0), not default 0. Got {result.pad_color}"


# =============================================================================
# Test Class 3: CONCAT Operations
# =============================================================================

class TestConcat:
    """Test CONCAT operations (repetition + gaps)."""

    def test_concat_rows_nogap_k2(self):
        """Verify vertical concatenation, k=2, no gap."""
        fixture = load_fixture("concat_rows_nogap.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]
        expected = fixture["expected"]

        result = infer_canvas(train_pairs)

        # PROVE concat detected
        assert result is not None, \
            "Vertical concat should be detected"

        assert result.operation == expected["operation"], \
            f"Operation must be '{expected['operation']}', got '{result.operation}'"

        # PROVE EXACT axis
        assert result.axis == expected["axis"], \
            f"axis must be EXACTLY '{expected['axis']}', got '{result.axis}'"

        # PROVE EXACT k
        assert result.k == expected["k"], \
            f"k must be EXACTLY {expected['k']}, got {result.k}"

        # PROVE EXACT gap
        assert result.gap == expected["gap"], \
            f"gap must be EXACTLY {expected['gap']}, got {result.gap}"

        # PROVE gap_color (null for gap=0)
        expected_gap_color = expected["gap_color"]
        if expected_gap_color is None:
            # gap_color could be None or 0 for no gap
            assert result.gap_color is None or result.gap_color == 0, \
                f"gap_color should be None or 0 for gap=0, got {result.gap_color}"
        else:
            assert result.gap_color == expected_gap_color, \
                f"gap_color must be {expected_gap_color}, got {result.gap_color}"

        # PROVE exact verification
        assert result.verified_exact == expected["verified_exact"], \
            "Transform must be verified exact"

        # PROVE forward mapping
        for input_grid, output_grid in train_pairs:
            computed = _apply_concat(
                input_grid, result.axis, result.k, result.gap, result.gap_color or 0
            )
            assert _grids_equal(computed, output_grid), \
                "Concat forward mapping must be pixel-perfect"

    def test_concat_rows_gap(self):
        """Verify vertical concatenation with gap."""
        fixture = load_fixture("concat_rows_gap.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]
        expected = fixture["expected"]

        result = infer_canvas(train_pairs)

        # PROVE concat with gap
        assert result is not None, "Concat with gap should be detected"
        assert result.operation == "concat", \
            f"Operation must be 'concat', got '{result.operation}'"

        assert result.axis == expected["axis"], \
            f"axis must be '{expected['axis']}', got '{result.axis}'"

        assert result.k == expected["k"], \
            f"k must be {expected['k']}, got {result.k}"

        assert result.gap == expected["gap"], \
            f"gap must be EXACTLY {expected['gap']}, got {result.gap}"

        # PROVE gap_color observed from Y0
        assert result.gap_color == expected["gap_color"], \
            f"gap_color must be {expected['gap_color']} (observed from Y0), got {result.gap_color}"

        # PROVE forward mapping
        for input_grid, output_grid in train_pairs:
            computed = _apply_concat(
                input_grid, result.axis, result.k, result.gap, result.gap_color
            )
            assert _grids_equal(computed, output_grid), \
                "Concat with gap forward mapping must be exact"

    def test_concat_cols_nogap_k2(self):
        """Verify horizontal concatenation, k=2, no gap."""
        fixture = load_fixture("concat_cols_nogap.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]
        expected = fixture["expected"]

        result = infer_canvas(train_pairs)

        # PROVE horizontal concat
        assert result is not None, "Horizontal concat should be detected"

        assert result.axis == "cols", \
            f"axis must be 'cols' for horizontal concat, got '{result.axis}'"

        assert result.k == expected["k"], \
            f"k must be {expected['k']}, got {result.k}"

        # PROVE forward mapping
        for input_grid, output_grid in train_pairs:
            computed = _apply_concat(
                input_grid, result.axis, result.k, result.gap, result.gap_color or 0
            )
            assert _grids_equal(computed, output_grid), \
                "Horizontal concat forward mapping must be exact"

    def test_concat_k_equals_3(self):
        """Verify triple concatenation (k=3)."""
        fixture = load_fixture("concat_k_equals_3.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]
        expected = fixture["expected"]

        result = infer_canvas(train_pairs)

        # PROVE k=3 detection
        assert result is not None, "k=3 concat should be detected"

        assert result.k == 3, \
            f"k must be EXACTLY 3, got {result.k}"

        # PROVE forward mapping
        for input_grid, output_grid in train_pairs:
            computed = _apply_concat(
                input_grid, result.axis, result.k, result.gap, result.gap_color or 0
            )
            assert _grids_equal(computed, output_grid), \
                "k=3 concat must produce exact output"

    def test_concat_k_bounded_max_10(self):
        """
        Verify k ∈ [2, 10] bound enforcement.

        Per spec: Bounded enumeration, k > 10 not enumerated.
        """
        # Create scenario where ONLY k=11 would match
        # 1×2 grid repeated 11 times → 11×2 output
        train_pairs = [
            (
                [[1, 2]],
                [[1, 2]] * 11  # k=11, height=11, width=2
            )
        ]

        result = infer_canvas(train_pairs)

        # PROVE k > 10 not enumerated → None
        # (k=2 through k=10 won't produce height=11)
        assert result is None, \
            f"k > 10 should NOT be enumerated (bounded at k ≤ 10). Got: {result}"

    def test_concat_gap_bounded_max_10(self):
        """Verify gap ∈ [0, 10] bound enforcement."""
        # Create scenario where gap=11 would match
        train_pairs = [
            ([[1]], [[1]] + [[0]] * 11 + [[1]])  # gap=11 would match
        ]

        result = infer_canvas(train_pairs)

        # PROVE gap > 10 not enumerated → None
        assert result is None, \
            "gap > 10 should NOT be enumerated (bounded at gap ≤ 10)"


# =============================================================================
# Test Class 4: FRAME Operations
# =============================================================================

class TestFrame:
    """
    Test FRAME vs RESIZE disambiguation.

    NOTE: Per spec enumeration order (IDENTITY → RESIZE → CONCAT → ... → FRAME),
    when both RESIZE and FRAME can produce the same output (symmetric padding with uniform color),
    RESIZE wins. These tests verify this behavior.
    """

    def test_frame_thickness_1(self):
        """
        Verify uniform border with thickness=1 returns RESIZE (not FRAME) per enumeration order.
        """
        fixture = load_fixture("frame_uniform.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]
        expected = fixture["expected"]

        result = infer_canvas(train_pairs)

        # PROVE border operation detected
        assert result is not None, "Border operation should be detected"

        assert result.operation == expected["operation"], \
            f"Operation must be '{expected['operation']}' (RESIZE wins by enumeration order), got '{result.operation}'"

        # PROVE EXACT pads_crops (RESIZE interpretation)
        assert result.pads_crops == tuple(expected["pads_crops"]), \
            f"pads_crops must be EXACTLY {tuple(expected['pads_crops'])}, got {result.pads_crops}"

        # PROVE pad_color observed
        assert result.pad_color == expected["pad_color"], \
            f"pad_color must be {expected['pad_color']}, got {result.pad_color}"

        # PROVE exact verification
        assert result.verified_exact == expected["verified_exact"], \
            "Transform must be verified exact"

        # PROVE forward mapping using RESIZE
        for input_grid, output_grid in train_pairs:
            computed = _apply_resize(input_grid, result.pads_crops, result.pad_color)
            assert _grids_equal(computed, output_grid), \
                "RESIZE forward mapping must be pixel-perfect"

    def test_frame_thickness_2(self):
        """Verify thick border (thickness=2) returns RESIZE per enumeration order."""
        fixture = load_fixture("frame_thick_2.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]
        expected = fixture["expected"]

        result = infer_canvas(train_pairs)

        # PROVE thick border detected
        assert result is not None, "Thick border should be detected"

        assert result.operation == "resize", \
            f"Operation must be 'resize' (enumeration order), got '{result.operation}'"

        assert result.pads_crops == tuple(expected["pads_crops"]), \
            f"pads_crops must be {tuple(expected['pads_crops'])}, got {result.pads_crops}"

        assert result.pad_color == expected["pad_color"], \
            f"pad_color must be {expected['pad_color']}, got {result.pad_color}"

        # PROVE forward mapping
        for input_grid, output_grid in train_pairs:
            computed = _apply_resize(input_grid, result.pads_crops, result.pad_color)
            assert _grids_equal(computed, output_grid), \
                "RESIZE forward mapping must be exact"

    def test_frame_color_observed(self):
        """
        Verify border color is observed from Y0, not invented.

        NOTE: Per enumeration order, this will be detected as RESIZE (not FRAME),
        but the pad_color should still be observed from Y0.
        """
        # Custom fixture: uniform border with color 3
        train_pairs = [
            ([[1, 2]], [[3, 3, 3, 3], [3, 1, 2, 3], [3, 3, 3, 3]]),
            ([[5]], [[3, 3, 3], [3, 5, 3], [3, 3, 3]])
        ]

        result = infer_canvas(train_pairs)

        # PROVE border operation detected (will be RESIZE per enumeration order)
        assert result is not None, "Border operation should be detected"
        assert result.operation == "resize", \
            f"Operation must be 'resize' (enumeration order), got '{result.operation}'"
        assert result.pad_color == 3, \
            f"pad_color must be 3 (observed from Y0), got {result.pad_color}"


# =============================================================================
# Test Class 5: Composition Operations
# =============================================================================

class TestComposition:
    """Test composition operations (RESIZE+FRAME, CONCAT+FRAME)."""

    def test_resize_then_frame(self):
        """
        Verify RESIZE+FRAME composition.

        Per spec (implementation_plan.md WO-07): "concat then frame" composition is required.
        Per worked_examples.md: CONCAT+FRAME and potentially RESIZE+FRAME compositions exist.

        This test verifies RESIZE+FRAME composition enumeration and verification.
        """
        fixture = load_fixture("resize_then_frame.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]
        expected = fixture["expected"]

        result = infer_canvas(train_pairs)

        # PROVE composition detected
        assert result is not None, \
            "RESIZE+FRAME composition should be detected per spec (see worked_examples.md Canvas CONCAT+FRAME)"

        assert result.operation == expected["operation"], \
            f"Operation must be '{expected['operation']}', got '{result.operation}'"

        # PROVE resize params
        assert result.pads_crops == tuple(expected["pads_crops"]), \
            f"Resize pads_crops must be {tuple(expected['pads_crops'])}, got {result.pads_crops}"

        assert result.pad_color == expected["pad_color"], \
            f"Pad color must be {expected['pad_color']}, got {result.pad_color}"

        # PROVE frame params
        assert result.frame_color == expected["frame_color"], \
            f"Frame color must be {expected['frame_color']}, got {result.frame_color}"

        assert result.frame_thickness == expected["frame_thickness"], \
            f"Frame thickness must be {expected['frame_thickness']}, got {result.frame_thickness}"

        # PROVE verified exact
        assert result.verified_exact == expected["verified_exact"], \
            "Composition must be verified exact"

        # PROVE forward mapping (apply resize, then frame)
        for input_grid, output_grid in train_pairs:
            # Step 1: Resize
            temp = _apply_resize(input_grid, result.pads_crops, result.pad_color)
            # Step 2: Frame
            computed = _apply_frame(temp, result.frame_color, result.frame_thickness)

            assert _grids_equal(computed, output_grid), \
                "RESIZE+FRAME composition must apply transforms in correct order"

    def test_concat_then_frame(self):
        """
        Verify CONCAT+FRAME composition.

        Per spec: Concat FIRST, then frame.
        """
        fixture = load_fixture("concat_then_frame.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]
        expected = fixture["expected"]

        result = infer_canvas(train_pairs)

        # PROVE composition detected
        assert result is not None, \
            "CONCAT+FRAME composition should be detected"

        assert result.operation == "concat+frame", \
            f"Operation must be 'concat+frame', got '{result.operation}'"

        # PROVE concat params
        assert result.axis == expected["axis"], \
            f"Concat axis must be '{expected['axis']}', got '{result.axis}'"

        assert result.k == expected["k"], \
            f"k must be {expected['k']}, got {result.k}"

        assert result.gap == expected["gap"], \
            f"gap must be {expected['gap']}, got {result.gap}"

        # PROVE frame params
        assert result.frame_color == expected["frame_color"], \
            f"Frame color must be {expected['frame_color']}, got {result.frame_color}"

        assert result.frame_thickness == expected["frame_thickness"], \
            f"Frame thickness must be {expected['frame_thickness']}, got {result.frame_thickness}"

        # PROVE forward mapping (apply concat, then frame)
        for input_grid, output_grid in train_pairs:
            # Step 1: Concat
            temp = _apply_concat(
                input_grid, result.axis, result.k, result.gap, result.gap_color or 0
            )
            # Step 2: Frame
            computed = _apply_frame(temp, result.frame_color, result.frame_thickness)

            assert _grids_equal(computed, output_grid), \
                "CONCAT+FRAME composition must apply transforms in correct order"


# =============================================================================
# Test Class 6: Verification Tests
# =============================================================================

class TestVerification:
    """Test exact forward verification."""

    def test_exact_match_all_trains(self):
        """Verify exact verification requires ALL trains to match."""
        # 3 trains, all match same RESIZE(1,1,1,1)
        train_pairs = [
            ([[1]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            ([[2]], [[0, 0, 0], [0, 2, 0], [0, 0, 0]]),
            ([[3]], [[0, 0, 0], [0, 3, 0], [0, 0, 0]]),
        ]

        result = infer_canvas(train_pairs)

        # PROVE all trains verified
        assert result is not None, \
            "All trains match → valid result"

        assert result.verified_exact == True, \
            "ALL trains verified → verified_exact must be True"

    def test_partial_match_rejected(self):
        """Verify partial match is rejected (all-or-nothing)."""
        # 2 trains: first matches RESIZE(1,1,1,1), second doesn't
        train_pairs = [
            ([[1]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]),  # Matches
            ([[2]], [[2, 2], [2, 2]]),  # Doesn't match
        ]

        result = infer_canvas(train_pairs)

        # PROVE partial match rejected
        # Could be None or a different transform that matches both
        # But RESIZE(1,1,1,1) should NOT be selected (only matches 1/2 trains)
        if result is not None and result.operation == "resize":
            # If resize is returned, it must verify both trains
            for input_grid, output_grid in train_pairs:
                computed = _apply_resize(input_grid, result.pads_crops, result.pad_color or 0)
                assert _grids_equal(computed, output_grid), \
                    "If resize is selected, it must verify ALL trains (not just partial match)"

    def test_pixel_mismatch_rejected(self):
        """Verify pixel-level mismatch rejects candidate."""
        # Train where output differs by 1 pixel from RESIZE(1,1,1,1)
        train_pairs = [
            ([[1]], [[0, 0, 0], [0, 1, 0], [0, 0, 5]])  # Last pixel wrong
        ]

        # Expected: RESIZE(1,1,1,1) would produce [[0,0,0], [0,1,0], [0,0,0]]
        # Actual output has [0,0,5] in last row

        result = infer_canvas(train_pairs)

        # PROVE pixel mismatch causes rejection
        # Could be None or a different transform, but NOT RESIZE(1,1,1,1)
        if result is not None and result.operation == "resize" and result.pads_crops == (1, 1, 1, 1):
            pytest.fail("Pixel mismatch should reject RESIZE(1,1,1,1) candidate")


# =============================================================================
# Test Class 7: Lex-Min Selection
# =============================================================================

class TestLexMin:
    """Test lex-min selection when multiple valid candidates."""

    def test_lex_min_multiple_valid_pads(self):
        """
        Verify lex-min selection among multiple valid padding distributions.

        Per spec: If multiple valid, choose lex-min using to_tuple() ordering.
        """
        fixture = load_fixture("multiple_valid_lex_min.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]
        expected = fixture["expected"]

        # For 1×1 → 3×3, multiple distributions are valid:
        # (1,1,1,1) symmetric
        # (2,0,1,1) top-heavy
        # (0,2,1,1) bottom-heavy
        # (1,1,2,0) left-heavy
        # (1,1,0,2) right-heavy

        result = infer_canvas(train_pairs)

        # PROVE lex-min selected
        assert result is not None, "Padding should be detected"

        assert result.pads_crops == tuple(expected["pads_crops"]), \
            f"Lex-min pads_crops must be {tuple(expected['pads_crops'])} (symmetric), got {result.pads_crops}"

        # PROVE to_tuple() ordering verifies lex-min
        # Symmetric (1,1,1,1) should be lex-min
        result_tuple = result.to_tuple()

        # Verify pads_crops field in tuple
        assert result_tuple[5] == (1, 1, 1, 1), \
            f"to_tuple() pads_crops must be (1,1,1,1), got {result_tuple[5]}"

    def test_to_tuple_ordering(self):
        """Verify to_tuple() produces correct ordering for lex comparison."""
        # Create two CanvasMap instances
        map1 = CanvasMap(
            operation="resize",
            pads_crops=(1, 1, 1, 1),
            pad_color=0,
            verified_exact=True
        )

        map2 = CanvasMap(
            operation="resize",
            pads_crops=(2, 0, 1, 1),
            pad_color=0,
            verified_exact=True
        )

        # PROVE to_tuple() ordering
        tuple1 = map1.to_tuple()
        tuple2 = map2.to_tuple()

        # (1,1,1,1) should be lex-less than (2,0,1,1)
        assert tuple1 < tuple2, \
            f"Symmetric (1,1,1,1) must be lex-min vs (2,0,1,1). Got tuple1={tuple1}, tuple2={tuple2}"


# =============================================================================
# Test Class 8: Edge Cases and Contracts
# =============================================================================

class TestEdgeCases:
    """Test edge cases and contract enforcement."""

    def test_no_valid_transform_returns_none(self):
        """Verify return None when no valid transform exists."""
        fixture = load_fixture("no_valid_transform.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]

        result = infer_canvas(train_pairs)

        # PROVE None returned (not empty CanvasMap)
        assert result is None, \
            "When no valid transform exists, must return None (not empty CanvasMap)"

    def test_determinism_same_inputs(self):
        """Verify deterministic output for same inputs."""
        train_pairs = [
            ([[1, 2], [3, 4]], [[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
        ]

        result1 = infer_canvas(train_pairs)
        result2 = infer_canvas(train_pairs)

        # PROVE determinism
        assert result1 == result2, \
            "Same inputs must produce identical CanvasMap (deterministic)"

        if result1 is not None:
            # PROVE to_tuple() determinism
            assert result1.to_tuple() == result2.to_tuple(), \
                "to_tuple() must be deterministic"

    def test_train_order_invariant(self):
        """Verify result is independent of train pair order."""
        train_pairs_1 = [
            ([[1]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
            ([[2]], [[0, 0, 0], [0, 2, 0], [0, 0, 0]]),
        ]

        train_pairs_2 = [
            ([[2]], [[0, 0, 0], [0, 2, 0], [0, 0, 0]]),
            ([[1]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        ]

        result1 = infer_canvas(train_pairs_1)
        result2 = infer_canvas(train_pairs_2)

        # PROVE order invariance
        assert result1 == result2, \
            "Train pair order should not affect result (deterministic enumeration)"

    def test_y_observation_only(self):
        """
        Verify Y is used only for verification, not parameter derivation.

        Per spec: Colors observed from Y0, not invented.
        """
        # Y has unique colors not in X
        train_pairs = [
            ([[1]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]),  # pad_color=0 from Y0
            ([[2]], [[0, 0, 0], [0, 2, 0], [0, 0, 0]])
        ]

        result = infer_canvas(train_pairs)

        # PROVE pad_color observed from Y0
        assert result is not None, "Padding should be detected"
        assert result.operation == "resize", \
            f"Operation must be 'resize', got '{result.operation}'"
        assert result.pad_color == 0, \
            "pad_color must be observed from Y0 borders (not invented)"

        # PROVE no arbitrary colors
        # If Y0 had different pad color, result would reflect that
        # This test verifies observation-guided enumeration

    def test_none_on_failure(self):
        """Verify explicit None return (not empty CanvasMap) on failure."""
        # Random grid transformation that doesn't match any operation
        train_pairs = [
            ([[1, 2], [3, 4]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]])
        ]

        result = infer_canvas(train_pairs)

        # PROVE explicit None
        assert result is None, \
            "No valid candidate → must return None (not empty CanvasMap)"


# =============================================================================
# Test Class 9: Receipt Validation
# =============================================================================

class TestReceipts:
    """Test receipt structure and invariants."""

    def test_receipt_fields_present(self):
        """Verify all required fields are present in CanvasMap."""
        train_pairs = [
            ([[1]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        ]

        result = infer_canvas(train_pairs)

        # PROVE required fields
        assert hasattr(result, "operation"), "CanvasMap must have 'operation' field"
        assert hasattr(result, "verified_exact"), "CanvasMap must have 'verified_exact' field"
        assert hasattr(result, "pads_crops"), "CanvasMap must have 'pads_crops' field"
        assert hasattr(result, "axis"), "CanvasMap must have 'axis' field"
        assert hasattr(result, "k"), "CanvasMap must have 'k' field"
        assert hasattr(result, "gap"), "CanvasMap must have 'gap' field"
        assert hasattr(result, "frame_color"), "CanvasMap must have 'frame_color' field"
        assert hasattr(result, "frame_thickness"), "CanvasMap must have 'frame_thickness' field"

    def test_verified_exact_flag(self):
        """Verify verified_exact flag is set correctly."""
        train_pairs = [
            ([[1, 2], [3, 4]], [[1, 2], [3, 4]])  # Identity
        ]

        result = infer_canvas(train_pairs)

        # PROVE verified_exact flag
        assert result is not None, "Identity should be detected"
        assert result.verified_exact == True, \
            "Valid transform must have verified_exact=True"

    def test_operation_logged(self):
        """
        Verify operation field correctly identifies transform type.

        NOTE: frame_uniform.json returns 'resize' due to enumeration order.
        """
        fixtures = [
            ("identity_2x2.json", "identity"),
            ("resize_pad_symmetric.json", "resize"),
            ("concat_rows_nogap.json", "concat"),
            ("frame_uniform.json", "resize"),  # RESIZE wins over FRAME by enumeration order
            ("resize_then_frame.json", "resize+frame"),  # Composition per spec
            ("concat_then_frame.json", "concat+frame"),
        ]

        for fixture_name, expected_op in fixtures:
            fixture = load_fixture(fixture_name)
            train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]

            result = infer_canvas(train_pairs)

            assert result is not None, \
                f"Fixture {fixture_name} should produce valid result (if failing, there's a BUG in composition enumeration)"

            assert result.operation == expected_op, \
                f"Fixture {fixture_name} operation must be '{expected_op}', got '{result.operation}'"

    def test_parameters_logged(self):
        """
        Verify parameters are correctly logged for each operation type.

        NOTE: frame_uniform.json returns RESIZE (not FRAME) due to enumeration order.
        """
        # RESIZE parameters
        fixture = load_fixture("resize_pad_symmetric.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]
        result = infer_canvas(train_pairs)

        assert result.pads_crops is not None, \
            "RESIZE must log pads_crops"
        assert result.pad_color is not None, \
            "RESIZE with padding must log pad_color"

        # CONCAT parameters
        fixture = load_fixture("concat_rows_nogap.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]
        result = infer_canvas(train_pairs)

        assert result.axis is not None, \
            "CONCAT must log axis"
        assert result.k is not None, \
            "CONCAT must log k"
        assert result.gap is not None, \
            "CONCAT must log gap"

        # Border operation (returns RESIZE per enumeration order)
        fixture = load_fixture("frame_uniform.json")
        train_pairs = [(pair["input"], pair["output"]) for pair in fixture["train_pairs"]]
        result = infer_canvas(train_pairs)

        # This will be RESIZE, not FRAME
        assert result.operation == "resize", \
            "Uniform border returns RESIZE (enumeration order)"
        assert result.pads_crops is not None, \
            "RESIZE must log pads_crops"
        assert result.pad_color is not None, \
            "RESIZE must log pad_color"
