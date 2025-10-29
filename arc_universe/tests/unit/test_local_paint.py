"""
Unit tests for WO-09: arc_laws/local_paint.py

Per test plan test_plans/WO-09-plan.md:
- Test role→color extraction from training pairs
- Test consistency filtering (multi-color roles)
- Test exactness verification (ALL trains)
- Test canonical space (D4 transforms)
- Test edge cases (empty, mismatch, no laws)

CRITICAL: All assertions verify EXACT values, not just existence.
Per WO-05/WO-07/WO-08 lessons: Battle-test the implementation.
"""

import json
import pytest
from pathlib import Path

from arc_laws.local_paint import build_local_paint, apply_local_paint
from arc_core.types import LocalPaintParams, Present, Pixel, RoleId
from arc_core.present import D4_TRANSFORMATIONS, D4_INVERSES


def load_fixture(filename: str) -> dict:
    """Load test fixture from WO-09 directory."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "WO-09" / filename
    with open(fixture_path, "r") as f:
        return json.load(f)


def build_params_from_fixture(fixture: dict) -> LocalPaintParams:
    """Convert fixture JSON to LocalPaintParams object."""
    # Convert train_pairs
    train_pairs = [
        (pair["X"], pair["Y"])
        for pair in fixture["train_pairs"]
    ]

    # Convert presents
    presents = []
    for pres in fixture["presents"]:
        # Create minimal Present object (only fields used by local_paint)
        present = Present(
            grid=pres["grid"],
            cbc3={},  # Not used by local_paint
            e4_neighbors={},  # Not used by local_paint
            row_members={},  # Not used by local_paint
            col_members={},  # Not used by local_paint
            g_inverse=pres["g_inverse"]
        )
        presents.append(present)

    # Convert role_map_list to role_map dict
    role_map = {}
    for entry in fixture["role_map_list"]:
        grid_id = entry["grid_id"]
        pixel = Pixel(entry["row"], entry["col"])
        role = RoleId(entry["role"])
        role_map[(grid_id, pixel)] = role

    return LocalPaintParams(
        train_pairs=train_pairs,
        presents=presents,
        role_map=role_map
    )


class TestBasicExtraction:
    """Test basic role→color extraction."""

    def test_LE01_simple_recolor(self):
        """LE-01: Simple 2-role recolor across 2 trains."""
        fixture = load_fixture("simple_recolor.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE exact count
        assert len(result) == expected["num_laws"], \
            f"Expected {expected['num_laws']} laws, got {len(result)}. " \
            f"BUG: Incorrect number of extracted laws."

        # PROVE exact laws
        laws_dict = {law.params["role"]: law.params["color"] for law in result}
        expected_dict = {law["role"]: law["color"] for law in expected["laws"]}
        assert laws_dict == expected_dict, \
            f"Expected laws {expected_dict}, got {laws_dict}. " \
            f"BUG: Incorrect role→color mappings."

        # PROVE law type
        for law in result:
            assert law.law_type == "LOCAL_PAINT_ROLE", \
                f"Law type must be LOCAL_PAINT_ROLE, got {law.law_type}. BUG: Wrong law type."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, \
            "Same input must produce identical output (deterministic). BUG: Non-deterministic behavior."

    def test_LE02_multi_role_recolor(self):
        """LE-02: Multi-role recolor - 5 roles, all distinct colors."""
        fixture = load_fixture("multi_role_recolor.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE all 5 roles extracted
        assert len(result) == 5, \
            f"Expected 5 laws (one per role), got {len(result)}. BUG: Not all roles extracted."

        # PROVE all roles present
        extracted_roles = {law.params["role"] for law in result}
        expected_roles = {law["role"] for law in expected["laws"]}
        assert extracted_roles == expected_roles, \
            f"Expected roles {expected_roles}, got {extracted_roles}. BUG: Missing or extra roles."

        # PROVE exact mappings
        laws_dict = {law.params["role"]: law.params["color"] for law in result}
        expected_dict = {law["role"]: law["color"] for law in expected["laws"]}
        assert laws_dict == expected_dict, \
            f"Expected {expected_dict}, got {laws_dict}. BUG: Incorrect mappings."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"

    def test_LE03_identity_mapping(self):
        """LE-03: Identity mapping - roles keep same colors."""
        fixture = load_fixture("identity_mapping.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE exact count (only roles appearing in ALL trains)
        assert len(result) == len(expected["laws"]), \
            f"Expected {len(expected['laws'])} laws, got {len(result)}. " \
            f"BUG: Should only extract roles appearing in all trains."

        # PROVE identity preserved for extracted roles
        laws_dict = {law.params["role"]: law.params["color"] for law in result}
        expected_dict = {law["role"]: law["color"] for law in expected["laws"]}
        assert laws_dict == expected_dict, \
            f"Expected {expected_dict}, got {laws_dict}. BUG: Identity not preserved."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"

    def test_LE04_single_pixel(self):
        """LE-04: Single pixel grid - minimal case."""
        fixture = load_fixture("single_pixel.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE exactly 1 law
        assert len(result) == 1, \
            f"Expected 1 law for single pixel, got {len(result)}. BUG: Edge case handling."

        # PROVE exact mapping
        law = result[0]
        expected_law = expected["laws"][0]
        assert law.params["role"] == expected_law["role"], \
            f"Expected role {expected_law['role']}, got {law.params['role']}"
        assert law.params["color"] == expected_law["color"], \
            f"Expected color {expected_law['color']}, got {law.params['color']}"

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"

    def test_LE05_all_same_role(self):
        """LE-05: All pixels have same role - uniform scenario."""
        fixture = load_fixture("all_same_role.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE exactly 1 law (all pixels same role)
        assert len(result) == 1, \
            f"Expected 1 law (single role), got {len(result)}. BUG: Single-role scenario."

        # PROVE exact mapping
        law = result[0]
        expected_law = expected["laws"][0]
        assert law.params["role"] == expected_law["role"], \
            f"Expected role {expected_law['role']}, got {law.params['role']}"
        assert law.params["color"] == expected_law["color"], \
            f"Expected color {expected_law['color']}, got {law.params['color']}"

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"


class TestConsistencyFiltering:
    """Test consistency filtering (multi-color roles)."""

    def test_CF01_inconsistent_mapping(self):
        """CF-01: Inconsistent mapping - role maps to different colors."""
        fixture = load_fixture("inconsistent_mapping.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE exact count (only consistent roles)
        assert len(result) == expected["num_laws"], \
            f"Expected {expected['num_laws']} laws (only consistent roles), got {len(result)}. " \
            f"BUG: Inconsistent roles not filtered."

        # PROVE inconsistent role NOT in output
        extracted_roles = {law.params["role"] for law in result}
        filtered_roles = set(expected.get("filtered_roles", []))
        for role in filtered_roles:
            assert role not in extracted_roles, \
                f"Role {role} is inconsistent (maps to multiple colors) but appears in output. " \
                f"BUG: Consistency filtering failed."

        # PROVE only consistent roles extracted
        expected_laws = {law["role"]: law["color"] for law in expected["laws"]}
        actual_laws = {law.params["role"]: law.params["color"] for law in result}
        assert actual_laws == expected_laws, \
            f"Expected {expected_laws}, got {actual_laws}. BUG: Wrong laws extracted."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"

    def test_CF02_multi_color_role(self):
        """CF-02: Multi-color role - role maps to 3 different colors."""
        fixture = load_fixture("multi_color_role.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE zero laws (all inconsistent)
        assert len(result) == 0, \
            f"Expected 0 laws (role maps to 3 colors), got {len(result)}. " \
            f"BUG: Multi-color role not filtered."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"

    def test_CF03_partial_consistency(self):
        """CF-03: Partial consistency - some roles consistent, some not."""
        fixture = load_fixture("partial_consistency.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE exact count (only consistent roles)
        assert len(result) == expected["num_laws"], \
            f"Expected {expected['num_laws']} consistent roles, got {len(result)}. " \
            f"BUG: Partial consistency not handled correctly."

        # PROVE inconsistent roles filtered
        extracted_roles = {law.params["role"] for law in result}
        filtered_roles = set(expected.get("filtered_roles", []))
        for role in filtered_roles:
            assert role not in extracted_roles, \
                f"Role {role} is inconsistent but appears in output. BUG: Filtering failed."

        # PROVE consistent roles extracted
        expected_laws = {law["role"]: law["color"] for law in expected["laws"]}
        actual_laws = {law.params["role"]: law.params["color"] for law in result}
        assert actual_laws == expected_laws, \
            f"Expected {expected_laws}, got {actual_laws}. BUG: Wrong subset extracted."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"

    def test_CF04_all_inconsistent(self):
        """CF-04: All inconsistent - every role maps to multiple colors."""
        fixture = load_fixture("all_inconsistent.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE empty list
        assert len(result) == 0, \
            f"Expected 0 laws (all roles inconsistent), got {len(result)}. " \
            f"BUG: All-inconsistent case not handled."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"


class TestExactnessVerification:
    """Test exactness on ALL trains."""

    def test_EV01_exact_on_all_trains(self):
        """EV-01: Exact on all trains - perfect match across 3 trains."""
        fixture = load_fixture("exact_on_all_trains.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE exact count
        assert len(result) == expected["num_laws"], \
            f"Expected {expected['num_laws']} laws, got {len(result)}. " \
            f"BUG: Not all exact laws extracted."

        # PROVE exact laws
        expected_laws = {law["role"]: law["color"] for law in expected["laws"]}
        actual_laws = {law.params["role"]: law.params["color"] for law in result}
        assert actual_laws == expected_laws, \
            f"Expected {expected_laws}, got {actual_laws}. BUG: Wrong laws extracted."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"

    def test_EV02_partial_match(self):
        """EV-02: Partial match - exact on train 1, wrong on train 2."""
        fixture = load_fixture("partial_match.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE zero laws (not exact on ALL trains)
        assert len(result) == 0, \
            f"Expected 0 laws (not exact on all trains), got {len(result)}. " \
            f"BUG: Partial match should be rejected (all-or-nothing)."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"

    def test_EV03_pixel_mismatch(self):
        """EV-03: Pixel mismatch - 1 pixel out of many mismatches."""
        fixture = load_fixture("pixel_mismatch.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE zero laws (single pixel mismatch fails exactness)
        assert len(result) == 0, \
            f"Expected 0 laws (single pixel mismatch), got {len(result)}. " \
            f"BUG: Single pixel failure should reject entire law."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"

    def test_EV04_dimension_exact(self):
        """EV-04: Dimension exact - after skipping mismatch, exact match."""
        fixture = load_fixture("dimension_exact.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE laws extracted from matching pairs
        assert len(result) == expected["num_laws"], \
            f"Expected {expected['num_laws']} laws (from dimension-matched pairs), got {len(result)}. " \
            f"BUG: Should skip mismatched pairs and extract from matching ones."

        # PROVE exact laws
        expected_laws = {law["role"]: law["color"] for law in expected["laws"]}
        actual_laws = {law.params["role"]: law.params["color"] for law in result}
        assert actual_laws == expected_laws, \
            f"Expected {expected_laws}, got {actual_laws}. BUG: Wrong laws after skipping."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"


class TestCanonicalSpace:
    """Test canonical space (D4 transforms)."""

    def test_CS01_d4_transform_applied(self):
        """CS-01: D4 transform applied - Y requires rotation to match."""
        fixture = load_fixture("d4_transform.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE all laws extracted (D4 transform applied correctly)
        assert len(result) == expected["num_laws"], \
            f"Expected {expected['num_laws']} laws, got {len(result)}. " \
            f"BUG: D4 transform not applied correctly to Y."

        # PROVE exact laws
        expected_laws = {law["role"]: law["color"] for law in expected["laws"]}
        actual_laws = {law.params["role"]: law.params["color"] for law in result}
        assert actual_laws == expected_laws, \
            f"Expected {expected_laws}, got {actual_laws}. BUG: Wrong mappings after D4."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"

    def test_CS02_dimension_mismatch_skip(self):
        """CS-02: Dimension mismatch skip - canvas transform case."""
        fixture = load_fixture("dimension_mismatch_skip.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE zero laws (dimension mismatch skipped)
        assert len(result) == 0, \
            f"Expected 0 laws (dimension mismatch), got {len(result)}. " \
            f"BUG: Dimension mismatch not skipped."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"

    def test_CS03_multiple_trains_mixed(self):
        """CS-03: Multiple trains mixed - some match, some don't."""
        fixture = load_fixture("multiple_trains_mixed.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE laws from matching trains only
        assert len(result) == expected["num_laws"], \
            f"Expected {expected['num_laws']} laws (from matching trains), got {len(result)}. " \
            f"BUG: Should skip mismatched, use matched trains."

        # PROVE exact laws
        expected_laws = {law["role"]: law["color"] for law in expected["laws"]}
        actual_laws = {law.params["role"]: law.params["color"] for law in result}
        assert actual_laws == expected_laws, \
            f"Expected {expected_laws}, got {actual_laws}. BUG: Wrong laws from mixed trains."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"

    def test_CS04_present_canonical(self):
        """CS-04: Present canonical - use present.grid not raw input."""
        fixture = load_fixture("present_canonical.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE all laws extracted (using canonical grid)
        assert len(result) == expected["num_laws"], \
            f"Expected {expected['num_laws']} laws, got {len(result)}. " \
            f"BUG: Not using present.grid (canonical)."

        # PROVE exact laws
        expected_laws = {law["role"]: law["color"] for law in expected["laws"]}
        actual_laws = {law.params["role"]: law.params["color"] for law in result}
        assert actual_laws == expected_laws, \
            f"Expected {expected_laws}, got {actual_laws}. BUG: Wrong canonical space."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"


class TestEdgeCases:
    """Test edge cases."""

    def test_EC01_empty_training(self):
        """EC-01: Empty training - no training pairs provided."""
        fixture = load_fixture("empty_training.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE empty list
        assert len(result) == 0, \
            f"Expected 0 laws (empty training), got {len(result)}. " \
            f"BUG: Empty training not handled gracefully."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"

    def test_EC03_no_role_map(self):
        """EC-03: No role map - empty role_map dictionary."""
        fixture = load_fixture("no_role_map.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE empty list
        assert len(result) == 0, \
            f"Expected 0 laws (no role map), got {len(result)}. " \
            f"BUG: Empty role map not handled."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"

    def test_EC04_no_valid_laws(self):
        """EC-04: No valid laws - all roles filtered."""
        fixture = load_fixture("no_valid_laws.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE empty list
        assert len(result) == 0, \
            f"Expected 0 laws (all filtered), got {len(result)}. " \
            f"BUG: Zero-laws case not handled."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"

    def test_EC05_single_train(self):
        """EC-05: Single train - only 1 training pair (trivially consistent)."""
        fixture = load_fixture("single_train.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE all laws extracted (trivially consistent)
        assert len(result) == expected["num_laws"], \
            f"Expected {expected['num_laws']} laws (single train), got {len(result)}. " \
            f"BUG: Single train case not handled."

        # PROVE exact laws
        expected_laws = {law["role"]: law["color"] for law in expected["laws"]}
        actual_laws = {law.params["role"]: law.params["color"] for law in result}
        assert actual_laws == expected_laws, \
            f"Expected {expected_laws}, got {actual_laws}. BUG: Wrong laws from single train."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"


class TestIntegration:
    """Test integration with WL/present."""

    def test_IT03_full_pipeline(self):
        """IT-03: Full pipeline - end-to-end with mixed consistency."""
        fixture = load_fixture("full_pipeline.json")
        params = build_params_from_fixture(fixture)
        expected = fixture["expected"]

        result = build_local_paint(params)

        # PROVE exact count (consistent subset)
        assert len(result) == expected["num_laws"], \
            f"Expected {expected['num_laws']} laws (consistent subset), got {len(result)}. " \
            f"BUG: Full pipeline integration failed."

        # PROVE inconsistent roles filtered
        extracted_roles = {law.params["role"] for law in result}
        filtered_roles = set(expected.get("filtered_roles", []))
        for role in filtered_roles:
            assert role not in extracted_roles, \
                f"Role {role} is inconsistent but appears in output. BUG: Integration filtering failed."

        # PROVE exact laws
        expected_laws = {law["role"]: law["color"] for law in expected["laws"]}
        actual_laws = {law.params["role"]: law.params["color"] for law in result}
        assert actual_laws == expected_laws, \
            f"Expected {expected_laws}, got {actual_laws}. BUG: Wrong laws in full pipeline."

        # PROVE determinism
        result2 = build_local_paint(params)
        assert result == result2, "Must be deterministic"


class TestDeterminism:
    """Test deterministic behavior."""

    def test_DT01_same_input_twice(self):
        """DT-01: Same input twice - byte-identical output."""
        fixture = load_fixture("simple_recolor.json")
        params = build_params_from_fixture(fixture)

        # Run twice
        result1 = build_local_paint(params)
        result2 = build_local_paint(params)

        # PROVE byte-identical
        assert result1 == result2, \
            "Same input must produce byte-identical output. BUG: Non-deterministic."

        # PROVE exact equality (not just set equality)
        assert len(result1) == len(result2), "Must have same count"
        for law1, law2 in zip(result1, result2):
            assert law1.law_type == law2.law_type, "Law types must match"
            assert law1.params == law2.params, "Params must match exactly"
            assert law1.domain_desc == law2.domain_desc, "Domain desc must match"

    def test_DT02_order_invariance(self):
        """DT-02: Order invariance - shuffling trains gives same laws (set equality)."""
        # Use simple fixture with multiple trains
        fixture = load_fixture("exact_on_all_trains.json")
        params1 = build_params_from_fixture(fixture)

        # Reverse train order and remap role_map grid_ids
        num_trains = len(params1.train_pairs)
        role_map_reversed = {}
        for (grid_id, pixel), role in params1.role_map.items():
            new_grid_id = num_trains - 1 - grid_id  # Reverse the grid_id
            role_map_reversed[(new_grid_id, pixel)] = role

        params2 = LocalPaintParams(
            train_pairs=list(reversed(params1.train_pairs)),
            presents=list(reversed(params1.presents)),
            role_map=role_map_reversed
        )

        result1 = build_local_paint(params1)
        result2 = build_local_paint(params2)

        # PROVE same laws (set equality - order may differ)
        laws1 = {(law.params["role"], law.params["color"]) for law in result1}
        laws2 = {(law.params["role"], law.params["color"]) for law in result2}
        assert laws1 == laws2, \
            f"Train order should not affect laws. Got {laws1} vs {laws2}. BUG: Order-dependent."

    def test_DT03_multiple_runs(self):
        """DT-03: Multiple runs - all identical."""
        fixture = load_fixture("multi_role_recolor.json")
        params = build_params_from_fixture(fixture)

        # Run 5 times
        results = [build_local_paint(params) for _ in range(5)]

        # PROVE all identical
        first = results[0]
        for i, result in enumerate(results[1:], start=2):
            assert result == first, \
                f"Run {i} differs from run 1. BUG: Non-deterministic across multiple runs."


class TestReceiptFormat:
    """Test receipt format correctness."""

    def test_RF01_law_type(self):
        """RF-01: Law type - verify law_type string."""
        fixture = load_fixture("simple_recolor.json")
        params = build_params_from_fixture(fixture)

        result = build_local_paint(params)

        # PROVE all laws have correct type
        for law in result:
            assert law.law_type == "LOCAL_PAINT_ROLE", \
                f"Expected law_type='LOCAL_PAINT_ROLE', got '{law.law_type}'. " \
                f"BUG: Wrong law type string."

    def test_RF02_params_format(self):
        """RF-02: Params format - verify dict structure."""
        fixture = load_fixture("simple_recolor.json")
        params = build_params_from_fixture(fixture)

        result = build_local_paint(params)

        # PROVE all laws have correct params format
        for law in result:
            assert isinstance(law.params, dict), \
                f"Params must be dict, got {type(law.params)}. BUG: Wrong params type."

            assert "role" in law.params, \
                "Params must contain 'role' key. BUG: Missing role in params."
            assert "color" in law.params, \
                "Params must contain 'color' key. BUG: Missing color in params."

            assert isinstance(law.params["role"], int), \
                f"Role must be int, got {type(law.params['role'])}. BUG: Wrong role type."
            assert isinstance(law.params["color"], int), \
                f"Color must be int, got {type(law.params['color'])}. BUG: Wrong color type."

            # PROVE colors in valid range [0-9]
            assert 0 <= law.params["color"] <= 9, \
                f"Color must be in [0-9], got {law.params['color']}. BUG: Invalid color value."


class TestApplyLocalPaint:
    """Test apply_local_paint helper function thoroughly."""

    def test_AP01_single_law_application(self):
        """AP-01: Apply single law to grid - verify pixel-perfect transformation."""
        from arc_core.types import LawInstance

        # Setup: grid with 2 roles
        grid = [[1, 2], [2, 1]]
        role_map = {
            Pixel(0, 0): RoleId(10),
            Pixel(0, 1): RoleId(20),
            Pixel(1, 0): RoleId(20),
            Pixel(1, 1): RoleId(10),
        }

        # Law: role 10 -> color 5
        law = LawInstance(
            law_type="LOCAL_PAINT_ROLE",
            params={"role": 10, "color": 5},
            domain_desc="Role 10"
        )

        result = apply_local_paint(law, grid, role_map)

        # PROVE: Only pixels with role 10 are recolored
        assert result[0][0] == 5, f"Pixel (0,0) has role 10, should be 5, got {result[0][0]}"
        assert result[0][1] == 2, f"Pixel (0,1) has role 20, should stay 2, got {result[0][1]}"
        assert result[1][0] == 2, f"Pixel (1,0) has role 20, should stay 2, got {result[1][0]}"
        assert result[1][1] == 5, f"Pixel (1,1) has role 10, should be 5, got {result[1][1]}"

        # PROVE: Original grid unchanged (deep copy)
        assert grid == [[1, 2], [2, 1]], "Original grid must not be modified"

    def test_AP02_multiple_laws_sequential(self):
        """AP-02: Apply multiple laws sequentially - verify composition."""
        from arc_core.types import LawInstance

        grid = [[0, 1, 2]]
        role_map = {
            Pixel(0, 0): RoleId(0),
            Pixel(0, 1): RoleId(1),
            Pixel(0, 2): RoleId(2),
        }

        law1 = LawInstance(
            law_type="LOCAL_PAINT_ROLE",
            params={"role": 0, "color": 5},
            domain_desc="Role 0"
        )
        law2 = LawInstance(
            law_type="LOCAL_PAINT_ROLE",
            params={"role": 1, "color": 6},
            domain_desc="Role 1"
        )
        law3 = LawInstance(
            law_type="LOCAL_PAINT_ROLE",
            params={"role": 2, "color": 7},
            domain_desc="Role 2"
        )

        # Apply sequentially
        result = grid
        for law in [law1, law2, law3]:
            result = apply_local_paint(law, result, role_map)

        # PROVE: All transformations applied
        assert result == [[5, 6, 7]], \
            f"Expected [[5,6,7]] after applying all laws, got {result}"

    def test_AP03_empty_role_map(self):
        """AP-03: Apply law with empty role_map - no changes."""
        from arc_core.types import LawInstance

        grid = [[1, 2], [3, 4]]
        role_map = {}

        law = LawInstance(
            law_type="LOCAL_PAINT_ROLE",
            params={"role": 10, "color": 9},
            domain_desc="Role 10"
        )

        result = apply_local_paint(law, grid, role_map)

        # PROVE: Grid unchanged (no pixels have roles)
        assert result == grid, "Grid should be unchanged with empty role_map"

    def test_AP04_no_matching_role(self):
        """AP-04: Apply law with no matching roles - grid unchanged."""
        from arc_core.types import LawInstance

        grid = [[1, 2]]
        role_map = {
            Pixel(0, 0): RoleId(5),
            Pixel(0, 1): RoleId(5),
        }

        # Law for role 99 (not in role_map)
        law = LawInstance(
            law_type="LOCAL_PAINT_ROLE",
            params={"role": 99, "color": 7},
            domain_desc="Role 99"
        )

        result = apply_local_paint(law, grid, role_map)

        # PROVE: Grid unchanged
        assert result == grid, "Grid should be unchanged when no pixels have the target role"


class TestLawReproduction:
    """CRITICAL: Test that extracted laws reproduce train outputs pixel-perfectly."""

    def test_LR01_simple_recolor_reproduction(self):
        """LR-01: Verify extracted laws reproduce BOTH train outputs exactly."""
        fixture = load_fixture("simple_recolor.json")
        params = build_params_from_fixture(fixture)

        # Extract laws
        laws = build_local_paint(params)

        # PROVE: Laws reproduce EACH train output
        for grid_id, (X_i, Y_i) in enumerate(params.train_pairs):
            present = params.presents[grid_id]
            X_canonical = present.grid

            # Build role_map for this specific grid
            grid_role_map = {}
            for (gid, pixel), role in params.role_map.items():
                if gid == grid_id:
                    grid_role_map[pixel] = role

            # Apply all laws sequentially to canonical input
            result = [row[:] for row in X_canonical]  # Deep copy
            for law in laws:
                result = apply_local_paint(law, result, grid_role_map)

            # Transform Y to canonical space
            transform_name = present.g_inverse
            forward_transform_name = D4_INVERSES[transform_name]
            forward_transform = D4_TRANSFORMATIONS[forward_transform_name]
            Y_canonical = forward_transform(Y_i)

            # PROVE: Pixel-perfect reproduction
            assert result == Y_canonical, \
                f"Train {grid_id}: Laws must reproduce output exactly.\n" \
                f"Expected: {Y_canonical}\n" \
                f"Got:      {result}\n" \
                f"BUG: Laws extracted but don't reproduce train output!"

    def test_LR02_multi_role_reproduction(self):
        """LR-02: Verify 5-role extraction reproduces output."""
        fixture = load_fixture("multi_role_recolor.json")
        params = build_params_from_fixture(fixture)

        laws = build_local_paint(params)

        # Verify reproduction
        grid_id = 0
        X_i, Y_i = params.train_pairs[grid_id]
        present = params.presents[grid_id]

        grid_role_map = {pixel: role for (gid, pixel), role in params.role_map.items() if gid == grid_id}

        result = [row[:] for row in present.grid]
        for law in laws:
            result = apply_local_paint(law, result, grid_role_map)

        transform_name = present.g_inverse
        forward_transform_name = D4_INVERSES[transform_name]
        Y_canonical = D4_TRANSFORMATIONS[forward_transform_name](Y_i)

        assert result == Y_canonical, \
            f"Multi-role laws must reproduce output exactly. Expected {Y_canonical}, got {result}"

    def test_LR03_partial_consistency_reproduction(self):
        """LR-03: Verify only consistent laws extracted, and they reproduce correctly."""
        fixture = load_fixture("partial_consistency.json")
        params = build_params_from_fixture(fixture)

        laws = build_local_paint(params)

        # PROVE: Only 2 laws extracted (roles 0,1 consistent)
        assert len(laws) == 2, f"Expected 2 consistent laws, got {len(laws)}"

        # PROVE: Laws reproduce BOTH trains
        for grid_id, (X_i, Y_i) in enumerate(params.train_pairs):
            present = params.presents[grid_id]
            grid_role_map = {pixel: role for (gid, pixel), role in params.role_map.items() if gid == grid_id}

            result = [row[:] for row in present.grid]
            for law in laws:
                result = apply_local_paint(law, result, grid_role_map)

            Y_canonical = D4_TRANSFORMATIONS[D4_INVERSES[present.g_inverse]](Y_i)

            # Note: result won't fully match Y_canonical because role 2 is not painted
            # But pixels with roles 0,1 MUST match
            for r in range(len(result)):
                for c in range(len(result[0])):
                    pixel = Pixel(r, c)
                    role = grid_role_map.get(pixel)
                    if role in [RoleId(0), RoleId(1)]:  # Only check painted roles
                        assert result[r][c] == Y_canonical[r][c], \
                            f"Train {grid_id}, pixel ({r},{c}) role {role}: " \
                            f"Expected {Y_canonical[r][c]}, got {result[r][c]}"

    def test_LR04_d4_transform_reproduction(self):
        """LR-04: Verify D4 transform applied correctly and laws reproduce."""
        fixture = load_fixture("d4_transform.json")
        params = build_params_from_fixture(fixture)

        laws = build_local_paint(params)

        # PROVE: 4 laws extracted
        assert len(laws) == 4, f"Expected 4 laws, got {len(laws)}"

        # PROVE: Laws reproduce output after D4 transform
        grid_id = 0
        X_i, Y_i = params.train_pairs[grid_id]
        present = params.presents[grid_id]
        grid_role_map = {pixel: role for (gid, pixel), role in params.role_map.items() if gid == grid_id}

        result = [row[:] for row in present.grid]
        for law in laws:
            result = apply_local_paint(law, result, grid_role_map)

        # Transform Y to canonical space
        Y_canonical = D4_TRANSFORMATIONS[D4_INVERSES[present.g_inverse]](Y_i)

        assert result == Y_canonical, \
            f"D4 transform case: Laws must reproduce canonical output.\n" \
            f"g_inverse={present.g_inverse}, forward={D4_INVERSES[present.g_inverse]}\n" \
            f"Expected: {Y_canonical}\n" \
            f"Got:      {result}\n" \
            f"BUG: D4 transform not applied correctly!"

    def test_LR05_exact_on_all_trains_reproduction(self):
        """LR-05: Verify laws exact on ALL 3 trains by applying them."""
        fixture = load_fixture("exact_on_all_trains.json")
        params = build_params_from_fixture(fixture)

        laws = build_local_paint(params)

        # PROVE: Reproduce ALL 3 trains
        for grid_id, (X_i, Y_i) in enumerate(params.train_pairs):
            present = params.presents[grid_id]
            grid_role_map = {pixel: role for (gid, pixel), role in params.role_map.items() if gid == grid_id}

            result = [row[:] for row in present.grid]
            for law in laws:
                result = apply_local_paint(law, result, grid_role_map)

            Y_canonical = D4_TRANSFORMATIONS[D4_INVERSES[present.g_inverse]](Y_i)

            assert result == Y_canonical, \
                f"Train {grid_id}: Laws must be exact on ALL trains.\n" \
                f"Expected: {Y_canonical}\n" \
                f"Got:      {result}\n" \
                f"BUG: Laws not exact on train {grid_id}!"

    def test_LR06_full_pipeline_reproduction(self):
        """LR-06: Full pipeline - verify extracted laws reproduce all matching trains."""
        fixture = load_fixture("full_pipeline.json")
        params = build_params_from_fixture(fixture)

        laws = build_local_paint(params)

        # PROVE: 4 laws extracted (roles 10-13)
        assert len(laws) == 4, f"Expected 4 consistent laws, got {len(laws)}"

        # PROVE: Reproduce all trains (for consistent roles only)
        for grid_id, (X_i, Y_i) in enumerate(params.train_pairs):
            present = params.presents[grid_id]
            grid_role_map = {pixel: role for (gid, pixel), role in params.role_map.items() if gid == grid_id}

            result = [row[:] for row in present.grid]
            for law in laws:
                result = apply_local_paint(law, result, grid_role_map)

            Y_canonical = D4_TRANSFORMATIONS[D4_INVERSES[present.g_inverse]](Y_i)

            # Check only pixels with consistent roles (10-13)
            for r in range(len(result)):
                for c in range(len(result[0])):
                    pixel = Pixel(r, c)
                    role = grid_role_map.get(pixel)
                    if role in [RoleId(10), RoleId(11), RoleId(12), RoleId(13)]:
                        assert result[r][c] == Y_canonical[r][c], \
                            f"Train {grid_id}, pixel ({r},{c}) role {role}: " \
                            f"Expected {Y_canonical[r][c]}, got {result[r][c]}. " \
                            f"BUG: Consistent law doesn't reproduce!"


class TestD4TransformVerification:
    """CRITICAL: Manually verify D4 transform math is correct."""

    def test_DV01_rot90_manual_verification(self):
        """DV-01: Verify rot90 (90° CW) transform manually."""
        grid = [[1, 2], [3, 4]]

        # Manual calculation: rot90 clockwise
        # result[0][0] = grid[1][0] = 3
        # result[0][1] = grid[0][0] = 1
        # result[1][0] = grid[1][1] = 4
        # result[1][1] = grid[0][1] = 2
        expected = [[3, 1], [4, 2]]

        result = D4_TRANSFORMATIONS["rot90"](grid)

        assert result == expected, \
            f"rot90 manual verification failed.\n" \
            f"Input:    {grid}\n" \
            f"Expected: {expected}\n" \
            f"Got:      {result}\n" \
            f"BUG: rot90 transform incorrect!"

    def test_DV02_rot270_manual_verification(self):
        """DV-02: Verify rot270 (90° CCW) transform manually."""
        grid = [[1, 2], [3, 4]]

        # Manual calculation: rot270 = rot90 counterclockwise
        # result[0][0] = grid[0][1] = 2
        # result[0][1] = grid[1][1] = 4
        # result[1][0] = grid[0][0] = 1
        # result[1][1] = grid[1][0] = 3
        expected = [[2, 4], [1, 3]]

        result = D4_TRANSFORMATIONS["rot270"](grid)

        assert result == expected, \
            f"rot270 manual verification failed.\n" \
            f"Input:    {grid}\n" \
            f"Expected: {expected}\n" \
            f"Got:      {result}\n" \
            f"BUG: rot270 transform incorrect!"

    def test_DV03_rot180_manual_verification(self):
        """DV-03: Verify rot180 (180°) transform manually."""
        grid = [[1, 2], [3, 4]]

        # Manual calculation: rot180
        # Reverse rows, then reverse each row
        # [[3, 4], [1, 2]] -> [[4, 3], [2, 1]]
        expected = [[4, 3], [2, 1]]

        result = D4_TRANSFORMATIONS["rot180"](grid)

        assert result == expected, \
            f"rot180 manual verification failed.\n" \
            f"Input:    {grid}\n" \
            f"Expected: {expected}\n" \
            f"Got:      {result}\n" \
            f"BUG: rot180 transform incorrect!"

    def test_DV04_d4_inverses_correctness(self):
        """DV-04: Verify D4 inverses are correct (T^-1(T(g)) = g)."""
        # Use square grid for D4 inverse verification
        # (Diagonal flips require square grids to be proper inverses)
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        for transform_name, inverse_name in D4_INVERSES.items():
            # Apply transform then inverse
            transformed = D4_TRANSFORMATIONS[transform_name](grid)
            restored = D4_TRANSFORMATIONS[inverse_name](transformed)

            assert restored == grid, \
                f"D4 inverse verification failed for {transform_name}.\n" \
                f"Original:    {grid}\n" \
                f"Transformed: {transformed}\n" \
                f"Restored:    {restored}\n" \
                f"BUG: {transform_name}^-1 = {inverse_name} is incorrect!"

    def test_DV05_fixture_d4_transform_manual_check(self):
        """DV-05: Manually verify d4_transform.json fixture calculations."""
        # From fixture: present.g_inverse = "rot90"
        # To canonicalize Y, we need forward = D4_INVERSES["rot90"] = "rot270"

        Y = [[5, 6], [7, 8]]

        # Manual rot270 calculation:
        # result[0][0] = Y[0][1] = 6
        # result[0][1] = Y[1][1] = 8
        # result[1][0] = Y[0][0] = 5
        # result[1][1] = Y[1][0] = 7
        expected_canonical = [[6, 8], [5, 7]]

        Y_canonical = D4_TRANSFORMATIONS["rot270"](Y)

        assert Y_canonical == expected_canonical, \
            f"d4_transform.json fixture verification failed.\n" \
            f"Y:                {Y}\n" \
            f"Expected Y_canon: {expected_canonical}\n" \
            f"Got Y_canon:      {Y_canonical}\n" \
            f"BUG: Fixture calculation error!"

        # Verify fixture expected values match
        fixture = load_fixture("d4_transform.json")
        expected_laws = {law["role"]: law["color"] for law in fixture["expected"]["laws"]}

        # Role 1 at (0,0) -> Y_canonical[0][0] = 6 ✓
        # Role 2 at (0,1) -> Y_canonical[0][1] = 8 ✓
        # Role 3 at (1,0) -> Y_canonical[1][0] = 5 ✓
        # Role 4 at (1,1) -> Y_canonical[1][1] = 7 ✓
        assert expected_laws == {1: 6, 2: 8, 3: 5, 4: 7}, \
            f"d4_transform.json fixture expected values incorrect. Got {expected_laws}"

    def test_DV06_fixture_present_canonical_manual_check(self):
        """DV-06: Manually verify present_canonical.json fixture calculations."""
        # From fixture: present.g_inverse = "rot180"
        # To canonicalize Y, we need forward = D4_INVERSES["rot180"] = "rot180"

        Y = [[1, 2], [3, 4]]

        # Manual rot180 calculation:
        # [[1,2],[3,4]] -> reverse rows -> [[3,4],[1,2]] -> reverse each -> [[4,3],[2,1]]
        expected_canonical = [[4, 3], [2, 1]]

        Y_canonical = D4_TRANSFORMATIONS["rot180"](Y)

        assert Y_canonical == expected_canonical, \
            f"present_canonical.json fixture verification failed.\n" \
            f"Y:                {Y}\n" \
            f"Expected Y_canon: {expected_canonical}\n" \
            f"Got Y_canon:      {Y_canonical}\n" \
            f"BUG: Fixture calculation error!"

        # Verify fixture expected values match
        fixture = load_fixture("present_canonical.json")
        expected_laws = {law["role"]: law["color"] for law in fixture["expected"]["laws"]}

        # Role 10 at (0,0) -> Y_canonical[0][0] = 4 ✓
        # Role 11 at (0,1) -> Y_canonical[0][1] = 3 ✓
        # Role 12 at (1,0) -> Y_canonical[1][0] = 2 ✓
        # Role 13 at (1,1) -> Y_canonical[1][1] = 1 ✓
        assert expected_laws == {10: 4, 11: 3, 12: 2, 13: 1}, \
            f"present_canonical.json fixture expected values incorrect. Got {expected_laws}"
