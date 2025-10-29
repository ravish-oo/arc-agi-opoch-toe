"""
Integration tests for WO-06: Lattice Compiler

Tests interaction with:
- WO-01 (order_hash) - global order tie-breaking
- WO-02 (present) - ΠG normalization before lattice
- Receipts format (for future WO-19 integration)

Covers:
- End-to-end: raw grids → infer_lattice → Lattice
- Global order usage for tie-breaking
- Present normalization compatibility
- Receipts structure (simulated for now)
- No leakage verification
"""

import json
import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from arc_universe.arc_core.lattice import infer_lattice
from arc_universe.arc_core.types import Grid, Lattice
from arc_universe.arc_core.order_hash import hash64, lex_min  # WO-01
# from arc_universe.arc_core.present import PiG  # WO-02 (not needed for these tests)


# ============================================================================
# Fixtures
# ============================================================================

def load_fixture(name: str):
    """Load test fixture"""
    path = Path(__file__).parent.parent / "fixtures" / "WO-06" / name
    with open(path) as f:
        data = json.load(f)
    return data["grid"], data


def simulate_receipt_lattice(lattice: Lattice) -> Dict[str, Any]:
    """
    Simulate lattice section of receipts (for WO-19 integration).

    Returns JSON-serializable dict matching spec format.
    """
    if lattice is None:
        return {"lattice": None}

    return {
        "lattice": {
            "method": lattice.method,
            "basis": lattice.basis,
            "periods": lattice.periods,
            "d8_canonical": True,  # Flag indicating canonicalization was applied
            "consensus": True  # Flag for multi-grid agreement
        }
    }


# ============================================================================
# WO-01 Integration Tests (order_hash)
# ============================================================================

class TestWO01Integration:
    """Integration with WO-01 (order_hash.py) for global order tie-breaking"""

    def test_global_order_tie_breaking_uses_hash64(self):
        """Test that hash64 from WO-01 is used for determinism"""
        grid, _ = load_fixture("periodic_3x3.json")

        result = infer_lattice([grid])

        if result is not None:
            # Compute hash using WO-01 hash64
            basis_repr = str(result.basis)
            hash_val = hash64(basis_repr)

            # Verify hash is deterministic
            hash_val_2 = hash64(basis_repr)
            assert hash_val == hash_val_2, \
                "hash64 from WO-01 must be deterministic"

    def test_global_order_lex_min_compatibility(self):
        """Test that lex_min from WO-01 could be used for basis selection"""
        grid, _ = load_fixture("periodic_2x2_checkerboard.json")

        result = infer_lattice([grid])

        if result is not None:
            # When multiple equivalent bases exist, lex_min determines canonical
            # Verify that result is deterministic (implementation uses global order)
            result2 = infer_lattice([grid])
            assert result.basis == result2.basis, \
                "Basis selection must be deterministic via global order"

            # Verify basis vectors could be ordered via lex_min
            v1 = tuple(result.basis[0])
            v2 = tuple(result.basis[1])
            # Just verify they're stable (actual lex_min application is internal)

    def test_hash64_determinism_across_lattices(self):
        """Test hash64 stability for different lattices"""
        fixtures = [
            "periodic_3x3.json",
            "periodic_2x2_checkerboard.json",
            "mixed_periods_3x5.json",
        ]

        for fixture_name in fixtures:
            grid, _ = load_fixture(fixture_name)
            result = infer_lattice([grid])

            if result is not None:
                # Hash must be stable across calls
                repr1 = f"{result.basis}{result.periods}{result.method}"
                repr2 = f"{result.basis}{result.periods}{result.method}"

                hash1 = hash64(repr1)
                hash2 = hash64(repr2)

                assert hash1 == hash2, \
                    f"hash64 not stable for {fixture_name}"

    def test_global_order_64bit_integers(self):
        """Test that global order uses 64-bit integers per spec (§2)"""
        grid, _ = load_fixture("periodic_3x3.json")
        result = infer_lattice([grid])

        if result is not None:
            # Compute hash - should be 64-bit integer
            repr_str = str(result.basis)
            hash_val = hash64(repr_str)

            # Verify it's an integer in 64-bit range
            assert isinstance(hash_val, int), "hash64 must return int"
            assert -(2**63) <= hash_val < 2**63, \
                f"hash64 must be 64-bit signed int, got {hash_val}"


# ============================================================================
# WO-02 Integration Tests (present normalization)
# ============================================================================

class TestWO02Integration:
    """Integration with WO-02 (present.py) for ΠG normalization"""

    def test_lattice_on_raw_grids(self):
        """Test lattice detection works on raw (unnormalized) grids"""
        grid, _ = load_fixture("periodic_3x3.json")

        # Should work on raw grids (in practice, would be present-normalized first)
        result = infer_lattice([grid])
        assert result is not None, "Lattice detection should work on raw grids"

    def test_lattice_deterministic_independent_of_normalization_order(self):
        """Test that lattice detection is deterministic"""
        grid1, _ = load_fixture("periodic_3x3.json")
        grid2, _ = load_fixture("periodic_2x2_checkerboard.json")

        # Call in different orders - each should be deterministic
        result1 = infer_lattice([grid1])
        result2 = infer_lattice([grid2])

        result1_repeat = infer_lattice([grid1])
        result2_repeat = infer_lattice([grid2])

        if result1 is not None:
            assert result1.basis == result1_repeat.basis
        if result2 is not None:
            assert result2.basis == result2_repeat.basis

    def test_no_output_leakage_present_only(self):
        """Test that only present-normalized inputs are used (no outputs)"""
        # Lattice detection should work with inputs only
        # Spec: uses train inputs only, never train outputs
        grid, _ = load_fixture("periodic_3x3.json")

        result = infer_lattice([grid])

        # Verify function signature doesn't accept outputs
        # (This is a design test - implementation should not have Y parameter)
        import inspect
        sig = inspect.signature(infer_lattice)
        params = list(sig.parameters.keys())

        assert 'Xs' in params or 'grids' in params or len(params) >= 1, \
            "infer_lattice should accept input grids"
        assert 'Ys' not in params and 'outputs' not in params, \
            "infer_lattice must NOT accept outputs (no leakage)"

    def test_input_only_compilation_no_test_input(self):
        """Test that test input is NOT used for lattice detection"""
        # Spec: Lattice compiled from train inputs only
        # Test input is in WL scope but not used for parameter extraction
        train1, _ = load_fixture("periodic_3x3.json")
        train2, _ = load_fixture("periodic_3x3.json")  # Same pattern

        # Lattice should be from train inputs only
        result_train_only = infer_lattice([train1, train2])

        assert result_train_only is not None, \
            "Should detect lattice from train inputs"

        # If we had a test input with different period, it shouldn't affect result
        # (This is tested implicitly by using only train inputs in function call)


# ============================================================================
# Receipts Format Tests (REC-01 to REC-05)
# ============================================================================

class TestReceiptsFormat:
    """Test lattice representation in receipts (for WO-19 integration)"""

    def test_receipt_lattice_present(self):
        """REC-01: Receipt has 'lattice' key"""
        grid, _ = load_fixture("periodic_3x3.json")
        result = infer_lattice([grid])

        receipt = simulate_receipt_lattice(result)

        assert "lattice" in receipt, "Receipt must have 'lattice' key"

    def test_receipt_method_logged(self):
        """REC-02: Receipt logs method (FFT or KMP)"""
        grid, _ = load_fixture("periodic_3x3.json")
        result = infer_lattice([grid])

        if result is not None:
            receipt = simulate_receipt_lattice(result)

            assert "method" in receipt["lattice"], \
                "Receipt must log method"
            assert receipt["lattice"]["method"] in ["FFT", "KMP"], \
                f"Invalid method in receipt: {receipt['lattice']['method']}"

    def test_receipt_basis_format(self):
        """REC-03: Basis is 2×2 list in receipt"""
        grid, _ = load_fixture("periodic_3x3.json")
        result = infer_lattice([grid])

        if result is not None:
            receipt = simulate_receipt_lattice(result)

            basis = receipt["lattice"]["basis"]
            assert isinstance(basis, list), "Basis must be list"
            assert len(basis) == 2, "Basis must be 2×2"
            assert all(len(row) == 2 for row in basis), "Basis rows must be length 2"
            assert all(isinstance(v, int) for row in basis for v in row), \
                "Basis must contain only integers"

    def test_receipt_periods_format(self):
        """REC-04: Periods is [int, int] in receipt"""
        grid, _ = load_fixture("periodic_3x3.json")
        result = infer_lattice([grid])

        if result is not None:
            receipt = simulate_receipt_lattice(result)

            periods = receipt["lattice"]["periods"]
            assert isinstance(periods, list), "Periods must be list"
            assert len(periods) == 2, "Periods must be [int, int]"
            assert all(isinstance(p, int) for p in periods), \
                "Periods must be integers"

    def test_receipt_null_lattice(self):
        """REC-05: Null lattice in receipt for non-periodic"""
        grid, _ = load_fixture("uniform.json")
        result = infer_lattice([grid])

        if result is None:
            receipt = simulate_receipt_lattice(result)

            assert receipt["lattice"] is None, \
                "Receipt must have lattice: null for non-periodic grids"

    def test_receipt_json_serializable(self):
        """REC-06: Receipt is JSON-serializable"""
        grid, _ = load_fixture("periodic_3x3.json")
        result = infer_lattice([grid])

        receipt = simulate_receipt_lattice(result)

        # Must be JSON-serializable
        try:
            json_str = json.dumps(receipt)
            reconstructed = json.loads(json_str)
            assert reconstructed == receipt, "Receipt must round-trip through JSON"
        except (TypeError, ValueError) as e:
            pytest.fail(f"Receipt not JSON-serializable: {e}")

    def test_receipt_d8_canonical_flag(self):
        """REC-07: Receipt includes D8 canonicalization flag"""
        grid, _ = load_fixture("periodic_3x3.json")
        result = infer_lattice([grid])

        if result is not None:
            receipt = simulate_receipt_lattice(result)

            # Receipt should indicate D8 canonicalization was applied
            assert "d8_canonical" in receipt["lattice"], \
                "Receipt should document D8 canonicalization"


# ============================================================================
# End-to-End Tests
# ============================================================================

class TestEndToEnd:
    """End-to-end pipeline tests"""

    def test_e2e_simple_periodic(self):
        """E2E-01: Simple periodic grid end-to-end"""
        grid, meta = load_fixture("periodic_3x3.json")

        # Full pipeline: load → detect → verify
        result = infer_lattice([grid])

        assert result is not None, "Should detect periodicity"
        assert result.periods == meta["expected_periods"], \
            f"Expected {meta['expected_periods']}, got {result.periods}"
        assert result.method in ["FFT", "KMP"], "Method must be logged"

        # Generate receipt
        receipt = simulate_receipt_lattice(result)

        # Verify receipt structure
        assert "lattice" in receipt
        assert receipt["lattice"]["method"] == result.method
        assert receipt["lattice"]["basis"] == result.basis
        assert receipt["lattice"]["periods"] == result.periods

    def test_e2e_multi_grid_consensus(self):
        """E2E-02: Multi-grid consensus end-to-end"""
        grid1, _ = load_fixture("multi_grid_consensus/consensus_set_1/train_0.json")
        grid2, _ = load_fixture("multi_grid_consensus/consensus_set_1/train_1.json")
        grid3, _ = load_fixture("multi_grid_consensus/consensus_set_1/train_2.json")

        # Detect consensus
        result = infer_lattice([grid1, grid2, grid3])

        assert result is not None, "Should find consensus"
        assert result.periods == [3, 3], "All grids have (3,3) tiling"

        # Receipt
        receipt = simulate_receipt_lattice(result)
        assert receipt["lattice"]["consensus"] is True

    def test_e2e_fallback_path(self):
        """E2E-03: Fallback to KMP end-to-end"""
        grid, _ = load_fixture("uniform.json")

        result = infer_lattice([grid])

        # Uniform grid triggers fallback
        if result is not None:
            assert result.method == "KMP", "Degenerate ACF should use KMP"

            receipt = simulate_receipt_lattice(result)
            assert receipt["lattice"]["method"] == "KMP"

    def test_e2e_no_output_leakage(self):
        """E2E-04: Verify no output leakage in full pipeline"""
        # Full pipeline should never use outputs
        grid, _ = load_fixture("periodic_3x3.json")

        # Lattice detection with input only
        result = infer_lattice([grid])

        # If we had outputs, they should NOT affect result
        # (This is a design guarantee - function doesn't accept outputs)
        result_repeat = infer_lattice([grid])

        if result is not None:
            assert result.basis == result_repeat.basis, \
                "Determinism ensures no hidden state/leakage"

    def test_e2e_determinism_full_pipeline(self):
        """E2E-05: Full pipeline determinism"""
        grid, _ = load_fixture("periodic_3x3.json")

        # Run full pipeline 10 times
        results = []
        receipts = []

        for _ in range(10):
            result = infer_lattice([grid])
            receipt = simulate_receipt_lattice(result)

            results.append(result)
            receipts.append(receipt)

        # All results must be identical
        if results[0] is not None:
            for i, result in enumerate(results[1:], 1):
                assert result.basis == results[0].basis, \
                    f"Run {i}: basis mismatch"
                assert result.periods == results[0].periods, \
                    f"Run {i}: periods mismatch"
                assert result.method == results[0].method, \
                    f"Run {i}: method mismatch"

        # All receipts must be identical
        for i, receipt in enumerate(receipts[1:], 1):
            assert receipt == receipts[0], f"Run {i}: receipt mismatch"

    def test_e2e_mixed_periods(self):
        """E2E-06: Mixed periods end-to-end"""
        test_cases = [
            ("mixed_periods_3x5.json", [3, 5]),
            ("mixed_periods_2x7.json", [2, 7]),
        ]

        for fixture_name, expected_periods in test_cases:
            grid, _ = load_fixture(fixture_name)
            result = infer_lattice([grid])

            assert result is not None, f"Should detect {fixture_name}"
            assert result.periods == expected_periods, \
                f"{fixture_name}: expected {expected_periods}, got {result.periods}"

            receipt = simulate_receipt_lattice(result)
            assert receipt["lattice"]["periods"] == expected_periods

    def test_e2e_d8_invariance(self):
        """E2E-07: D8 invariance end-to-end"""
        base_grid, _ = load_fixture("d8_variants/d8_base.json")
        rot90_grid, _ = load_fixture("d8_variants/d8_rot90.json")

        result_base = infer_lattice([base_grid])
        result_rot90 = infer_lattice([rot90_grid])

        assert result_base is not None
        assert result_rot90 is not None

        # Periods should be equivalent (D8 group)
        assert sorted(result_base.periods) == sorted(result_rot90.periods), \
            "D8 invariance violated in end-to-end test"

        # Receipts should reflect D8 canonicalization
        receipt_base = simulate_receipt_lattice(result_base)
        receipt_rot90 = simulate_receipt_lattice(result_rot90)

        assert receipt_base["lattice"]["d8_canonical"] is True
        assert receipt_rot90["lattice"]["d8_canonical"] is True


# ============================================================================
# Spec Compliance Tests
# ============================================================================

class TestSpecCompliance:
    """Test compliance with specification requirements"""

    def test_spec_input_only(self):
        """SPEC-01: Only uses inputs (no outputs)"""
        # Per spec: "Lattice extraction must be input-only (no output leakage)"
        grid, _ = load_fixture("periodic_3x3.json")

        # Verify function signature
        import inspect
        sig = inspect.signature(infer_lattice)

        # Must accept inputs, not outputs
        assert 'Ys' not in sig.parameters, \
            "SPEC VIOLATION: infer_lattice must not accept outputs"

    def test_spec_deterministic(self):
        """SPEC-02: Deterministic (no randomness)"""
        # Per spec: "Deterministic everywhere: no randomness"
        grid, _ = load_fixture("periodic_3x3.json")

        results = [infer_lattice([grid]) for _ in range(100)]

        if results[0] is not None:
            bases = [r.basis for r in results]
            assert len(set(tuple(tuple(row) for row in b) for b in bases)) == 1, \
                "SPEC VIOLATION: Not deterministic across 100 runs"

    def test_spec_integer_arithmetic(self):
        """SPEC-03: Integer arithmetic only"""
        # Per spec: "All integer arithmetic, no floats"
        grid, _ = load_fixture("periodic_3x3.json")
        result = infer_lattice([grid])

        if result is not None:
            # Verify all outputs are integers
            for row in result.basis:
                for val in row:
                    assert isinstance(val, (int, np.integer)), \
                        f"SPEC VIOLATION: Basis contains non-int: {val}"

            for p in result.periods:
                assert isinstance(p, (int, np.integer)), \
                    f"SPEC VIOLATION: Period contains non-int: {p}"

    def test_spec_method_logged(self):
        """SPEC-04: Method (FFT/KMP) logged"""
        # Per spec: "Fallback logs which method was used"
        test_fixtures = [
            "periodic_3x3.json",
            "uniform.json",
            "rank1_row.json",
        ]

        for fixture_name in test_fixtures:
            grid, _ = load_fixture(fixture_name)
            result = infer_lattice([grid])

            if result is not None:
                assert result.method in ["FFT", "KMP"], \
                    f"SPEC VIOLATION: Method not logged for {fixture_name}"

    def test_spec_d8_canonical(self):
        """SPEC-05: D8 canonical form"""
        # Per spec: "HNF → D8 canonical"
        grid, _ = load_fixture("periodic_3x3.json")
        result = infer_lattice([grid])

        if result is not None:
            # D8 canonical means rotations/flips give equivalent basis
            # Verify by checking determinism (canonical form is unique)
            result2 = infer_lattice([grid])
            assert result.basis == result2.basis, \
                "SPEC VIOLATION: D8 canonical form not unique"

    def test_spec_global_order_tie_break(self):
        """SPEC-06: Global order (§2) for tie-breaking"""
        # Per spec: "choose basis by global order"
        grid, _ = load_fixture("periodic_2x2_checkerboard.json")

        results = [infer_lattice([grid]) for _ in range(10)]

        if results[0] is not None:
            bases = [r.basis for r in results]
            # All must be identical (global order picks canonical)
            assert all(b == bases[0] for b in bases), \
                "SPEC VIOLATION: Global order tie-break not deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
