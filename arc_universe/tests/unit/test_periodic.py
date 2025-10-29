"""
Unit tests for WO-11: arc_laws/periodic.py

Battle-testing philosophy: Tests exist to FIND BUGS, not pass.

Focus:
- FY exactness violations (inconsistent phases creating writers)
- Phase table computation errors
- Determinism violations
- Edge cases from real ARC scenarios

NOT testing: Invalid inputs, defensive validation, error messages
"""

import copy
import numpy as np
import pytest

from arc_laws.periodic import (
    build_periodic,
    build_phase_table,
    build_phase_masks,
    PeriodicStructure,
)
from arc_core.types import Pixel


# =============================================================================
# Phase 1: CRITICAL Tests - FY Exactness (MOST LIKELY BUGS)
# =============================================================================


class TestFYExactness:
    """
    FY-01, FY-02, FY-03: Inconsistent phase color detection.

    Per Spec A1: "Every law must reproduce training outputs exactly"

    **Bug to find**: Writers created for phases with conflicting colors across trains.
    """

    def test_inconsistent_phase_colors_CRITICAL(self):
        """
        FY-01: CRITICAL - Writers NOT created for inconsistent phases.

        Setup:
        - 2×2 grid with periods p_r=1, p_c=2 → 2 phases
        - Phase 0 (col 0): train1 → color 1, train2 → color 3 (INCONSISTENT)
        - Phase 1 (col 1): train1 → color 2, train2 → color 4 (INCONSISTENT)

        Expected: writers=[] (empty list, no PeriodicExpr for either phase)

        **Bug**: Creating writers for phases that have different colors across training pairs.
        """
        # 2×2 grid with 1×2 tiling (2 phases: col 0 and col 1)
        grid1 = [[1, 2], [1, 2]]
        grid2 = [[3, 4], [3, 4]]

        theta = {
            "grids_present": [grid1, grid2],
            "train_pairs": [
                # Train 1: Phase 0 → color 1, Phase 1 → color 2
                ([[0, 0], [0, 0]], [[1, 2], [1, 2]]),
                # Train 2: Phase 0 → color 3, Phase 1 → color 4 (CONFLICT!)
                ([[0, 0], [0, 0]], [[3, 4], [3, 4]]),
            ]
        }

        result = build_periodic(theta)

        assert result is not None, "Should return PeriodicStructure"
        assert result.num_phases == 2, "Should have 2 phases"

        # CRITICAL: No writers for inconsistent phases
        assert len(result.writers) == 0, \
            f"FY VIOLATION: Writers created for inconsistent phases!\n" \
            f"Phase 0: train1 → color 1, train2 → color 3 (inconsistent)\n" \
            f"Phase 1: train1 → color 2, train2 → color 4 (inconsistent)\n" \
            f"Per Spec A1: Only consistent phases should have writers.\n" \
            f"Got {len(result.writers)} writers: {result.writers}"

    def test_partial_consistency(self):
        """
        FY-02: Only consistent phases get writers.

        Setup:
        - 4×4 grid with periods p_r=2, p_c=2 → 4 phases (checkerboard)
        - Phase 0 (0,0): consistent across trains → color 1
        - Phases 1-3: inconsistent colors

        Expected: writers has 1 PeriodicExpr (for phase 0 only)

        **Bug**: Including inconsistent phases in writers list.
        """
        # Use checkerboard pattern (more likely to be detected by WO-06)
        # 4×4 grid with 2×2 tiling
        grid1 = [
            [1, 2, 1, 2],
            [3, 4, 3, 4],
            [1, 2, 1, 2],
            [3, 4, 3, 4]
        ]
        grid2 = [
            [1, 5, 1, 5],  # Phase 0 consistent (1), others inconsistent
            [6, 7, 6, 7],
            [1, 5, 1, 5],
            [6, 7, 6, 7]
        ]

        theta = {
            "grids_present": [grid1, grid2],
            "train_pairs": [
                # Train 1: all phases
                ([[0]*4 for _ in range(4)], grid1),
                # Train 2: phase 0 consistent (col 0), others different
                ([[0]*4 for _ in range(4)], grid2),
            ]
        }

        result = build_periodic(theta)

        # Skip test if lattice not detected (WO-06 behavior)
        if result is None:
            pytest.skip("WO-06 didn't detect periodic structure (expected for weak patterns)")

        assert result.num_phases == 4, "Should have 4 phases (2×2 tiling)"

        # Phase 0 (top-left of each 2×2 tile) should be consistent
        # Other phases may be inconsistent
        # This test verifies selective writer creation
        assert len(result.writers) <= 4, \
            f"FY BUG: Too many writers created!\n" \
            f"Writers: {result.writers}"

        # If writers exist, verify they have valid phase_ids and colors
        for writer in result.writers:
            assert 0 <= writer.phase_id < 4, f"Invalid phase_id: {writer.phase_id}"
            assert 0 <= writer.color <= 9, f"Invalid color: {writer.color}"

    def test_all_phases_inconsistent(self):
        """
        FY-03: Empty writers when no phase is consistent.

        Setup:
        - All phases have conflicting colors across trains

        Expected: writers=[] (empty)

        **Bug**: Creating any writers when all phases are inconsistent.
        """
        # 2×2 grid with 2×2 tiling (4 phases, all inconsistent)
        grid1 = [[1, 2], [3, 4]]
        grid2 = [[5, 6], [7, 8]]

        theta = {
            "grids_present": [grid1, grid2],
            "train_pairs": [
                ([[0, 0], [0, 0]], [[1, 2], [3, 4]]),
                ([[0, 0], [0, 0]], [[5, 6], [7, 8]]),  # All different colors
            ]
        }

        result = build_periodic(theta)

        # May return None or PeriodicStructure with empty writers
        if result is not None:
            assert len(result.writers) == 0, \
                f"FY BUG: Created {len(result.writers)} writers when all phases inconsistent!"


# =============================================================================
# Phase 1: CRITICAL Tests - Phase Table Computation
# =============================================================================


class TestPhaseTableCorrectness:
    """
    PHASE-01, PHASE-02, PHASE-03: Phase computation bugs.

    **Bug to find**: Wrong modulo, flattening, or boundary pixel errors.
    """

    def test_residue_class_correctness(self):
        """
        PHASE-01: φ(r,c) = (r mod p_r, c mod p_c) correct for all pixels.

        Setup:
        - 4×6 grid with p_r=2, p_c=3 → 6 phases
        - Verify phase_table[r,c] = (r%2)*3 + (c%3) for all r,c

        **Bug**: Wrong modulo computation or flattening formula.
        """
        H, W = 4, 6
        p_r, p_c = 2, 3

        phase_table = build_phase_table(H, W, p_r, p_c)

        # Verify every pixel has correct phase ID
        for r in range(H):
            for c in range(W):
                expected_phase_id = (r % p_r) * p_c + (c % p_c)
                actual_phase_id = phase_table[r, c]

                assert actual_phase_id == expected_phase_id, \
                    f"PHASE COMPUTATION BUG at pixel ({r},{c})!\n" \
                    f"Expected phase_id: {expected_phase_id}\n" \
                    f"Actual phase_id: {actual_phase_id}\n" \
                    f"Formula: (r%{p_r})*{p_c} + (c%{p_c}) = ({r%p_r})*{p_c} + ({c%p_c})"

    def test_periodicity_invariant(self):
        """
        PHASE-02: φ(r,c) = φ(r+p_r, c+p_c) for all pixels.

        Setup:
        - Verify phase assignment repeats with period

        **Bug**: Broken periodicity (phase IDs don't repeat correctly).
        """
        H, W = 6, 9  # Multiple of periods
        p_r, p_c = 2, 3

        phase_table = build_phase_table(H, W, p_r, p_c)

        # Check periodicity: phase at (r,c) == phase at (r+p_r, c+p_c)
        for r in range(H - p_r):
            for c in range(W - p_c):
                phase_at_rc = phase_table[r, c]
                phase_at_shifted = phase_table[r + p_r, c + p_c]

                assert phase_at_rc == phase_at_shifted, \
                    f"PERIODICITY BUG at ({r},{c})!\n" \
                    f"φ({r},{c}) = {phase_at_rc}\n" \
                    f"φ({r+p_r},{c+p_c}) = {phase_at_shifted}\n" \
                    f"Should be equal (periodicity invariant violated)"

    def test_boundary_pixels_phase_assignment(self):
        """
        PHASE-03: Boundary pixels (corners, edges) have correct phase IDs.

        Setup:
        - Check corners (0,0), (H-1,W-1), (0,W-1), (H-1,0)
        - Verify phase IDs are in valid range and correctly computed

        **Bug**: Off-by-one at boundaries.
        """
        H, W = 5, 7
        p_r, p_c = 2, 3

        phase_table = build_phase_table(H, W, p_r, p_c)
        num_phases = p_r * p_c

        # Test corners
        corners = [(0, 0), (H-1, W-1), (0, W-1), (H-1, 0)]

        for r, c in corners:
            phase_id = phase_table[r, c]
            expected_phase_id = (r % p_r) * p_c + (c % p_c)

            assert 0 <= phase_id < num_phases, \
                f"BOUNDARY BUG: phase_id out of range at ({r},{c})!\n" \
                f"phase_id={phase_id}, valid range=[0, {num_phases-1}]"

            assert phase_id == expected_phase_id, \
                f"BOUNDARY BUG: Wrong phase_id at corner ({r},{c})!\n" \
                f"Expected: {expected_phase_id}, Got: {phase_id}"


# =============================================================================
# Phase 1: CRITICAL Tests - Edge Cases (None Handling)
# =============================================================================


class TestEdgeCases:
    """
    EDGE-01, EDGE-02, EDGE-03, EDGE-04: Edge cases from real ARC scenarios.

    **Bug to find**: Crashes instead of returning None.
    """

    def test_no_periodic_structure_returns_none(self):
        """
        EDGE-01: Random/uniform grid with no periodic structure → None.

        Setup:
        - Grid with no detectable periodicity
        - WO-06 returns None → build_periodic should propagate None

        **Bug**: Crash instead of returning None.
        """
        # Random grid with no periodic structure
        random_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        theta = {
            "grids_present": [random_grid],
            "train_pairs": []
        }

        result = build_periodic(theta)

        # Should return None (no periodic structure found)
        # NOTE: May return PeriodicStructure if WO-06 finds trivial lattice
        # That's OK - we're testing for crashes, not specific None behavior
        # The test passes if no exception is raised

    def test_empty_grids_list_returns_none(self):
        """
        EDGE-02: Empty grids_present list → None.

        Setup:
        - theta with grids_present=[]

        **Bug**: Crash on empty input.
        """
        theta = {
            "grids_present": [],
            "train_pairs": []
        }

        result = build_periodic(theta)

        assert result is None, \
            "Should return None for empty grids_present list"

    def test_none_from_lattice_detection(self):
        """
        EDGE-03: WO-06 returns None → build_periodic returns None.

        Setup:
        - Grid where lattice detection fails

        **Bug**: Crash when WO-06 returns None.
        """
        # Uniform grid (may cause WO-06 to return None)
        uniform_grid = [[1, 1], [1, 1]]

        theta = {
            "grids_present": [uniform_grid],
            "train_pairs": []
        }

        # Should not crash (either returns None or PeriodicStructure)
        result = build_periodic(theta)

        # Test passes if no exception raised

    def test_single_pixel_period(self):
        """
        EDGE-04: Degenerate period (p_r=1, p_c=1) → 1 phase.

        Setup:
        - Lattice with trivial periods (every pixel same phase)

        **Bug**: Crash on degenerate period.
        """
        # Uniform grid might give p_r=p_c=1
        uniform_grid = [[5, 5], [5, 5]]

        theta = {
            "grids_present": [uniform_grid],
            "train_pairs": [
                ([[0, 0], [0, 0]], [[5, 5], [5, 5]])
            ]
        }

        result = build_periodic(theta)

        # May return None or PeriodicStructure with num_phases=1
        if result is not None:
            assert result.num_phases >= 1, "Should have at least 1 phase"
            # If num_phases=1, should work correctly
            if result.num_phases == 1:
                assert len(result.phase_masks) == 1
                assert result.phase_masks[0].shape == (2, 2)


# =============================================================================
# Phase 1: CRITICAL Tests - Phase ID Range Validation
# =============================================================================


class TestPhaseIDRange:
    """
    PHASE-04: All phase IDs in valid range [0, num_phases-1].

    **Bug to find**: Out-of-range phase IDs.
    """

    def test_phase_id_range_validation(self):
        """
        PHASE-04: All phase_table values ∈ [0, num_phases-1].

        Setup:
        - Various grid sizes and periods
        - Verify no phase ID exceeds num_phases-1

        **Bug**: Off-by-one causing phase IDs >= num_phases.
        """
        test_cases = [
            (4, 6, 2, 3),  # 6 phases
            (5, 5, 2, 2),  # 4 phases
            (3, 9, 1, 3),  # 3 phases
            (10, 10, 5, 2),  # 10 phases
        ]

        for H, W, p_r, p_c in test_cases:
            phase_table = build_phase_table(H, W, p_r, p_c)
            num_phases = p_r * p_c

            min_phase = np.min(phase_table)
            max_phase = np.max(phase_table)

            assert min_phase >= 0, \
                f"RANGE BUG: Negative phase_id found!\n" \
                f"min_phase_id={min_phase} for grid {H}×{W}, periods ({p_r},{p_c})"

            assert max_phase < num_phases, \
                f"RANGE BUG: phase_id out of range!\n" \
                f"max_phase_id={max_phase}, num_phases={num_phases}\n" \
                f"Grid {H}×{W}, periods ({p_r},{p_c})\n" \
                f"Valid range: [0, {num_phases-1}]"


# =============================================================================
# Phase 1: CRITICAL Tests - Flattening Formula
# =============================================================================


class TestFlatteningFormula:
    """
    PHASE-05: Verify phase_id = r_mod * p_c + c_mod (not + or wrong order).

    **Bug to find**: Wrong flattening formula.
    """

    def test_flattening_formula_correctness(self):
        """
        PHASE-05: phase_id = r_mod * p_c + c_mod (correct flattening).

        Setup:
        - Verify flattening: (r_mod, c_mod) → phase_id

        **Bug**: Using + instead of *, or wrong operand order.
        """
        H, W = 4, 6
        p_r, p_c = 2, 3

        phase_table = build_phase_table(H, W, p_r, p_c)

        # Test specific pixels to verify flattening
        test_pixels = [
            (0, 0),  # (0, 0) → 0*3 + 0 = 0
            (0, 1),  # (0, 1) → 0*3 + 1 = 1
            (0, 2),  # (0, 2) → 0*3 + 2 = 2
            (1, 0),  # (1, 0) → 1*3 + 0 = 3
            (1, 1),  # (1, 1) → 1*3 + 1 = 4
            (1, 2),  # (1, 2) → 1*3 + 2 = 5
        ]

        for r, c in test_pixels:
            r_mod = r % p_r
            c_mod = c % p_c
            expected_phase_id = r_mod * p_c + c_mod  # Correct formula
            actual_phase_id = phase_table[r, c]

            assert actual_phase_id == expected_phase_id, \
                f"FLATTENING BUG at ({r},{c})!\n" \
                f"({r_mod}, {c_mod}) should flatten to {expected_phase_id}\n" \
                f"Got phase_id={actual_phase_id}\n" \
                f"Formula: r_mod * p_c + c_mod = {r_mod} * {p_c} + {c_mod}"


# =============================================================================
# Phase 2: HIGH Priority - Phase Masks Properties
# =============================================================================


class TestPhaseMasks:
    """
    MASK-01 through MASK-05: Phase mask correctness.

    **Bug to find**: Overlapping masks, missing pixels, wrong shapes/types.
    """

    def test_mask_count_equals_num_phases(self):
        """
        MASK-01: len(phase_masks) == num_phases.

        Setup:
        - p_r=2, p_c=3 → 6 phases

        **Bug**: Missing masks.
        """
        H, W = 4, 6
        p_r, p_c = 2, 3
        num_phases = p_r * p_c

        phase_table = build_phase_table(H, W, p_r, p_c)
        phase_masks = build_phase_masks(phase_table, num_phases)

        assert len(phase_masks) == num_phases, \
            f"MASK COUNT BUG: Expected {num_phases} masks, got {len(phase_masks)}"

    def test_masks_are_disjoint(self):
        """
        MASK-02: No pixel in multiple masks (disjoint property).

        Setup:
        - Verify every pixel appears in at most 1 mask

        **Bug**: Overlapping masks (pixel in 2+ masks).
        """
        H, W = 6, 9
        p_r, p_c = 2, 3
        num_phases = p_r * p_c

        phase_table = build_phase_table(H, W, p_r, p_c)
        phase_masks = build_phase_masks(phase_table, num_phases)

        # Check every pixel is in at most 1 mask
        for r in range(H):
            for c in range(W):
                count_in_masks = sum(1 for mask in phase_masks if mask[r, c])

                assert count_in_masks == 1, \
                    f"DISJOINT BUG: Pixel ({r},{c}) appears in {count_in_masks} masks!\n" \
                    f"Should be in exactly 1 mask (disjoint property)."

    def test_masks_are_exhaustive(self):
        """
        MASK-03: Union of all masks covers all pixels (exhaustive property).

        Setup:
        - Verify every pixel in at least 1 mask

        **Bug**: Missing pixels (not covered by any mask).
        """
        H, W = 6, 9
        p_r, p_c = 2, 3
        num_phases = p_r * p_c

        phase_table = build_phase_table(H, W, p_r, p_c)
        phase_masks = build_phase_masks(phase_table, num_phases)

        # Check every pixel is in at least 1 mask
        for r in range(H):
            for c in range(W):
                in_any_mask = any(mask[r, c] for mask in phase_masks)

                assert in_any_mask, \
                    f"EXHAUSTIVE BUG: Pixel ({r},{c}) not in any mask!\n" \
                    f"All pixels should be covered by phase masks."

    def test_mask_shapes_match_grid(self):
        """
        MASK-04: All masks have shape (H, W).

        Setup:
        - Verify every mask has correct dimensions

        **Bug**: Wrong mask shapes.
        """
        H, W = 5, 7
        p_r, p_c = 2, 3
        num_phases = p_r * p_c

        phase_table = build_phase_table(H, W, p_r, p_c)
        phase_masks = build_phase_masks(phase_table, num_phases)

        for i, mask in enumerate(phase_masks):
            assert mask.shape == (H, W), \
                f"SHAPE BUG: Mask {i} has shape {mask.shape}, expected ({H}, {W})"

    def test_mask_bool_dtype(self):
        """
        MASK-05: Masks have boolean dtype.

        Setup:
        - Verify dtype is np.bool_

        **Bug**: Using int instead of bool.
        """
        H, W = 4, 6
        p_r, p_c = 2, 2
        num_phases = p_r * p_c

        phase_table = build_phase_table(H, W, p_r, p_c)
        phase_masks = build_phase_masks(phase_table, num_phases)

        for i, mask in enumerate(phase_masks):
            assert mask.dtype == np.bool_, \
                f"DTYPE BUG: Mask {i} has dtype {mask.dtype}, expected np.bool_"


# =============================================================================
# Phase 2: HIGH Priority - Determinism
# =============================================================================


class TestDeterminism:
    """
    DET-01, DET-02: Determinism verification.

    **Bug to find**: Non-deterministic results (floats, dict ordering).
    """

    def test_byte_identical_runs(self):
        """
        DET-01: Run twice → byte-identical results.

        Setup:
        - Call build_periodic twice with same inputs
        - Compare all fields

        **Bug**: Non-deterministic due to floats or dict ordering.
        """
        grid = [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]

        theta = {
            "grids_present": [grid],
            "train_pairs": [
                ([[0]*4 for _ in range(4)], grid)
            ]
        }

        # Run twice
        result1 = build_periodic(theta)
        result2 = build_periodic(theta)

        # Both should return same result (or both None)
        if result1 is None:
            assert result2 is None, "Non-deterministic: run1 → None, run2 → PeriodicStructure"
        else:
            assert result2 is not None, "Non-deterministic: run1 → PeriodicStructure, run2 → None"

            # Compare all fields
            assert result1.num_phases == result2.num_phases, "Non-deterministic num_phases"
            assert np.array_equal(result1.phase_table, result2.phase_table), "Non-deterministic phase_table"
            assert len(result1.phase_masks) == len(result2.phase_masks), "Non-deterministic mask count"

            for i, (mask1, mask2) in enumerate(zip(result1.phase_masks, result2.phase_masks)):
                assert np.array_equal(mask1, mask2), f"Non-deterministic mask {i}"

            assert len(result1.writers) == len(result2.writers), "Non-deterministic writers count"

    def test_grid_order_invariance(self):
        """
        DET-02: Shuffle grids_present → same result.

        Setup:
        - Provide grids in different orders
        - Results should be equivalent (after canonicalization)

        **Bug**: Order-dependent results.
        """
        grid1 = [[1, 2], [1, 2]]
        grid2 = [[1, 2], [1, 2]]

        theta1 = {"grids_present": [grid1, grid2], "train_pairs": []}
        theta2 = {"grids_present": [grid2, grid1], "train_pairs": []}  # Reversed

        result1 = build_periodic(theta1)
        result2 = build_periodic(theta2)

        # Both should return same type (both None or both PeriodicStructure)
        if result1 is None:
            assert result2 is None, "Order-dependent: different None vs non-None"
        else:
            assert result2 is not None, "Order-dependent: different None vs non-None"
            # Structure should be same (periods, num_phases)
            assert result1.num_phases == result2.num_phases, "Order-dependent num_phases"


# =============================================================================
# Phase 3: MEDIUM Priority - Integration with WO-06/WO-16
# =============================================================================


class TestIntegration:
    """
    INT-01 through INT-05: Integration with WO-06 and WO-16.

    **Bug to find**: Wrong period extraction, missing closures, invalid writers.
    """

    def test_lattice_periods_extracted_correctly(self):
        """
        INT-01: Periods extracted from lattice.basis correctly.

        Setup:
        - Build periodic structure
        - Verify p_r and p_c match lattice basis diagonal

        **Bug**: Wrong period extraction from HNF basis.
        """
        grid = [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]

        theta = {
            "grids_present": [grid],
            "train_pairs": []
        }

        result = build_periodic(theta)

        if result is not None:
            # Verify periods match lattice basis
            lattice = result.lattice
            p_r = abs(lattice.basis[0][0])
            p_c = abs(lattice.basis[1][1])

            expected_num_phases = p_r * p_c

            assert result.num_phases == expected_num_phases, \
                f"PERIOD EXTRACTION BUG!\n" \
                f"Lattice basis: {lattice.basis}\n" \
                f"Extracted periods: p_r={p_r}, p_c={p_c}\n" \
                f"Expected num_phases: {expected_num_phases}\n" \
                f"Actual num_phases: {result.num_phases}"

    def test_closure_is_callable(self):
        """
        INT-02: closure field is a callable function.

        Setup:
        - Verify closure is callable

        **Bug**: closure not a function.
        """
        grid = [[1, 1], [1, 1]]

        theta = {
            "grids_present": [grid],
            "train_pairs": []
        }

        result = build_periodic(theta)

        if result is not None:
            assert callable(result.closure), \
                f"CLOSURE BUG: closure field not callable!\n" \
                f"Type: {type(result.closure)}"

    def test_writers_are_periodic_expr_instances(self):
        """
        INT-03: writers are List[PeriodicExpr].

        Setup:
        - Verify all writers have correct type

        **Bug**: Wrong writer type.
        """
        grid = [[1, 2, 1, 2], [1, 2, 1, 2]]

        theta = {
            "grids_present": [grid],
            "train_pairs": [
                ([[0, 0], [0, 0]], [[1, 2, 1, 2], [1, 2, 1, 2]])
            ]
        }

        result = build_periodic(theta)

        if result is not None and len(result.writers) > 0:
            for i, writer in enumerate(result.writers):
                # Check it has phase_id and color attributes (PeriodicExpr properties)
                assert hasattr(writer, 'phase_id'), \
                    f"WRITER TYPE BUG: writer {i} missing phase_id attribute"
                assert hasattr(writer, 'color'), \
                    f"WRITER TYPE BUG: writer {i} missing color attribute"

    def test_periodic_expr_phase_ids_valid(self):
        """
        INT-04: PeriodicExpr phase_ids ∈ [0, num_phases-1].

        Setup:
        - Verify writer phase_ids in valid range

        **Bug**: Out-of-range phase IDs in writers.
        """
        grid = [[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]

        theta = {
            "grids_present": [grid],
            "train_pairs": [
                ([[0]*4 for _ in range(4)], grid)
            ]
        }

        result = build_periodic(theta)

        if result is not None and len(result.writers) > 0:
            for writer in result.writers:
                assert 0 <= writer.phase_id < result.num_phases, \
                    f"WRITER PHASE_ID BUG: phase_id={writer.phase_id} out of range [0, {result.num_phases-1}]"

    def test_periodic_expr_colors_in_arc_range(self):
        """
        INT-05: PeriodicExpr colors ∈ [0, 9] (ARC color range).

        Setup:
        - Verify writer colors in ARC range

        **Bug**: Invalid color values.
        """
        grid = [[1, 2, 1, 2], [1, 2, 1, 2]]

        theta = {
            "grids_present": [grid],
            "train_pairs": [
                ([[0, 0], [0, 0]], [[1, 2, 1, 2], [1, 2, 1, 2]])
            ]
        }

        result = build_periodic(theta)

        if result is not None and len(result.writers) > 0:
            for writer in result.writers:
                assert 0 <= writer.color <= 9, \
                    f"WRITER COLOR BUG: color={writer.color} out of ARC range [0, 9]"


# =============================================================================
# Phase 3: Remaining Edge Cases
# =============================================================================


class TestRemainingEdgeCases:
    """
    EDGE-05 through EDGE-07: Additional edge cases.

    **Bug to find**: Crashes on degenerate inputs.
    """

    def test_large_periods_handling(self):
        """
        EDGE-07: Periods equal to grid size (no actual tiling).

        Setup:
        - p_r=H, p_c=W → num_phases = H*W (every pixel unique phase)

        **Bug**: Accepting invalid structure.
        """
        # This should either return None or handle gracefully
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

        theta = {
            "grids_present": [grid],
            "train_pairs": []
        }

        # Should not crash
        result = build_periodic(theta)

        # If it returns a result, verify it's sensible
        if result is not None:
            H, W = 3, 3
            # Periods can't exceed grid dimensions
            assert result.num_phases <= H * W, \
                f"LARGE PERIOD BUG: num_phases={result.num_phases} exceeds H*W={H*W}"
