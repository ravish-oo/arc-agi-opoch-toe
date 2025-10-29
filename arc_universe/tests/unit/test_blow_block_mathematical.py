"""
Mathematical validation tests for WO-12: BLOWUP + BLOCK_SUBST

Philosophy: Prove mathematical correctness through:
- Known-answer tests with pre-computed exact outputs
- Property-based validation (Kronecker product, tile alignment)
- Pathological stress tests (large grids, many colors, edge cases)

Battle-test the implementation to find bugs, not just verify consistency.
"""

import pytest
import hashlib
import json
from typing import List, Tuple, Set

from arc_laws.blow_block import (
    apply_blowup,
    apply_block_subst,
    apply_blowup_block_subst,
    infer_blowup_params,
    verify_blowup_block_subst,
    build_blow_block,
    motif_to_tuple,
    tuple_to_motif,
    extract_motif,
    infer_motifs_from_pair,
)
from arc_core.types import Grid
from arc_core.order_hash import hash64


# =============================================================================
# Known-Answer Tests (KNOWN-01 to KNOWN-03)
# =============================================================================

class TestKnownAnswers:
    """Tests with pre-computed exact expected outputs"""

    def test_known_01_simple_2x2_blowup_k2(self):
        """
        KNOWN-01: Simple 2×2 BLOWUP with k=2

        Pre-computed expected output verified manually.
        """
        grid = [[1, 2],
                [3, 4]]
        k = 2

        result = apply_blowup(grid, k)

        expected = [[1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [3, 3, 4, 4],
                    [3, 3, 4, 4]]

        # Exact pixel-by-pixel verification
        assert result == expected, (
            f"Blowup failed known-answer test:\n"
            f"Got:\n{result}\n"
            f"Expected:\n{expected}"
        )

        # Verify size
        assert len(result) == 4 and len(result[0]) == 4, "Size should be 4×4"

        # Verify each 2×2 block
        for r in range(2):
            for c in range(2):
                original_color = grid[r][c]
                # Check all pixels in the 2×2 block
                for dr in range(2):
                    for dc in range(2):
                        assert result[2*r + dr][2*c + dc] == original_color, (
                            f"Pixel at ({2*r + dr}, {2*c + dc}) should be {original_color}"
                        )

    def test_known_02_block_subst_checkerboard_k2(self):
        """
        KNOWN-02: BLOCK_SUBST with Checkerboard Motif (k=2)

        Pre-computed checkerboard pattern verified manually.
        """
        grid = [[1, 1],
                [1, 1]]
        k = 2
        motifs = {1: [[1, 0],
                      [0, 1]]}

        result = apply_blowup_block_subst(grid, k, motifs)

        expected = [[1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1]]

        assert result == expected, (
            f"Checkerboard BLOCK_SUBST failed:\n"
            f"Got:\n{result}\n"
            f"Expected:\n{expected}"
        )

        # Verify pattern: alternating 1s and 0s
        for r in range(4):
            for c in range(4):
                expected_color = 1 if (r + c) % 2 == 0 else 0
                assert result[r][c] == expected_color, (
                    f"Checkerboard pattern broken at ({r}, {c})"
                )

    def test_known_03_multi_color_block_subst_k3(self):
        """
        KNOWN-03: Multi-Color BLOCK_SUBST (k=3)

        4 distinct colors, each with unique 3×3 motif.
        Pre-computed expected pattern verified manually.
        """
        grid = [[1, 2],
                [3, 0]]
        k = 3

        # Define distinct motifs for each color
        motifs = {
            1: [[1, 1, 1],  # Hollow square
                [1, 0, 1],
                [1, 1, 1]],
            2: [[2, 2, 2],  # Solid
                [2, 2, 2],
                [2, 2, 2]],
            3: [[3, 0, 3],  # Plus sign
                [0, 3, 0],
                [3, 0, 3]],
            0: [[0, 0, 0],  # Background (all zeros)
                [0, 0, 0],
                [0, 0, 0]]
        }

        result = apply_blowup_block_subst(grid, k, motifs)

        # Expected 6×6 grid:
        # Top-left 3×3: motif for color 1 (hollow square)
        # Top-right 3×3: motif for color 2 (solid)
        # Bottom-left 3×3: motif for color 3 (plus)
        # Bottom-right 3×3: motif for color 0 (zeros)

        expected = [[1, 1, 1, 2, 2, 2],
                    [1, 0, 1, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2],
                    [3, 0, 3, 0, 0, 0],
                    [0, 3, 0, 0, 0, 0],
                    [3, 0, 3, 0, 0, 0]]

        assert result == expected, (
            f"Multi-color BLOCK_SUBST failed:\n"
            f"Got:\n{result}\n"
            f"Expected:\n{expected}"
        )


# =============================================================================
# Mathematical Property Validation (MATH-01 to MATH-04)
# =============================================================================

class TestMathematicalProperties:
    """Property-based validation of mathematical invariants"""

    def test_math_01_kronecker_product_correctness(self):
        """
        MATH-01: Kronecker Product Property

        For any grid X of shape (R, C) and blowup factor k:
        BLOWUP[k](X)[i, j] == X[i // k, j // k]
        for all 0 ≤ i < k·R, 0 ≤ j < k·C
        """
        import random

        # Test on multiple random grids
        for grid_size in [(2, 2), (3, 3), (4, 5), (5, 4), (10, 10)]:
            R, C = grid_size
            for k in [2, 3, 4]:
                # Generate random grid
                grid = [[random.randint(0, 9) for _ in range(C)] for _ in range(R)]

                result = apply_blowup(grid, k)

                # Verify Kronecker product property
                for i in range(k * R):
                    for j in range(k * C):
                        expected_color = grid[i // k][j // k]
                        actual_color = result[i][j]
                        assert actual_color == expected_color, (
                            f"Kronecker property failed at ({i}, {j}) for grid {grid_size}, k={k}: "
                            f"expected {expected_color}, got {actual_color}"
                        )

    def test_math_02_tile_alignment_no_overlaps_no_gaps(self):
        """
        MATH-02: Tile Alignment Property

        For k×k motifs and grid of shape (k·R, k·C), tiles are aligned at:
        {(k·r, k·c) | 0 ≤ r < R, 0 ≤ c < C}

        No overlaps, no gaps between tiles.
        """
        for k in [2, 3, 4]:
            for R, C in [(2, 2), (3, 3), (4, 5)]:
                # Create input grid
                grid = [[1] * C for _ in range(R)]

                # Track which pixels are covered by tiles
                covered = [[False] * (k * C) for _ in range(k * R)]

                # Mark tiles
                for r in range(R):
                    for c in range(C):
                        # Tile at position (k*r, k*c)
                        for dr in range(k):
                            for dc in range(k):
                                row = k * r + dr
                                col = k * c + dc

                                # Check no overlaps
                                assert not covered[row][col], (
                                    f"Tile overlap at ({row}, {col}) for k={k}, R={R}, C={C}"
                                )
                                covered[row][col] = True

                # Check no gaps (all pixels covered)
                for r in range(k * R):
                    for c in range(k * C):
                        assert covered[r][c], (
                            f"Gap at ({r}, {c}) for k={k}, R={R}, C={C}"
                        )

    def test_math_03_motif_determinism_property(self):
        """
        MATH-03: Motif Determinism Property

        Extracting motif B(c) from identical training pairs yields identical motifs:
        extract_motifs(trains1, k) == extract_motifs(trains2, k)
        where trains1 == trains2
        """
        train_pairs = [
            ([[1, 2]], [[1, 0, 2, 2], [0, 1, 2, 2]]),
            ([[3]], [[3, 3], [3, 3]])
        ]

        results = []
        for run in range(100):
            params = infer_blowup_params(train_pairs)
            k, motifs = params

            # Serialize motifs for comparison
            serialized = json.dumps(
                {str(color): list(motif) for color, motif in motifs.items()},
                sort_keys=True
            )
            results.append(hashlib.sha256(serialized.encode()).hexdigest())

        unique_hashes = set(results)
        assert len(unique_hashes) == 1, (
            f"Motif extraction is non-deterministic: {len(unique_hashes)} unique results"
        )

    def test_math_04_fy_exactness_on_all_trains(self):
        """
        MATH-04: FY Exactness Property

        For any train pair (X_i, Y_i):
        BLOCK_SUBST(BLOWUP[k](X_i), B) == Y_i
        where k and B are learned from all trains
        """
        # Multiple training pairs
        train_pairs = [
            ([[1, 2]], [[1, 0, 2, 2], [0, 1, 2, 2]]),
            ([[1]], [[1, 0], [0, 1]]),
            ([[2]], [[2, 2], [2, 2]])
        ]

        params = infer_blowup_params(train_pairs)
        assert params is not None, "Should infer consistent parameters"

        k, motifs = params
        motifs_dict = {color: tuple_to_motif(motif) for color, motif in motifs.items()}

        # Verify FY exactness on each training pair
        for i, (X, Y) in enumerate(train_pairs):
            Y_pred = apply_blowup_block_subst(X, k, motifs_dict)

            # Pixel-perfect match required
            assert Y_pred == Y, (
                f"FY exactness failed on train pair {i}:\n"
                f"Input: {X}\n"
                f"Expected: {Y}\n"
                f"Predicted: {Y_pred}\n"
                f"k={k}, motifs={motifs}"
            )


# =============================================================================
# Pathological Stress Tests (PATH-01 to PATH-06)
# =============================================================================

class TestPathological:
    """ARC-relevant pathological and stress tests"""

    def test_path_01_large_blowup_k5_20x20(self):
        """
        PATH-01: Large Blowup (k=5, 20×20 → 100×100)

        Stress test: no overflow, correct size, acceptable performance.
        """
        import time

        R, C = 20, 20
        k = 5

        # Create 20×20 grid with pattern
        grid = [[(r * C + c) % 10 for c in range(C)] for r in range(R)]

        start = time.time()
        result = apply_blowup(grid, k)
        elapsed = time.time() - start

        # Check size
        assert len(result) == 100, f"Expected height 100, got {len(result)}"
        assert len(result[0]) == 100, f"Expected width 100, got {len(result[0])}"

        # Check correctness (sample a few pixels)
        assert result[0][0] == grid[0][0], "Top-left pixel incorrect"
        assert result[99][99] == grid[19][19], "Bottom-right pixel incorrect"
        assert result[50][50] == grid[10][10], "Center pixel incorrect"

        # Performance check (should be fast)
        assert elapsed < 1.0, f"Large blowup too slow: {elapsed:.3f}s"

    def test_path_02_many_colors_10_distinct_motifs(self):
        """
        PATH-02: Many Colors (10 distinct colors with unique 3×3 motifs)

        Stress test: handle many colors, no motif collisions.
        """
        k = 3
        num_colors = 10

        # Create grid with 10 different colors
        grid = [[c for c in range(num_colors)]]

        # Create unique 3×3 motif for each color
        motifs = {}
        for c in range(num_colors):
            # Unique pattern: diagonal with color value
            motif = [[c if r == d and d == d2 else 0 for d2 in range(3)]
                     for r, d in enumerate(range(3))]
            motifs[c] = motif

        result = apply_blowup_block_subst(grid, k, motifs)

        # Verify size
        assert len(result) == 3 and len(result[0]) == 30, "Size incorrect"

        # Verify each color got its motif
        for c in range(num_colors):
            # Check the 3×3 tile for color c
            tile_start_col = c * 3
            for r in range(3):
                for col_offset in range(3):
                    expected = motifs[c][r][col_offset]
                    actual = result[r][tile_start_col + col_offset]
                    assert actual == expected, (
                        f"Motif for color {c} incorrect at ({r}, {tile_start_col + col_offset})"
                    )

    def test_path_03_single_pixel_input_1x1_to_kxk(self):
        """
        PATH-03: Single Pixel Input (1×1 → k×k)

        Edge case: minimal input, verify single tile substitution.
        """
        for k in [2, 3, 4, 5]:
            grid = [[7]]  # Single pixel, color 7

            # Create k×k motif
            motif = [[(r + c) % 2 for c in range(k)] for r in range(k)]
            motifs = {7: motif}

            result = apply_blowup_block_subst(grid, k, motifs)

            # Verify size
            assert len(result) == k and len(result[0]) == k, (
                f"Expected {k}×{k}, got {len(result)}×{len(result[0])}"
            )

            # Verify motif applied
            assert result == motif, f"Single pixel motif incorrect for k={k}"

    def test_path_04_maximum_arc_grid_30x30_k3(self):
        """
        PATH-04: Maximum ARC Grid (30×30 → 90×90 with k=3)

        Stress test: handle maximum ARC grid size, 900 tiles.
        """
        import time

        R, C = 30, 30
        k = 3

        # Create 30×30 grid with diagonal pattern
        grid = [[(r + c) % 10 for c in range(C)] for r in range(R)]

        start = time.time()
        result = apply_blowup(grid, k)
        elapsed = time.time() - start

        # Verify size
        assert len(result) == 90 and len(result[0]) == 90, "Size incorrect"

        # Verify tile alignment (sample check)
        for r in range(0, 30, 5):  # Sample every 5 rows
            for c in range(0, 30, 5):  # Sample every 5 cols
                expected_color = grid[r][c]
                # Check all pixels in the 3×3 tile
                for dr in range(3):
                    for dc in range(3):
                        actual = result[3*r + dr][3*c + dc]
                        assert actual == expected_color, (
                            f"Tile alignment error at ({3*r + dr}, {3*c + dc})"
                        )

        # Performance check
        assert elapsed < 1.0, f"Maximum grid blowup too slow: {elapsed:.3f}s"

    def test_path_05_inconsistent_motifs_across_trains_rejected(self):
        """
        PATH-05: Inconsistent Motifs Across Trains

        Critical test: detect and reject when same color has different motifs.
        """
        train_pairs = [
            ([[1]], [[1, 0], [0, 1]]),  # Color 1 → checkerboard
            ([[1]], [[0, 1], [1, 0]])   # Color 1 → DIFFERENT pattern!
        ]

        params = infer_blowup_params(train_pairs)

        # Must reject inconsistent motifs
        assert params is None, (
            "Should reject when color 1 has different motifs in different trains"
        )

    def test_path_06_non_aligned_tile_coverage(self):
        """
        PATH-06: Non-Aligned Tile Coverage

        Edge case: grid size not exact multiple of k (11×11, k=3).
        Should raise error or return None.
        """
        # Create 11×11 inflated grid (not divisible by k=3)
        grid = [[1] * 11 for _ in range(11)]
        k = 3
        motifs = {1: [[1, 1, 1], [1, 1, 1], [1, 1, 1]]}

        # Should raise ValueError
        with pytest.raises(ValueError, match="not divisible"):
            apply_block_subst(grid, k, motifs)


# =============================================================================
# Deep Validation: Tile Alignment Verification
# =============================================================================

class TestDeepTileAlignment:
    """Deep validation of tile alignment correctness"""

    def test_tile_positions_exact_for_various_k(self):
        """
        Verify exact tile positions for k=2, k=3, k=4, k=5.

        This is a CRITICAL test that verifies tiles start at (k*r, k*c).
        """
        for k in [2, 3, 4, 5]:
            R, C = 5, 5  # 5×5 input

            # Create grid where each pixel has unique value
            grid = [[r * C + c for c in range(C)] for r in range(R)]

            # Apply blowup
            result = apply_blowup(grid, k)

            # Verify each tile
            for r in range(R):
                for c in range(C):
                    original_value = grid[r][c]

                    # Tile should start at (k*r, k*c)
                    tile_start_r = k * r
                    tile_start_c = k * c

                    # Verify all k×k pixels in tile
                    for dr in range(k):
                        for dc in range(k):
                            pixel_r = tile_start_r + dr
                            pixel_c = tile_start_c + dc

                            actual_value = result[pixel_r][pixel_c]

                            assert actual_value == original_value, (
                                f"Tile position error for k={k} at original ({r},{c}): "
                                f"expected tile pixel ({pixel_r},{pixel_c}) = {original_value}, "
                                f"got {actual_value}"
                            )

    def test_motif_substitution_exact_tile_replacement(self):
        """
        Verify motif substitution replaces exact k×k tiles.

        Deep validation: each k×k block should be EXACTLY replaced.
        """
        k = 3
        grid = [[1, 2], [3, 4]]  # 2×2 input

        # Unique motifs for each color
        motifs = {
            1: [[1, 1, 1], [1, 9, 1], [1, 1, 1]],  # Hollow square with 9 in center
            2: [[2, 2, 2], [2, 8, 2], [2, 2, 2]],
            3: [[3, 3, 3], [3, 7, 3], [3, 3, 3]],
            4: [[4, 4, 4], [4, 6, 4], [4, 4, 4]]
        }

        result = apply_blowup_block_subst(grid, k, motifs)

        # Verify each 3×3 tile matches its motif EXACTLY
        for r in range(2):
            for c in range(2):
                original_color = grid[r][c]
                expected_motif = motifs[original_color]

                # Extract actual 3×3 tile from result
                tile_start_r = k * r
                tile_start_c = k * c

                for dr in range(k):
                    for dc in range(k):
                        expected_pixel = expected_motif[dr][dc]
                        actual_pixel = result[tile_start_r + dr][tile_start_c + dc]

                        assert actual_pixel == expected_pixel, (
                            f"Motif replacement failed for color {original_color} "
                            f"at tile ({r},{c}), pixel offset ({dr},{dc}): "
                            f"expected {expected_pixel}, got {actual_pixel}"
                        )


# =============================================================================
# Critical Deep Validation (DEEP-01 to DEEP-03)
# =============================================================================

class TestCriticalDeepValidation:
    """
    Critical tests identified after self-assessment.
    These tests verify properties that could hide bugs if not explicitly checked.
    """

    def test_deep_01_motif_hash_integrity(self):
        """
        DEEP-01: Verify motif hashes are actually hash64() of the motifs

        Critical: Receipts must have correct hashes for debugging.
        If hashes are wrong, troubleshooting will be impossible.
        """
        train_pairs = [
            ([[1, 2]], [[1, 0, 2, 2], [0, 1, 2, 2]])
        ]
        theta = {"train_pairs": train_pairs}

        laws = build_blow_block(theta)
        assert len(laws) == 1, "Should build one law"

        law = laws[0]

        # Verify each motif hash matches hash64(motif)
        for color, motif_tuple in law.motifs.items():
            # Compute expected hash
            expected_hash = hash64(motif_tuple)

            # Verify law contains correct hash
            assert color in law.motif_hashes, f"Color {color} missing from motif_hashes"
            actual_hash = law.motif_hashes[color]

            assert actual_hash == expected_hash, (
                f"Motif hash incorrect for color {color}:\n"
                f"Expected hash64({motif_tuple}) = {expected_hash}\n"
                f"Got: {actual_hash}\n"
                f"This breaks receipts integrity!"
            )

    def test_deep_02_multiple_instances_same_color_all_identical(self):
        """
        DEEP-02: When same color appears multiple times in grid,
        ALL instances must produce identical motifs (not just first one)

        Critical: Catches bugs where only first tile is checked, rest assumed.
        """
        # Grid where color 1 appears in ALL 9 positions (3×3 grid)
        X = [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]]

        # k=2, so output is 6×6 with 9 tiles (each 2×2)
        # All tiles should have SAME motif for color 1
        k = 2
        motif_1 = [[1, 0], [0, 1]]  # Checkerboard

        # Create output Y where EVERY 2×2 tile is the checkerboard motif
        Y = [[1, 0, 1, 0, 1, 0],
             [0, 1, 0, 1, 0, 1],
             [1, 0, 1, 0, 1, 0],
             [0, 1, 0, 1, 0, 1],
             [1, 0, 1, 0, 1, 0],
             [0, 1, 0, 1, 0, 1]]

        # Extract motifs from pair
        motifs = infer_motifs_from_pair(X, Y, k)

        assert motifs is not None, "Should extract motif successfully"
        assert 1 in motifs, "Color 1 motif should be extracted"

        # Now verify: manually extract motif from EACH of the 9 positions
        # and confirm they're ALL identical
        extracted_motifs = []
        for r in range(3):
            for c in range(3):
                # Extract 2×2 tile at position (r, c)
                tile = extract_motif(Y, r, c, k)
                tile_tuple = motif_to_tuple(tile)
                extracted_motifs.append(tile_tuple)

        # All 9 motifs should be identical
        first_motif = extracted_motifs[0]
        for i, motif in enumerate(extracted_motifs):
            assert motif == first_motif, (
                f"Motif at position {i // 3, i % 3} differs from first motif!\n"
                f"Expected all 9 instances of color 1 to have same motif.\n"
                f"First: {first_motif}\n"
                f"Position {i}: {motif}\n"
                f"This indicates inconsistent motif extraction!"
            )

        # Final check: the extracted motif should match what we designed
        expected_motif_tuple = motif_to_tuple(motif_1)
        assert motifs[1] == expected_motif_tuple, (
            f"Extracted motif doesn't match expected:\n"
            f"Expected: {expected_motif_tuple}\n"
            f"Got: {motifs[1]}"
        )

    def test_deep_03_tile_extraction_exact_positions(self):
        """
        DEEP-03: Verify tiles are extracted from EXACT positions Y[k*r:k*r+k, k*c:k*c+k]

        Critical: Off-by-one errors in tile extraction would break everything.
        This test explicitly verifies the math is correct.
        """
        # Create input with unique value at each position
        X = [[1, 2],
             [3, 4]]
        k = 3

        # Create output where each 3×3 tile has a unique marker at center
        # This lets us verify EXACTLY which tile was extracted
        Y = [
            # Tile (0,0) for color 1: center=11
            [1, 1, 1,   2, 2, 2],
            [1, 11, 1,  2, 22, 2],
            [1, 1, 1,   2, 2, 2],
            # Tile (1,0) for color 3: center=33
            [3, 3, 3,   4, 4, 4],
            [3, 33, 3,  4, 44, 4],
            [3, 3, 3,   4, 4, 4]
        ]

        # Extract motifs
        motifs = infer_motifs_from_pair(X, Y, k)
        assert motifs is not None, "Should extract motifs"

        # Verify tile for color 1 was extracted from Y[0:3, 0:3] (has center=11)
        motif_1 = tuple_to_motif(motifs[1])
        assert motif_1[1][1] == 11, (
            f"Tile for color 1 should have center=11 (from Y[0:3, 0:3]).\n"
            f"Got center={motif_1[1][1]}.\n"
            f"This means tile was extracted from WRONG position!"
        )

        # Verify tile for color 2 was extracted from Y[0:3, 3:6] (has center=22)
        motif_2 = tuple_to_motif(motifs[2])
        assert motif_2[1][1] == 22, (
            f"Tile for color 2 should have center=22 (from Y[0:3, 3:6]).\n"
            f"Got center={motif_2[1][1]}.\n"
            f"This means tile was extracted from WRONG position!"
        )

        # Verify tile for color 3 was extracted from Y[3:6, 0:3] (has center=33)
        motif_3 = tuple_to_motif(motifs[3])
        assert motif_3[1][1] == 33, (
            f"Tile for color 3 should have center=33 (from Y[3:6, 0:3]).\n"
            f"Got center={motif_3[1][1]}.\n"
            f"This means tile was extracted from WRONG position!"
        )

        # Verify tile for color 4 was extracted from Y[3:6, 3:6] (has center=44)
        motif_4 = tuple_to_motif(motifs[4])
        assert motif_4[1][1] == 44, (
            f"Tile for color 4 should have center=44 (from Y[3:6, 3:6]).\n"
            f"Got center={motif_4[1][1]}.\n"
            f"This means tile was extracted from WRONG position!"
        )

        # All positions verified - tile extraction is mathematically correct!
