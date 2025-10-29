"""
Test script for WO-12 (blow_block.py) - Manual verification.

Tests BLOWUP and BLOCK_SUBST operations:
1. BLOWUP[k] - Kronecker inflation
2. BLOCK_SUBST[B(c)] - Per-color motif substitution
3. Composition BLOWUP + BLOCK_SUBST
4. Parameter inference from training pairs
5. FY exactness verification

All tests verify deterministic behavior and exact matching.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_laws.blow_block import (
    apply_blowup,
    apply_block_subst,
    apply_blowup_block_subst,
    infer_k_from_pair,
    infer_motifs_from_pair,
    infer_blowup_params,
    build_blow_block,
    verify_blowup_block_subst,
    motif_to_tuple,
    tuple_to_motif,
)


def print_grid(grid, label=""):
    """Helper to print grids."""
    if label:
        print(f"\n{label}:")
    for row in grid:
        print("  " + " ".join(str(c) for c in row))


def test_blowup_simple():
    """Test BLOWUP[k=2] on 2×2 grid."""
    print("\n" + "=" * 60)
    print("TEST 1: BLOWUP[k=2] - Simple Inflation")
    print("=" * 60)

    grid = [
        [1, 2],
        [3, 4]
    ]

    print_grid(grid, "Original 2×2")

    result = apply_blowup(grid, k=2)

    print_grid(result, "After BLOWUP[k=2] → 4×4")

    # Expected: each pixel becomes 2×2 block
    expected = [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ]

    assert result == expected, f"BLOWUP failed"
    print("✅ PASSED")
    return True


def test_blowup_3x3_to_9x9():
    """Test BLOWUP[k=3] on 3×3 grid (spec example)."""
    print("\n" + "=" * 60)
    print("TEST 2: BLOWUP[k=3] - 3×3 → 9×9")
    print("=" * 60)

    grid = [
        [1, 2, 1],
        [2, 0, 2],
        [1, 2, 1]
    ]

    print_grid(grid, "Original 3×3")

    result = apply_blowup(grid, k=3)

    print_grid(result, "After BLOWUP[k=3] → 9×9")

    # Verify size
    assert len(result) == 9 and len(result[0]) == 9, "Wrong size"

    # Verify first 3×3 block (should be all 1s)
    for r in range(3):
        for c in range(3):
            assert result[r][c] == 1, f"Block (0,0) wrong at ({r},{c})"

    # Verify center 3×3 block (should be all 0s)
    for r in range(3, 6):
        for c in range(3, 6):
            assert result[r][c] == 0, f"Block (1,1) wrong at ({r},{c})"

    print("✅ PASSED")
    return True


def test_block_subst_uniform():
    """Test BLOCK_SUBST with uniform motifs."""
    print("\n" + "=" * 60)
    print("TEST 3: BLOCK_SUBST - Uniform Motifs")
    print("=" * 60)

    # 4×4 grid from BLOWUP[k=2] of [[1, 2], [3, 4]]
    grid = [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ]

    print_grid(grid, "After BLOWUP[k=2]")

    # Define motifs (2×2 blocks)
    motifs = {
        1: [[5, 5], [5, 5]],  # Color 1 → all 5s
        2: [[6, 6], [6, 6]],  # Color 2 → all 6s
        3: [[7, 7], [7, 7]],  # Color 3 → all 7s
        4: [[8, 8], [8, 8]],  # Color 4 → all 8s
    }

    result = apply_block_subst(grid, k=2, motifs=motifs)

    print_grid(result, "After BLOCK_SUBST")

    # Expected: blocks replaced with uniform colors
    expected = [
        [5, 5, 6, 6],
        [5, 5, 6, 6],
        [7, 7, 8, 8],
        [7, 7, 8, 8],
    ]

    assert result == expected, "BLOCK_SUBST failed"
    print("✅ PASSED")
    return True


def test_block_subst_patterns():
    """Test BLOCK_SUBST with patterned motifs."""
    print("\n" + "=" * 60)
    print("TEST 4: BLOCK_SUBST - Patterned Motifs")
    print("=" * 60)

    # 6×6 grid from BLOWUP[k=3] of [[1, 2], [3, 4]]
    grid = apply_blowup([[1, 2], [3, 4]], k=3)

    print_grid(grid, "After BLOWUP[k=3]")

    # Define patterned motifs (3×3 blocks with patterns)
    motifs = {
        1: [[1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]],  # Cross pattern
        2: [[2, 2, 0],
            [2, 2, 0],
            [0, 0, 0]],  # Top-left block
        3: [[0, 0, 0],
            [3, 3, 3],
            [0, 0, 0]],  # Horizontal line
        4: [[4, 0, 4],
            [0, 4, 0],
            [4, 0, 4]],  # X pattern
    }

    result = apply_block_subst(grid, k=3, motifs=motifs)

    print_grid(result, "After BLOCK_SUBST with patterns")

    # Verify top-left block has cross pattern
    assert result[0][0] == 1 and result[0][2] == 1, "Cross pattern wrong"
    assert result[1][1] == 1, "Cross center wrong"

    # Verify top-right block has top-left block pattern
    assert result[0][3] == 2 and result[0][4] == 2, "Top-left block wrong"

    print("✅ PASSED")
    return True


def test_blowup_block_subst_composition():
    """Test full BLOWUP + BLOCK_SUBST composition."""
    print("\n" + "=" * 60)
    print("TEST 5: BLOWUP + BLOCK_SUBST Composition")
    print("=" * 60)

    grid = [
        [1, 2],
        [3, 1]
    ]

    print_grid(grid, "Original 2×2")

    motifs = {
        1: [[0, 1], [1, 0]],  # Checkerboard
        2: [[2, 2], [2, 2]],  # Solid
        3: [[3, 0], [0, 3]],  # Diagonal
    }

    result = apply_blowup_block_subst(grid, k=2, motifs=motifs)

    print_grid(result, "After BLOWUP[k=2] + BLOCK_SUBST")

    # Verify size
    assert len(result) == 4 and len(result[0]) == 4, "Wrong size"

    # Verify top-left block has checkerboard pattern
    assert result[0][0] == 0 and result[0][1] == 1, "Checkerboard wrong"
    assert result[1][0] == 1 and result[1][1] == 0, "Checkerboard wrong"

    # Verify top-right block is solid 2s
    assert result[0][2] == 2 and result[0][3] == 2, "Solid wrong"

    print("✅ PASSED")
    return True


def test_infer_k():
    """Test k inference from training pair."""
    print("\n" + "=" * 60)
    print("TEST 6: Infer k from Size Ratio")
    print("=" * 60)

    # 2×2 → 4×4 (k=2)
    X1 = [[1, 2], [3, 4]]
    Y1 = [[0] * 4 for _ in range(4)]
    k1 = infer_k_from_pair(X1, Y1)
    assert k1 == 2, f"Expected k=2, got {k1}"
    print(f"2×2 → 4×4: k={k1} ✅")

    # 3×3 → 9×9 (k=3)
    X2 = [[1] * 3 for _ in range(3)]
    Y2 = [[0] * 9 for _ in range(9)]
    k2 = infer_k_from_pair(X2, Y2)
    assert k2 == 3, f"Expected k=3, got {k2}"
    print(f"3×3 → 9×9: k={k2} ✅")

    # 2×2 → 6×4 (inconsistent, should fail)
    X3 = [[1, 2], [3, 4]]
    Y3 = [[0] * 4 for _ in range(6)]
    k3 = infer_k_from_pair(X3, Y3)
    assert k3 is None, f"Expected None for inconsistent, got {k3}"
    print(f"2×2 → 6×4: k={k3} (inconsistent) ✅")

    print("✅ PASSED")
    return True


def test_infer_motifs():
    """Test motif inference from training pair."""
    print("\n" + "=" * 60)
    print("TEST 7: Infer Motifs from Training")
    print("=" * 60)

    # Create input
    X = [[1, 2], [3, 1]]

    # Create output with known motifs (k=2)
    motifs_true = {
        1: [[0, 1], [1, 0]],  # Checkerboard
        2: [[2, 2], [2, 2]],  # Solid
        3: [[3, 0], [0, 3]],  # Diagonal
    }

    Y = apply_blowup_block_subst(X, k=2, motifs=motifs_true)

    print_grid(X, "Input X")
    print_grid(Y, "Output Y (from known motifs)")

    # Infer motifs
    motifs_inferred = infer_motifs_from_pair(X, Y, k=2)

    assert motifs_inferred is not None, "Failed to infer motifs"

    # Convert back to lists for comparison
    for color in motifs_true:
        motif_true = motif_to_tuple(motifs_true[color])
        motif_inferred = motifs_inferred[color]
        assert motif_inferred == motif_true, f"Motif for color {color} wrong"

    print("All motifs inferred correctly ✅")
    print("✅ PASSED")
    return True


def test_infer_blowup_params():
    """Test full parameter inference from multiple training pairs."""
    print("\n" + "=" * 60)
    print("TEST 8: Infer k and Motifs from Training Pairs")
    print("=" * 60)

    # Define motifs
    motifs = {
        1: [[1, 0], [0, 1]],  # Diagonal
        2: [[2, 2], [0, 0]],  # Top solid
        3: [[0, 3], [3, 0]],  # Anti-diagonal
    }

    # Create training pairs
    X1 = [[1, 2], [3, 1]]
    Y1 = apply_blowup_block_subst(X1, k=2, motifs=motifs)

    X2 = [[2, 1], [1, 3]]
    Y2 = apply_blowup_block_subst(X2, k=2, motifs=motifs)

    train_pairs = [(X1, Y1), (X2, Y2)]

    print(f"Training pairs: {len(train_pairs)}")

    # Infer parameters
    params = infer_blowup_params(train_pairs)

    assert params is not None, "Failed to infer parameters"

    k_inferred, motifs_inferred = params

    assert k_inferred == 2, f"Expected k=2, got {k_inferred}"
    print(f"Inferred k={k_inferred} ✅")

    # Verify motifs
    for color in motifs:
        motif_true = motif_to_tuple(motifs[color])
        motif_inferred = motifs_inferred[color]
        assert motif_inferred == motif_true, f"Motif for color {color} wrong"

    print(f"Inferred {len(motifs_inferred)} motifs correctly ✅")
    print("✅ PASSED")
    return True


def test_fy_exactness():
    """Test FY exactness verification."""
    print("\n" + "=" * 60)
    print("TEST 9: FY Exactness Verification")
    print("=" * 60)

    # Create training pairs with consistent motifs
    motifs = {
        1: [[1, 0], [0, 1]],
        2: [[2, 2], [2, 2]],
    }

    X1 = [[1, 2], [2, 1]]
    Y1 = apply_blowup_block_subst(X1, k=2, motifs=motifs)

    X2 = [[2, 1], [1, 2]]
    Y2 = apply_blowup_block_subst(X2, k=2, motifs=motifs)

    train_pairs = [(X1, Y1), (X2, Y2)]

    # Convert motifs to tuples
    motifs_tuple = {c: motif_to_tuple(m) for c, m in motifs.items()}

    # Verify FY exactness
    is_exact = verify_blowup_block_subst(2, motifs_tuple, train_pairs)

    assert is_exact, "FY exactness failed (should pass)"
    print("FY exactness verified ✅")

    # Test with wrong motif (should fail)
    wrong_motifs = {
        1: motif_to_tuple([[9, 9], [9, 9]]),  # Wrong motif
        2: motifs_tuple[2],
    }

    is_exact_wrong = verify_blowup_block_subst(2, wrong_motifs, train_pairs)

    assert not is_exact_wrong, "FY exactness passed (should fail)"
    print("FY exactness correctly rejected wrong motifs ✅")

    print("✅ PASSED")
    return True


def test_build_blow_block():
    """Test build_blow_block with full pipeline."""
    print("\n" + "=" * 60)
    print("TEST 10: build_blow_block - Full Pipeline")
    print("=" * 60)

    # Create training pairs
    motifs = {
        1: [[0, 1, 0], [1, 1, 1], [0, 1, 0]],  # Plus sign
        2: [[2, 2, 2], [2, 0, 2], [2, 2, 2]],  # Frame
        3: [[3, 0, 3], [0, 3, 0], [3, 0, 3]],  # X pattern
    }

    X1 = [[1, 2], [3, 1]]
    Y1 = apply_blowup_block_subst(X1, k=3, motifs=motifs)

    X2 = [[2, 3], [1, 2]]
    Y2 = apply_blowup_block_subst(X2, k=3, motifs=motifs)

    train_pairs = [(X1, Y1), (X2, Y2)]

    print_grid(X1, "Training X1")
    print_grid(Y1, "Training Y1")

    # Build laws
    theta = {"train_pairs": train_pairs}
    laws = build_blow_block(theta)

    assert len(laws) == 1, f"Expected 1 law, got {len(laws)}"

    law = laws[0]
    assert law.k == 3, f"Expected k=3, got {law.k}"
    assert len(law.motifs) == 3, f"Expected 3 motifs, got {len(law.motifs)}"

    print(f"\nExtracted law:")
    print(f"  Operation: {law.operation}")
    print(f"  k: {law.k}")
    print(f"  Motifs: {len(law.motifs)} colors")
    print(f"  Motif hashes: {law.motif_hashes}")

    print("✅ PASSED")
    return True


def test_determinism():
    """Test determinism across multiple runs."""
    print("\n" + "=" * 60)
    print("TEST 11: Determinism (10 runs)")
    print("=" * 60)

    grid = [[1, 2], [3, 4]]
    motifs = {
        1: [[1, 0], [0, 1]],
        2: [[2, 2], [0, 0]],
        3: [[3, 0], [0, 3]],
        4: [[0, 4], [4, 0]],
    }

    # Run 10 times
    results = []
    for _ in range(10):
        result = apply_blowup_block_subst(grid, k=2, motifs=motifs)
        results.append(result)

    # Verify all identical
    for i in range(1, 10):
        assert results[i] == results[0], f"Run {i+1} differs from run 1"

    print("All 10 runs produced identical results ✅")
    print("✅ PASSED")
    return True


def run_all_tests():
    """Run all test cases."""
    print("\n" + "=" * 70)
    print(" WO-12 BLOWUP + BLOCK_SUBST - Test Suite")
    print("=" * 70)

    tests = [
        test_blowup_simple,
        test_blowup_3x3_to_9x9,
        test_block_subst_uniform,
        test_block_subst_patterns,
        test_blowup_block_subst_composition,
        test_infer_k,
        test_infer_motifs,
        test_infer_blowup_params,
        test_fy_exactness,
        test_build_blow_block,
        test_determinism,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except AssertionError as e:
            print(f"\n❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        except Exception as e:
            print(f"\n💥 ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f" Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
