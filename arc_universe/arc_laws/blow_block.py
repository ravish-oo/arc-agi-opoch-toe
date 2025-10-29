"""
BLOWUP and BLOCK_SUBST laws (WO-12).

Per implementation_plan.md lines 292-296 and math_spec.md §6.4:
- BLOWUP[k]: Kronecker inflate canvas by k in each axis (each pixel → k×k block)
- BLOCK_SUBST[B(c)]: Per-color k×k motif substitution

Key principles:
1. Learn k from size ratio: k = H_Y / H_X = W_Y / W_X
2. Extract motifs B(c) for each color c by sampling k×k tiles from Y
3. Verify ALL tiles for color c are IDENTICAL (exact matching)
4. Verify consistency across ALL training pairs
5. FY exactness: Applying learned params must reproduce all training outputs

All operations deterministic, integer-only math, exact verification.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from arc_core.types import Grid, LawInstance
from arc_core.order_hash import hash64


# =============================================================================
# Types
# =============================================================================

@dataclass(frozen=True)
class BlowBlockLaw:
    """
    BLOWUP + BLOCK_SUBST law instance.

    Each instance represents learned parameters (k, motifs) that passed
    FY exactness verification on all training pairs.
    """
    operation: str  # "blowup", "block_subst", or "blowup+block_subst"
    k: int  # Inflation factor (each pixel → k×k block)
    motifs: Dict[int, Tuple[Tuple[int, ...]]]  # color → k×k motif (as nested tuples for hashability)
    motif_hashes: Dict[int, int]  # color → hash of motif (for receipts)


# =============================================================================
# BLOWUP Operation (Kronecker Inflation)
# =============================================================================

def apply_blowup(grid: Grid, k: int) -> Grid:
    """
    Apply BLOWUP[k]: Kronecker inflate canvas by k.

    Each pixel at (r,c) with color v becomes a k×k block of color v,
    positioned at (k*r, k*c) to (k*r+k-1, k*c+k-1).

    Args:
        grid: Input grid H×W
        k: Inflation factor

    Returns:
        Inflated grid (k*H)×(k*W) where each pixel is replicated k×k times
    """
    if not grid or not grid[0]:
        return []

    H, W = len(grid), len(grid[0])
    H_new, W_new = k * H, k * W

    # Create output grid
    result = [[0] * W_new for _ in range(H_new)]

    # Inflate each pixel
    for r in range(H):
        for c in range(W):
            color = grid[r][c]
            # Fill k×k block
            for dr in range(k):
                for dc in range(k):
                    result[k * r + dr][k * c + dc] = color

    return result


# =============================================================================
# BLOCK_SUBST Operation (Motif Substitution)
# =============================================================================

def apply_block_subst(grid: Grid, k: int, motifs: Dict[int, List[List[int]]]) -> Grid:
    """
    Apply BLOCK_SUBST[B(c)]: Replace k×k blocks with per-color motifs.

    Assumes grid has already been inflated by BLOWUP[k], so each original
    pixel corresponds to a k×k block. Replaces each block with motif B(c)
    where c is the original pixel color.

    Args:
        grid: Inflated grid (k*H)×(k*W) from BLOWUP
        k: Block size
        motifs: Dict mapping color → k×k motif grid

    Returns:
        Grid with motifs substituted
    """
    if not grid or not grid[0]:
        return []

    H, W = len(grid), len(grid[0])

    # Grid must be divisible by k
    if H % k != 0 or W % k != 0:
        raise ValueError(f"Grid size ({H}×{W}) not divisible by k={k}")

    # Create output grid
    result = [row[:] for row in grid]  # Deep copy

    # Process each k×k block
    for r_block in range(H // k):
        for c_block in range(W // k):
            # Get color from first pixel of block (assumes uniform after BLOWUP)
            r_start = r_block * k
            c_start = c_block * k
            color = grid[r_start][c_start]

            # Get motif for this color
            if color not in motifs:
                continue  # No motif for this color, leave unchanged

            motif = motifs[color]

            # Replace k×k block with motif
            for dr in range(k):
                for dc in range(k):
                    result[r_start + dr][c_start + dc] = motif[dr][dc]

    return result


def apply_blowup_block_subst(grid: Grid, k: int, motifs: Dict[int, List[List[int]]]) -> Grid:
    """
    Apply BLOWUP[k] followed by BLOCK_SUBST[B(c)] composition.

    This is the most common usage: inflate then substitute motifs.

    Args:
        grid: Input grid H×W
        k: Inflation factor
        motifs: Dict mapping color → k×k motif grid

    Returns:
        Result grid (k*H)×(k*W) with motifs substituted
    """
    # First, apply BLOWUP
    inflated = apply_blowup(grid, k)

    # Then, apply BLOCK_SUBST
    result = apply_block_subst(inflated, k, motifs)

    return result


# =============================================================================
# Parameter Inference (Learning k and Motifs)
# =============================================================================

def infer_k_from_pair(X: Grid, Y: Grid) -> Optional[int]:
    """
    Infer inflation factor k from a single training pair.

    k must satisfy: H_Y = k * H_X and W_Y = k * W_X

    Args:
        X: Input grid H_X × W_X
        Y: Output grid H_Y × W_Y

    Returns:
        k if consistent integer inflation exists, None otherwise
    """
    if not X or not X[0] or not Y or not Y[0]:
        return None

    H_X, W_X = len(X), len(X[0])
    H_Y, W_Y = len(Y), len(Y[0])

    # Check if Y is larger
    if H_Y <= H_X or W_Y <= W_X:
        return None

    # Check if divisible
    if H_Y % H_X != 0 or W_Y % W_X != 0:
        return None

    # Compute k
    k_h = H_Y // H_X
    k_w = W_Y // W_X

    # k must be uniform (same in both dimensions)
    if k_h != k_w:
        return None

    return k_h


def extract_motif(Y: Grid, r: int, c: int, k: int) -> List[List[int]]:
    """
    Extract k×k motif from Y at block position (r, c).

    Args:
        Y: Output grid
        r: Block row index (0-indexed)
        c: Block column index (0-indexed)
        k: Block size

    Returns:
        k×k motif as 2D list
    """
    motif = []
    for dr in range(k):
        row = []
        for dc in range(k):
            row.append(Y[r * k + dr][c * k + dc])
        motif.append(row)
    return motif


def motif_to_tuple(motif: List[List[int]]) -> Tuple[Tuple[int, ...]]:
    """Convert motif to nested tuples for hashability."""
    return tuple(tuple(row) for row in motif)


def tuple_to_motif(motif_tuple: Tuple[Tuple[int, ...]]) -> List[List[int]]:
    """Convert nested tuples back to motif."""
    return [list(row) for row in motif_tuple]


def infer_motifs_from_pair(X: Grid, Y: Grid, k: int) -> Optional[Dict[int, Tuple[Tuple[int, ...]]]]:
    """
    Infer per-color motifs B(c) from a single training pair.

    For each color c in X, extracts all k×k tiles from Y that correspond
    to pixels of color c, and verifies they are ALL IDENTICAL.

    Args:
        X: Input grid H_X × W_X
        Y: Output grid (k*H_X) × (k*W_X)
        k: Inflation factor

    Returns:
        Dict mapping color → motif (as tuple), or None if inconsistent
    """
    if not X or not X[0]:
        return None

    H_X, W_X = len(X), len(X[0])

    # Collect all motifs per color
    color_motifs: Dict[int, List[Tuple[Tuple[int, ...]]]] = defaultdict(list)

    for r in range(H_X):
        for c in range(W_X):
            color = X[r][c]
            motif = extract_motif(Y, r, c, k)
            motif_tuple = motif_to_tuple(motif)
            color_motifs[color].append(motif_tuple)

    # Verify all motifs for each color are identical
    consistent_motifs = {}
    for color, motif_list in color_motifs.items():
        if not motif_list:
            continue

        # Check all motifs are identical
        first_motif = motif_list[0]
        if not all(m == first_motif for m in motif_list):
            return None  # Inconsistent motifs for this color

        consistent_motifs[color] = first_motif

    return consistent_motifs


def infer_blowup_params(train_pairs: List[Tuple[Grid, Grid]]) -> Optional[Tuple[int, Dict[int, Tuple[Tuple[int, ...]]]]]:
    """
    Learn BLOWUP parameters (k, motifs) from training pairs.

    Algorithm:
    1. Infer k from first training pair
    2. Verify k is consistent across ALL training pairs
    3. Infer motifs from first training pair
    4. Verify motifs are consistent across ALL training pairs

    Args:
        train_pairs: List of (input_grid, output_grid) training pairs

    Returns:
        (k, motifs) if consistent across all trains, None otherwise
    """
    if not train_pairs:
        return None

    X0, Y0 = train_pairs[0]

    # Infer k from first pair
    k = infer_k_from_pair(X0, Y0)
    if k is None:
        return None

    # Verify k consistent across all pairs
    for X, Y in train_pairs:
        k_pair = infer_k_from_pair(X, Y)
        if k_pair != k:
            return None  # Inconsistent k

    # Infer motifs from first pair
    motifs = infer_motifs_from_pair(X0, Y0, k)
    if motifs is None:
        return None

    # Verify motifs consistent across all pairs
    for X, Y in train_pairs[1:]:
        motifs_pair = infer_motifs_from_pair(X, Y, k)
        if motifs_pair is None:
            return None  # Inconsistent motifs in this pair

        # Check all colors have same motifs
        for color, motif in motifs.items():
            if color in motifs_pair and motifs_pair[color] != motif:
                return None  # Motif for color changed

    return k, motifs


# =============================================================================
# Verification (FY Exactness)
# =============================================================================

def verify_blowup_block_subst(k: int, motifs: Dict[int, Tuple[Tuple[int, ...]]],
                               train_pairs: List[Tuple[Grid, Grid]]) -> bool:
    """
    Verify that BLOWUP+BLOCK_SUBST reproduces all training outputs exactly.

    FY exactness: fy_gap = 0 for all training pairs.

    Args:
        k: Inflation factor
        motifs: Per-color motifs (as tuples)
        train_pairs: List of (input_grid, output_grid) training pairs

    Returns:
        True if exact on all trains, False otherwise
    """
    # Convert motifs from tuples to lists for application
    motifs_list = {color: tuple_to_motif(motif) for color, motif in motifs.items()}

    for X, Y in train_pairs:
        Y_pred = apply_blowup_block_subst(X, k, motifs_list)

        # Check exact equality
        if Y_pred != Y:
            return False

    return True


# =============================================================================
# Main Entry Point
# =============================================================================

def build_blow_block(theta: dict) -> List[BlowBlockLaw]:
    """
    Build BLOWUP + BLOCK_SUBST law instances from compiled parameters.

    Learns k and motifs B(c) from training pairs and verifies FY exactness.

    Args:
        theta: Compiled parameters containing:
            - train_pairs: List[(Grid, Grid)]

    Returns:
        List of BlowBlockLaw instances that passed FY verification
    """
    laws = []

    train_pairs = theta.get("train_pairs", [])
    if not train_pairs:
        return laws

    # Infer parameters
    params = infer_blowup_params(train_pairs)
    if params is None:
        return laws  # No consistent BLOWUP+BLOCK_SUBST detected

    k, motifs = params

    # Verify FY exactness
    if not verify_blowup_block_subst(k, motifs, train_pairs):
        return laws  # Failed FY exactness

    # Compute motif hashes for receipts
    motif_hashes = {}
    for color, motif_tuple in motifs.items():
        # Hash the motif for receipts
        motif_hashes[color] = hash64(motif_tuple)

    # Create law instance
    law = BlowBlockLaw(
        operation="blowup+block_subst",
        k=k,
        motifs=motifs,
        motif_hashes=motif_hashes
    )
    laws.append(law)

    return laws


# =============================================================================
# Utilities
# =============================================================================

def get_motif_for_color(law: BlowBlockLaw, color: int) -> Optional[List[List[int]]]:
    """
    Get motif for a specific color from a BlowBlockLaw.

    Args:
        law: BlowBlockLaw instance
        color: Color to get motif for

    Returns:
        k×k motif as 2D list, or None if no motif for this color
    """
    if color not in law.motifs:
        return None
    return tuple_to_motif(law.motifs[color])


def print_motif(motif: List[List[int]], label: str = ""):
    """Helper to print motif for debugging."""
    if label:
        print(f"\n{label}:")
    for row in motif:
        print("  " + " ".join(str(c) for c in row))


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "BlowBlockLaw",
    "apply_blowup",
    "apply_block_subst",
    "apply_blowup_block_subst",
    "infer_blowup_params",
    "build_blow_block",
    "verify_blowup_block_subst",
    "get_motif_for_color",
]
