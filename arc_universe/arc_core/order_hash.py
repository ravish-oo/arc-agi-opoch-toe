"""
Global total order and deterministic hashing (WO-01).

Per implementation_plan.md lines 97-126 and clarifications §2, §7.

Provides:
- hash64: SHA-256 canonical hash truncated to 64-bit int
- lex_min: Returns lex-min item by global order
- boundary_hash: Hash of 4-connected (E4) boundary pixels

All functions are deterministic and stable across runs.
No use of Python's built-in hash() (non-deterministic).
"""

import hashlib
import json
from typing import Any, Callable, Iterable, TypeVar

from .types import Grid, Hash64, Pixel

T = TypeVar("T")


def hash64(obj: Any) -> Hash64:
    """
    Deterministic 64-bit hash using SHA-256 on canonical JSON.

    Per clarifications §2 and §7:
    - Uses canonical JSON serialization (sorted keys, no whitespace)
    - SHA-256 for cryptographic-grade determinism
    - Truncates to 64-bit integer (first 8 bytes)

    Args:
        obj: Any JSON-serializable Python object

    Returns:
        64-bit integer hash (0 to 2^64-1)

    Acceptance:
        - Stable across runs (same input → same hash)
        - Independent of dict key order
        - Works with nested structures

    Examples:
        >>> hash64([1, 2, 3]) == hash64([1, 2, 3])
        True
        >>> hash64({"a": 1, "b": 2}) == hash64({"b": 2, "a": 1})
        True
    """
    # Serialize to canonical JSON (sorted keys, compact)
    canonical_json = json.dumps(obj, sort_keys=True, separators=(",", ":"))

    # SHA-256 hash
    sha = hashlib.sha256(canonical_json.encode("utf-8"))

    # Take first 8 bytes (64 bits) as integer
    hash_bytes = sha.digest()[:8]
    hash_int = int.from_bytes(hash_bytes, byteorder="big", signed=False)

    return Hash64(hash_int)


def lex_min(items: Iterable[T], key: Callable[[T], tuple] = lambda x: (x,)) -> T:
    """
    Returns the lexicographically minimal item by global order.

    Per clarifications §7:
    Global order on tuples:
    1. coordinates (r,c) in row-major
    2. color index after palette canon
    3. present component ID
    4. integer hashes ascending
    5. matrices lex on flattened entries

    Args:
        items: Iterable of items to compare
        key: Function extracting comparison tuple (default: identity)

    Returns:
        The lex-min item

    Raises:
        ValueError: If items is empty

    Acceptance:
        - Invariant under permutation of input
        - Uses Python's built-in tuple comparison (lex order)

    Examples:
        >>> lex_min([Pixel(1, 2), Pixel(0, 5), Pixel(1, 0)])
        Pixel(row=0, col=5)
        >>> lex_min([3, 1, 2])
        1
    """
    items_list = list(items)
    if not items_list:
        raise ValueError("lex_min requires non-empty iterable")

    return min(items_list, key=key)


def boundary_hash(component: set[Pixel], grid: Grid) -> Hash64:
    """
    Hash of 4-connected (E4) boundary pixels.

    Per implementation_plan.md lines 109-116 and clarifications §2, §3:
    - Components are 8-connected (8-CC)
    - Boundary detection uses 4-neighbors (E4)
    - Pixel is on boundary if ANY 4-neighbor has different color
    - Returns hash64 of sorted boundary coordinates

    Args:
        component: Set of pixels in the component (8-connected)
        grid: Grid to check neighbor colors

    Returns:
        64-bit hash of boundary pixel coordinates

    Acceptance:
        - Uses E4 (4-connected) neighbors for boundary detection
        - Deterministic (sorts coordinates before hashing)
        - Stable under component re-enumeration

    Examples:
        Plus-shape component:
            . X .
            X X X
            . X .
        Boundary = all 5 pixels (each has at least one different-color 4-neighbor)
    """
    if not component:
        return hash64([])

    # Get the color of the component (assume all pixels same color)
    sample_pixel = next(iter(component))
    component_color = grid[sample_pixel.row][sample_pixel.col]

    # Find boundary pixels (have at least one 4-neighbor with different color)
    boundary_pixels: list[tuple[int, int]] = []

    for pixel in component:
        r, c = pixel.row, pixel.col

        # Check 4-connected neighbors (up, down, left, right)
        neighbors_4 = [
            (r - 1, c),  # up
            (r + 1, c),  # down
            (r, c - 1),  # left
            (r, c + 1),  # right
        ]

        # Pixel is on boundary if any 4-neighbor has different color
        is_boundary = False
        for nr, nc in neighbors_4:
            # Check bounds
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                neighbor_color = grid[nr][nc]
                if neighbor_color != component_color:
                    is_boundary = True
                    break
            else:
                # Out of bounds = different "color" (implicitly boundary)
                is_boundary = True
                break

        if is_boundary:
            boundary_pixels.append((r, c))

    # Sort boundary pixels in row-major order (deterministic)
    boundary_pixels.sort()

    # Hash the sorted coordinates
    return hash64(boundary_pixels)
