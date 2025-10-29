"""Debug skeleton algorithm."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_laws.object_arith import skeleton_zhang_suen
from arc_core.types import Pixel
from arc_core.components import extract_components


def test_tiny_skeleton():
    """Test skeleton on tiny 3x3 block (should reduce to single pixel or cross)."""
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]

    print("Original 3x3 block:")
    for row in grid:
        print(" ", row)

    comps = extract_components(grid)
    comp_1 = [c for c in comps if c.color == 1][0]

    print(f"\nComponent area: {comp_1.area}")
    print(f"Component pixels: {sorted(comp_1.pixels, key=lambda p: (p.row, p.col))}")

    # Skeletonize with iteration limit
    print("\nRunning Zhang-Suen (max 5 iterations)...")
    skeleton_pixels = skeleton_zhang_suen(grid, comp_1.pixels, max_iterations=5)

    print(f"\nSkeleton area: {len(skeleton_pixels)}")
    print(f"Skeleton pixels: {sorted(skeleton_pixels, key=lambda p: (p.row, p.col))}")

    # Visualize
    skel_grid = [[0] * 5 for _ in range(5)]
    for pixel in skeleton_pixels:
        skel_grid[pixel.row][pixel.col] = 1

    print("\nSkeleton visualization:")
    for row in skel_grid:
        print(" ", row)

    assert len(skeleton_pixels) > 0, "Skeleton is empty"
    assert len(skeleton_pixels) < comp_1.area, "Skeleton not thinner"

    print("\n✅ Skeleton test passed")


if __name__ == "__main__":
    test_tiny_skeleton()
