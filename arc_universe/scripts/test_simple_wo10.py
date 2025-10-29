"""Simple test to debug WO-10."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_laws.object_arith import (
    apply_translate,
    bresenham_4conn,
    bresenham_8conn,
)
from arc_core.types import Pixel
from arc_core.components import extract_components


def test_translate():
    """Test TRANSLATE."""
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]

    print("Original:")
    for row in grid:
        print(" ", row)

    comps = extract_components(grid)
    comp_1 = [c for c in comps if c.color == 1][0]

    result = apply_translate(grid, comp_1.pixels, (1, 2), bg_color=0, clear_source=True)

    print("\nAfter translate by (1,2):")
    for row in result:
        print(" ", row)

    print("\n✅ TRANSLATE works")


def test_bresenham():
    """Test Bresenham."""
    p1 = Pixel(0, 0)
    p2 = Pixel(4, 4)

    line_4 = bresenham_4conn(p1, p2)
    print(f"\n4-conn line {p1} to {p2}:")
    print(f"  {line_4}")

    line_8 = bresenham_8conn(p1, p2)
    print(f"\n8-conn line {p1} to {p2}:")
    print(f"  {line_8}")

    print("\n✅ BRESENHAM works")


if __name__ == "__main__":
    test_translate()
    test_bresenham()
    print("\n✅ All simple tests passed")
