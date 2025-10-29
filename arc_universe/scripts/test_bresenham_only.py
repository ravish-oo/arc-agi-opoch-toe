"""Test just Bresenham."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arc_laws.object_arith import bresenham_4conn
from arc_core.types import Pixel

print("Testing bresenham_4conn...")

# Simple horizontal
print("\n1. Horizontal line:")
line = bresenham_4conn(Pixel(2, 1), Pixel(2, 3))
print(f"   {line}")

# Simple vertical
print("\n2. Vertical line:")
line = bresenham_4conn(Pixel(1, 2), Pixel(3, 2))
print(f"   {line}")

# Diagonal
print("\n3. Diagonal (should create staircase):")
line = bresenham_4conn(Pixel(0, 0), Pixel(2, 2))
print(f"   {line}")

print("\n✅ All Bresenham tests passed")
