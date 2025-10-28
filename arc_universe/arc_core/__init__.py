"""
arc_core: Core primitives for the ARC-AGI solver.

Provides:
- types: Grid, Pixel, RoleId, and other fundamental types
- order_hash: Global total order and deterministic hashing (SHA-256)
- present: Π_G canonicalization and present structure builders
- wl: Weisfeiler-Leman fixed point on disjoint union
- shape: Shape unification (R×C meet)
- palette: Palette canonicalization and orbit CPRQ
- lattice: FFT/HNF lattice inference with KMP fallback
- canvas: Canvas transform inference (RESIZE/CONCAT/FRAME)
- components: Component extraction and Hungarian matching
"""

__all__ = [
    "order_hash",
    "types",
]
