"""
ARC Laws: Deterministic transformation law families (WO-09 through WO-14).

Per implementation_plan.md Phase C and engineering_spec.md §5.

Each law module builds instances from training data that:
- Are exact on trains (reproduce all train pixels)
- Are total on their domain
- Are parameterized deterministically (no search)

Law families:
- local_paint.py (WO-09): Per-role recolor
- object_arith.py (WO-10): Copy/move/Δ, lines, skeleton
- periodic.py (WO-11): Tiling from lattice & phases
- blow_block.py (WO-12): BLOWUP[k], BLOCK_SUBST[B]
- selectors.py (WO-13): ARGMAX, UNIQUE, MODE, PARITY
- connect_fill.py (WO-14): CONNECT_ENDPOINTS, REGION_FILL
"""

from .local_paint import build_local_paint
from .selectors import apply_selector_on_test

__all__ = ["build_local_paint", "apply_selector_on_test"]
