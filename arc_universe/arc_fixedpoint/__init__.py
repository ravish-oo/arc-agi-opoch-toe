"""
Fixed-point solver for ARC-AGI (WO-15, WO-16, WO-17).

Per implementation_plan.md Phase D and engineering_spec.md §8.

Modules:
- expressions.py (WO-15): Expression representation E_q(θ), domains Dom(e)
- closures.py (WO-16): 8 closure functions (T_def, T_canvas, ..., T_Γ)
- lfp.py (WO-17): Least fixed-point computation on product lattice
"""

from .expressions import Expr, init_expressions

__all__ = ["Expr", "init_expressions"]
