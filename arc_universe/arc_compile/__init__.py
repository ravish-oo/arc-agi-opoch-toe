"""
End-to-end compilation: ΠG → WL union → shape meet → param extraction → U0 → closures (WO-19).

Per implementation_plan.md lines 480-487 and clarifications §1-5.

Modules:
- compile_theta.py: End-to-end compile from train_pairs + test_input

All functions deterministic, no search.
"""

from .compile_theta import compile_theta

__all__ = [
    "compile_theta",
]
