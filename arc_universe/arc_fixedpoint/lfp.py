"""
Least fixed-point (LFP) computation for ARC-AGI (WO-17).

Per implementation_plan.md lines 419-469 and engineering_spec.md §8.

Goal: Worklist LFP on product lattice with fixed 8-stage closure order.

Key principles:
1. Finite product lattice: U0 = ∏_q 𝒫(E_q(θ)) - per-pixel candidate expression sets
2. Monotone closures: All closures only REMOVE expressions, never add
3. Unique LFP: Guaranteed by finite lattice + monotone operators
4. Convergence-based: Loop until no changes (no max_passes limit)
5. Fixed order: T_def → T_canvas → T_lattice → T_block → T_object → T_select → T_local → T_Γ

Mathematical foundation (math_spec.md lines 11, 119-121):
- "every closure is monotone (removes only)"
- "the composite operator has a unique least fixed point (lfp)"
- "At the lfp, each pixel's set is a singleton and defined"

Termination guarantee (math_spec.md line 139):
- Passes ≤ S0 - S* (bounded by total removals)
- Polynomial complexity in grid size, components, lattice peaks

Order independence (clarifications line 256):
- "Order is mathematically independent (lfp is unique on finite lattice)"
- "but we fix a canonical order for byte-stable, reproducible runs"

Acceptance criteria:
- Ends with singletons: |U*[q]| = 1 for all q
- Monotone removals only (never adds expressions)
- Convergence-based termination (no max_passes parameter)
- Passes ≤ S0 - S* (bounded by total removals)
- Fixed closure order matches clarifications §5
"""

from typing import Dict, Set, Tuple
from dataclasses import dataclass

from arc_core.types import Grid, Pixel
from arc_fixedpoint.expressions import Expr
from arc_fixedpoint.closures import (
    apply_definedness_closure,
    apply_canvas_closure,
    apply_lattice_closure,
    apply_block_closure,
    apply_object_closure,
    apply_selector_closure,
    apply_local_paint_closure,
    apply_interface_closure,
)


# =============================================================================
# Types
# =============================================================================


@dataclass
class LFPReceipt:
    """
    LFP computation receipt (for task receipts).

    Per implementation_plan.md line 467:
    "Receipts log passes, removals per stage, final singletons count"

    Per clarifications lines 521-524:
    {
        "passes": 4,
        "removals": 1834,
        "singletons": 400
    }
    """
    passes: int                      # Number of worklist iterations until convergence
    total_removals: int              # Total expressions removed across all passes
    removals_per_stage: Dict[str, int]  # Removals per closure (T_def, T_canvas, etc.)
    singletons: int                  # Final singleton count (must equal n_pixels)


# =============================================================================
# Main Entry Point
# =============================================================================


def compute_lfp(
    U0: Dict[Pixel, Set[Expr]],
    theta: dict,
    X_test_present: Grid
) -> Tuple[Dict[Pixel, Set[Expr]], LFPReceipt]:
    """
    Compute least fixed point of closure composition.

    Per implementation_plan.md lines 423-456:
    Fixed-point loop applying 8 closure functions in fixed order (clarifications §5):
    T_def → T_canvas → T_lattice → T_block → T_object → T_select → T_local → T_Γ

    Terminates when no removals occur in a full pass (convergence-based).
    Returns U_star where |U_star[q]| = 1 for all q (singletons).

    Args:
        U0: Initial product lattice U0 = ∏_q 𝒫(E_q(θ))
            Dictionary mapping each pixel q to set of candidate expressions
        theta: Compiled task parameters (from WO-19 compile_theta)
            Contains train_pairs, components, lattice, canvas, etc.
        X_test_present: Canonical test grid ΠG(X_test) for definedness checks

    Returns:
        (U_star, receipt) where:
        - U_star: Fixed-point state with |U_star[q]| = 1 for all q
        - receipt: LFPReceipt with passes, removals, singletons

    Raises:
        AssertionError: If convergence fails (non-singletons remain after LFP)

    Acceptance:
        - Ends with singletons: |U*[q]| = 1 for all q
        - Monotone removals only (never adds expressions)
        - Convergence-based termination (no max_passes parameter)
        - Passes ≤ S0 - S* (bounded by total removals)
        - Fixed closure order matches clarifications §5

    Algorithm:
        1. Initialize U = U0.copy()
        2. While changed:
            a. Apply 8 closures in fixed order
            b. Check if U changed from previous iteration
        3. Verify all pixels are singletons
        4. Return U_star and receipt

    Example:
        >>> U0 = {Pixel(0,0): {expr1, expr2}, Pixel(0,1): {expr3}}
        >>> U_star, receipt = compute_lfp(U0, theta, X_test)
        >>> assert all(len(U_star[q]) == 1 for q in U_star.keys())
        >>> print(f"Converged in {receipt.passes} passes")
    """
    # Initialize working state
    U = _deep_copy_lattice(U0)
    changed = True
    passes = 0

    # Track removals per stage for receipts
    removals_per_stage: Dict[str, int] = {
        "T_def": 0,
        "T_canvas": 0,
        "T_lattice": 0,
        "T_block": 0,
        "T_object": 0,
        "T_select": 0,
        "T_local": 0,
        "T_Gamma": 0,
    }

    # Worklist loop: apply closures until convergence
    while changed:
        U_prev = _deep_copy_lattice(U)

        # Fixed 8-stage closure pipeline (clarifications §5, line 17)
        # Order: T_def → T_canvas → T_lattice → T_block → T_object → T_select → T_local → T_Γ

        U, removed_def = apply_definedness_closure(U, X_test_present, theta)
        removals_per_stage["T_def"] += removed_def

        U, removed_canvas = apply_canvas_closure(U, theta)
        removals_per_stage["T_canvas"] += removed_canvas

        U, removed_lattice = apply_lattice_closure(U, theta)
        removals_per_stage["T_lattice"] += removed_lattice

        U, removed_block = apply_block_closure(U, theta)
        removals_per_stage["T_block"] += removed_block

        U, removed_object = apply_object_closure(U, theta)
        removals_per_stage["T_object"] += removed_object

        U, removed_select = apply_selector_closure(U, theta)
        removals_per_stage["T_select"] += removed_select

        U, removed_local = apply_local_paint_closure(U, theta)
        removals_per_stage["T_local"] += removed_local

        U, removed_gamma = apply_interface_closure(U, theta)
        removals_per_stage["T_Gamma"] += removed_gamma

        # Check convergence: did any closure remove expressions?
        changed = (U != U_prev)
        passes += 1

    # Compute total removals
    total_removals = sum(removals_per_stage.values())

    # Count singletons (should equal total number of pixels)
    singletons = sum(1 for q in U.keys() if len(U[q]) == 1)

    # CRITICAL VERIFICATION (implementation_plan line 454):
    # "every pixel has exactly one defined expression"
    assert all(len(U[q]) == 1 for q in U.keys()), \
        f"Non-singletons remain after LFP: {singletons}/{len(U)} pixels are singletons"

    # Build receipt
    receipt = LFPReceipt(
        passes=passes,
        total_removals=total_removals,
        removals_per_stage=removals_per_stage,
        singletons=singletons
    )

    return U, receipt


# =============================================================================
# Helper Functions
# =============================================================================


def _deep_copy_lattice(U: Dict[Pixel, Set[Expr]]) -> Dict[Pixel, Set[Expr]]:
    """
    Deep copy product lattice for comparison.

    Creates a new dictionary with new sets (shallow copy of Expr objects,
    which are immutable frozen dataclasses).

    Args:
        U: Product lattice to copy

    Returns:
        Deep copy of U with new dict and new sets
    """
    return {q: set(exprs) for q, exprs in U.items()}


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "compute_lfp",
    "LFPReceipt",
]
