"""
Evaluate singletons to Y^; unpresent by g_test^{-1} (WO-18).

Per implementation_plan.md lines 473-477 and math_spec.md §1, §8.

Provides:
- evaluate_and_unpresent(U_star, Xhat, theta, g_test): Final output Y*

All evaluation is deterministic and total after LFP guarantees.
"""

from typing import Dict, Set

from arc_core.present import D4_TRANSFORMATIONS
from arc_core.types import Grid, Pixel
from arc_fixedpoint.expressions import Expr


def evaluate_and_unpresent(
    U_star: Dict[Pixel, Set[Expr]],
    Xhat: Grid,
    theta: dict,
    g_test: str
) -> Grid:
    """
    Evaluate singletons to Y^; unpresent by g_test^{-1}.

    Per math_spec.md line 125:
        Y^(q) = eval(e_q, ΠG(X*))
        Y* = g_test^{-1}(Y^)

    Implementation follows two-step process:
    1. Evaluate: For each pixel q, extract singleton expression e_q and evaluate
       on canonical grid Xhat to produce Y^[q]
    2. Unpresent: Apply inverse transformation g_test to Y^ to produce final Y*

    Args:
        U_star: LFP result (singleton expression per pixel)
        Xhat: Canonical test grid (ΠG(X*))
        theta: Compiled parameters from training (needed by Expr.eval())
        g_test: Inverse transformation name (from Present.g_inverse)

    Returns:
        Final output grid Y*

    Acceptance:
        - All pixels have singletons (|U_star[q]| == 1)
        - Evaluation is total (every expression defined at its pixel)
        - Unpresent produces grid with correct dimensions
        - Deterministic: same inputs → same output

    Examples:
        >>> # After LFP converges to singletons
        >>> U_star = {Pixel(0,0): {LocalPaintExpr(role=1, color=3)}, ...}
        >>> Xhat = [[0, 1], [2, 3]]
        >>> g_test = "rot270"  # Inverse of rot90
        >>> Y_star = evaluate_and_unpresent(U_star, Xhat, theta, g_test)
    """
    # Step 1: Assert singletons (per math_spec.md line 11)
    # "At the lfp, each pixel's set is a singleton"
    singletons = sum(1 for q in U_star.keys() if len(U_star[q]) == 1)
    total_pixels = len(U_star)

    assert singletons == total_pixels, \
        f"Non-singletons remain: {singletons}/{total_pixels} pixels are singletons"

    # Step 2: Evaluate expressions → Y^ (per math_spec.md line 125)
    # Y^(q) = eval(e_q, ΠG(X*))
    rows, cols = len(Xhat), len(Xhat[0])
    Y_hat = [[0 for _ in range(cols)] for _ in range(rows)]

    for q in U_star.keys():
        # Extract unique expression from singleton set
        expr_set = U_star[q]
        e_q = next(iter(expr_set))  # Get the single expression

        # Evaluate expression on canonical grid
        # Per expressions.py lines 57-75: eval returns int (color 0-9)
        # Per implementation_clarifications §8: evaluation is total at lfp
        color = e_q.eval(q, Xhat, theta)
        Y_hat[q.row][q.col] = color

    # Step 3: Unpresent → Y* (per math_spec.md line 125)
    # Y* = g_test^{-1}(Y^)
    #
    # Note: g_test is already the inverse transformation name
    # (from Present.g_inverse = D4_INVERSES[transform_name])
    # So we apply it directly to unpresent the canonical grid
    Y_star = D4_TRANSFORMATIONS[g_test](Y_hat)

    return Y_star
