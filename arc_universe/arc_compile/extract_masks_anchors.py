"""
Extract masks and anchors from PRESENT for CONNECT/FILL laws (WO-19).

GROUNDED IN engineering_spec.md §5.5, §5.8, §6:
- §5.8 line 108: "Masks are present-definable (band roles, component classes, periodic classes)"
- §5.8 line 121: "closures drop any variant not matching train"
- §5.7: "CONNECT_ENDPOINTS: anchors from present"

Key principle (A0): "Work only in canonical present... never mint new facts"

Strategy per engineering_spec.md line 113 (Canvas enumeration):
"enumerate finite parameter candidates... closures drop any variant not matching train"

We ENUMERATE all (mask_type, selector_type) combinations.
Closures (T_select, FY verification) will filter to only those that work.
"""

from typing import Dict, List, Set, Tuple, Optional

from arc_core.types import Grid, Pixel, Present
from arc_core.components import extract_components
from arc_compile.mask_eval import MaskSpec


def extract_fill_masks(
    train_pairs: List[Tuple[Grid, Grid]],
    presents_train: List[Present],
    theta: dict
) -> List[dict]:
    """
    Extract REGION_FILL mask candidates by ENUMERATING all combinations.

    Per engineering_spec.md line 121:
    "closures drop any variant not matching train"

    Per engineering_spec.md line 113 (Canvas example):
    "enumerate finite parameter candidates from train sizes"

    Strategy:
    1. Enumerate ALL mask types (band_role, component_interior, periodic_class)
    2. For EACH mask, enumerate ALL selector types
    3. Return all (mask, selector) combinations
    4. Let build_connect_fill() + FY verification filter to valid laws

    Args:
        train_pairs: List[(X_i, Y_i)] - canonicalized training pairs
        presents_train: List[Present] - present structures for train inputs
        theta: Compiled parameters (contains shape, lattice, etc.)

    Returns:
        List of mask specs: [{"mask_spec": MaskSpec, "selector_type": str, "k": int}]

    Grounding:
        - engineering_spec.md line 108: Three mask types
        - engineering_spec.md line 121: Closure-based filtering
        - math_spec.md line 106: "histogram on ΠG(X_train) over present masks"
    """
    mask_specs = []

    # Per engineering_spec.md line 108, three mask types:
    # 1. Band roles
    # 2. Component classes
    # 3. Periodic classes

    # =========================================================================
    # A) BAND ROLE MASKS
    # =========================================================================
    # Per engineering_spec.md line 180:
    # "1-D WL bands yields band roles (masks)"

    shape_result = theta.get("shape")
    if shape_result:
        num_row_classes = shape_result.num_row_classes
        num_col_classes = shape_result.num_col_classes

        # Enumerate all (row_class, col_class) pairs
        for r_class in range(num_row_classes):
            for c_class in range(num_col_classes):
                mask_spec = MaskSpec(
                    mask_type="band_role",
                    params={"band": (r_class, c_class)}
                )
                mask_specs.append({"mask_spec": mask_spec})

    # =========================================================================
    # B) COMPONENT INTERIOR MASKS
    # =========================================================================
    # Per engineering_spec.md line 108:
    # "component classes" as mask type

    # Get all colors seen in training inputs
    colors_seen = set()
    for X_i, _ in train_pairs:
        for row in X_i:
            for color in row:
                if color != 0:  # Exclude background
                    colors_seen.add(color)

    # For each color, enumerate interior/boundary/all regions
    for color in colors_seen:
        for region in ["interior", "boundary", "all"]:
            mask_spec = MaskSpec(
                mask_type="component_interior",
                params={"color": color, "region": region}
            )
            mask_specs.append({"mask_spec": mask_spec})

    # =========================================================================
    # C) PERIODIC CLASS MASKS (if lattice detected)
    # =========================================================================
    # Per engineering_spec.md line 108:
    # "periodic classes" as mask type

    lattice_result = theta.get("lattice")
    if lattice_result and hasattr(lattice_result, "periods"):
        periods = lattice_result.periods
        if periods:
            period_r, period_c = periods

            # Enumerate all residue classes
            for r_offset in range(period_r):
                for c_offset in range(period_c):
                    mask_spec = MaskSpec(
                        mask_type="periodic_class",
                        params={"residue_class": (r_offset, c_offset)}
                    )
                    mask_specs.append({"mask_spec": mask_spec})

    # =========================================================================
    # D) ENUMERATE SELECTOR TYPES FOR EACH MASK
    # =========================================================================
    # Per engineering_spec.md §5.5:
    # "ARGMAX_COLOR, ARGMIN_NONZERO, UNIQUE_COLOR, MODE_k×k, PARITY_COLOR"

    selector_types = [
        "ARGMAX",
        "ARGMIN_NONZERO",
        "UNIQUE",
        "PARITY",
        # TODO: MODE_kxk requires enumerating k values
    ]

    # Create all (mask, selector) combinations
    fill_candidates = []
    for mask_dict in mask_specs:
        mask_spec = mask_dict["mask_spec"]
        for selector_type in selector_types:
            fill_candidates.append({
                "mask_spec": mask_spec,
                "selector_type": selector_type,
                "k": None  # TODO: enumerate k for MODE_kxk
            })

    return fill_candidates


def extract_connect_anchors(
    train_pairs: List[Tuple[Grid, Grid]],
    presents_train: List[Present]
) -> List[dict]:
    """
    Extract CONNECT_ENDPOINTS anchors from training pairs.

    Per user clarifications (lines 113-143), three detection methods in priority order:
    1. Markers (area=1 components with unique colors)
    2. Centroids of role components
    3. Diff-path endpoints from outputs

    Per engineering_spec.md §5.7:
    "Unique shortest 4/8-connected path between two present-definable anchors"

    Returns:
        List of anchor specs: [{"anchor1": Pixel, "anchor2": Pixel, "metric": str, "color": int}]

    TODO: Implement anchor detection methods
    For now, returns empty list (most tasks use FILL not CONNECT)
    """
    # TODO: Implement markers detection
    # TODO: Implement centroids detection
    # TODO: Implement endpoints detection
    return []


__all__ = [
    "extract_fill_masks",
    "extract_connect_anchors",
]
