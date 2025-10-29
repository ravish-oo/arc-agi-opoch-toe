"""
Expression representation E_q(θ), domains Dom(e) (WO-15).

Per implementation_plan.md lines 329-337 and engineering_spec.md §7.

Expressions represent transformation rules in the fixed-point system:
- Each expression has a kind (law family) and parameters
- eval(q, grid, theta): Evaluate expression at pixel q → color
- Dom(q, grid, theta): Check if expression is defined at pixel q → bool
- Composition: Dom(e∘f) = Dom(e) ∩ f⁻¹(Dom(f))

Expression types (8 law families per math_spec.md §6):
1. LOCAL_PAINT: Per-role/component/band/phase recolor (WO-09)
2. TRANSLATE/COPY/RECOLOR: Object arithmetic (WO-10)
3. PERIODIC/TILE: Tiling from lattice (WO-11)
4. BLOWUP/BLOCK_SUBST: Motif substitution (WO-12)
5. SELECTOR: Histogram selectors (WO-13)
6. RESIZE/CONCAT/FRAME: Canvas operations (WO-07)
7. CONNECT_ENDPOINTS: Path drawing (WO-14)
8. REGION_FILL: Flood fill (WO-14)
9. COMPOSE: Expression composition

All expressions are immutable (frozen dataclasses) for use in sets.
Equality and hashing support set operations in U (product lattice).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple, List, Any
from abc import ABC, abstractmethod

from arc_core.types import Grid, Pixel, RoleId, ComponentId


# =============================================================================
# Base Expression Class
# =============================================================================


@dataclass(frozen=True)
class Expr(ABC):
    """
    Base expression class (per engineering_spec.md §7).

    Each expression is a parameterized transformation that:
    - Has a domain Dom(e) ⊆ U (pixels where it's defined)
    - Evaluates to a color at any pixel in its domain
    - Composes with other expressions

    Subclasses implement specific law families.

    Per A0 (no minting): Only expressions derivable from θ are created.
    Per A1 (FY exactness): All laws reproduce training outputs exactly.
    """

    kind: str  # Expression type (law family)

    @abstractmethod
    def eval(self, q: Pixel, grid: Grid, theta: dict) -> int:
        """
        Evaluate expression at pixel q on grid.

        Args:
            q: Target pixel
            grid: Canonical grid (ΠG(X*))
            theta: Compiled parameters from training

        Returns:
            Color (0-9) at pixel q

        Precondition:
            q ∈ Dom(self) (checked by T_def closure before evaluation)

        Note:
            Evaluation is total on domain (never undefined after T_def).
        """
        pass

    @abstractmethod
    def Dom(self, q: Pixel, grid: Grid, theta: dict) -> bool:
        """
        Check if expression is defined at pixel q.

        Args:
            q: Pixel to check
            grid: Canonical grid (ΠG(X*))
            theta: Compiled parameters from training

        Returns:
            True if expression is defined at q, False otherwise

        Note:
            Used by T_def closure (WO-16) to remove undefined expressions.
        """
        pass


# =============================================================================
# 1. Local Paint (WO-09)
# =============================================================================


@dataclass(frozen=True)
class LocalPaintExpr(Expr):
    """
    Per-role/component/band/phase recolor (WO-09).

    Paints pixels belonging to a specific role with a fixed color.

    Per engineering_spec.md §5.1:
    - role_id: WL role from present (WO-03)
    - color: Target color (0-9)
    - mask_type: "role" | "component" | "band" | "phase"
    """

    kind: str = field(default="LOCAL_PAINT", init=False)
    role_id: RoleId
    color: int  # Target color (0-9)
    mask_type: str  # "role" | "component" | "band" | "phase"

    def eval(self, q: Pixel, grid: Grid, theta: dict) -> int:
        """Paint pixel q with the fixed color."""
        return self.color

    def Dom(self, q: Pixel, grid: Grid, theta: dict) -> bool:
        """Defined if q belongs to this role."""
        role_map = theta.get("role_map", {})
        # role_map: Dict[(grid_id, Pixel), RoleId]
        # For test grid, grid_id is typically -1 or "test"
        grid_id = theta.get("test_grid_id", -1)
        return role_map.get((grid_id, q)) == self.role_id


# =============================================================================
# 2. Object Arithmetic (WO-10)
# =============================================================================


@dataclass(frozen=True)
class TranslateExpr(Expr):
    """
    Translate component by Δ (WO-10).

    Per engineering_spec.md §5.2:
    - component_id: Component to translate
    - delta: (dr, dc) displacement vector
    - color: Optional recolor (None = keep original)
    """

    kind: str = field(default="TRANSLATE", init=False)
    component_id: ComponentId
    delta: Tuple[int, int]  # (dr, dc)
    color: Optional[int] = None  # None = keep original, int = recolor

    def eval(self, q: Pixel, grid: Grid, theta: dict) -> int:
        """Evaluate translated component at q."""
        # Read from source position (q - delta)
        dr, dc = self.delta
        source_pixel = Pixel(q.row - dr, q.col - dc)

        # Check if source is in bounds
        rows, cols = len(grid), len(grid[0])
        if not (0 <= source_pixel.row < rows and 0 <= source_pixel.col < cols):
            # Undefined (should be caught by Dom)
            return 0

        # Read color from source
        source_color = grid[source_pixel.row][source_pixel.col]

        # Apply recolor if specified
        if self.color is not None:
            return self.color
        else:
            return source_color

    def Dom(self, q: Pixel, grid: Grid, theta: dict) -> bool:
        """Defined if q is in the translated component's bounding box."""
        # Check if this pixel is a valid target for this component
        components = theta.get("components", {})
        comp = components.get(self.component_id)
        if comp is None:
            return False

        # Check if q - delta is in the original component
        dr, dc = self.delta
        source_pixel = Pixel(q.row - dr, q.col - dc)

        # Check bounds
        rows, cols = len(grid), len(grid[0])
        if not (0 <= source_pixel.row < rows and 0 <= source_pixel.col < cols):
            return False

        # Check if source pixel is in this component
        # BUG FIX (Gate E): comp is Component dataclass, not dict
        # Per arc_core/components.py:52, pixels is a frozenset attribute
        comp_pixels = comp.pixels
        return source_pixel in comp_pixels


@dataclass(frozen=True)
class CopyExpr(Expr):
    """
    Copy component to new location (WO-10).

    Similar to TRANSLATE but explicitly a copy operation.
    """

    kind: str = field(default="COPY", init=False)
    component_id: ComponentId
    target_pixel: Pixel  # Where to copy to
    color: Optional[int] = None

    def eval(self, q: Pixel, grid: Grid, theta: dict) -> int:
        """Evaluate copied component at q."""
        # Similar to TRANSLATE but with explicit target
        # Implementation depends on how copy is parameterized in theta
        # For now, delegate to component data
        if self.color is not None:
            return self.color
        else:
            return grid[q.row][q.col]

    def Dom(self, q: Pixel, grid: Grid, theta: dict) -> bool:
        """Defined if q is in the copy target region."""
        # Check if q is in the target region for this copy
        components = theta.get("components", {})
        comp = components.get(self.component_id)
        if comp is None:
            return False

        # Check if q is in target region (implementation-specific)
        return True  # Placeholder


# =============================================================================
# 3. Periodic/Tiling (WO-11)
# =============================================================================


@dataclass(frozen=True)
class PeriodicExpr(Expr):
    """
    Periodic tiling from lattice (WO-11).

    Per engineering_spec.md §5.3:
    - phase_id: Phase in the periodic structure
    - color: Color for this phase
    """

    kind: str = field(default="PERIODIC", init=False)
    phase_id: int
    color: int

    def eval(self, q: Pixel, grid: Grid, theta: dict) -> int:
        """Paint pixel with phase color."""
        return self.color

    def Dom(self, q: Pixel, grid: Grid, theta: dict) -> bool:
        """Defined if q belongs to this phase."""
        periodic_structure = theta.get("periodic_structure")
        if periodic_structure is None:
            return False

        phase_table = periodic_structure.get("phase_table")
        if phase_table is None:
            return False

        # Check bounds
        rows, cols = phase_table.shape
        if not (0 <= q.row < rows and 0 <= q.col < cols):
            return False

        # Convert numpy boolean to Python bool
        return bool(phase_table[q.row, q.col] == self.phase_id)


# =============================================================================
# 4. Blowup/Block Substitution (WO-12)
# =============================================================================


@dataclass(frozen=True)
class BlowupExpr(Expr):
    """
    Kronecker inflate by k (WO-12).

    Per engineering_spec.md §5.4:
    - k: Blowup factor (each pixel → k×k block)
    """

    kind: str = field(default="BLOWUP", init=False)
    k: int

    def eval(self, q: Pixel, grid: Grid, theta: dict) -> int:
        """Read from source position (q // k)."""
        source_row = q.row // self.k
        source_col = q.col // self.k

        # Check bounds
        rows, cols = len(grid), len(grid[0])
        if not (0 <= source_row < rows and 0 <= source_col < cols):
            return 0

        return grid[source_row][source_col]

    def Dom(self, q: Pixel, grid: Grid, theta: dict) -> bool:
        """Defined if q maps to a valid source pixel."""
        source_row = q.row // self.k
        source_col = q.col // self.k

        rows, cols = len(grid), len(grid[0])
        return 0 <= source_row < rows and 0 <= source_col < cols


@dataclass(frozen=True)
class BlockSubstExpr(Expr):
    """
    Per-color k×k motif substitution (WO-12).

    Per engineering_spec.md §5.4:
    - k: Motif size
    - motifs: Dict[color, k×k grid] mapping colors to motifs
    """

    kind: str = field(default="BLOCK_SUBST", init=False)
    k: int
    source_color: int  # Which color this motif replaces

    def eval(self, q: Pixel, grid: Grid, theta: dict) -> int:
        """Read from motif at (q mod k) offset."""
        # Get motifs from theta
        motifs = theta.get("motifs", {})
        motif = motifs.get(self.source_color)

        if motif is None:
            return 0

        # Position within k×k block
        r_offset = q.row % self.k
        c_offset = q.col % self.k

        # Check bounds
        if not (0 <= r_offset < len(motif) and 0 <= c_offset < len(motif[0])):
            return 0

        return motif[r_offset][c_offset]

    def Dom(self, q: Pixel, grid: Grid, theta: dict) -> bool:
        """Defined if motif exists for source color at this position."""
        motifs = theta.get("motifs", {})
        return self.source_color in motifs


# =============================================================================
# 5. Selector (WO-13)
# =============================================================================


@dataclass(frozen=True)
class SelectorExpr(Expr):
    """
    Histogram selector on present-definable mask (WO-13).

    Per engineering_spec.md §5.5:
    - selector_type: "ARGMAX" | "ARGMIN_NONZERO" | "UNIQUE" | "MODE_kxk" | "PARITY"
    - mask: Set of pixels defining the mask
    - k: Optional window size for MODE_kxk
    """

    kind: str = field(default="SELECTOR", init=False)
    selector_type: str
    mask: frozenset  # Frozen for hashing (will be Set[Pixel] at runtime)
    k: Optional[int] = None  # For MODE_kxk

    def eval(self, q: Pixel, grid: Grid, theta: dict) -> int:
        """
        Evaluate selector at pixel q.

        Note: Selector evaluates to the selected color (same for all pixels).
        """
        # Import here to avoid circular dependency
        from arc_laws.selectors import apply_selector_on_test

        # Convert frozenset back to set
        mask_set = set(self.mask)

        # Apply selector
        color, empty_mask = apply_selector_on_test(
            selector_type=self.selector_type,
            mask=mask_set,
            X_test=grid,
            histogram=None,
            k=self.k
        )

        if empty_mask or color is None:
            # Should be removed by T_select closure
            return 0

        return color

    def Dom(self, q: Pixel, grid: Grid, theta: dict) -> bool:
        """
        Defined if mask is non-empty on grid.

        Note: T_select closure will remove this expression if mask is empty.
        """
        # Convert frozenset to set
        mask_set = set(self.mask)

        if not mask_set:
            return False

        # Check if any mask pixel is in bounds
        rows, cols = len(grid), len(grid[0])
        for pixel in mask_set:
            if 0 <= pixel.row < rows and 0 <= pixel.col < cols:
                return True

        return False


# =============================================================================
# 6. Canvas Operations (WO-07)
# =============================================================================


@dataclass(frozen=True)
class ResizeExpr(Expr):
    """
    Resize with padding/cropping (WO-07).

    Per engineering_spec.md §5.6:
    - pads: (top, bottom, left, right) - positive = pad, negative = crop
    """

    kind: str = field(default="RESIZE", init=False)
    pads: Tuple[int, int, int, int]  # (top, bottom, left, right)

    def eval(self, q: Pixel, grid: Grid, theta: dict) -> int:
        """Read from source position after accounting for padding."""
        top, bottom, left, right = self.pads

        # Map target pixel to source pixel
        source_row = q.row - top
        source_col = q.col - left

        # Check if source is in bounds
        rows, cols = len(grid), len(grid[0])
        if not (0 <= source_row < rows and 0 <= source_col < cols):
            # Outside source grid (padding area)
            return 0  # Or padding color from theta

        return grid[source_row][source_col]

    def Dom(self, q: Pixel, grid: Grid, theta: dict) -> bool:
        """Defined for all pixels in resized canvas."""
        # Resize is always defined (pads with color or crops)
        return True


@dataclass(frozen=True)
class ConcatExpr(Expr):
    """
    Concatenate grids along axis (WO-07).

    Per engineering_spec.md §5.6:
    - axis: 0 (vertical) or 1 (horizontal)
    - gaps: List of gap sizes between concatenated parts
    """

    kind: str = field(default="CONCAT", init=False)
    axis: int  # 0 = vertical, 1 = horizontal
    gaps: Tuple[int, ...]  # Frozen for hashing

    def eval(self, q: Pixel, grid: Grid, theta: dict) -> int:
        """Read from appropriate source grid after accounting for offsets."""
        # Implementation depends on how concat sources are stored in theta
        # Placeholder
        return grid[q.row][q.col] if 0 <= q.row < len(grid) and 0 <= q.col < len(grid[0]) else 0

    def Dom(self, q: Pixel, grid: Grid, theta: dict) -> bool:
        """Defined for all pixels in concatenated canvas."""
        return True


@dataclass(frozen=True)
class FrameExpr(Expr):
    """
    Draw frame around canvas (WO-07).

    Per engineering_spec.md §5.6:
    - color: Frame color
    - thickness: Frame thickness (in pixels)
    """

    kind: str = field(default="FRAME", init=False)
    color: int
    thickness: int

    def eval(self, q: Pixel, grid: Grid, theta: dict) -> int:
        """Return frame color if q is in frame, else read from interior."""
        rows, cols = len(grid), len(grid[0])

        # Check if q is in frame region
        in_top = q.row < self.thickness
        in_bottom = q.row >= rows - self.thickness
        in_left = q.col < self.thickness
        in_right = q.col >= cols - self.thickness

        if in_top or in_bottom or in_left or in_right:
            return self.color
        else:
            # Interior - read from source
            source_row = q.row - self.thickness
            source_col = q.col - self.thickness
            return grid[source_row][source_col]

    def Dom(self, q: Pixel, grid: Grid, theta: dict) -> bool:
        """Defined for all pixels in framed canvas."""
        return True


# =============================================================================
# 7. Connect Endpoints (WO-14)
# =============================================================================


@dataclass(frozen=True)
class ConnectExpr(Expr):
    """
    Draw shortest path between anchors (WO-14).

    Per engineering_spec.md §5.7:
    - anchors: [start_pixel, end_pixel]
    - metric: "4conn" or "8conn"
    - path_color: Color to draw path with
    """

    kind: str = field(default="CONNECT", init=False)
    anchors: Tuple[Pixel, Pixel]  # (start, end) - frozen for hashing
    metric: str  # "4conn" or "8conn"
    path_color: int

    def eval(self, q: Pixel, grid: Grid, theta: dict) -> int:
        """Return path color if q is on path, else keep original."""
        # Check if q is on the computed path
        path_pixels = theta.get("connect_paths", {}).get((self.anchors, self.metric), set())
        if q in path_pixels:
            return self.path_color
        else:
            return grid[q.row][q.col]

    def Dom(self, q: Pixel, grid: Grid, theta: dict) -> bool:
        """Defined if path exists between anchors."""
        # Path must be pre-computed in theta
        path_pixels = theta.get("connect_paths", {}).get((self.anchors, self.metric))
        return path_pixels is not None


# =============================================================================
# 8. Region Fill (WO-14)
# =============================================================================


@dataclass(frozen=True)
class RegionFillExpr(Expr):
    """
    Flood fill region with selector color (WO-14).

    Per engineering_spec.md §5.8:
    - mask: Present-definable mask (region to fill)
    - selector: SelectorExpr that determines fill color
    """

    kind: str = field(default="REGION_FILL", init=False)
    mask: frozenset  # Frozen for hashing (will be Set[Pixel] at runtime)
    fill_color: int  # Pre-computed from selector

    def eval(self, q: Pixel, grid: Grid, theta: dict) -> int:
        """Return fill color if q is in mask, else keep original."""
        mask_set = set(self.mask)
        if q in mask_set:
            return self.fill_color
        else:
            return grid[q.row][q.col]

    def Dom(self, q: Pixel, grid: Grid, theta: dict) -> bool:
        """Defined if mask is non-empty."""
        return len(self.mask) > 0


# =============================================================================
# 9. Composition
# =============================================================================


@dataclass(frozen=True)
class ComposeExpr(Expr):
    """
    Expression composition e∘f (WO-15).

    Per engineering_spec.md §7:
    - Dom(e∘f) = Dom(e) ∩ f⁻¹(Dom(f))
    - eval(e∘f, q) = eval(e, eval(f, q))

    Note: This is a placeholder for future composition support.
    Most ARC tasks use simple expressions, not deep composition.
    """

    kind: str = field(default="COMPOSE", init=False)
    outer: "Expr"
    inner: "Expr"

    def eval(self, q: Pixel, grid: Grid, theta: dict) -> int:
        """Evaluate composition: outer(inner(q))."""
        # First evaluate inner expression
        inner_result = self.inner.eval(q, grid, theta)

        # Then apply outer expression (interpretation depends on expression type)
        # For simplicity, assume outer reads from modified grid
        # Full implementation would require more sophisticated evaluation context
        return self.outer.eval(q, grid, theta)

    def Dom(self, q: Pixel, grid: Grid, theta: dict) -> bool:
        """
        Domain of composition: Dom(e∘f) = Dom(e) ∩ f⁻¹(Dom(f)).

        For simplicity, check both expressions are defined at q.
        Full implementation would require pullback computation.
        """
        return self.inner.Dom(q, grid, theta) and self.outer.Dom(q, grid, theta)


# =============================================================================
# Initialize Expression Sets U₀
# =============================================================================


def init_expressions(theta: dict) -> Dict[Pixel, Set[Expr]]:
    """
    Initialize U₀ = ∏_q 𝒫(E_q(θ)).

    Per engineering_spec.md §8 (lines 150-151):
    "Initialize expression sets E_q(θ) and the product U_0=∏_q 𝒫(E_q)."

    For each pixel q in the canvas, create the set E_q of all candidate
    expressions that could produce a color at q.

    Args:
        theta: Compiled parameters from training containing:
            - canvas_shape: (H, W) - output canvas size
            - role_map: Dict[(grid_id, Pixel), RoleId] - WL roles
            - components: Dict[ComponentId, component_data] - components
            - periodic_structure: PeriodicStructure - lattice/phases
            - motifs: Dict[int, Grid] - block substitution motifs
            - selectors: List of (selector_type, mask, k) tuples
            - connect_paths: Dict - pre-computed paths
            - Other law-specific parameters

    Returns:
        Dict mapping each pixel to its initial set of candidate expressions.

    Note:
        This creates the initial universe U₀. Closures (WO-16) will prune
        this to U* where |U*[q]| = 1 for all q (singletons).

    Acceptance:
        - All expressions derivable from θ (A0 - no minting)
        - Deterministic (same θ → same U₀)
        - Per-pixel sets contain all legal candidates
    """
    # Extract canvas shape
    canvas_shape = theta.get("canvas_shape", (10, 10))  # Default 10×10
    H, W = canvas_shape

    # Initialize U₀: Dict[Pixel, Set[Expr]]
    U0: Dict[Pixel, Set[Expr]] = {}

    # For each pixel q in canvas
    for r in range(H):
        for c in range(W):
            q = Pixel(r, c)
            candidates: Set[Expr] = set()

            # 1. LOCAL_PAINT expressions (WO-09)
            role_map = theta.get("role_map", {})
            for (grid_id, pixel), role_id in role_map.items():
                if pixel == q:
                    # Create LOCAL_PAINT expression for each color
                    for color in range(10):
                        expr = LocalPaintExpr(
                            role_id=role_id,
                            color=color,
                            mask_type="role"
                        )
                        candidates.add(expr)

            # 2. TRANSLATE expressions (WO-10)
            components = theta.get("components", {})
            deltas = theta.get("deltas", [])
            for comp_id in components.keys():
                for delta in deltas:
                    expr = TranslateExpr(
                        component_id=comp_id,
                        delta=delta,
                        color=None  # Keep original
                    )
                    candidates.add(expr)

            # 3. PERIODIC expressions (WO-11)
            periodic_structure = theta.get("periodic_structure")
            if periodic_structure:
                num_phases = periodic_structure.get("num_phases", 0)
                for phase_id in range(num_phases):
                    for color in range(10):
                        expr = PeriodicExpr(
                            phase_id=phase_id,
                            color=color
                        )
                        candidates.add(expr)

            # 4. BLOWUP/BLOCK_SUBST expressions (WO-12)
            k_blowup = theta.get("k_blowup")
            if k_blowup:
                expr = BlowupExpr(k=k_blowup)
                candidates.add(expr)

            motifs = theta.get("motifs", {})
            for source_color in motifs.keys():
                expr = BlockSubstExpr(
                    k=theta.get("k_blowup", 3),
                    source_color=source_color
                )
                candidates.add(expr)

            # 5. SELECTOR expressions (WO-13)
            selectors = theta.get("selectors", [])
            for selector_data in selectors:
                selector_type = selector_data.get("type")
                mask = frozenset(selector_data.get("mask", set()))
                k = selector_data.get("k")

                expr = SelectorExpr(
                    selector_type=selector_type,
                    mask=mask,
                    k=k
                )
                candidates.add(expr)

            # 6. RESIZE/CONCAT/FRAME expressions (WO-07)
            pads = theta.get("resize_pads")
            if pads:
                expr = ResizeExpr(pads=pads)
                candidates.add(expr)

            concat_params = theta.get("concat_params")
            if concat_params:
                expr = ConcatExpr(
                    axis=concat_params.get("axis", 0),
                    gaps=tuple(concat_params.get("gaps", []))
                )
                candidates.add(expr)

            frame_params = theta.get("frame_params")
            if frame_params:
                expr = FrameExpr(
                    color=frame_params.get("color", 0),
                    thickness=frame_params.get("thickness", 1)
                )
                candidates.add(expr)

            # 7. CONNECT expressions (WO-14)
            connect_paths = theta.get("connect_paths", {})
            for (anchors, metric), path_pixels in connect_paths.items():
                expr = ConnectExpr(
                    anchors=anchors,
                    metric=metric,
                    path_color=theta.get("path_color", 1)
                )
                candidates.add(expr)

            # 8. REGION_FILL expressions (WO-14)
            region_fills = theta.get("region_fills", [])
            for fill_data in region_fills:
                mask = frozenset(fill_data.get("mask", set()))
                fill_color = fill_data.get("fill_color", 0)

                expr = RegionFillExpr(
                    mask=mask,
                    fill_color=fill_color
                )
                candidates.add(expr)

            # CRITICAL FIX (BUG-15-01): Ensure non-empty E_q for every pixel
            # Per engineering_spec.md §7 line 142: "At lfp U*, every pixel is singleton"
            # Per clarifications §5 line 549: "lfp.singletons == n_pixels"
            # Empty set cannot converge to singleton → violates spec
            # LFP only removes (monotone), never adds → empty stays empty
            if not candidates:
                # Fallback: Add LOCAL_PAINT for all 10 colors
                # At least one will be correct (enforced by T_local closure per A1)
                # Use default role_id=0 for pixels without explicit role assignments
                for color in range(10):
                    expr = LocalPaintExpr(
                        role_id=RoleId(0),  # Default fallback role
                        color=color,
                        mask_type="role"
                    )
                    candidates.add(expr)

            # Store candidates for this pixel
            U0[q] = candidates

    return U0


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "Expr",
    "LocalPaintExpr",
    "TranslateExpr",
    "CopyExpr",
    "PeriodicExpr",
    "BlowupExpr",
    "BlockSubstExpr",
    "SelectorExpr",
    "ResizeExpr",
    "ConcatExpr",
    "FrameExpr",
    "ConnectExpr",
    "RegionFillExpr",
    "ComposeExpr",
    "init_expressions",
]
