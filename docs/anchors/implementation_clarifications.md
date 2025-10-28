# Implementation Clarifications

**Purpose**: This document locks down every ambiguity in the math and engineering specs with precise, unambiguous answers. When you drift during coding, read this first to re-align immediately.

**Status**: Final. All answers below are deterministic and implementation-ready.

---

## Quick Reference: The Five Critical Questions

| # | Question | Answer | Where Used |
|---|----------|--------|------------|
| 1 | WL scope | `{train inputs} ∪ {test input}` (NO outputs) | §3 WL, present roles |
| 2 | Palette canon | Per-task, pooled train∪test inputs; boundary = 4-conn pixels | §7.3, WL seeding |
| 3 | Component IDs | Sort by (lex-min pixel → area → centroid → boundary hash) | §5.2, object arithmetic |
| 4 | E4→E8 escalation | Once only, when push-back proves necessary | §3, WL iteration |
| 5 | Closure order | T_def → T_canvas → T_lattice → T_block → T_object → T_select → T_local → T_Γ | §8, lfp worklist |

---

## 1) WL Scope (No Test-Only Roles)

### Question
Do we run WL on train inputs only, or include the test input? Do we include train outputs?

### Answer (Final)
Run 1-WL on:
```
𝒱 = {X₁, X₂, ..., Xₘ} ∪ {X*}
```
where `Xᵢ` are **train inputs** and `X*` is the **test input**.

**Do NOT include** train outputs `{Y₁, Y₂, ..., Yₘ}`.

### Why This Is Safe
- All WL features are **input-only**: CBC₃ patches, E4 neighbors, SameRow/SameCol bags use only pixel colors
- No output information leaks into the present
- Including X* guarantees **no test-only roles**: every role in X* appears in train support

### Why Train-Only Would Fail
If a color/structure appears only in X* (the test input), train-only WL produces a "role gap" → would need witness/search. Including X* closes this gap deterministically.

### Implementation Checklist
- [x] Collect all train inputs `{X₁, ..., Xₘ}` and test input `X*`
- [x] Apply `Π_G` (present) to each grid individually
- [x] Build disjoint union `𝒱` with pixel IDs tagged by (grid_id, row, col)
- [x] Run WL on the entire union with shared ID space
- [x] Result: global role IDs `c: 𝒱 → ℕ` shared across train and test

---

## 2) Palette Canonicalization (Per-Task, Input-Only)

### Question
Do we canonicalize palette per-grid or per-task? What exactly is "boundary hash"?

### Answer (Final)
**Per-task**, pooled across all inputs (train ∪ test), **not outputs**.

### Algorithm (Deterministic)
```python
def canonicalize_palette_for_task(train_inputs, test_input):
    """
    Returns a single palette mapping used for all grids in the task.
    """
    all_inputs = train_inputs + [test_input]

    # Step 1: Pool color statistics across all inputs
    color_stats = {}
    for grid in all_inputs:
        for color in grid.unique_colors():
            if color not in color_stats:
                color_stats[color] = {
                    'count': 0,
                    'first_appearance': float('inf'),
                    'boundary_hash': compute_boundary_hash(grid, color)
                }
            color_stats[color]['count'] += count_pixels(grid, color)
            color_stats[color]['first_appearance'] = min(
                color_stats[color]['first_appearance'],
                first_scanline_index(grid, color)
            )

    # Step 2: Sort colors by (count ↓, first_appearance ↑, boundary_hash ↑)
    sorted_colors = sorted(
        color_stats.keys(),
        key=lambda c: (-color_stats[c]['count'],
                       color_stats[c]['first_appearance'],
                       color_stats[c]['boundary_hash'])
    )

    # Step 3: Map to canonical digits
    palette_map = {old: new for new, old in enumerate(sorted_colors)}

    return palette_map
```

### Boundary Hash Definition
```python
def compute_boundary_hash(grid, color):
    """
    Hash of 4-connected boundary pixels of all components of this color.
    """
    components = find_8connected_components(grid, color)
    boundary_pixels = []

    for comp in components:
        for pixel in comp:
            # Pixel is on boundary if any 4-neighbor has different color
            if any(neighbor_color != color for neighbor_color in get_E4_neighbors(grid, pixel)):
                boundary_pixels.append(pixel)

    # Sort boundary pixels in row-major order for determinism
    boundary_pixels.sort(key=lambda p: (p.row, p.col))

    # Hash the sorted coordinate list
    return sha256_hash(boundary_pixels)  # 64-bit
```

### Why Per-Task (Not Per-Grid)
- WL bags (SameRow/SameCol) compare colors across grids
- Per-grid palette would destabilize WL seeds → inconsistent role IDs
- Per-task palette keeps WL ID space coherent

### Train Output Handling
When comparing our canonical train outputs to given `Yᵢ`:
- Print palette permutation `πᵢ` that maps our canon to given `Yᵢ`
- Verify `cells_wrong_after_π = 0` (isomorphic by palette)

### Implementation Checklist
- [x] Canonicalize once per task using train∪test **inputs**
- [x] Apply same palette map to all grids (train inputs, test input, train outputs for verification)
- [x] Store boundary hash as 64-bit integer (SHA-256 truncated)
- [x] Use 4-connectivity for boundary detection

---

## 3) Component IDs (Deterministic Ordering)

### Question
How do we assign component IDs deterministically? What breaks ties?

### Answer (Final)
Components = equivalence classes of **(SameColor ∧ 8-connected)** in the present grid.

### Ordering Algorithm (Deterministic)
```python
def assign_component_ids(grid):
    """
    Returns a map from component → deterministic ID.
    """
    components = find_8connected_components_per_color(grid)

    # Sort components by 4-level tie-break
    sorted_components = sorted(
        components,
        key=lambda comp: (
            lex_min_pixel(comp),        # 1. Lex-min pixel (row-major)
            -len(comp),                  # 2. Area (larger first; flip if law needs smaller)
            centroid(comp),              # 3. Centroid (row-major)
            compute_boundary_hash_comp(grid, comp)  # 4. Boundary hash
        )
    )

    # Assign IDs in sorted order
    comp_ids = {id(comp): i for i, comp in enumerate(sorted_components)}
    return comp_ids

def lex_min_pixel(comp):
    """Returns (row, col) of lex-min pixel in component."""
    return min(comp, key=lambda p: (p.row, p.col))

def centroid(comp):
    """Returns (mean_row, mean_col) as tuple for sorting."""
    return (sum(p.row for p in comp) / len(comp),
            sum(p.col for p in comp) / len(comp))
```

### Why These Tie-Breaks
1. **Lex-min pixel**: Most components differ here (top-left corner)
2. **Area**: Separates small/large objects (flip sign if law needs smallest-first)
3. **Centroid**: Geometric center for symmetric objects
4. **Boundary hash**: Final deterministic separator for identical shapes

### Component IDs Are Present-Only
- Never leak component IDs as features into WL
- Use only for:
  - Matching (Hungarian)
  - Serialization (output order)
  - Law parameters (TRANSLATE[comp_id, Δ])

### Implementation Checklist
- [x] Find 8-connected components per color
- [x] Sort by 4-level key (lex-min → area → centroid → boundary)
- [x] Assign IDs 0, 1, 2, ... in sorted order
- [x] Keep component IDs stable across pipeline (recompute deterministically if needed)

---

## 4) E4→E8 Escalation (Once Only, When Necessary)

### Question
When do we escalate from E4 to E8 adjacency in WL? Preemptively or on-demand?

### Answer (Final)
Escalate **once only**, and **only when necessary**.

### Escalation Trigger (Push-Back)
```python
def check_escalation_needed(train_inputs, train_outputs, wl_roles_E4):
    """
    Returns True if E4 alone cannot satisfy training label equalities.
    """
    # Build label kernel from train outputs
    label_kernel = compute_label_kernel(train_outputs)

    # Try to push label kernel back to present (E4 roles)
    interior = compute_interior(label_kernel, wl_roles_E4)

    # Check if E4 present can separate all necessary distinctions
    for train_idx, (X, Y) in enumerate(zip(train_inputs, train_outputs)):
        for pixel_pair in get_must_distinguish_pairs(Y):
            if not can_separate_with_E4(pixel_pair, interior):
                return True  # Escalation needed

    return False  # E4 is sufficient
```

### Escalation Rules
1. **Default**: Use E4 adjacency in WL
2. **Test once**: After E4 WL stabilizes, check if push-back forces escalation
3. **Escalate once**: If needed, add E8 bag to WL iteration and re-run to stability
4. **Never escalate twice**: E8 is the single lawful refinement
5. **Record**: Set `E8: true` in receipts when escalated

### Why Once Only
- A0 (no minted differences): Keep present as coarse as possible
- E8 is the finest reasonable adjacency (diagonal separation)
- If E8 still fails, the task needs different laws (selectors, canvas ops), not finer adjacency

### Implementation Checklist
- [x] Run WL with E4 adjacency first
- [x] Check push-back: can E4 roles express all train label equalities?
- [x] If no: add E8 bag, re-run WL from scratch
- [x] If yes: stop, use E4 roles
- [x] Record boolean `E8_used` in receipts

---

## 5) Closure Composition Order (Fixed Pipeline)

### Question
In what order do we apply closures? Does order matter?

### Answer (Final)
**Order is mathematically independent** (lfp is unique on finite lattice), but we fix a **canonical order** for byte-stable, reproducible runs.

### Fixed Closure Order (Worklist Loop)
```python
def compute_lfp(U0, theta, X_test_present):
    """
    Compute least fixed point of closure composition.
    """
    U = U0.copy()
    changed = True

    while changed:
        changed = False
        U_prev = U.copy()

        # Apply closures in fixed order
        U = apply_definedness_closure(U, X_test_present, theta)    # T_def
        U = apply_canvas_closure(U, theta)                         # T_canvas
        U = apply_lattice_closure(U, theta)                        # T_lattice
        U = apply_block_closure(U, theta)                          # T_block
        U = apply_object_closure(U, theta)                         # T_object
        U = apply_selector_closure(U, theta)                       # T_select
        U = apply_local_paint_closure(U, theta)                    # T_local
        U = apply_interface_closure(U, theta)                      # T_Gamma

        if U != U_prev:
            changed = True

    # Verify: every pixel has exactly one defined expression
    assert all(len(U[q]) == 1 for q in U.keys())
    assert all(is_defined_at(U[q][0], q, X_test_present) for q in U.keys())

    return U
```

### Closure Definitions (What Each Removes)

| Closure | Symbol | Removes |
|---------|--------|---------|
| **Definedness** | T_def | Expressions where `q ∉ Dom(e)` on `Π_G(X_test)` |
| **Canvas** | T_canvas | RESIZE/CONCAT/FRAME that don't reproduce trains |
| **Lattice** | T_lattice | PERIODIC/TILE with wrong phases on trains |
| **Block** | T_block | BLOWUP[k], BLOCK_SUBST[B] that fail train tiles |
| **Object** | T_object | TRANSLATE/COPY/RECOLOR with wrong Δ or colors |
| **Selector** | T_select | ARGMAX/UNIQUE/MODE/CONNECT/FILL that fail trains |
| **Local paint** | T_local | Per-role recolors that mismatch train pixels |
| **Interface** | T_Γ | Expression pairs violating overlap/junction equality |

### Why This Order
1. **T_def first**: Remove undefined expressions immediately (saves work)
2. **Canvas → Lattice → Block**: Canvas establishes domain; lattice/block depend on it
3. **Object → Selector → Local**: Object moves; selectors compute from objects; local paint fills
4. **T_Γ last**: Interface closure needs all other laws to stabilize first

### Termination Guarantee
- Finite lattice: `U₀` has finite size
- Monotone: Each pass removes ≥0 expressions, never adds
- Termination: ≤ `|U₀| - n_pixels` passes (bounded by total removals)
- Typical: Converges in 3-12 passes

### Implementation Checklist
- [x] Initialize `U₀ = ∏_q 𝒫(E_q(θ))` with all legal expressions per pixel
- [x] Loop: apply 8 closures in fixed order
- [x] Check convergence: `U == U_prev`
- [x] Verify: `|U*(q)| == 1` and `is_defined(e_q, q)` for all q
- [x] Record: `lfp.passes`, `lfp.removals`, `lfp.singletons` in receipts

---

## 6) Law Basis (Complete Catalogue)

### The Eight Families (Exhaustive)

Every ARC transform is generated by these modulo G:

1. **Local paint & recolor**
   - SetColor[role → c]
   - Recolor by present role, component class, row/col band, periodic residue

2. **Object arithmetic**
   - TRANSLATE[comp, Δ], COPY[comp, Δ], DELETE[comp]
   - Lines: DRAWLINE[anchor1, anchor2, metric={4conn,8conn}]
   - Skeleton/thinning for 1-px guides

3. **Periodic/tiling**
   - PERIODIC[lattice L, phase φ]
   - Lattice: FFT ACF → HNF → D8 canonical
   - Fallback: 2-D KMP if ACF degenerate

4. **BLOWUP + BLOCK_SUBST**
   - BLOWUP[k]: Kronecker inflate by k×k
   - BLOCK_SUBST[B]: per-color motif B(c) learned on aligned tiles

5. **Histogram/selector**
   - ARGMAX_COLOR[mask], ARGMIN_NONZERO, UNIQUE_COLOR
   - MODE_k×k, PARITY_COLOR
   - Mask = present-definable region; histogram recomputed on test

6. **Canvas arithmetic**
   - RESIZE[pads/crops], CONCAT[axis, gaps], FRAME[color, thickness]
   - Finite candidate enumeration → exact verification → lex-min tie

7. **CONNECT_ENDPOINTS**
   - Shortest 4/8-conn path between two anchors
   - Anchors from present; path = lex-min among geodesics

8. **REGION_FILL**
   - Flood region inside mask with selector color
   - REGION_FILL[mask, UNIQUE_COLOR]

### No Other Laws
If a transform doesn't fit these 8 families, it's either:
- Expressible as a composition (e.g., move-then-recolor)
- Not an official ARC task
- A bug in understanding (revisit task)

---

## 7) Global Order (All Ties Funnel Here)

### Definition (§2 of Math Spec)
Total order on tuples:

```python
def global_order_key(item):
    """
    Universal tie-breaker key for all comparisons.
    """
    if isinstance(item, Pixel):
        return (item.row, item.col)  # Row-major

    elif isinstance(item, Color):
        return item.canonical_index  # After palette canon

    elif isinstance(item, Component):
        return (lex_min_pixel(item), -len(item), centroid(item), boundary_hash(item))

    elif isinstance(item, Hash):
        return int(item)  # Hashes as 64-bit ascending

    elif isinstance(item, Matrix):
        return tuple(item.flatten())  # Lex on flattened entries

    else:
        raise TypeError(f"No global order defined for {type(item)}")
```

### Where Used (Everywhere)
- Π_G: Lex-min D4 orientation for present
- Component matching: Hungarian cost = (inertia ↓, area ↓/↑, boundary hash ↑)
- Δ selection: Lex-min displacement vector
- Lattice basis: Lex-min HNF basis
- Path selection: Lex-min geodesic (CONNECT_ENDPOINTS)
- Histogram ties: Lex-min color among modes
- Canvas ties: Lex-min parameter tuple

### Determinism Guarantee
Using a single global order everywhere ensures:
- Π_G is idempotent: `Π_G(Π_G(X)) = Π_G(X)`
- All tie-breaks are consistent
- Byte-stable outputs across runs

---

## 8) Definedness and Totality

### Domain Tracking
Every expression carries a domain `Dom(e) ⊆ U`:

```python
class Expression:
    def __init__(self, law, params):
        self.law = law
        self.params = params
        self.domain = self.compute_domain()

    def compute_domain(self):
        """
        Returns the set of pixels where this expression is defined.
        """
        if self.law == 'COPY':
            return self.params['source_region']

        elif self.law == 'PERIODIC':
            lattice = self.params['lattice']
            return pixels_covered_by_lattice(lattice)

        elif self.law == 'TRANSLATE':
            comp = self.params['component']
            delta = self.params['delta']
            return shifted_region(comp, delta)

        # ... all laws compute their domain exactly
```

### Composition Domain (Pullback)
```python
def compose(e, f):
    """
    Compose two expressions: (e ∘ f)(x) = e(f(x))
    """
    composed_domain = e.domain.intersection(f.inverse_image(f.domain))
    return Expression(
        law='COMPOSE',
        params={'outer': e, 'inner': f},
        domain=composed_domain
    )
```

### Definedness Closure (T_def)
```python
def apply_definedness_closure(U, X_test_present, theta):
    """
    Remove expressions that are undefined at their target pixel.
    """
    for q in U.keys():
        U[q] = {e for e in U[q] if q in e.compute_domain_on(X_test_present, theta)}
    return U
```

### Totality at LFP
After lfp, every pixel has:
1. Exactly one expression: `|U*(q)| = 1`
2. That expression is defined at q: `q ∈ Dom(e_q)`

Therefore, evaluation `Y^(q) = eval(e_q, Π_G(X*))` is **total** (never undefined).

---

## 9) Receipts (Per-Task JSON)

### Full Receipt Schema
```json
{
  "task_id": "00d62c1b",
  "present": {
    "CBC3": true,
    "E4": true,
    "E8": false,
    "Row1D": false,
    "Col1D": false
  },
  "wl": {
    "iters": 5,
    "roles_train": 47,
    "roles_test": 52,
    "unseen_roles": 0
  },
  "basis_used": [
    "BLOCK_SUBST",
    "REGION_FILL",
    "INTERFACE"
  ],
  "params": {
    "lattice": null,
    "canvas": "unchanged",
    "block": {"k": 3, "motifs": {"4": "hash_a3b9"}},
    "selectors": [{"type": "UNIQUE_COLOR", "mask": "role_17"}],
    "deltas": [],
    "anchors": []
  },
  "fy_glue": {
    "fy_gap": 0,
    "interfaces_ok": true
  },
  "lfp": {
    "passes": 4,
    "removals": 1834,
    "singletons": 400
  },
  "totality": {
    "undefined_reads": 0
  },
  "ties": {
    "component_matching": 2,
    "delta_selection": 0,
    "lattice_basis": 0,
    "path_selection": 0,
    "histogram_mode": 1
  },
  "palette": {
    "orbit_used": false,
    "train_permutations": [],
    "test_isomorphic": null
  },
  "test_diffs": [0, 0, 0, 0, 0, 0, ...]  // All zeros for correct answer
}
```

### Critical Receipt Invariants (Must Hold)
- `wl.unseen_roles == 0` (no test-only roles)
- `fy_glue.fy_gap == 0` (laws reproduce trains exactly)
- `fy_glue.interfaces_ok == true` (no gluing violations)
- `lfp.singletons == n_pixels` (every pixel has unique expression)
- `totality.undefined_reads == 0` (evaluation is total)
- `test_diffs == [0, 0, ...]` OR `palette.test_isomorphic == {"π": [...], "cells_wrong_after_π": 0}`

If any invariant fails, the implementation has a bug.

---

## 10) Common Pitfalls (Avoid These)

### Pitfall 1: Leaking Outputs into Present
**Wrong**: Including train outputs `{Y₁, ..., Yₘ}` in WL union
**Right**: WL on train **inputs** ∪ test input only

### Pitfall 2: Per-Grid Palette
**Wrong**: Canonicalizing palette separately per grid
**Right**: Per-task palette pooled across train∪test inputs

### Pitfall 3: Minting New Expressions
**Wrong**: Adding expressions during lfp that weren't in `U₀`
**Right**: Closures only **remove** expressions; monotone operators only

### Pitfall 4: Random Tie-Breaking
**Wrong**: Using `random.choice()`, set iteration order, or hash randomness
**Right**: All ties resolved by §2 global order (deterministic)

### Pitfall 5: Ignoring Definedness
**Wrong**: Evaluating expressions without checking `q ∈ Dom(e)`
**Right**: T_def closure removes undefined expressions before lfp

### Pitfall 6: Search/Backtracking
**Wrong**: Trying multiple parameter combinations and picking best
**Right**: Deterministic compilation; exactly one θ(S) per task

### Pitfall 7: Incomplete Law Basis
**Wrong**: Omitting selectors, canvas ops, or CONNECT/FILL
**Right**: All 8 families implemented; official tasks span this basis

### Pitfall 8: Test-Dependent Compilation
**Wrong**: Using test input size/colors to guide parameter extraction
**Right**: θ compiled from train inputs only; test used only for WL roles and definedness

---

## 11) Debugging Checklist (When Drifting)

If implementation produces wrong answers or UNSAT, check:

- [ ] **WL scope**: Did you include test input? Exclude train outputs?
- [ ] **Palette**: Per-task? Pooled across inputs? Boundary hash correct?
- [ ] **Component IDs**: Deterministic order? Lex-min → area → centroid → boundary?
- [ ] **E4→E8**: Escalated only when necessary? Recorded boolean?
- [ ] **Closure order**: Fixed 8-stage pipeline? Loop to convergence?
- [ ] **Law basis**: All 8 families? No extra heuristics?
- [ ] **Global order**: All ties use §2 key? No randomness?
- [ ] **Definedness**: T_def removes undefined? Domains tracked correctly?
- [ ] **Totality**: Every pixel singleton at lfp? All defined?
- [ ] **Receipts**: All invariants hold? `unseen_roles=0`, `fy_gap=0`, `undefined_reads=0`?

---

## 12) Quick Re-Alignment (30-Second Read)

**If you've drifted and need to re-anchor immediately, read this:**

1. **WL on train inputs ∪ test input** (no outputs)
2. **Palette per-task** (pooled inputs, boundary = 4-conn)
3. **Component IDs by lex-min pixel** (then area, centroid, boundary)
4. **E8 only when push-back proves necessary** (once max)
5. **Closure order: T_def → canvas → lattice → block → object → select → local → interface** (loop to lfp)
6. **8 law families exhaustive** (local, object, periodic, blowup, selector, canvas, connect, fill)
7. **Global order §2 for all ties** (row-major, canon color, comp ID, hash, matrix)
8. **Definedness tracked** (T_def removes undefined; lfp guarantees totality)
9. **Receipts must show**: `unseen_roles=0`, `fy_gap=0`, `undefined_reads=0`, `singletons=n_pixels`
10. **No search, no randomness, no heuristics** (deterministic compilation → monotone closures → unique lfp)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-28 | Initial locked specification |

---

**End of Implementation Clarifications**

When in doubt during coding, re-read sections 1-5 (the five critical questions) and section 12 (quick re-alignment). These are your anchor points.
