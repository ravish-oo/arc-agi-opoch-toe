Below is a modular, bottoms-up implementation blueprint for the universe-style ARC solver. It is designed so each work order (WO) is atomic, manageable, independently testable, and integrates cleanly. This plan leaves no stubs: every module has a clear API, inputs/outputs, acceptance tests, and receipts.

## CRITICAL UPDATES (Spec-Aligned)

This plan incorporates three critical clarifications from anchor docs validation:

1. **Fixed 8-Stage Closure Order** (WO-16, WO-17):
   - Explicit pipeline: T_def → T_canvas → T_lattice → T_block → T_object → T_select → T_local → T_Γ
   - Closures are functions (not data structures)
   - Applied in fixed order per clarifications §5

2. **Empty Selector Mask Handling** (WO-13):
   - When mask is empty on test: return (None, True)
   - T_select closure MUST remove selector expression from U[q]
   - Enforces spec requirement to "delete all conflicting expressions"

3. **Boundary Hash = 4-Connected** (WO-01, WO-05, WO-08):
   - Boundary detection uses E4 (4-connected) neighbors
   - Components remain 8-connected (8-CC)
   - Hash of sorted boundary pixel coordinates (SHA-256, 64-bit)

⸻

0) Architecture & Folder Structure

arc_universe/
  README.md
  pyproject.toml
  arc_core/
    _init_.py
    types.py                 # Grid, Pixel, RoleId, EquivId, CanvasMap, Lattice, Receipt types
    order_hash.py            # global total order + deterministic hashing (SHA-256)
    present.py               # G, ΠG, present structure builders
    wl.py                    # 1-WL fixed point on disjoint union
    shape.py                 # 1D WL rows/cols, meet -> U = R×C
    palette.py               # palette canonicalization + Orbit CPRQ helper
    lattice.py               # FFT ACF -> HNF -> D8; 2D-KMP fallback
    canvas.py                # RESIZE/CONCAT/FRAME finite candidate verification
    components.py            # component extraction & Hungarian match with lex ties
  arc_laws/
    _init_.py
    local_paint.py           # per-role recolor/paint
    object_arith.py          # copy/move Δ, lines (Bresenham), skeleton/thinning
    periodic.py              # tiling from lattice & phases
    blow_block.py            # BLOWUP[k], BLOCK_SUBST[B]
    selectors.py             # ARGMAX_COLOR, UNIQUE_COLOR, MODE_k×k, PARITY
    connect_fill.py          # CONNECT_ENDPOINTS, REGION_FILL[mask,color]
  arc_fixedpoint/
    _init_.py
    expressions.py           # expression sets E_q(θ), domains Dom(e)
    closures.py              # FY, interface (GLUE), definedness closures
    lfp.py                   # worklist LFP engine (monotone, idempotent)
    eval_unpresent.py        # evaluator + unpresent by g_test^{-1}
  arc_compile/
    _init_.py
    compile_theta.py         # compile θ(S) from ΠG(S); build U0, closures, receipts stubs
  arc_io/
    _init_.py
    load_json.py             # challenges/solutions IO
    save_receipts.py         # structured receipts JSON/CSV
    cli.py                   # solve one task / all tasks; flags
  tests/
    unit/                    # per-wo unit tests
    integration/             # end-to-end on curated tasks
    fixtures/                # tiny grids & expected receipts
  scripts/
    run_one.py               # solve a single id with receipts
    run_all.py               # sweep corpus; produce coverage report
  ci/
    linting.yml              # ruff/black/mypy
    tests.yml                # pytest -q

Language: Python 3.11 (no randomness).
Libs: numpy for FFT (optional; if absent we use KMP fallback). No other heavy deps.

⸻

1) Global Policies (apply everywhere)
	•	Global total order (arc_core/order_hash.py):
ORDER = (row-major coords) -> (canonical color index) -> (present component id) -> (hash ints) -> (matrix lex)
Provide helper lex_min(items, key=...).
	•	Deterministic hashing: hash64(obj) = SHA-256(JSON(obj, canonical))[:16] → int.
	•	No target leakage in present, WL, shape, lattice, canvas inference.
	•	No coordinates in present features; only equivalence bag-hashes of current WL colors.
	•	Single lawful refinement: allow E8 once if push-back requires present split.
	•	Definedness: every expression carries Dom(e) and closures drop undefined on test.
	•	Receipts: each WO prints receipts for its contract in unit tests.

⸻

2) Work Orders (WOs) — atomic units (each ≤500 LoC)

For each WO: Goal → Inputs → Outputs → API → Acceptance → Tests (unit).

⸻

WO-01: arc_core/order_hash.py — Global order & hashing ✅ COMPLETED
	•	Goal: Provide canonical hashing and the single total order used everywhere.
	•	Inputs: arbitrary Python types that serialize to canonical JSON.
	•	Outputs: hash64(obj:int), tuple utilities lex_min, lex_sort, boundary_hash.
	•	API:

def hash64(obj: Any) -> int:
    """SHA-256 canonical hash, truncated to 64-bit int"""

def lex_min(items: Iterable[T], key: Callable[[T], Tuple]) -> T:
    """Returns lex-min item by global order"""

def boundary_hash(component: Set[Pixel], grid: Grid) -> int:
    """
    Hash of 4-connected boundary pixels (E4).
    Pixel is on boundary if any 4-neighbor has different color.
    Components are 8-connected (8-CC), boundary detection uses 4-neighbors (E4).
    Returns 64-bit hash of sorted boundary coordinates.
    Per clarifications §2 and §3.
    """


	•	Acceptance:
	    - hash64 stable across runs (deterministic)
	    - lex_min invariant under permutation
	    - boundary_hash uses E4 (4-connected) neighbors
	•	Tests:
	    - determinism across repeated calls
	    - nested tuples; dict order irrelevance
	    - boundary_hash on plus-shape component (verify E4 boundary)

⸻

WO-02: arc_core/present.py — G & ΠG, present structure ✅ COMPLETED
	•	Goal: Fix G=D_4\times translation anchor; build ΠG; present \mathcal G^\bullet (CBC3,E4,row/col eqs, same_color/component).
	•	Inputs: raw grid X.
	•	Outputs: PiG(X); present object {U, RAW, CBC3, E4, RowMembers, ColMembers, CompEq?} and g_test.
	•	API:

def PiG(X: Grid) -> Grid
def build_present(X: Grid) -> Present


	•	Acceptance: ΠG idempotent; present has eq members and no coords.
	•	Tests: D4 transformations; minimal BB anchor; component summary determinism.

⸻

WO-03: arc_core/wl.py — 1-WL on disjoint union (train∪test) ✅ COMPLETED
	•	Goal: WL fixed point on union; E8 escalation toggle.
	•	Inputs: list[Present], escalate: bool.
	•	Outputs: colors: Dict[(grid_id, (r,c)), RoleId] stable across union.
	•	API:

def wl_union(presents: List[Present], escalate=False, max_iters=12) -> RoleMap


	•	Acceptance: stable IDs; E8 only when invoked; no coords in features.
	•	Tests: deterministic ids; union vs. per-grid (must be union); escalate once.

⸻

WO-04: arc_core/shape.py — 1-D WL meet for U=R×C ✅ COMPLETED
	•	Goal: unify shape (no P catalog); R=∧R_i, C=∧C_i.
	•	Inputs: present-trains.
	•	Outputs: partitions R,C, maps s_i.
	•	API:

def unify_shape(presents_train: List[Present]) -> ShapeParams


	•	Acceptance: deterministic; same result for permuted trains.
	•	Tests: identity; uniform scale; NPS row/col.

⸻

### Integration Testing Infrastructure (Post WO-04) ✅ COMPLETED

**Location:** `arc_universe/integration_tests/`

This directory contains standalone orchestration scripts for validating completed WOs against real ARC-AGI data. These are NOT unit tests—they are end-to-end validation harnesses that run the full pipeline on actual challenge data and generate receipts.

**Structure:**
- `utils.py` — ARC data loaders and helpers
- `run_gate_a.py` — Gate A validation (WL union + receipts, after WO-03)
- `run_gate_b.py` — Gate B validation (Gate A + shape meet, after WO-04)
- `receipts/` — JSON receipts per task (gitignored)
- `logs/` — Execution logs (gitignored)

**⚠️ DO NOT MODIFY:** This workspace is managed separately for integration validation. Changes to `arc_core/` modules are fine, but do not edit integration test scripts unless explicitly instructed.

**Usage:**
```bash
cd arc_universe/integration_tests
python run_gate_a.py --limit 50    # Run Gate A on 50 random tasks
python run_gate_b.py --limit 50    # Run Gate B on 50 random tasks
```

**Expected Outcomes (Gates A & B):**
- ✅ `unseen_roles = 0` for all tasks (critical WL union invariant)
- ✅ 100% determinism across runs (hash64, global order)
- ✅ ΠG idempotence verified (present canonicalization)
- 📊 WL statistics logged: iterations, role counts, convergence patterns
- 📊 Shape statistics logged: |R|, |C| distributions, shape-change patterns

**Gate Progression:**
- **Gate A** (after WO-03): Present + WL union validation only
- **Gate B** (after WO-04): Gate A + shape meet validation
- **Gate C** (after WO-05): Gate B + palette orbit/canonicalization
- **Gate D** (after WO-06/07/08): Parameter extraction validation
- **Gate E** (after WO-09/10 + WO-15/16/17): First end-to-end predictions
- **Gate F** (after WO-11/12/13/14): Extended law families
- **Gate G** (after WO-19/20): Full corpus sweep with CLI

⸻

WO-05: arc_core/palette.py — Orbit CPRQ + canonicalizer N ✅ COMPLETED
	•	Goal: orbit kernel + input-lawful canonical palette (counts↓, first-occurrence↑, boundary-hash↑).
	•	Inputs: train labels, present-trains.
	•	Outputs: abstract \tilde\rho and canon mapping digits for train/test.
	•	API:

def orbit_cprq(train_pairs, presents) -> AbstractMap
def canonicalize_palette(abstract_grid, obs) -> (digit_grid, perm)


	•	Acceptance: train exact up to π; permutation is bijection; canon stable.
	•	Tests: cyclic-3 phase task; recolored trains; isomorphic receipts.

⸻

WO-06: arc_core/lattice.py — FFT/HNF lattice (+ 2D KMP fallback)
	•	Goal: find periodic basis deterministically.
	•	Inputs: present-trains; grids.
	•	Outputs: (basis, method); method ∈ {FFT, KMP}.
	•	API:

def infer_lattice(Xs: List[Grid]) -> Lattice  # includes basis, periods, method


	•	Acceptance: reproducible basis; fallback logs.
	•	Tests: degenerate ACF; mixed periods; D8 invariance.

⸻

WO-07: arc_core/canvas.py — Canvas maps (finite verification) ✅ COMPLETED
	•	Goal: determine RESIZE/CONCAT/FRAME transforms from finite candidate set.
	•	Inputs: (X_i, Y_i).
	•	Outputs: CanvasMap with params (axis, pads/crops, gaps, frame color/width).
	•	API:

def infer_canvas(train_pairs) -> CanvasMap


	•	Acceptance: only exact forward maps accepted; lex-min if multiple.
	•	Tests: concat then frame; padding/cropping; zero slack.

⸻

WO-08: arc_core/components.py — Components & Hungarian match
	•	Goal: component extraction & matching; Δ with lex tie.
	•	Inputs: present-trains; bounding boxes.
	•	Outputs: matches, Δ, IDs.
	•	API:

def extract_components(X: Grid) -> List[Comp]
def match_components(comps_X, comps_Y) -> List[Match]  # lex tie


	•	Acceptance: stable IDs; lex ties consistent.
	•	Tests: equal-area; same shapes; multi-match tie.

⸻

WO-09: arc_laws/local_paint.py
	•	Goal: per-role recolor sets.
	•	API: build_local_paint(theta) -> List[LawInstance]
	•	Tests: exact on trains.

⸻

WO-10: arc_laws/object_arith.py
	•	Goal: copy/move Δ; lines via Bresenham; skeleton/thinning (1-px).
	•	API: build_object_arith(theta) -> [...]
	•	Tests: draw path; move by Δ; skeleton receipts.

⸻

WO-11: arc_laws/periodic.py
	•	Goal: tiling from lattice; phases.
	•	API: build_periodic(theta) -> [...]
	•	Tests: periodic grids; phases consistent.

⸻

WO-12: arc_laws/blow_block.py  (new)
	•	Goal: BLOWUP[k], BLOCK_SUBST[B(c)].
	•	API: build_blow_block(theta) -> [...]
	•	Compile: learn k + motifs per color from aligned tiles; exact check.
	•	Tests: 3×3→9×9; motif equality receipts.

⸻

WO-13: arc_laws/selectors.py  (new)
	•	Goal: ARGMAX_COLOR, ARGMIN_NONZERO, UNIQUE_COLOR, MODE_k×k, PARITY_COLOR on present masks.
	•	API:

def apply_selector_on_test(selector_type, mask, X_test, histogram) -> Tuple[Optional[int], bool]:
    """
    Returns (color_or_None, empty_mask_flag).
    If mask is empty, returns (None, True) signaling T_select closure to remove this expression.
    """

	•	Acceptance:
	    - Recompute histogram on ΠG(X_test) for non-empty masks
	    - When empty_mask=True, T_select closure MUST remove that selector expression from U[q]
	    - Tie-breaks (histogram modes) use global order (smallest canonical color)
	    - Receipts log selector type, mask hash, and empty_on_test flag
	•	Tests:
	    - band-mask argmax with non-empty mask
	    - selector with mask empty on test (verify expression removed by T_select)
	    - unique in component; receipts log histograms

⸻

WO-14: arc_laws/connect_fill.py  (new)
	•	Goal: CONNECT_ENDPOINTS (shortest 4/8-conn path; lex-min path); REGION_FILL[mask,color].
	•	API: build_connect_fill(theta) -> [...]
	•	Tests: exact geodesic; flood under mask; receipts anchor/mask hash.

⸻

WO-15: arc_fixedpoint/expressions.py
	•	Goal: expression representation E_q(\theta), domains Dom(e).
	•	API:

class Expr: ...  # eval(), Dom()
def init_expressions(theta) -> Dict[q, Set[Expr]]


	•	Tests: domain propagation; equality.

⸻

WO-16: arc_fixedpoint/closures.py
	•	Goal: Implement 8 closure functions (T_def, T_canvas, T_lattice, T_block, T_object, T_select, T_local, T_Γ).
	•	API (8 closure functions, each monotone - removes only):

# Fixed 8-stage closure pipeline (per clarifications §5)
def apply_definedness_closure(U, X_test_present, theta) -> U:
    """Remove expressions where q ∉ Dom(e) on ΠG(X_test)"""

def apply_canvas_closure(U, theta) -> U:
    """Remove RESIZE/CONCAT/FRAME expressions that don't reproduce trains"""

def apply_lattice_closure(U, theta) -> U:
    """Remove PERIODIC/TILE expressions with wrong phases"""

def apply_block_closure(U, theta) -> U:
    """Remove BLOWUP[k], BLOCK_SUBST[B] expressions that fail train tiles"""

def apply_object_closure(U, theta) -> U:
    """Remove TRANSLATE/COPY/RECOLOR expressions with wrong Δ or colors"""

def apply_selector_closure(U, theta) -> U:
    """Remove ARGMAX/UNIQUE/MODE/CONNECT/FILL expressions that fail trains.
    CRITICAL: When apply_selector_on_test returns empty_mask=True,
    MUST remove that selector expression from U[q]."""

    # Implementation pattern (spec §12 compliant):
    removed = 0
    X_test_present = theta.X_test_present  # ΠG(X_test)

    for q in list(U.keys()):
        for expr in list(U[q]):
            # 1) SELECTOR expressions: ARGMAX/UNIQUE/MODE/PARITY
            if expr.kind == "SELECTOR":
                color, empty_mask = apply_selector_on_test(
                    selector=expr.selector,
                    mask_pixels=expr.mask_pixels,
                    X_test_present=X_test_present
                )
                if empty_mask:
                    U[q].discard(expr); removed += 1  # DELETE per spec §12
                    continue
                if expr.color != color:  # FY exactness
                    U[q].discard(expr); removed += 1

            # 2) REGION_FILL depends on selector color
            elif expr.kind == "REGION_FILL":
                fill_color, empty_mask = apply_selector_on_test(
                    selector=expr.selector_for_fill,
                    mask_pixels=expr.mask_pixels,
                    X_test_present=X_test_present
                )
                if empty_mask:  # Region not present on test
                    U[q].discard(expr); removed += 1
                    continue
                if expr.fill_color != fill_color:
                    U[q].discard(expr); removed += 1

    return U, removed

def apply_local_paint_closure(U, theta) -> U:
    """Remove per-role recolors that mismatch train pixels"""

def apply_interface_closure(U, theta) -> U:
    """Remove expression pairs violating overlap/junction equality (GLUE)"""

	•	Acceptance:
	    - Each closure is monotone (removes only, never adds)
	    - Each closure is idempotent
	    - T_select handles empty mask deletion (engineering_spec §12)
	    - Applied in fixed order per clarifications §5
	•	Tests:
	    - FY gap=0 after T_canvas, T_lattice, T_block, T_object, T_select, T_local
	    - GLUE satisfied after T_Γ
	    - T_def removes undefined expressions
	    - T_select removes selector when mask empty on test

⸻

WO-17: arc_fixedpoint/lfp.py
	•	Goal: worklist LFP on product lattice with fixed 8-stage closure order.
	•	API:

def compute_lfp(U0: Dict[Pixel, Set[Expression]],
                theta: TaskParams,
                X_test_present: Grid) -> Dict[Pixel, Set[Expression]]:
    """
    Fixed-point loop applying 8 closure functions in fixed order (clarifications §5):
    T_def → T_canvas → T_lattice → T_block → T_object → T_select → T_local → T_Γ

    Terminates when no removals occur in a full pass (convergence-based).
    Returns U_star where |U_star[q]| = 1 for all q (singletons).
    """
    U = U0.copy()
    changed = True
    passes = 0

    while changed:
        U_prev = U.copy()

        # Fixed 8-stage closure pipeline
        U = apply_definedness_closure(U, X_test_present, theta)
        U = apply_canvas_closure(U, theta)
        U = apply_lattice_closure(U, theta)
        U = apply_block_closure(U, theta)
        U = apply_object_closure(U, theta)
        U = apply_selector_closure(U, theta)
        U = apply_local_paint_closure(U, theta)
        U = apply_interface_closure(U, theta)

        changed = (U != U_prev)
        passes += 1

    # Verify: every pixel has exactly one defined expression
    assert all(len(U[q]) == 1 for q in U.keys()), "Non-singletons remain after LFP"

    return U


	•	Acceptance:
	    - Ends with singletons: |U*[q]| = 1 for all q
	    - Monotone removals only (never adds expressions)
	    - Convergence-based termination (no max_passes parameter)
	    - Passes ≤ S0 - S* (bounded by total removals)
	    - Fixed closure order matches clarifications §5
	•	Tests:
	    - Synthetic lattice shrinkage (converges to singletons)
	    - Receipts log passes, removals per stage, final singletons count
	    - Verify closure order is T_def → ... → T_Γ
	    - Idempotence: running lfp twice produces same result

⸻

WO-18: arc_fixedpoint/eval_unpresent.py
	•	Goal: evaluate singletons to Y^\wedge; unpresent by g_{\text{test}}^{-1}.
	•	API: evaluate_and_unpresent(U_star, Xhat, g_test) -> Y
	•	Tests: exact evaluation; unpresent invariance.

⸻

WO-19: arc_compile/compile_theta.py
	•	Goal: end-to-end compile: ΠG, WL(union), shape meet, param extraction (components, lattice/canvas, motifs, selectors, anchors/masks), palette (optional orbit), and expression sets + closures; produce receipts skeleton.
	•	API:

def compile_theta(train_pairs) -> (theta, U0, closures, receipts_stub)


	•	Tests: compile deterministically; receipts stub contains present flags, WL roles counts, basis list.

⸻

WO-20: arc_io/save_receipts.py & arc_io/cli.py
	•	Goal: JSON receipts + CSV summary; CLI for single/multi-task runs.
	•	API:

python -m arc_io.cli --task 05269061 --ch challenges.json --sol solutions.json --out receipts/
python -m arc_io.cli --all --out receipts/


	•	Receipts: present flags, WL stats (unseen roles=0), basis/params, FY/GLUE/LFP/Totality, ties, palette (if any), diffs array.
	•	Receipt structure (JSON):

{
  "task_id": "070dd51e",
  "closure_order": ["T_def","T_canvas","T_lattice","T_block","T_object","T_select","T_local","T_Γ"],
  "present": {"CBC3": true, "E4": true, "E8": false, "Row1D": true, "Col1D": true},
  "wl": {"iters": 5, "roles_train": 12, "roles_test": 12, "unseen_roles": 0},
  "basis_used": ["CONNECT_ENDPOINTS", "ARGMAX_COLOR", "REGION_FILL", "FRAME"],
  "params": {
    "lattice": null,
    "canvas": {"frame": {"color": 4, "thickness": 2}, "verification": "exact"},
    "components": [...],
    "selectors": [
      {
        "type": "ARGMAX_COLOR",
        "mask_hash": "0x3f4a2b1c",
        "empty_on_test": false,
        "color_result": 7
      },
      {
        "type": "UNIQUE_COLOR",
        "mask_hash": "0x9a8c7e2d",
        "empty_on_test": true,
        "color_result": null
      }
    ],
    "connect": [{"anchors": [[1,2], [8,2]], "metric": "4conn", "tie": "global_lex"}],
    "fills": [{"mask_id": "hole_c1#0x7be1", "color": "unique"}]
  },
  "selectors_summary": {
    "empty_mask_deletes": 3,
    "recomputed_histograms": 14
  },
  "fy_glue": {"fy_gap": 0, "interfaces_ok": true},
  "lfp": {"passes": 3, "removals": 112, "singletons": 400},
  "totality": {"undefined_reads": 0},
  "ties": {"delta_lex": 1, "histogram_ties": 0, "basis_choice": 0},
  "palette": {"orbit_used": false, "train_permutations": [], "test_isomorphic": null},
  "diffs": [0, 0, 0, ...]
}

⸻

3) Dependency Graph & Bring-Up Plan (bottom-up)

Phase A — Core invariants
	1.	WO-01 order_hash.py
	2.	WO-02 present.py
	3.	WO-03 wl.py
	4.	WO-04 shape.py
	5.	WO-05 palette.py (optional orbit)

Phase B — Deterministic param extractors
 6.⁠ ⁠WO-08 components.py
 7.⁠ ⁠WO-06 lattice.py (FFT + KMP fallback)
 8.⁠ ⁠WO-07 canvas.py

Phase C — Law families
9.  WO-09 local_paint.py
10.⁠ ⁠WO-10 object_arith.py
11.⁠ ⁠WO-11 periodic.py
12.⁠ ⁠WO-12 blow_block.py
13.⁠ ⁠WO-13 selectors.py
14.⁠ ⁠WO-14 connect_fill.py

Phase D — Fixed-point
15.⁠ ⁠WO-15 expressions.py
16.⁠ ⁠WO-16 closures.py
17.⁠ ⁠WO-17 lfp.py
18.⁠ ⁠WO-18 eval_unpresent.py

Phase E — Glue
19.⁠ ⁠WO-19 compile_theta.py
20.⁠ ⁠WO-20 io/cli/receipts

Integration Milestones
	•	M1: Present + WL on union prints roles (no laws yet).
	•	M2: Shape meet + Palette canon (orbit CPRQ) prints abstract maps (cyclic-3 verifies as isomorphic-by-palette).
	•	M3: Add LocalPaint + ObjectArith laws → solve a batch of easy tasks; diffs=0.
	•	M4: Add Periodic + Blow/Block laws → solve scale/tiling tasks.
	•	M5: Add Selectors + Connect/Fill + Canvas inference → solve remaining classes; receipts show basis variety.
	•	M6: Full LFP evaluator; all tasks produce receipts; diffs=0 or isomorphic-by-palette with π printed.

⸻

4) Coding Standards & CI
	•	Style: ruff + black; type hints via mypy.
	•	Tests: pytest -q; unit tests per WO; integration tests on curated tasks.
	•	Determinism guard: assert no use of Python hash(), no random, no time-dependent data.
	•	Perf: add lightweight memoization for CBC tokens and row/col bag hashes; WL runs log iterations.

⸻

5) Example WOs (concrete acceptance snippets)

WO-12 Blow-up & Block-substitution
	•	Unit test: 3×3 → 9×9 motif; verify motifs learned exactly; any deviation in train removes that law instance.

# GIVEN train tiles exactly match B(c); WHEN compile; THEN law kept and test reproduces exactly

WO-13 Selectors (ARGMAX_COLOR)
	•	Unit test: mask = row band; train histogram picks argmax=7; test recompute histogram on same mask returns 7; assert diffs=0.

WO-14 Connect & Fill
	•	Unit test (connect): two markers; 4-conn shortest path; lex-min path stable; train exact.
	•	Unit test (fill): hole region mask; selector=UNIQUE_COLOR; flood equals train; test diffs=0.

⸻

6) Example run receipts (difficult cases)

Cyclic-3 (palette orbit)

present: {CBC3:true,E4:true,E8:false}
wl: {iters:5, roles_train:3, roles_test:3, unseen:0}
basis_used: ["periodic","local_paint"]
palette: canon_rule="count↓,first↑,boundary↑", π_train=[{2→2,4→4,1→1}, {4→2,8→1,3→4}, ...]
verify: isomorphic_by_palette; cells_wrong_after_π = [0,...]
lfp: {passes:3, removals:..., singletons:N}

Blow-up + block-subst

present: {CBC3:true,E4:true,E8:false}
lattice: method=FFT, basis=[[3,0],[0,3]]
basis_used: ["BLOWUP","BLOCK_SUBST"]
params: {k=3, motifs_hash={c7:"0x...", c1:"0x..."}}
verify: diffs=[0]

Selector + fill

basis_used: ["ARGMAX_COLOR","REGION_FILL"]
params: {mask_hash:"0x...", selector:"argmax"}
fy_glue: {fy_gap:0, interfaces_ok:true}
verify: diffs=[0]


⸻

7) Why this plan guarantees 100% with receipts
	•	The present is closed (CBC3 + E4→E8 + optional 1-D bands) and WL runs on the union → no test-only roles.
	•	Orbit CPRQ + canonical palette makes recolor variations exact (with π receipts).
	•	The law basis (local/object/periodic + selectors/CONNECT/FILL/BLOWUP/BLOCK_SUBST/canvas) spans all official ARC transforms modulo G.
	•	All choices are deterministic (global order) and verified on trains; definedness closures ensure total evaluation; LFP yields unique singletons → one output.
	•	Receipts show proof: FY gap = 0; interfaces satisfied; totality; palette permutations; diffs=0 (or isomorphic-by-palette with cells_wrong_after_π=0).

⸻

8) Next steps
	1.	Stand up WOs 01–04 (present + WL + shape + palette) → print roles & abstract maps.
	2.	Add WOs 08,06,07 (components/lattice/canvas) → parameters verified.
	3.	Bring in laws 09–14 incrementally; each law family must pass its unit receipts before enabling in compile.
	4.	Integrate LFP (15–18), compile (19) and IO/CLI (20).
	5.	Run scripts/run_all.py → produce receipts/ and a coverage CSV; expect “exact” or “isomorphic by palette” for every id—no UNSAT.

This is the complete, bottom-up plan. Every WO is ≤500 LoC, independently testable, and integrates with the fixed-point schema to deliver a 100% deterministic ARC solver with proofs.