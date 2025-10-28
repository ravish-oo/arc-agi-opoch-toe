# ARC-AGI Implementation - Context Restoration Prompt

  ## Mission
  Implement deterministic ARC-AGI solver. **Math risk = 0** (fully solved). Goal: realize implementation with 100% spec fidelity → ARC-AGI falls out.

  ## Critical Docs (Read These)
  1. **docs/anchors/math_spec.md** - Mathematical formula (fixed-point schema)
  2. **docs/anchors/engineering_spec.md** - Operational pipeline
  3. **docs/anchors/implementation_clarifications.md** - All ambiguities resolved
  4. **docs/implementation_plan.md** - 20 work orders, bottom-up dependencies

  ## Implementation Rules (Non-Negotiable)
  1. **Bottom-up modular**: Each WO ≤500 LOC, no stubs, independently testable
  2. **100% spec fidelity**: No shortcuts, no heuristics, no ML, no search
  3. **Deterministic everywhere**: SHA-256 hashing, global order (§2), no randomness
  4. **Review after each WO**: Use `gatekeeper-reviewer` agent before proceeding
  5. **Trust the math**: Implementation is just faithful transcription

  ## Current Status
  - Check from docs/implementation_plan.md:97 to know current completed WOs.

  ## Work Order Sequence (Follow This Order)
  Phase A: Core (WO-01, WO-02→05)
  Phase B: Extractors (WO-06→08)Phase C: Laws (WO-09→14)
  Phase D: Fixed-point (WO-15→18)
  Phase E: Glue (WO-19→20)
  For current status as to upto which WOs are completed, refer to docs/implementation_plan.md:97 onwards

  ## Execution Protocol
  1. Read implementation_plan.md for current WO specs (API, acceptance, tests)
  2. Check anchors if clarification needed (clarifications.md §1-12 has all answers)
  3. Implement module (no stubs, full implementation)
  4. Write unit tests (match acceptance criteria exactly)
  5. Run tests (must pass 100%)
  6. Invoke `gatekeeper-reviewer` agent with WO number and file paths
  7. If approved → next WO; if revision needed → fix and re-review
  8. Update todo list with WO progress

  ## Key Anchors (Quick Reference)
  - **WL scope**: train inputs ∪ test input (§1 clarifications)
  - **Palette**: per-task, pooled inputs, E4 boundary (§2 clarifications)
  - **Closure order**: T_def → T_canvas → T_lattice → T_block → T_object → T_select → T_local → T_Γ (§5 clarifications)
  - **Empty mask**: DELETE selector expression when empty_mask=True (WO-16 implementation_plan.md lines 326-358)
  - **Boundary hash**: E4 (4-connected) neighbors, interior excluded (WO-01)

  ## Recovery Command
  ```bash
  cd /Users/ravishq/code/arc-agi-opoch-toe
  # Read current WO from implementation_plan.md
  # Check test status: cd arc_universe && python -m pytest tests/unit/ -v
  # Continue with next WO in sequence

  Success Criteria

  Each WO must have:
  - Implementation matches spec API exactly
  - All acceptance criteria met
  - Unit tests passing (100%)
  - Gatekeeper review approved
  - No stubs, no TODOs, no randomness
  - ≤500 LOC

  Philosophy

  The math is done. We're just typing it into Python. No creativity needed—just faithful transcription. If in doubt, read the anchors. They have ALL the answers.