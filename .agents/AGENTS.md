# AI Project Instructions

## 1. Core Philosophy & Operating Standard
**Core Philosophy:** Treat every function and line of code as a liability. Abstract, combine, and eliminate logic whenever possible.
**Goal:** Minimize architectural exceptions. Do not optimize for single prompts, fixtures, or local examples.

### 1.1 Before Editing
- **Analyze:** Read current implementation and nearby tests.
- **Scope:** Identify the smallest behavior boundaries.
- **Classify:** Categorize changes using the core boundaries defined in vocab.md.
- **Scan:** Look for duplicate, stale, compatibility-only, or additive code.
- **Plan:** State intended behavioral change in one sentence before substantial edits.

### 1.2 While Editing
- **Prioritize:** Deletion > Merge > Simplification > Reuse > Adding new branches.
- **Reuse:** Prefer existing patterns over new abstractions.
- **Scope:** Isolate changes to shared behavior under test.
- **Clean:** Do not leave old/new implementations running in parallel (unless requested).
- **Document Fallbacks:** Explain why they exist and the future replacement plan.
- **Comments:** Use hash (`#`) comments extensively. Describe methods fully. **NO DOCSTRINGS**.

### 1.3 Testing
- **Coverage:** Add/update behavior tests alongside changes.
- **Focus:** Assert observable behavior, not implementation trivia.
- **Execution:** Run the smallest focused test/harness proving the change. Escalate to broader tests only across subsystem boundaries.
- **Validation:** Do not report completion without running focused verification (unless explicitly blocked).

### 1.4 Reporting
- Explain architectural change.
- List changed files.
- Include exact commands run and results.
- Identify weak spots.
- Call out temporary scaffolding/hardcoding.

## 2. Architecture Bias
**Rule:** Prefer reusable mechanisms over prompt-specific handlers.
**Preferred Abstractions:** `data`, `examples`, `graph structure`, `learned relations`, `traversal pressure`, `reusable scoring`, `candidate memory`, `proximity`, `history`.

### Strictly Avoid:
- Prompt substring branches.
- Symbolic exceptions for single examples.
- Domain-specific task handlers disguised as general logic.
- Duplicated scoring/traversal systems.
- Hidden metadata bypassing observable evidence.
- Unjustified compatibility surfaces.

*Note: Scaffolding must be small, named, and isolated. Replace in the same pass if feasible.*

## 3. Configuration Sources of Truth
- **Architecture Specification (CANONICAL):** `.agents/field_gather_spec.md` — the north star for the architecture. When making a non-trivial deviation, always consult me first.
- **Active build/test plan:** `.agents/field_gather_build_plan.md`.
- **SUPERSEDED (do NOT follow — kept for history only):** `.agents/architecture_spec.md`, `.agents/build_roadmap.md`, `.agents/implementation_spec.md` describe the dissipative-Hamiltonian *itinerancy* model that was falsified at gate G3 (see `.agents/G3_escalation.md`). They are replaced by the two files above.
- **Vocabulary/Terms:** `.agents/vocab.md` (If code and docs disagree, inspect code to determine which is stale).
- **Formatting/Style:** `.agents/formatting.md` (Overrides local habits unless a strong file pattern exists).
- **Execution Skills:** `.agents/SKILLS/` (Before performing strategic assessments, schema design, code reviews, or complex workflows, list the contents of this directory and apply the relevant skill document).

## 4. Verification Discovery
**Rule:** Prefer documented repo commands over global defaults.

### Check locations:
- `pytest.ini`, `tests/`, `scripts/`, `.agents/vocab.md`, `.agents/formatting.md`, `AGENTS.md`

### Common Focused Commands:
```bash
python -m pytest -q
python tests/behavior/one_shot_harness.py
```
*(Use `venv/bin/python` if local virtual environment exists).*
