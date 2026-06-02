# Project Formatting & Style Guide

**Purpose:** Source of truth for code style, documentation shape, comments, naming, and LLM response formatting.

## 1. Code Style

- **Import Ordering:** Combine imports into one line where possible (`import os, time, hashlib`). Keep the import block compact without empty lines. Condense `try/except` for fallback imports onto single lines.
- **Type Hints:** Use modern Python type hints (`str | None` instead of `Optional[str]`). Include `from __future__ import annotations`.
- **Inline Statements:** Compress simple control flows to a single line (`if path == ":memory:": target = ":memory:"`). Put `if`, `else`, `with`, or `except` bodies on the same line if short.
- **Line Length & Spacing:** Max 2 consecutive blank lines (never 3+). Code must be vertically compact; max 1 blank line between related methods/functions.
- **Bracket & String Layout:** Use compact bracket layout. The closing parenthesis `)` must remain on the same line as the final argument. For multi-line strings (like SQL), place closing triple quotes `"""` directly at the end of the last line of text.
- **Wrapping & Parameters:** Keep parameters **horizontally dense**. **DO NOT** spread parameters onto individual lines (e.g., `method(\n  param1,\n  param2\n)` is strictly forbidden). Pack multiple parameters onto the same line. Prefer balanced splits (~70/70) over lopsided ones (100/40) when wrapping long lines.
- **Headers:** **ALWAYS** keep function headers on one line. Never wrap signatures.
- **Comments:** **HASH COMMENTS ONLY (`#`) FOR INTERNAL LOGIC.** No docstrings for private methods. Use concise, one-line docstrings exclusively for public API boundaries. Keep comments concise. Place function descriptions exactly one line above the header. Use inline comments directly next to relevant code (`try: # Load sqlite-vec`).
- **Naming:** Reference `.agents/vocab.md`.

## 2. LLM / Codex Response Rules

### 2.1 Progress Updates
- **Actionable:** Keep updates short and concrete.
- **Contextual:** State what is inspected, changed, or verified, and why it matters. Include important findings immediately.
- **Intent-Driven:** State intended behavioral change in one sentence before substantial edits.
- **No Fluff:** Avoid filler, praise, motivational language, and repetitive status phrasing.

### 2.2 Final Answers
- **Structure:** Lead with changes or findings. Mention changed files.
- **Proof:** Include exact verification commands and results.
- **Brevity:** Keep answers concise unless a detailed review is requested.
- **Reviews:** List findings by severity first, then assumptions/gaps.
- **Closing:** Do not end with generic follow-up offers.

### 2.3 File References
- **Links:** Use clickable absolute file links for local files.
- **Precision:** Include line numbers for specific implementations.
- **Efficiency:** Do not repeat the same path unnecessarily. Group related files.

### 2.4 Test Reporting
- **Format:** Report commands exactly as run.
- **Status:** Mark results as `passed`, `failed`, `skipped`, or `blocked`.
- **Failures:** Summarize the first relevant failure and the likely boundary affected.
- **Skips/Blocks:** Explain why and name the remaining risk.

### 2.5 Risk Reporting
- **Explicit Callouts:** Highlight unverified behavior, scaffolding, hardcoding, compatibility paths, and stale docs.
- **Clarity:** Distinguish assumptions from confirmed facts.
- **Classification:** Categorize risks using the core boundaries defined in vocab.md.
- **Tone:** Prefer direct language over reassurance.

## 3. Strict Prohibitions

**DO NOT:**
- [ ] Indulge in formatting churn unrelated to the task.
- [ ] Mix style-only rewrites with behavior changes.
- [ ] Introduce new naming patterns conflicting with `vocab.md`.
