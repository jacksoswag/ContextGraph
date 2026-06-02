# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent Guidance Sources

`.agents/` is the source of truth for this project. Read these before non-trivial work:

- `.agents/AGENTS.md` — operating standard (architecture-first, prefer deletion/reuse over new branches, classify changes by subsystem, name temporary scaffolding).
- `.agents/vocab.md` — current preferred names (e.g. "behavior", "evidence", "candidate", "scaffold") and deprecated terms.
- `.agents/formatting.md` — code style: combined imports on one line, hash comments only (NO docstrings), max 2 consecutive blank lines (never 3+), compact bracket layout, function headers on one line, prefer balanced ~70/70 splits over 100/40 when wrapping.

Treat every function and line of code as a liability. Abstract and combine logic whenever possible.