# Update Version Working Notes

1. Read `pyproject.toml` and extract the current `version` field (e.g., `1.5.0`).
2. Open `doc/v{version}_changes.md` (e.g., `doc/v1.5.0.md`). If it doesn't exist, create it with the full structure below. If it exists, update it — append to existing sections, never overwrite previous session log entries.
3. Review all changes made in this session — files modified, functions added/removed/rewritten, design decisions discussed — and write them into the document.

## Document Structure

Use this exact layout:


## What to capture

- **Why**, not just what — record rejected alternatives and the reasoning behind every decision
- **Traced reasoning** — when a formula, invariant, or architectural choice is derived, show the walkthrough (steady-state trace, counterexample that breaks a naive version)
- **Honest boundaries** — what this phase does NOT fix; assumptions that could break; what the next phase must handle
- **Debug/flag state** — what's toggled on/off in the working tree right now and why
- **Corrections** — when a design sketched in a previous session turns out wrong, record the original, the failure mode, and the fix. Do NOT silently overwrite the old text — keep both
- **Measured data** — timings, throughput numbers, failure rates. Always note the conditions (dataset size, hardware, config flags)
- **Diff from previous version** — reference what changed relative to v{previous_version} where it helps explain the motivation

## Rules

- Every item is in exactly one state: designed / implemented / verified / shipped / rejected
- Never delete previous session log entries — they are the narrative record
- When something is "implemented but not verified," say so explicitly
- The session log entry for today should be the LAST thing you write, summarizing everything from this session
- If `doc/v{version}.md` already has content, read it fully before updating — merge, don't duplicate
- If this is the first session for this version, look at the previous version's doc in `doc/` for context on what's changing and why