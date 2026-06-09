---
title: Foundation Decisions
slug: foundation
type: decision
status: live
created: 2026-06-09
updated: 2026-06-09
owner: Wei Alexander Xin
scope: project
project: Big_Data_Analytics_Midterm_Project
tags: [architecture, notebooks, cli, compatibility, eva-x]
canonical: true
sources:
  - docs/retrospectives/2026-06-09_notebook-refactor-runnable-scripts-closeout.md
related:
  - docs/knowledge-base/learnings/2026-06-09_notebook-package-extraction.md
metadata:
  high_water_mark: 1
---

# Foundation Decisions

## [DEC-FND-001] Treat notebook logic as packaged CLI surfaces with compatibility shims

**Date:** 2026-03-07
**Status:** Approved
**Context:** The project’s training and diagnosis logic lived primarily in Colab notebooks with duplicated code paths, backend-specific differences, and Colab-only assumptions. The refactor session needed runnable scripts without breaking existing notebook workflows.
**Decision:** Extract shared logic into packaged Python modules with `python -m src.train` and `python -m src.diagnose` entry points, while retaining notebook compatibility shims such as the root `eva_x.py`.
**Consequences:**

- Core classifier and diagnosis logic gained a reusable module boundary instead of notebook-cell duplication.
- Heavy backend dependencies must remain lazy so core imports and CLI help work on CPU-only environments.
- Compatibility-sensitive vendored files should stay verbatim, with project-specific behavior layered around them.
- Checkpoint construction, inference restore, and training resume need separate interfaces.

**Alternatives considered:**

1. Keep training and diagnosis logic inside the notebooks. Rejected because verification, reuse, and automation remained fragile.
2. Add root-level scripts with `sys.path` manipulation. Rejected because import semantics would depend on invocation style and working directory.
3. Extract packaged modules plus CLI entry points while preserving notebook compatibility shims. Chosen because it gave a clean reusable surface without forcing an abrupt notebook break.

**Sources:** `76a6766`, `27156ef`, `fe383bc`, and `docs/retrospectives/2026-06-09_notebook-refactor-runnable-scripts-closeout.md`.
