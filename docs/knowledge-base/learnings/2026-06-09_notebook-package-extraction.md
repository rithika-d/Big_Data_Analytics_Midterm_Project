---
title: Notebook Package Extraction
slug: notebook-package-extraction
type: learning
status: live
created: 2026-06-09
updated: 2026-06-09
owner: Wei Alexander Xin
scope: project
project: Big_Data_Analytics_Midterm_Project
tags: [notebooks, cli, lazy-imports, checkpoints, refactor]
canonical: true
sources:
  - docs/retrospectives/2026-06-09_notebook-refactor-runnable-scripts-closeout.md
related:
  - docs/knowledge-base/decisions/foundation.md
  - docs/knowledge-base/learnings/2026-06-09_vendored-model-edit-discipline.md
---

# Notebook Package Extraction

## Summary

The notebook-to-package refactor succeeded because the session treated package
boundaries, dependency loading, and checkpoint semantics as core design
decisions instead of cleanup details.

## Key Learnings

- **CPU-safe imports are part of the contract.**
  `--help`, module imports, and plan-level verification should work without
  GPU-only or heavyweight LLM dependencies installed.
- **Split backend dependencies deliberately.**
  Separate requirements surfaces made it possible to keep the core classifier
  path usable while Llama and CheXagent stayed optional.
- **Do not overload restore helpers.**
  Fresh model construction, inference restore, and training resume should have
  distinct code paths so verification and future refactors stay predictable.
- **Treat checkpoint metadata compatibility as real functionality.**
  PyTorch defaults changed under this repo’s historical checkpoints, and the
  loader needed to opt into full metadata restore explicitly.

## Evidence

- `git show --stat 76a6766`
- `git show --stat 27156ef`
- `git show --stat fe383bc`
- `docs/retrospectives/2026-06-09_notebook-refactor-runnable-scripts-closeout.md`
