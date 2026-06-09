---
title: Vendored Model Edit Discipline
slug: vendored-model-edit-discipline
type: learning
status: live
created: 2026-06-09
updated: 2026-06-09
owner: Wei Alexander Xin
scope: project
project: Big_Data_Analytics_Midterm_Project
tags: [vendored-code, eva-x, review, formatting]
canonical: true
sources:
  - docs/retrospectives/2026-06-09_streamlit-ui-delivery-closeout.md
  - docs/retrospectives/2026-06-09_repo-hygiene-closeout.md
  - docs/retrospectives/2026-06-09_notebook-refactor-runnable-scripts-closeout.md
related:
  - docs/knowledge-base/learnings/2026-06-09_streamlit-inference-wiring.md
---

# Vendored Model Edit Discipline

## Summary

When a repo carries vendored upstream model code like `eva_x.py`, reviews stay
healthy only if local changes are surgical. Functional additions should be easy
to isolate from unchanged upstream code. Large formatting rewrites make PR
review, blame, and future rebases harder than they need to be.

## Key Learnings

- **Keep vendored diffs narrow.**
  Add the new helper or hook, but avoid broad style rewrites that touch every
  function in the file.
- **If formatter pressure conflicts with vendored readability, isolate it.**
  Localized formatter guards are preferable to obscuring the actual functional
  change across an entire upstream file.
- **Review feedback about diff shape is substantive.**
  The follow-up commit here improved maintainability without changing behavior
  by reducing churn in `eva_x.py`.

## Evidence

- `git show --stat 27156ef`
- `git show --stat 77f6f40`
- `docs/retrospectives/2026-06-09_notebook-refactor-runnable-scripts-closeout.md`
- `docs/retrospectives/2026-06-09_streamlit-ui-delivery-closeout.md`
- `docs/retrospectives/2026-06-09_repo-hygiene-closeout.md`
