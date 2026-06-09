---
title: Open Questions
slug: open-questions
type: qa
status: live
created: 2026-06-09
updated: 2026-06-09
owner: Alex Xin
scope: project
project: Big_Data_Analytics_Midterm_Project
tags: [qa, github, ci, review]
canonical: true
sources:
  - /Users/wax/coding/Big_Data_Analytics_Midterm_Project/docs/retrospectives/2026-06-09_big-refactor-consolidation-closeout.md
related:
  - /Users/wax/coding/Big_Data_Analytics_Midterm_Project/docs/knowledge-base/learnings/2026-06-09_refactor-review-guardrails.md
---

# Open Questions

### What minimum automated checks should gate future refactor PRs in this repo?

This remains open. PR #5 merged with `statusCheckRollup: []`, so current merge confidence depends on manual review and ad hoc local verification.

The next pass should define a small required baseline. At minimum, this repo likely wants:

- package import validation
- CLI smoke coverage
- one notebook/package consistency guard, or an explicitly documented alternative

**Source:** `docs/retrospectives/2026-06-09_big-refactor-consolidation-closeout.md`
