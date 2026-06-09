---
title: Streamlit Inference Wiring
slug: streamlit-inference-wiring
type: learning
status: live
created: 2026-06-09
updated: 2026-06-09
owner: Wei Alexander Xin
scope: project
project: Big_Data_Analytics_Midterm_Project
tags: [streamlit, inference, eva-x, llm, testing]
canonical: true
sources:
  - docs/retrospectives/2026-06-09_streamlit-ui-delivery-closeout.md
related:
  - docs/knowledge-base/learnings/2026-06-09_vendored-model-edit-discipline.md
---

# Streamlit Inference Wiring

## Summary

The notebook-to-app migration worked once the inference contract was treated as
one end-to-end path: reconstruct the EVA-X architecture without external MIM
weights, load the binary checkpoint once, keep threshold as runtime UI state,
persist classifier plus reasoning context into the downloadable payload, and
verify the non-Colab path with a CPU smoke test.

## Key Learnings

- **Checkpoint-first inference is the right boundary here.**
  `create_eva_x_tiny()` is enough to rebuild the architecture; local inference
  should not require the upstream pretrained MIM checkpoint.
- **Do not put slider-controlled threshold inside cached bundle state.**
  Threshold belongs in the inference call so the model cache stays valid while
  UI state changes.
- **Ask Agent needs the full report context, not just a label.**
  Payloads should carry `prediction`, `p_abnormal`, `threshold`, reasoning,
  reasoning model, source filename, checkpoint, and device metadata.
- **The smoke test should cover the non-happy path that is most likely to break.**
  In this repo that meant CPU checkpoint restore, synthetic inference, payload
  schema checks, and missing-key behavior for the optional OpenAI path.

## Evidence

- `git show --stat 23bc745`
- `git show --stat 77f6f40`
- `docs/retrospectives/2026-06-09_streamlit-ui-delivery-closeout.md`
