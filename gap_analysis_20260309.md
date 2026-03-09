# Gap Analysis: Notebooks vs Streamlit UI — LLM Functionality

**Date:** 2026-03-09

## Feature Comparison

| Feature | `Midterm2.ipynb` | `Midterm_Project.ipynb` | `Evaluation.ipynb` | Streamlit UI |
|---|---|---|---|---|
| **LLM for reasoning** | Llama-3.2-11B-Vision-Radiology-mini (via unsloth) | CheXagent-2-3b | OpenAI gpt-4.1 | Llama (default) or OpenAI (toggle) |
| **Confidence-tiered prompts** | Yes (3 tiers in final cell) | No (single generic prompt) | No | Yes (3 tiers, matching Midterm2) |
| **Q&A / chat** | No | No | Yes (via `answer_question_about_report`, OpenAI) | Yes (Llama or OpenAI) |
| **Evaluation/judge** | No | No | Yes (MedGemma judge) | No |
| **Image passed to LLM** | Yes (vision model) | Yes (CheXagent vision) | No (text-only QA) | Yes for reasoning, text-only for QA |

## Key Gaps

1. **`Midterm_Project.ipynb` has no tiered prompting** — it uses a single generic prompt regardless of confidence level, unlike `Midterm2.ipynb` and the Streamlit UI which both use borderline/moderate/high tiers.

2. **CheXagent is broken in `Midterm_Project.ipynb`** — cell 42 crashes with `ValueError: Passing along a device_map requires low_cpu_mem_usage=True`, and cell 46's output is just the prompt echoed back (empty generation). CheXagent never actually produces useful output in this notebook.

3. **Neither notebook has Q&A/chat** — that's Streamlit-only (and `Evaluation.ipynb` for OpenAI).

4. **`Evaluation.ipynb` uses OpenAI for QA, not Llama** — it imports `answer_question_about_report` which hits the OpenAI API, not the Llama model.

5. **`Midterm2.ipynb` decodes full output including input tokens** — `tokenizer.decode(out[0], skip_special_tokens=True)` includes the user prompt in the output (visible in cell 13/15 output). The Streamlit UI's Llama backend slices to only new tokens, which is cleaner.

## Summary

`Midterm2.ipynb` is the closest to the Streamlit UI — same model (Llama), same tiered prompting. The main differences are the input-token decoding issue and no chat capability. `Midterm_Project.ipynb`'s CheXagent integration is essentially non-functional.
