# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Added Streamlit delivery and repo-hygiene closeout records in `docs/retrospectives/`.
- Added project KB learnings for Streamlit inference wiring and vendored model edit discipline in `docs/knowledge-base/learnings/`.
- Added `pm/done.md` as a durable index for closeout records.
- Added a notebook-refactor closeout retrospective in `docs/retrospectives/2026-06-09_notebook-refactor-runnable-scripts-closeout.md`.
- Added project KB entries for notebook-package extraction and the historical notebook-to-CLI architecture decision under `docs/knowledge-base/`.
- Added the historical-session retrospective `docs/retrospectives/2026-06-09_big-refactor-consolidation-closeout.md`.
- Added project KB entries for refactor-review guardrails and the open CI question under `docs/knowledge-base/`.
- Added minimal PM tracking surfaces (`pm/backlog.md`, `pm/issues.md`, `pm/ideas.md`, `pm/done.md`) seeded from the closeout capture.
- Added Llama-3.2-11B-Vision-Radiology-mini as the default LLM provider in the Streamlit UI, with a dropdown to toggle to OpenAI.
- Added MedGemma evaluation judge (`src/rav/evaluation.py`) to score LLM reasoning on a 1–5 correctness scale, adapted from the Evaluation notebook.
- Added MedGemma evaluation toggle in the Streamlit sidebar.
- Added `.env` placeholder for `OPENAI_API_KEY` (gitignored).
- Added gap analysis comparing LLM functionality across notebooks and Streamlit UI.
- Added a `rav` package with EVA-X model loading, binary reporting, LLM helpers (Llama, CheXagent, OpenAI), training, evaluation metrics, QA evaluator, and shared utilities.
- Added a Streamlit app with Inference, Model Info, and Ask Agent pages for the chest X-ray workflow.
- Added `python -m src.train` and `python -m src.diagnose` CLI entry points for training and diagnosis workflows.
- Added `scripts/smoke_test.py` for CPU checkpoint restore, inference, payload, import, and LLM-key validation checks.
- Added canonical `requirements.txt` and backend-specific requirements files.
- Added a tracked `.gitignore` for prompt artifacts, local caches, and workspace directories.

### Changed
- Documented the repo's durable closeout and KB surfaces in `CLAUDE.md`.
- Renamed package from `bda_chest` to `rav` to match project name.
- Consolidated `cxr_pipeline` into `rav` — one canonical package for all consumers (Streamlit, CLI, evaluation scripts).
- Rewrote README with final paper results, demo screenshots, ROC curve, and team names.
- Added final paper PDF and demo video frame screenshots to `screens/`.
- Rewrote `src/diagnose.py` with lazy LLM loading — LLM backend only loaded when classifier predicts abnormal.
- Dropped `--pretrained-weights` from `src/diagnose.py` (inference checkpoint is self-contained).
- Renamed display name from "BDA" to "RAV" in the Streamlit UI.
- Made OpenAI import lazy in `llm.py` so the module loads without the `openai` package installed (required for Llama-only environments).
- Added `create_eva_x_tiny()` to `eva_x.py` so inference can reconstruct EVA-X without the external MIM checkpoint.
- Avoided double checkpoint deserialization when building the cached inference bundle.
- Fixed checkpoint loading to explicitly allow full PyTorch checkpoint metadata on PyTorch 2.6+.
- Updated AMP usage to the current `torch.amp` API to avoid deprecation warnings.
- Consolidated and updated `README.md` to document all three ways to run the project (Streamlit, CLI, notebooks).
