# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Added Llama-3.2-11B-Vision-Radiology-mini as the default LLM provider in the Streamlit UI, with a dropdown to toggle to OpenAI.
- Added MedGemma evaluation judge (`src/bda_chest/evaluation.py`) to score LLM reasoning on a 1–5 correctness scale, adapted from the Evaluation notebook.
- Added MedGemma evaluation toggle in the Streamlit sidebar.
- Added `.env` placeholder for `OPENAI_API_KEY` (gitignored).
- Added gap analysis comparing LLM functionality across notebooks and Streamlit UI.
- Added a `bda_chest` inference package with EVA-X model loading, binary reporting, LLM helpers, and shared utilities.
- Added a `src.cxr_pipeline` package with reusable model, data, training, inference, evaluation, and diagnosis modules extracted from the notebooks.
- Added a Streamlit app with Inference, Model Info, and Ask Agent pages for the chest X-ray workflow.
- Added `python -m src.train` and `python -m src.diagnose` CLI entry points for training and diagnosis workflows.
- Added `scripts/smoke_test.py` for CPU checkpoint restore, inference, payload, import, and LLM-key validation checks.
- Added canonical `requirements.txt` and backend-specific requirements files.
- Added a tracked `.gitignore` for prompt artifacts, local caches, and workspace directories.

### Changed
- Made OpenAI import lazy in `llm.py` so the module loads without the `openai` package installed (required for Llama-only environments).
- Added `create_eva_x_tiny()` to `eva_x.py` so inference can reconstruct EVA-X without the external MIM checkpoint.
- Avoided double checkpoint deserialization when building the cached inference bundle.
- Copied `eva_x.py` into `src/cxr_pipeline/` while keeping the root module for notebook compatibility.
- Fixed checkpoint loading to explicitly allow full PyTorch checkpoint metadata on PyTorch 2.6+.
- Updated AMP usage to the current `torch.amp` API to avoid deprecation warnings.
- Consolidated and updated `README.md` to document all three ways to run the project (Streamlit, CLI, notebooks).
