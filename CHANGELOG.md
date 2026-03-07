# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Added a `bda_chest` inference package with EVA-X model loading, binary reporting, OpenAI vision/Q&A helpers, and shared utilities.
- Added a Streamlit app with Inference, Model Info, and Ask Agent pages for the chest X-ray workflow.
- Added `scripts/smoke_test.py` for CPU checkpoint restore, inference, payload, import, and LLM-key validation checks.
- Added a canonical `requirements.txt` for the Streamlit application environment.
- Added a tracked `.gitignore` for prompt artifacts and local Python caches.

### Changed
- Added `create_eva_x_tiny()` to `eva_x.py` so inference can reconstruct EVA-X without the external MIM checkpoint.
- Updated `README.md` with Streamlit setup instructions and research-only disclaimers.
- Avoided double checkpoint deserialization when building the cached inference bundle.
- Expanded `.gitignore` with the existing local-only paths and log patterns used in this repo.
## [Unreleased]

### Added
- Added a `src.cxr_pipeline` package with reusable model, data, training, inference, evaluation, and diagnosis modules extracted from the notebooks.
- Added `python -m src.train` and `python -m src.diagnose` CLI entry points for training and diagnosis workflows.
- Added core and backend-specific requirements files for reproducible local setup.

### Changed
- Updated `README.md` with the new package layout, install options, and CLI usage.
- Copied `eva_x.py` into `src/cxr_pipeline/` while keeping the root module for notebook compatibility.
- Fixed checkpoint loading to explicitly allow full PyTorch checkpoint metadata on PyTorch 2.6+.
- Simplified training resume to restore model and optimizer state from a single checkpoint load.
- Kept `src/cxr_pipeline/eva_x.py` in sync as a verbatim copy of the root vendored module.
- Updated AMP usage to the current `torch.amp` API to avoid deprecation warnings during training.
- Added a repo `.gitignore` for local prompts, caches, logs, and Codex/Claude workspace artifacts.
