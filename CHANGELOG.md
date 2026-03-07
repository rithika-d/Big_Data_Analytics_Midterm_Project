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
