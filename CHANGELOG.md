# Changelog

## [Unreleased]

### Added
- Added a `src.cxr_pipeline` package with reusable model, data, training, inference, evaluation, and diagnosis modules extracted from the notebooks.
- Added `python -m src.train` and `python -m src.diagnose` CLI entry points for training and diagnosis workflows.
- Added core and backend-specific requirements files for reproducible local setup.

### Changed
- Updated `README.md` with the new package layout, install options, and CLI usage.
- Copied `eva_x.py` into `src/cxr_pipeline/` while keeping the root module for notebook compatibility.
