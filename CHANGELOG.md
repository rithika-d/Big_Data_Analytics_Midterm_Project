# Changelog

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
