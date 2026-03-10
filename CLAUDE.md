# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EECS E6895 Big Data Analytics midterm project: a two-stage chest X-ray diagnostic pipeline.

1. **Stage 1 — Binary classifier**: EVA-X Tiny Vision Transformer fine-tuned for normal vs. pneumonia classification on chest X-rays.
2. **Stage 2 — Reasoning LLM**: When the classifier predicts abnormal, a vision-language model generates radiologic findings. Two LLM backends are used across notebooks:
   - **Llama-3.2-11B-Vision-Radiology-mini** (via `unsloth`, in `Big_Data_Analytics_Midterm2.ipynb`)
   - **CheXagent-2-3b** (via `transformers`, in `Big_Data_Analytics_Midterm_Project.ipynb`)

## Runtime Environment

Both notebooks are designed for **Google Colab with GPU** (tested on A100). They mount Google Drive for dataset and checkpoint access. Paths like `/content/drive/MyDrive/chest_xray/` are Colab-specific.

## Key Files

| File | Purpose |
|---|---|
| `eva_x.py` | EVA-X model definitions (tiny/small/base). Subclasses `timm.models.eva.Eva`. Includes checkpoint weight-key remapping in `checkpoint_filter_fn`. |
| `src/bda_chest/` | Canonical Python package — models, pipeline, LLM backends, training, metrics, evaluation, reporting, utils. |
| `src/train.py` | CLI entry point for classifier training. |
| `src/diagnose.py` | CLI entry point for classifier + LLM diagnosis (lazy LLM loading). |
| `app/streamlit_app.py` | Streamlit UI (inference, chat, evaluation). |
| `Big_Data_Analytics_Midterm_Project.ipynb` | Training notebook: data loading, `Eva_X_Model` wrapper, `Trainer` class, training loop with early stopping, evaluation metrics, CheXagent integration. |
| `Big_Data_Analytics_Midterm2.ipynb` | Inference notebook: loads trained checkpoint, `diagnose_chest_xray()` pipeline with confidence-tiered prompting, Llama radiology model. |
| `eva_x_tiny_binary_best.pt` | Trained checkpoint (epoch 12, best_val_loss=0.0313). Contains `model_state_dict`, `optimizer_state_dict`, `class_to_idx`. |
| `EVA-X_requirements .txt` | Dependencies for the EVA-X component (note: filename has a trailing space before `.txt`). |

## Architecture Details

### Eva_X_Model (wrapper)
- Loads `eva_x_tiny_patch16` with pretrained MIM weights (`eva_x_tiny_patch16_merged520k_mim.pt`, downloaded from HuggingFace)
- Replaces head: `nn.Linear(192, 1)` for binary classification
- Freezes entire backbone, then unfreezes: `head`, `blocks.11` (last transformer block), `norm`, `fc_norm`
- Input: `(B, 3, 224, 224)` → Output: `(B, 1)` logits
- Training uses `BCEWithLogitsLoss` with `pos_weight=0.70` for class imbalance (1341 normal / 1912 pneumonia)

### Image Preprocessing
- Training: Resize(256) → RandomCrop(224) → RandomHorizontalFlip → ImageNet normalization
- Inference: Resize(256) → CenterCrop(224) → ImageNet normalization
- ImageNet stats: `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`

### diagnose_chest_xray() Pipeline
Confidence-tiered prompting based on `p_abnormal`:
- `<= 0.5`: Normal — LLM not invoked
- `0.5–0.7`: Borderline — emphasize uncertainty, subtle findings
- `0.7–0.8`: Moderate — standard differential diagnosis
- `> 0.8`: High confidence — dominant patterns, likely explanations

## External Data & Models

- **Dataset**: [Kaggle Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data) — 3253 train, 16 val, 624 test
- **EVA-X pretrained weights**: Download `eva_x_tiny_patch16_merged520k_mim.pt` from [HuggingFace MapleF/eva_x](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_tiny_patch16_merged520k_mim.pt)
- **Llama radiology model**: `0llheaven/Llama-3.2-11B-Vision-Radiology-mini` (HuggingFace)
- **CheXagent**: `StanfordAIMI/CheXagent-2-3b` and `StanfordAIMI/CheXagent-2-3b-srrg-findings` (HuggingFace)

## Dependencies

Core: `torch`, `torchvision`, `timm>=0.9.0`, `transformers`, `Pillow`, `scikit-learn`, `numpy`, `pandas`

For Llama radiology model: `unsloth`, `accelerate`

For CheXagent: `transformers` with `trust_remote_code=True`

The `timm` version matters — `eva_x.py` imports from `timm.models.eva` and `timm.layers`, which require `timm>=0.9.0`.
