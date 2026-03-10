# Big Data Analytics Midterm Project: Summary

Version: 0.2.1
Date: 2026-03-09

---

## Overview

**Course:** EECS E6895 Big Data Analytics — Midterm Project
**Goal:** Two-stage chest X-ray diagnostic pipeline for pneumonia detection.
**Runtime:** Google Colab with GPU (training/notebooks), local GPU or CPU (Streamlit UI, CLI).

Research prototype only. Not for clinical use.

The system operates in two stages:
1. **Stage 1 — Binary Classifier:** An EVA-X Tiny Vision Transformer, fine-tuned on the Kaggle Chest X-Ray Pneumonia dataset, classifies images as normal vs. abnormal.
2. **Stage 2 — Reasoning LLM:** When the classifier predicts abnormal, a vision-language model generates radiologic findings. Three LLM backends are supported:
   - **Llama-3.2-11B-Vision-Radiology-mini** (local GPU, default in Streamlit UI)
   - **CheXagent-2-3b** (local GPU, CLI and notebook)
   - **OpenAI API** (gpt-4.1, Streamlit UI)

An optional **MedGemma evaluation judge** scores LLM reasoning on a 1–5 correctness scale.

---

## Project Layout

```
app/streamlit_app.py              Streamlit UI (inference, chat, evaluation)
src/bda_chest/                    Core package for the Streamlit app
  llm.py                          LLM backends (Llama local + OpenAI API)
  evaluation.py                   MedGemma judge for scoring LLM responses
  models.py, pipeline.py, ...     EVA-X loading, inference, reporting
  version.py                      App version constant
src/cxr_pipeline/                 Package extracted from original notebooks
  model.py, trainer.py, data.py   Training-side model wrapper, trainer, dataset
  inference.py, diagnosis.py      Inference and diagnosis pipeline
  chexagent.py, llama_backend.py  LLM backend adapters
  evaluation.py, qa_evaluator.py  Evaluation helpers
  eva_x.py, transforms.py        EVA-X copy and image transforms
src/train.py                      CLI: classifier training
src/diagnose.py                   CLI: classifier + LLM diagnosis
scripts/
  smoke_test.py                   Integration smoke test (CPU, no LLM)
  download_test_images.py         Download Kaggle test images
  generate_test_json.py           Generate evaluation Q&A samples
  evaluate_radiology_assistant.py End-to-end evaluation script
eva_x.py                          Root EVA-X module (notebook compatibility)
Big_Data_Analytics_Midterm_Project.ipynb   Training notebook (Colab)
Big_Data_Analytics_Midterm2.ipynb          Inference notebook (Colab)
Radiology_Assistant_Evaluation.ipynb      Evaluation notebook (Colab)
```

---

## Model Architecture

### EVA-X Tiny (Stage 1 backbone)
- Patch size 16, embed dim 192, depth 12 blocks, 3 attention heads
- SwiGLU MLP, Rotary Position Embeddings (RoPE)
- Input: `(B, 3, 224, 224)` images
- Pretrained via Masked Image Modeling (MIM) on 520K medical images

### Eva_X_Model (fine-tuning wrapper)
- Loads `eva_x_tiny_patch16` with MIM pretrained weights
- Replaces head: `nn.Linear(192, 1)` for binary classification
- Freezes entire backbone, then selectively unfreezes:
  - `head` (classification layer)
  - `blocks.11` (last transformer block)
  - `norm`, `fc_norm` (layer normalization)
- ~454k trainable parameters out of ~5.7M total
- Output: `(B, 1)` logits, trained with `BCEWithLogitsLoss`

The root `eva_x.py` includes a `create_eva_x_tiny()` factory so inference can reconstruct the model without the external MIM checkpoint.

---

## Dataset

**Kaggle Chest X-Ray Pneumonia** (`paultimothymooney/chest-xray-pneumonia`):
- Train: 3,253 images (1,341 Normal / 1,912 Pneumonia)
- Val: 16 images (very small)
- Test: 624 images
- Binary labels: Normal (0) vs. Pneumonia (1)

---

## Image Preprocessing

| Stage | Pipeline |
|---|---|
| Training | `Resize(256)` → `RandomCrop(224)` → `RandomHorizontalFlip(0.5)` → ImageNet normalize |
| Inference | `Resize(256)` → `CenterCrop(224)` → ImageNet normalize |

ImageNet normalization: `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`

---

## Training Configuration

| Parameter | Value |
|---|---|
| Loss | `BCEWithLogitsLoss`, `pos_weight=0.70` (handles 1:1.4 class imbalance) |
| Optimizer | AdamW, `lr=1e-4`, `weight_decay=1e-4` |
| Scheduler | None |
| Epochs | 20 max |
| Early stopping | Patience=4, `min_delta=1e-3` |
| Precision | Mixed (AMP via `torch.amp`) |
| Batch size | 32 |

---

## Training Results

**Best checkpoint:** Epoch 12 with `val_loss = 0.0313`

| Image Type | p_abnormal | Prediction | Note |
|---|---|---|---|
| Pneumonia (radiopaedia) | 0.9988 | Abnormal | In-scope positive-style spot check |
| Normal (radiopaedia) | 0.0865 | Normal | In-scope negative-style spot check |
| Bronchitis | 0.9405 | Abnormal | Out-of-distribution example |
| TB | 0.9986 | Abnormal | Out-of-distribution example |
| Lung cancer | 0.949 | Abnormal | Out-of-distribution example |
| Normal (Getty) | 0.4394 | Normal | External normal spot check |

These are illustrative spot checks from the notebooks, not a labeled benchmark. They suggest strong separation on a handful of clearly normal vs. clearly abnormal external examples, but do not characterize the full score distribution.

---

## Evaluation Metrics

The `evaluate_full()` function computes:
- Accuracy, Sensitivity (Recall/Pneumonia), Specificity (Recall/Normal)
- AUROC (Area Under ROC Curve)
- Confusion Matrix (TN, FP, FN, TP)

---

## Two-Stage Inference Pipeline: `diagnose_chest_xray()`

**Confidence-tiered prompting** based on classifier probability `p_abnormal`:

| p_abnormal Range | Behavior |
|---|---|
| ≤ 0.5 | **Normal** — LLM not invoked, returns classification only |
| 0.5–0.7 | **Borderline** — LLM prompt emphasizes uncertainty, subtle findings, normal variants |
| 0.7–0.8 | **Moderate** — Standard findings + short differential diagnosis |
| > 0.8 | **High confidence** — Dominant patterns, 2–3 likely explanations |

**Design rationale:**
- Efficiency: LLM is expensive; only invoked when needed
- Safety: Avoids LLM hallucinations on obviously normal cases
- Interpretability: Classifier decision triggers graded LLM explanation

---

## Stage 2 LLM Backends

### Llama-3.2-11B-Vision-Radiology-mini (default in Streamlit UI)
- Model: `0llheaven/Llama-3.2-11B-Vision-Radiology-mini` via Unsloth
- 4-bit quantization, gradient checkpointing
- Generation: `max_new_tokens=128`, `temperature=0.6`, `top_p=0.9` (nucleus sampling)
- Image loading: resized to 448×448

### CheXagent-2-3b (notebook + CLI)
- Model: `StanfordAIMI/CheXagent-2-3b-srrg-findings`
- dtype: `bfloat16`, device: GPU 0
- Generation: `max_new_tokens=512`, `num_beams=1` (greedy decoding)

### OpenAI API (Streamlit UI)
- Model: `gpt-4.1`
- Requires `OPENAI_API_KEY` in `.env` or environment
- Text-based analysis using classifier output and prompt context

---

## Evaluation Pipeline

The evaluation pipeline assesses the Q&A capabilities of the radiology assistant:

1. **MedGemma Judge** (`google/medgemma-1.5-4b-it`): Scores LLM reasoning on a 1–5 correctness scale. Togglable in the Streamlit sidebar. Requires GPU and HuggingFace gated access.
2. **Quantitative metrics**: BLEU and ROUGE scores for response quality.
3. **End-to-end script**: `scripts/evaluate_radiology_assistant.py` runs the full pipeline with ground truth samples (`qa_test_samples.json`).

The `Radiology_Assistant_Evaluation.ipynb` notebook provides the Colab-native version of this pipeline.

---

## Streamlit Web UI

The recommended way to interact with the project. Three functional areas:

1. **Inference** — Upload a chest X-ray, run the EVA-X classifier, view probability and prediction. Optionally invoke the LLM for radiologic findings.
2. **Ask Agent** — Q&A chatbot grounded in the current inference report context. Ask follow-up questions about findings.
3. **MedGemma Evaluation** — Score the LLM's reasoning output on a 1–5 scale using the MedGemma judge.

**Sidebar settings:**
- **LLM Provider**: Toggle between Llama (local GPU) and OpenAI (API).
- **MedGemma evaluation**: Enable/disable the evaluation judge.
- LLM features are optional — the EVA-X classifier works without them.

---

## Running the Project

### Option 1: Streamlit UI

```bash
pip install -r requirements.txt        # core deps
pip install -r requirements-llama.txt   # optional: Llama backend
streamlit run app/streamlit_app.py
```

### Option 2: CLI Scripts

```bash
# Train the binary classifier
python -m src.train \
  --data-dir ./chest_xray \
  --pretrained-weights ./eva_x_tiny_patch16_merged520k_mim.pt \
  --checkpoint-dir ./checkpoints

# Run diagnosis (classifier + LLM reasoning)
python -m src.diagnose \
  --image ./test_image.jpeg \
  --checkpoint ./checkpoints/eva_x_tiny_binary_best.pt \
  --pretrained-weights ./eva_x_tiny_patch16_merged520k_mim.pt \
  --backend llama
```

### Option 3: Colab Notebooks

- `Big_Data_Analytics_Midterm_Project.ipynb` — Training + CheXagent inference
- `Big_Data_Analytics_Midterm2.ipynb` — Llama inference with confidence-tiered prompting
- `Radiology_Assistant_Evaluation.ipynb` — End-to-end evaluation with MedGemma judge

---

## Model Assets

| Asset | Source |
|---|---|
| EVA-X pretrained MIM weights | [MapleF/eva_x](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_tiny_patch16_merged520k_mim.pt) |
| Trained binary checkpoint | `eva_x_tiny_binary_best.pt` (included in repo) |
| Llama radiology model | [0llheaven/Llama-3.2-11B-Vision-Radiology-mini](https://huggingface.co/0llheaven/Llama-3.2-11B-Vision-Radiology-mini) |
| CheXagent findings model | [StanfordAIMI/CheXagent-2-3b-srrg-findings](https://huggingface.co/StanfordAIMI/CheXagent-2-3b-srrg-findings) |
| MedGemma evaluation judge | [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) |
| Image dataset | [Kaggle Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data) |

---

## Dependencies

Core (Streamlit app + CLI):
```bash
pip install -r requirements.txt
```

Llama backend (requires GPU):
```bash
pip install -r requirements-llama.txt
```

CheXagent backend (CLI only):
```bash
pip install -r requirements-chexagent.txt
```

Key packages: `torch`, `torchvision`, `timm>=0.9.0`, `transformers`, `streamlit`, `Pillow`, `scikit-learn`, `numpy`, `pandas`

---

## Environment Variables

| Variable | Required for |
|---|---|
| `OPENAI_API_KEY` | OpenAI LLM provider in Streamlit UI; evaluation pipeline |
| `HF_TOKEN` | Downloading gated models (MedGemma) |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Downloading dataset for evaluation |

---

## Known Issues

1. **LLM hallucinations on normal images**: Both CheXagent and Llama can fabricate pathology when given normal X-rays. The confidence-tiered prompting mitigates this by not invoking the LLM for clearly normal predictions (p ≤ 0.5).
2. **Tiny validation set**: Only 16 validation images creates high variance in early stopping decisions.
3. **Borderline prompting under-tested**: Very few test cases fall in the 0.5–0.7 range, so that tier's prompting logic has limited validation.

---

## Strengths

- Medical-domain pretrained backbone (EVA-X with MIM on 520K medical images)
- Multimodal LLM integration — vision-language models directly see the X-ray
- Creative confidence-tiered prompting design
- Multiple LLM backends (Llama, CheXagent, OpenAI) with easy switching
- MedGemma evaluation judge for automated reasoning assessment
- Streamlit web UI for interactive inference and Q&A
- CLI and notebook entry points for flexibility across environments
- Modular package structure (`bda_chest`, `cxr_pipeline`) with reusable components
- Simple to run — Streamlit app works on CPU for the classifier, GPU only needed for LLM
