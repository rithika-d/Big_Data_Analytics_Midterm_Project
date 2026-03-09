# Big Data Analytics Midterm Project

EECS E6893: Big Data Analytics midterm project.

Research prototype only. Not for clinical use.

## Overview

A two-stage chest X-ray diagnostic pipeline:

1. **Stage 1 — Binary classifier**: EVA-X Tiny Vision Transformer fine-tuned for normal vs. pneumonia classification.
2. **Stage 2 — Reasoning LLM**: When the classifier predicts abnormal, a vision-language model generates radiologic findings using confidence-tiered prompting.

EVA-X model code is derived and adapted from [hustvl/EVA-X](https://github.com/hustvl/EVA-X).

## Project Layout

```
app/streamlit_app.py              Streamlit UI (inference, chat, evaluation)
src/bda_chest/                    Core package for the Streamlit app
  llm.py                          LLM backends (Llama local + OpenAI API)
  evaluation.py                   MedGemma judge for scoring LLM responses
  models.py, pipeline.py, ...     EVA-X loading, inference, reporting
src/cxr_pipeline/                 Package extracted from original notebooks
src/train.py                      CLI: classifier training
src/diagnose.py                   CLI: classifier + LLM diagnosis
scripts/smoke_test.py             Integration smoke test (CPU, no LLM)
eva_x.py                          Root EVA-X module (notebook compatibility)
Big_Data_Analytics_Midterm_Project.ipynb   Training notebook (Colab)
Big_Data_Analytics_Midterm2.ipynb          Inference notebook (Colab)
Radiology_Assistant_Evaluation.ipynb      Evaluation notebook (Colab)
```

## Model Assets

| Asset | Source |
|---|---|
| EVA-X pretrained MIM weights | [MapleF/eva_x](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_tiny_patch16_merged520k_mim.pt) |
| Trained binary checkpoint | `eva_x_tiny_binary_best.pt` (included in repo) |
| Llama radiology model | [0llheaven/Llama-3.2-11B-Vision-Radiology-mini](https://huggingface.co/0llheaven/Llama-3.2-11B-Vision-Radiology-mini) |
| CheXagent findings model | [StanfordAIMI/CheXagent-2-3b-srrg-findings](https://huggingface.co/StanfordAIMI/CheXagent-2-3b-srrg-findings) |
| MedGemma evaluation judge | [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) |
| Image dataset | [Kaggle Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data) |

## Installation

Core dependencies (Streamlit app + CLI):

```bash
pip install -r requirements.txt
```

Llama backend (requires NVIDIA/AMD/Intel GPU):

```bash
pip install -r requirements-llama.txt
```

CheXagent backend (CLI only):

```bash
pip install -r requirements-chexagent.txt
```

## Running the Project

### Option 1: Streamlit UI

The recommended way to interact with the project. Supports image upload, LLM reasoning, Q&A chat, and MedGemma evaluation.

```bash
streamlit run app/streamlit_app.py
```

**Sidebar settings:**
- **LLM Provider**: Llama (Local) runs `0llheaven/Llama-3.2-11B-Vision-Radiology-mini` on GPU via `unsloth`. OpenAI (API) uses `gpt-4.1` and requires `OPENAI_API_KEY` in `.env` or the environment.
- **MedGemma evaluation**: Scores LLM reasoning on a 1–5 correctness scale using `google/medgemma-1.5-4b-it` (requires GPU).
- LLM features are optional — the EVA-X classifier works without them.

### Option 2: CLI Scripts

Headless training and diagnosis, intended for Colab or GPU servers.

Train the binary classifier:

```bash
python -m src.train \
  --data-dir ./chest_xray \
  --pretrained-weights ./eva_x_tiny_patch16_merged520k_mim.pt \
  --checkpoint-dir ./checkpoints
```

Run diagnosis (classifier + LLM reasoning):

```bash
python -m src.diagnose \
  --image ./test_image.jpeg \
  --checkpoint ./checkpoints/eva_x_tiny_binary_best.pt \
  --pretrained-weights ./eva_x_tiny_patch16_merged520k_mim.pt \
  --backend llama
```

The `--backend` flag accepts `llama` or `chexagent`.

### Option 3: Colab Notebooks

The original notebooks are designed for Google Colab with GPU. They mount Google Drive for dataset and checkpoint access.

- `Big_Data_Analytics_Midterm_Project.ipynb` — Training + CheXagent inference
- `Big_Data_Analytics_Midterm2.ipynb` — Llama inference with confidence-tiered prompting
- `Radiology_Assistant_Evaluation.ipynb` — End-to-end evaluation with MedGemma judge

## Environment Variables

| Variable | Required for |
|---|---|
| `OPENAI_API_KEY` | OpenAI LLM provider in Streamlit UI; Evaluation notebook |
| `HF_TOKEN` | Downloading gated models (MedGemma) |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Downloading dataset in Evaluation notebook |

Create a `.env` file in the project root (already gitignored):

```
OPENAI_API_KEY=sk-...
```
