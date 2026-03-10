# Big_Data_Analytics_Midterm_Project

EECS E6893: Big Data Analytics midterm project.

Research prototype only. Not for clinical use.

## Overview

This repo contains a two-stage chest X-ray workflow:

- Stage 1: EVA-X Tiny binary classifier for normal vs. pneumonia-like abnormality.
- Stage 2: optional vision LLM reasoning when the classifier predicts abnormal.

The original training and exploration notebooks remain in the repo:

- `Big_Data_Analytics_Midterm_Project.ipynb`
- `Big_Data_Analytics_Midterm2.ipynb`

EVA-X model code is derived and adapted from [hustvl/EVA-X](https://github.com/hustvl/EVA-X).

The pretrained EVA-X MIM weights referenced by the notebooks can be downloaded from [MapleF/eva_x](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_tiny_patch16_merged520k_mim.pt).

The image data used for the project are sourced from the [Kaggle Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data).

## Streamlit App

The repo now includes a lightweight Streamlit interface for local inference and report Q&A.

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run app/streamlit_app.py
```

### Notes

- `OPENAI_API_KEY` in the environment or a local `.env` file is required for LLM features.
- LLM features are optional. The EVA-X classifier works without an API key.
- `requirements.txt` is the canonical dependency file for the Streamlit app.
- `EVA-X_requirements .txt` is kept only as a historical upstream reference.

## Models

- EVA-X Tiny checkpoint: `eva_x_tiny_binary_best.pt`
- Optional notebook-era LLM reference: [0llheaven/Llama-3.2-11B-Vision-Radiology-mini](https://huggingface.co/0llheaven/Llama-3.2-11B-Vision-Radiology-mini)
EECS E6893 Big Data Analytics midterm project. The repository now includes the original Colab notebooks and a runnable Python package under `src/` for training and diagnosis workflows.

## Project Layout

- `Big_Data_Analytics_Midterm_Project.ipynb`: original training notebook
- `Big_Data_Analytics_Midterm2.ipynb`: original inference notebook
- `src/cxr_pipeline/`: shared package extracted from the notebooks
- `src/train.py`: CLI entry point for classifier training
- `src/diagnose.py`: CLI entry point for classifier + LLM diagnosis
- `eva_x.py`: original root EVA-X module kept for notebook compatibility

## Model Assets

- EVA-X pretrained weights: [MapleF/eva_x](https://huggingface.co/MapleF/eva_x/blob/main/eva_x_tiny_patch16_merged520k_mim.pt)
- Llama radiology model: [0llheaven/Llama-3.2-11B-Vision-Radiology-mini](https://huggingface.co/0llheaven/Llama-3.2-11B-Vision-Radiology-mini)
- CheXagent findings model: [StanfordAIMI/CheXagent-2-3b-srrg-findings](https://huggingface.co/StanfordAIMI/CheXagent-2-3b-srrg-findings)

Image data (training, validation, test) used for the project are sourced from the [Kaggle Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data).

## Installation

Core dependencies:

```bash
pip install -r requirements.txt
```

Llama backend:

```bash
pip install -r requirements-llama.txt
```

CheXagent backend:

```bash
pip install -r requirements-chexagent.txt
```

## Usage

Train the binary EVA-X classifier:

```bash
python -m src.train \
  --data-dir ./chest_xray \
  --pretrained-weights ./eva_x_tiny_patch16_merged520k_mim.pt \
  --checkpoint-dir ./checkpoints
```

Run diagnosis with the Llama backend:

```bash
python -m src.diagnose \
  --image ./test_image.jpeg \
  --checkpoint ./checkpoints/eva_x_tiny_binary_best.pt \
  --pretrained-weights ./eva_x_tiny_patch16_merged520k_mim.pt \
  --backend llama
```

## Evaluation Pipeline

The project includes a comprehensive evaluation pipeline to assess the Q&A capabilities of the radiology assistant using both qualitative (Med-Gemma judge) and quantitative (BLEU, ROUGE) metrics.

### 1. Setup Credentials
Ensure your `.env` file contains the following keys:
- `OPENAI_API_KEY`: For the base model and Q&A.
- `HF_TOKEN`: For downloading Med-Gemma (requires [gated access](https://huggingface.co/google/medgemma-1.5-4b-it)).
- `KAGGLE_USERNAME` & `KAGGLE_KEY`: For downloading test images.

### 2. Download Test Data
Download a sample of the [Kaggle Chest X-Ray Pneumonia dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data):
```bash
python scripts/download_test_images.py
```
This saves images to the `test_images/` directory.

### 3. Generate Evaluation Samples
Create the `qa_test_samples.json` file with ground truth and expert context:
```bash
python scripts/generate_test_json.py
```

### 4. Run Evaluation
Run the evaluator using the Med-Gemma 1.5-4b-it judge:
```bash
python scripts/evaluate_radiology_assistant.py --model openai --use-judge
```
Options for `--model`: `openai` (default RAG), `chexagent`, or `llama`.
Results will be saved to `evaluation_report.json`.

The root `eva_x.py` module remains available so the original notebooks can still import it without changes.
