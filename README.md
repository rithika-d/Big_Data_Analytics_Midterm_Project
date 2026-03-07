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
