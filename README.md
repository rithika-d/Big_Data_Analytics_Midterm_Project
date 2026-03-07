# Big_Data_Analytics_Midterm_Project

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

The root `eva_x.py` module remains available so the original notebooks can still import it without changes.
