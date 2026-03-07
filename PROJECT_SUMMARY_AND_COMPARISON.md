# Big Data Analytics Midterm Project: Summary & Comparison with RAV

Date: 2026-03-07

---

## Part 1: Project Summary (Big Data Analytics Midterm)

### Overview

**Course:** EECS E6893 Big Data Analytics — Midterm Project
**Goal:** Two-stage chest X-ray diagnostic pipeline for pneumonia detection.
**Runtime:** Google Colab with GPU (tested on A100).

The system operates in two stages:
1. **Stage 1 — Binary Classifier:** An EVA-X Tiny Vision Transformer, fine-tuned on the Kaggle Chest X-Ray Pneumonia dataset, classifies images as normal vs. abnormal.
2. **Stage 2 — Reasoning LLM:** When the classifier predicts abnormal, a vision-language model generates radiologic findings. Two LLM backends are explored across notebooks:
   - **CheXagent-2-3b** (in `Big_Data_Analytics_Midterm_Project.ipynb`)
   - **Llama-3.2-11B-Vision-Radiology-mini** (in `Big_Data_Analytics_Midterm2.ipynb`)

### Repository Contents

| File | Purpose |
|---|---|
| `eva_x.py` | EVA-X model definitions (Tiny/Small/Base). Subclasses `timm.models.eva.Eva`. Includes checkpoint weight-key remapping. |
| `Big_Data_Analytics_Midterm_Project.ipynb` | Training notebook: data loading, `Eva_X_Model` wrapper, `Trainer` class, training loop, evaluation, CheXagent integration. |
| `Big_Data_Analytics_Midterm2.ipynb` | Inference notebook: loads trained checkpoint, `diagnose_chest_xray()` pipeline, Llama radiology model. |
| `eva_x_tiny_binary_best.pt` | Trained checkpoint (epoch 12, best_val_loss=0.0313). |
| `EVA-X_requirements .txt` / `requirements.txt` | Dependency references are inconsistent in this checkout: the tracked legacy filename has a trailing space, and an untracked `requirements.txt` is also present locally. |
| `README.md` | Project overview with resource links. |

### Model Architecture

**EVA-X Tiny (Stage 1 backbone):**
- Patch size 16, embed dim 192, depth 12 blocks, 3 attention heads
- SwiGLU MLP, Rotary Position Embeddings (RoPE)
- Input: `(B, 3, 224, 224)` images
- Pretrained via Masked Image Modeling (MIM) on 520K medical images

**Eva_X_Model (fine-tuning wrapper):**
- Loads `eva_x_tiny_patch16` with MIM pretrained weights
- Replaces head: `nn.Linear(192, 1)` for binary classification
- Freezes entire backbone, then selectively unfreezes:
  - `head` (classification layer)
  - `blocks.11` (last transformer block)
  - `norm`, `fc_norm` (layer normalization)
- ~454k trainable parameters out of ~5.7M total in the checked-in Tiny wrapper configuration
- Output: `(B, 1)` logits, trained with `BCEWithLogitsLoss`

### Dataset

**Kaggle Chest X-Ray Pneumonia** (`paultimothymooney/chest-xray-pneumonia`):
- Train: 3,253 images (1,341 Normal / 1,912 Pneumonia)
- Val: 16 images (very small)
- Test: 624 images
- Binary labels: Normal (0) vs. Pneumonia (1)

### Image Preprocessing

| Stage | Pipeline |
|---|---|
| Training | `Resize(256)` -> `RandomCrop(224)` -> `RandomHorizontalFlip(0.5)` -> ImageNet normalize |
| Inference | `Resize(256)` -> `CenterCrop(224)` -> ImageNet normalize |

ImageNet normalization: `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`

### Training Configuration

| Parameter | Value |
|---|---|
| Loss | `BCEWithLogitsLoss`, `pos_weight=0.70` (handles 1:1.4 class imbalance) |
| Optimizer | AdamW, `lr=1e-4`, `weight_decay=1e-4` |
| Scheduler | None |
| Epochs | 20 max |
| Early stopping | Patience=4, `min_delta=1e-3` |
| Precision | Mixed (AMP via GradScaler) |
| Batch size | 32 |

### Training Results

**Best checkpoint:** Epoch 12 with `val_loss = 0.0313`

| Image Type | p_abnormal | Prediction | Note |
|---|---|---|---|
| Pneumonia (radiopaedia) | 0.9988 | Abnormal | In-scope positive-style spot check |
| Normal (radiopaedia) | 0.0865 | Normal | In-scope negative-style spot check |
| Bronchitis | 0.9405 | Abnormal | Out-of-distribution example; not task-aligned ground truth for this classifier |
| TB | 0.9986 | Abnormal | Out-of-distribution example; suggests generic abnormal sensitivity, not pneumonia specificity |
| Lung cancer | 0.949 | Abnormal | Out-of-distribution example; should not be reported as task-level correctness |
| Normal (Getty) | 0.4394 | Normal | External normal spot check |

These rows are illustrative spot checks from the notebooks, not a labeled benchmark. They suggest strong separation on a small set of clearly normal vs. clearly abnormal external examples, but they do not establish correctness on non-pneumonia pathologies or characterize the full score distribution.

### Evaluation Metrics

The `evaluate_full()` function computes:
- Accuracy, Sensitivity (Recall/Pneumonia), Specificity (Recall/Normal)
- AUROC (Area Under ROC Curve)
- Confusion Matrix (TN, FP, FN, TP)

### Two-Stage Inference Pipeline: `diagnose_chest_xray()`

**Confidence-tiered prompting** based on classifier probability `p_abnormal`:

| p_abnormal Range | Behavior |
|---|---|
| <= 0.5 | **Normal** — LLM not invoked, returns classification only |
| 0.5 - 0.7 | **Borderline** — LLM prompt emphasizes uncertainty, subtle findings, normal variants |
| 0.7 - 0.8 | **Moderate** — Standard findings + short differential diagnosis |
| > 0.8 | **High confidence** — Dominant patterns, 2-3 likely explanations |

**Design rationale:**
- Efficiency: LLM is expensive; only invoked when needed
- Safety: Avoids LLM hallucinations on obviously normal cases
- Interpretability: Classifier decision triggers graded LLM explanation

### Stage 2 LLM Backends

**CheXagent-2-3b** (Midterm_Project notebook):
- Model: `StanfordAIMI/CheXagent-2-3b-srrg-findings`
- dtype: `bfloat16`, device: GPU 0
- Generation: `max_new_tokens=512`, `num_beams=1` (greedy decoding)

**Llama-3.2-11B-Vision-Radiology-mini** (Midterm2 notebook):
- Model: `0llheaven/Llama-3.2-11B-Vision-Radiology-mini` via Unsloth
- 4-bit quantization, gradient checkpointing
- Generation: `max_new_tokens=128`, `temperature=0.6`, `top_p=0.9` (nucleus sampling)
- Image loading: resized to 448x448

### Known Issues

1. **LLM hallucinations on normal images**: Both CheXagent and Llama fabricate pathology when given normal X-rays (e.g., inventing fractures, pneumothorax). Models appear optimized for abnormal cases.
2. **Tiny validation set**: Only 16 validation images creates high variance in early stopping decisions.
3. **Borderline prompting under-tested**: Very few test cases fall in the 0.5-0.7 range, so that tier's prompting logic has limited validation.

---

## Part 2: RAV Project Summary

> Note: This section summarizes RAV from a separate repository snapshot that is not present in this checkout. Exact versions, file counts, cost figures, and in-progress status notes should be treated as approximate unless pinned to a specific RAV commit.

### Overview

**Context:** Separate chest-radiology project used here for comparison.
**Goal:** Production-grade agentic AI prototype for chest X-ray diagnosis with structured reporting, LLM integration, web UI, and cloud training infrastructure.
**Snapshot:** External RAV inspection, not locally verifiable from this repository.

RAV is a chest-first radiology project with brain imaging deferred. It has two active tracks:
- **POC Track:** Binary classification on Kaggle Chest X-Ray Pneumonia (same dataset as the Midterm project)
- **Primary Track:** Multi-label classification on CheXpert (14 thoracic findings) — in progress

### Repository Structure (Key Components)

```
RAV/
  src/rav_chest/         # Core Python package (models, data, pipeline, reporting, metrics, llm, utils)
  scripts/               # Training, eval, inference, GCP adapters, data prep, monitoring
  configs/               # YAML configs for primary (CheXpert) and POC tracks
  app/                   # Streamlit web UI (inference, metrics, Ask Agent Q&A)
  gcp/                   # GCP spot-runner integration (Dockerfile, entrypoint, state machine, reconciler)
  tests/                 # BATS shell integration tests
  docs/                  # Runbook, hardware sizing, documentation index
  data/                  # Raw and processed datasets (gitignored)
  outputs/               # Checkpoints, metrics, reports (gitignored)
  private/               # Development notes, GCP operational notes
  PDF/                   # Proposal and plan PDFs
```

### Model Architecture

**Classification backbone:** DenseNet121 (default), also supports ResNet50 and EfficientNet-B0.
- Input: 320x320 chest X-rays (configurable)
- Head: `Dropout` -> `nn.Linear(num_classes)` — 1 class for POC, 14 for CheXpert
- Pretrained ImageNet weights, frozen base with trainable head
- Loss: `BCEWithLogitsLoss` with per-class `pos_weight`
- Multi-label classification (CheXpert) or binary (POC)

### Dataset Strategy

| Track | Dataset | Labels | Status |
|---|---|---|---|
| POC | Kaggle Chest X-Ray Pneumonia | Binary (Pneumonia / No Finding) | Complete |
| Primary | CheXpert-v1.0-small (Kaggle mirror) | 14 thoracic pathologies | In progress |
| Deferred | CheXpert Full (~471 GB), MIMIC-CXR-JPG, VinDr-CXR, BraTS/UPENN-GBM (brain) | Various | Future |

### Training Pipeline

- YAML-driven configuration (optimizer, LR, augmentation, early stopping, etc.)
- Mixed precision (AMP) support
- Early stopping on validation AUROC/F1
- Checkpoint management: `best.pt` + `last.pt` every epoch
- Resume from checkpoint on restart
- `skip_none_collate` filters corrupt/unreadable images at runtime
- Data sanity checking (class balance, missing files, split leakage)
- ETA monitoring from `history.jsonl`

### Inference & Reporting Pipeline

1. Load checkpoint + config
2. Image -> RGB tensor -> resize/normalize -> forward pass -> sigmoid -> per-class probabilities
3. Apply per-class thresholds (configurable, with overrides for critical findings)
4. Build structured findings JSON with confidence scores and critical flags
5. Generate template-based impression (deterministic, grammar-based)
6. Optional LLM rewrite via OpenAI API (`gpt-4.1-mini` default) with hallucination guardrails
7. Save/display report

**Report schema:**
```json
{
  "findings": [{"name": "...", "confidence": 0.87, "threshold": 0.5}],
  "critical_flags": ["Pneumonia"],
  "impression": "...",
  "llm_rewrite": {"enabled": true, "model": "gpt-4.1-mini", "rewritten_impression": "..."}
}
```

### Streamlit Web UI

Three-page app:
1. **Inference** — Upload CXR, select config/checkpoint, run prediction, view findings, download report JSON
2. **Model Metrics** — AUROC/F1 plots, per-class tables, confusion matrices from training runs
3. **Ask Agent** — Q&A chatbot grounded in the current inference report context

### GCP Spot Runner Integration

- External `gcp-spot-runner` project with RAV-specific thin adapter wrappers
- Unified CLI: `./scripts/rav-gcp.sh {build|submit|status|monitor|...}`
- Container: Docker with CUDA 12.4.1, deployed via Cloud Build to Artifact Registry
- Spot VMs (preemptible) with auto-restart and checkpoint sync to GCS
- State machine for failure recovery (RUNNING, COMPLETE, FAILED, PREEMPTED)
- Cloud reconciler for stale heartbeat detection and VM restart
- Dataset sync: GCS -> VM at job start; Azure -> GCS one-time transfer for CheXpert Full

### Evaluation Metrics

- Per-class AUROC (macro and individual)
- Per-class F1 score
- Per-class Brier score (calibration)
- Confusion matrices: TP, TN, FP, FN, sensitivity, specificity, precision, NPV

### Private Directory Findings

**`private/NOTES.md`** — Development notes and operational runbook:
- MVP remaining items: finish training, confirm best.pt, run held-out eval, verify Streamlit demo
- GCP monitoring commands (status, serial logs, SSH, training metrics tail)
- GCP setup steps (dataset upload, job submission, monitoring)
- Bug fixes (RUNNER_DIR relative symlink fix, gcloud Python version pinning)
- Azure -> GCS dataset transfer detailed steps with `azcopy`
- Service account and IAM role setup
- TODO list (most items marked DONE)
- Primary track status: Data prepared, Training WIP, Eval queued

**`private/GCP_Notes.md`** — Quick-reference GCP monitoring commands:
- Canonical status check via `rav-gcp.sh`
- Watch, serial, events, and list commands

**`private/Big_Data_Analytics_Midterm.ipynb`** — Reference copy of the Midterm project notebook within the RAV repo. Contains the EVA-X model code, dataset classes (MultiClassImageDataset, MultiClassImageTestDataset), ViT preprocessing, and Eva_X_Model wrapper. This appears to be an earlier/reference version used for cross-referencing between the two projects.

---

## Part 3: Deep Comparison

### 3.1 Project Scope & Ambition

| Dimension | Midterm Project | RAV |
|---|---|---|
| **Course / context** | EECS E6893 (Big Data Analytics) | Separate chest-radiology project; exact course context not verified in this repo |
| **Scope** | Single notebook pipeline, proof-of-concept | Production-grade prototype with ops infrastructure |
| **Tracks** | Single (binary pneumonia) | Two active (POC binary + CheXpert 14-label), plus deferred tracks |
| **Codebase size** | Small notebook-centric repo | Larger multi-directory application repo in the inspected snapshot |
| **Code organization** | Monolithic notebooks + 1 Python file | Modular Python package (`src/rav_chest/`) + scripts + configs + app |
| **Version control** | Basic git workflow with lightweight changelog | Semantic versioning and a detailed CHANGELOG in the inspected snapshot |
| **Documentation** | README + CLAUDE.md + project summary | README plus runbook and ops documentation in the inspected snapshot |

### 3.2 Model Architecture

| Dimension | Midterm Project | RAV |
|---|---|---|
| **Backbone** | EVA-X Tiny (ViT, 192-dim, 12 blocks, RoPE) | DenseNet121 (default); also ResNet50, EfficientNet-B0 |
| **Architecture family** | Vision Transformer | CNN |
| **Pretrained weights** | MIM (Masked Image Modeling) on 520K medical images | ImageNet (general domain) |
| **Domain specificity** | Medical-domain pretrained (EVA-X designed for medical imaging) | General-purpose pretrained (ImageNet) |
| **Fine-tuning strategy** | Freeze all, unfreeze last block + head + norms | Freeze base, unfreeze head (dropout + linear) |
| **Trainable params** | ~454k (last block + norms + head) | Head only (much fewer) |
| **Input size** | 224x224 | 320x320 (configurable) |
| **Output** | Binary (1 logit) | Binary (POC) or 14-label multi-label (CheXpert) |
| **Model variants** | Single (Tiny only) | Three backbone options via factory |

**Analysis:** The Midterm project uses a more specialized medical-domain backbone (EVA-X with medical MIM pretraining), while RAV uses a general-purpose but well-proven CNN (DenseNet121) with broader backbone flexibility. EVA-X's medical pretraining may give it an edge on medical features, but DenseNet121 is a strong, established baseline for CheXpert-style tasks.

### 3.3 Dataset & Labels

| Dimension | Midterm Project | RAV |
|---|---|---|
| **Primary dataset** | Kaggle Chest X-Ray Pneumonia | CheXpert-v1.0-small (primary) + Kaggle (POC) |
| **Label cardinality** | Binary (Normal / Pneumonia) | 14-label multi-label (CheXpert) + binary (POC) |
| **Training samples** | 3,253 | Kaggle-scale POC plus a much larger CheXpert-scale primary dataset |
| **Validation set** | 16 images (problematic) | Proper val split |
| **Uncertainty labels** | N/A (binary) | CheXpert's -1 (uncertain) with configurable mapping |
| **Data sanity checks** | None | Script for class balance, missing files, split leakage |
| **Data loading robustness** | Standard | `skip_none_collate` filters corrupt images at runtime |

**Analysis:** RAV has a significantly more mature data pipeline. The Midterm project's 16-image validation set is a notable weakness — RAV addresses this with proper splits. CheXpert's 14-label schema is far more clinically useful than binary pneumonia detection, though it introduces the complexity of uncertain labels.

### 3.4 Training Pipeline

| Dimension | Midterm Project | RAV |
|---|---|---|
| **Environment** | Google Colab (A100 GPU) | Local (.venv) + GCP Spot VMs (L4/T4) |
| **Configuration** | Hardcoded in notebook | YAML config files, CLI arguments |
| **Mixed precision** | AMP via GradScaler | AMP (auto-enabled on GPU) |
| **Early stopping** | Patience=4, min_delta=1e-3 | Configurable patience, selection metric (AUROC/F1/val_loss) |
| **Checkpoint management** | Single best checkpoint | `best.pt` + `last.pt` every epoch, resume from checkpoint |
| **Class imbalance** | `pos_weight=0.70` | Per-class `pos_weight` |
| **Scheduler** | None | Configurable |
| **Augmentation** | RandomCrop + HorizontalFlip | Configurable (RandomHorizontalFlip, RandomAffine, ColorJitter) |
| **Reproducibility** | Manual Colab execution | `make` targets, one-command training |
| **Cloud training** | Colab (manual) | GCP Spot with auto-restart, checkpoint sync, reconciler |
| **Monitoring** | Manual observation | ETA monitor, GCS metrics tail, serial log viewer |

**Analysis:** RAV's training infrastructure is dramatically more robust. The YAML-driven configuration, checkpoint resume, GCP spot integration with auto-restart, and monitoring tools make it production-capable. The Midterm project relies on manual Colab execution with no resume capability.

### 3.5 Inference & Reporting

| Dimension | Midterm Project | RAV |
|---|---|---|
| **Inference flow** | `predict_image()` -> confidence tier -> LLM prompt | `infer_chest_single.py` -> findings JSON -> template report -> optional LLM rewrite |
| **LLM integration** | Direct VLM (CheXagent / Llama) sees the image | OpenAI API text-only rewrite of structured findings |
| **LLM models** | CheXagent-2-3b, Llama-3.2-11B-Vision | gpt-4.1-mini (OpenAI API, text-only) |
| **LLM input** | Image + prompt (multimodal) | Structured findings JSON (text-only, no image) |
| **Hallucination control** | Confidence-tiered prompting (skip LLM for normals) | Constrained rewrite: LLM can only rephrase predicted findings, system prompt forbids inventing |
| **Report format** | Free-text LLM output | Structured JSON with findings, confidence, thresholds, critical flags, impression |
| **Critical flags** | None | Automatic flagging (Pneumothorax, Pleural Effusion, Edema) |
| **UI** | Notebook output | Streamlit web app (inference + metrics + Q&A) |

**Analysis:** These represent fundamentally different approaches to LLM integration:

- **Midterm:** The LLM directly sees the X-ray image (multimodal VLM). This is more powerful in theory — the LLM can identify features the classifier missed. But it leads to hallucinations on normal images and is harder to control.
- **RAV:** The LLM only rewrites/rephrases classifier outputs (text-only). This is more controlled — it cannot hallucinate findings the classifier didn't predict — but cannot add insights beyond the classifier's capability.

The Midterm's confidence-tiered prompting is a creative middle ground: it avoids LLM hallucinations on clearly normal images (p <= 0.5) while adapting the prompt intensity for borderline vs. confident abnormal cases. RAV's approach is more conservative and production-safe.

### 3.6 Evaluation

| Dimension | Midterm Project | RAV |
|---|---|---|
| **Metrics** | Accuracy, Sensitivity, Specificity, AUROC, Confusion Matrix | Per-class AUROC, F1, Brier score, Confusion Matrix (TP/TN/FP/FN), Sensitivity, Specificity, Precision, NPV |
| **Calibration** | Not measured | Brier score per class |
| **Threshold tuning** | Fixed at 0.5 | Per-class thresholds with configurable overrides |
| **Artifacts** | In-notebook output | JSON/CSV files saved to `outputs/.../metrics/` |
| **Reporting** | LLM quality not formally evaluated | Fact grounding score, hallucination rate, clinical coherence rubric planned |

### 3.7 LLM Strategy Comparison

| Dimension | Midterm Project | RAV |
|---|---|---|
| **Approach** | Vision-Language Model (sees image directly) | Text-only LLM (rewrites structured findings) |
| **Models used** | CheXagent-2-3b, Llama-3.2-11B-Vision-Radiology | gpt-4.1-mini (OpenAI API) |
| **Model hosting** | Self-hosted (Colab GPU, 4-bit quantized) | API call (OpenAI) |
| **Compute cost** | High (11B model on GPU) | Low (API call, small model) |
| **Multimodal** | Yes (image + text input) | No (text-only rewrite) |
| **Hallucination risk** | High (generates findings from image, hallucinates on normals) | Low (constrained to classifier outputs) |
| **Medical domain specificity** | High (radiology-specific VLMs) | Low (general-purpose LLM) |
| **Agentic Q&A** | No | Yes (Ask Agent, grounded in report context) |

### 3.8 Infrastructure & Operations

| Dimension | Midterm Project | RAV |
|---|---|---|
| **Runtime** | Google Colab (free/paid tier) | Local dev (.venv) + GCP Spot VMs |
| **Containerization** | None | Docker with CUDA 12.4.1 |
| **Cloud ops** | Manual Colab | Unified CLI (`rav-gcp.sh`), automated spot management |
| **Fault tolerance** | None (lost on disconnect) | Auto-restart on preemption, checkpoint sync, state machine |
| **Cost optimization** | Free tier / Colab Pro | Spot VMs intended to reduce cost relative to on-demand instances |
| **CI/Testing** | None | BATS integration tests in the inspected snapshot |
| **Makefile** | No | Yes (train, eval, streamlit, sanity, monitoring targets) |
| **Web UI** | No | Streamlit (3-page app) |

### 3.9 Code Quality & Engineering

| Dimension | Midterm Project | RAV |
|---|---|---|
| **Code organization** | Monolithic notebooks | Modular package with separation of concerns |
| **Configuration** | Hardcoded constants | YAML config files with schema |
| **Error handling** | Basic (notebook-level try/except) | Robust (corrupt image handling, collate filtering, device auto-detect) |
| **Reproducibility** | Manual (re-run cells) | One-command (`make train-poc`) |
| **Testing** | None | BATS shell tests |
| **Documentation** | README, inline comments, project summary | README, CHANGELOG, runbook, hardware sizing, GCP guides |
| **Version tracking** | Git commits plus lightweight changelog | Semantic versioning with a detailed CHANGELOG |
| **Dependencies** | Dependency filenames currently inconsistent in this checkout | requirements.txt + YAML config schema |

### 3.10 Strengths & Weaknesses

#### Midterm Project

**Strengths:**
- Medical-domain pretrained backbone (EVA-X with MIM on medical images)
- Multimodal LLM integration — VLM actually sees the X-ray
- Creative confidence-tiered prompting design
- Explores multiple VLM backends (CheXagent, Llama)
- Simple, self-contained — runs in a single Colab session
- Illustrative spot checks show strong separation on a handful of clearly normal vs. clearly abnormal external examples

**Weaknesses:**
- Tiny validation set (16 images) undermines model selection reliability
- LLM hallucinations on normal images are significant
- No structured output format — free-text LLM responses
- No web UI, no cloud ops, no checkpoint resume
- Single binary label (Normal/Pneumonia) limits clinical utility
- No calibration metrics, no threshold tuning
- Not reproducible without manual Colab interaction

#### RAV

**Strengths:**
- Production-grade infrastructure (GCP spot, Docker, auto-restart, checkpoint sync)
- Modular, well-structured codebase with separation of concerns
- Multi-label classification (14 CheXpert findings) — clinically meaningful
- Structured reporting with hallucination guardrails
- Web UI with inference, metrics, and Q&A
- Comprehensive evaluation (AUROC, F1, Brier, per-class thresholds)
- YAML-driven configuration, one-command workflows
- Testing, documentation, and versioning

**Weaknesses:**
- General-purpose backbone (DenseNet121 with ImageNet weights) — no medical-domain pretraining
- Text-only LLM rewrite — cannot identify visual features the classifier missed
- LLM (gpt-4.1-mini) is general-purpose, not radiology-specific
- Primary CheXpert training still incomplete (WIP)
- Significant infrastructure complexity for a course project
- No multimodal VLM capability

### 3.11 Cross-Pollination Opportunities

**What RAV could learn from Midterm:**
1. **EVA-X backbone**: Medical-domain pretrained ViT could replace or complement DenseNet121 — RAV's model factory already supports swapping backbones
2. **Multimodal VLM integration**: Direct image-to-LLM pipeline could augment RAV's text-only rewrite approach for richer clinical descriptions
3. **Confidence-tiered prompting**: Could be adapted to RAV's LLM rewrite step — adjust rewrite aggressiveness based on classifier confidence

**What Midterm could learn from RAV:**
1. **Structured reporting**: Replace free-text LLM output with structured findings JSON + constrained rewrite
2. **Multi-label classification**: Expand from binary to CheXpert-style 14-label taxonomy
3. **Infrastructure**: YAML configs, checkpoint resume, proper val splits, make targets
4. **Calibration**: Add Brier score and per-class threshold tuning
5. **Web UI**: Streamlit app for demonstration and interaction

### 3.12 Summary Matrix

| Dimension | Midterm | RAV | Winner |
|---|---|---|---|
| Medical backbone | EVA-X (medical MIM) | DenseNet121 (ImageNet) | Midterm |
| Label richness | Binary | 14-label multi-label | RAV |
| LLM approach | Multimodal VLM (sees image) | Text-only rewrite (controlled) | Different tradeoffs |
| Hallucination safety | Tiered prompting (partial) | Constrained rewrite (strong) | RAV |
| Clinical specificity | Pneumonia only | 14 thoracic findings + critical flags | RAV |
| Training infrastructure | Colab (manual) | GCP Spot (automated) | RAV |
| Code quality | Notebook-based | Modular package | RAV |
| Data pipeline | Basic, 16-image val | Robust, proper splits | RAV |
| Web UI | None | Streamlit (3 pages) | RAV |
| Evaluation depth | Basic metrics | Per-class AUROC/F1/Brier, threshold tuning | RAV |
| Simplicity | High (runs in 1 notebook) | Low (many moving parts) | Midterm |
| Domain-specific VLM | CheXagent + Llama-Rad | None | Midterm |
| Innovation | Confidence-tiered prompting | Spot VM resilience, structured reporting | Tie |

### 3.13 Relationship Between Projects

The RAV `private/` directory contains a copy of the Midterm notebook (`Big_Data_Analytics_Midterm.ipynb`), suggesting the Midterm project served as an initial exploration or reference for RAV's development. Both projects share:
- The same Kaggle Chest X-Ray Pneumonia dataset (used as POC in RAV)
- Interest in EVA-X as a medical backbone (referenced in RAV's README)
- The same fundamental goal: chest X-ray diagnosis with AI

The Midterm project can be seen as a focused research prototype exploring medical VLMs and confidence-tiered prompting, while RAV is the production-oriented evolution with proper engineering practices, multi-label support, and cloud infrastructure. They complement each other — the Midterm's medical-domain innovations (EVA-X, VLM integration) could strengthen RAV's more robust but less specialized pipeline.
