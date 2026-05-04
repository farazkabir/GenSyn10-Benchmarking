# Synth10: A Synthetic CIFAR-10 Dataset from Vision-Language Models

A complete pipeline for generating, evaluating, and benchmarking a **synthetic image dataset** aligned with [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) using state-of-the-art text-to-image models. The project spans prompt engineering, multi-model image synthesis, distributional quality metrics, 17-architecture classification benchmarks, and real-vs-synthetic detection.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Pipeline Stages](#pipeline-stages)
  - [Stage 1: Prompt Generation](#stage-1-prompt-generation)
  - [Stage 2: Image Generation](#stage-2-image-generation)
  - [Stage 3: Dataset Quality Evaluation (FID / IS / KID)](#stage-3-dataset-quality-evaluation)
  - [Stage 4: CIFAR-10 Model Initialization and Baseline Benchmark](#stage-4-cifar-10-model-initialization-and-baseline-benchmark)
  - [Stage 5: Four-Stage Cross-Domain Evaluation (CIFAR-10 vs Synth10)](#stage-5-four-stage-cross-domain-evaluation)
  - [Stage 6: Binary Real-vs-Synthetic Classification (SynthCIFAR)](#stage-6-binary-real-vs-synthetic-classification)
  - [Stage 7: Single-Model (SDXL) Generalization Studies](#stage-7-single-model-sdxl-generalization-studies)
- [Models Benchmarked](#models-benchmarked)
- [Key Results](#key-results)
- [Prompt Grammar](#prompt-grammar)
- [Setup and Dependencies](#setup-and-dependencies)
- [Usage](#usage)
- [Hardware Used](#hardware-used)
- [License](#license)

---

## Overview

This repository provides:

1. **Grammar-based prompt generation**: a combinatorial prompt builder producing diverse, photorealistic prompts across 10 CIFAR-10 classes with controllable difficulty tiers ("clean" and "hard").
2. **Multi-model image synthesis**: Jupyter notebooks for four text-to-image backends that render 512x512 images from prompts, then downsample to 32x32 for CIFAR-style evaluation.
3. **Distributional quality metrics**: FID, Inception Score, KID, Precision, and Recall computed against original CIFAR-10 using `torch-fidelity`.
4. **17-architecture classification benchmarks**: a comprehensive evaluation of CNNs and vision transformers on both CIFAR-10 and the synthetic dataset (Synth10), including a four-stage domain-shift study.
5. **Real-vs-synthetic binary detection**: all 17 models adapted for binary classification on the SynthCIFAR dataset (real vs. synthetic images), with a parallel **SD 3.5 Large-only** variant (SDCIFAR) that isolates a single generator.
6. **Single-model generalization study**: Synth10 fine-tuned checkpoints evaluated against an SDXL-only test set to measure cross-generator transfer.

The final dataset, **Synth10**, contains **60,000 synthetic images** (6,000 per class) at 32x32 resolution, structured identically to CIFAR-10 with a 50,000/10,000 train/test split.

---

## Repository Structure

```
.
├── README.md
│
├── Prompt_Generation.ipynb                        # Prompt grammar + generation (6k/10k/60k prompts)
├── prompts_10k.jsonl                              # Pre-generated 10k prompts (JSONL)
├── prompts_10k.csv                                # Same prompts in CSV
├── per_class_prompts/                             # Per-class prompt files (airplane.jsonl, bird.csv, ...)
│
├── Image_Generation_SDXL.ipynb                    # Stable Diffusion 3.5 Large
├── Image_Generation_FLUX2_4bit.ipynb              # FLUX.2-dev (4-bit quantized)
├── Image_Generation_QWEN.ipynb                    # Qwen-Image-2512
├── Image_Generation_ HUANYUAN3_4bit.ipynb         # HunyuanImage-3 NF4 v2
│
├── synth10_fid_inception_metrics_eval.ipynb        # FID, IS, KID, Precision, Recall evaluation
├── fid_inception score_results.txt                 # Saved FID / IS / KID / Precision / Recall values
│
├── cifar10_model_init.ipynb                        # Load & benchmark 17 architectures on CIFAR-10
├── cifar10_benchmark_eval_with_synth10_four_stage_with_results (final).ipynb
│                                                   # Four-stage CIFAR-10 vs Synth10 cross-evaluation
├── SynthCIFAR_binary_classification_with_results (final).ipynb
│                                                   # Binary real-vs-synthetic detection benchmark (Synth10)
├── SDCIFAR_binary_classification_with_results.ipynb
│                                                   # Binary real-vs-SDXL detection (single-model variant)
├── sdxl_dataset_synth10_eval.ipynb                 # Synth10 fine-tuned models evaluated on SDL-only test set
│
└── Synth10(Unsplitted)/                                        # Output: 32x32 images per class 
    ├── airplane/
    ├── automobile/
    ├── bird/
    ├── cat/
    ├── deer/
    ├── dog/
    ├── frog/
    ├── horse/
    ├── ship/
    └── truck/
```

---

## Pipeline Stages

### Stage 1: Prompt Generation

**Notebook:** `Prompt_Generation.ipynb`

Generates structured, photorealistic prompts using a combinatorial grammar with the following variation slots:

| Slot | Description | Count |
|------|-------------|-------|
| **Classes** | CIFAR-10 categories | 10 |
| **Instances** | Per-class subtypes (e.g., "passenger jet", "sedan", "tabby cat") | 6 to 7 per class |
| **Scenes** | Background settings (e.g., "an urban street", "a sandy beach") | 14 |
| **Views (clean)** | Standard camera angles | 8 |
| **Views (hard)** | Clean views plus challenging framing | 10 |
| **Lighting** | Illumination conditions (e.g., "golden hour", "foggy morning") | 8 |
| **Actions** | Per-class poses/activities | 4 to 6 per class |
| **Hard modifiers** | Difficulty-increasing phrases (occlusion, motion blur, etc.) | 5 |
| **Prompt openers** | Opening phrase variety | 6 |

Each prompt follows this template:

```
{opener} of a {subject_phrase} {action} in {scene}, {view}, {lighting}. [hard_modifier.] {CONSTRAINTS}.
```

A fixed set of **constraints** is appended to every prompt to enforce photorealism, and a global **negative prompt** is provided for models that support it (e.g., SD).

**Difficulty tiers:**

- **Clean** (default 80%): standard views, no hard modifiers.
- **Hard** (default 20%): extended views plus one hard modifier per prompt (e.g., "partially occluded by a foreground object", "slight motion blur").

**Output formats:** JSONL and CSV, with per-prompt metadata (`id`, `class_label`, `instance`, `action`, `scene`, `view`, `lighting`, `tier`, `prompt_opener`, `prompt`, `negative_prompt`).

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--out` | (required) | Output file path |
| `--format` | `jsonl` | `jsonl` or `csv` |
| `--total` | `60000` | Total prompts (must be divisible by 10) |
| `--seed` | `42` | Random seed |
| `--clean-ratio` | `0.8` | Fraction of clean-tier prompts |

---

### Stage 2: Image Generation

Four notebooks, one per text-to-image backend. All share a common pipeline architecture:

1. Load per-class prompt JSONL files.
2. Sample `N` prompts per class.
3. Batch-generate images at 512x512 using the model pipeline.
4. Downsample to 32x32 via Lanczos resampling.
5. Save high-resolution and CIFAR-sized images, plus a metadata JSONL log per run.

| Notebook | Model | Model ID | Quantization | Notes |
|----------|-------|----------|-------------|-------|
| `Image_Generation_SDXL.ipynb` | **Stable Diffusion 3.5 Large** | `stabilityai/stable-diffusion-3.5-large` | bfloat16 | Uses `StableDiffusion3Pipeline`; supports negative prompts |
| `Image_Generation_FLUX2_4bit.ipynb` | **FLUX.2-dev** | `black-forest-labs/FLUX.2-dev` | 4-bit (BnB) | Loads 4-bit transformer and text encoder from `diffusers/FLUX.2-dev-bnb-4bit`; uses `Flux2Pipeline` with Mistral3 text encoder |
| `Image_Generation_QWEN.ipynb` | **Qwen-Image-2512** | `Qwen/Qwen-Image-2512` | bfloat16 | Uses `DiffusionPipeline`; supports `true_cfg_scale` parameter |
| `Image_Generation_ HUANYUAN3_4bit.ipynb` | **HunyuanImage-3 NF4 v2** | `EricRollei/HunyuanImage-3-NF4-v2` | NF4 (4-bit) | Uses `AutoModelForCausalLM` with SDPA attention; MoE architecture with `device_map="auto"` |

**Common `GenConfig` parameters:**

```python
GenConfig(
    jsonl_paths=["airplane.jsonl", ...],  # per-class prompt files
    out_root="./cifar10_synth_<model>",
    samples_per_class=1000,
    seed=123,
    gen_width=512, gen_height=512,
    cifar_size=32,
    steps=30,
    guidance=7.5,       # guidance_scale (or true_cfg_scale for Qwen)
    batch_size=50-100,  # tune for available VRAM
)
```

**Output layout:**

```
cifar10_synth_<model>/
├── cifar_hires/<class>/   # 512x512 PNGs
├── cifar32/<class>/       # 32x32 PNGs
└── metadata_<backend>.jsonl
```

Each generation run supports **resume**: if both the high-res and CIFAR image already exist for a task, it is skipped.

---

### Stage 3: Dataset Quality Evaluation

**Notebook:** `synth10_fid_inception_metrics_eval.ipynb`

Computes distributional quality metrics comparing the full 60,000-image Synth10 dataset against the original CIFAR-10 training set using `torch-fidelity`:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Inception Score (IS)** | 8.77 ± 0.11 | Measures class diversity and image clarity (higher is better) |
| **Frechet Inception Distance (FID)** | 35.17 | Measures distributional similarity to real CIFAR-10 (lower is better) |
| **Kernel Inception Distance (KID)** | 0.0183 ± 0.0010 | Unbiased alternative to FID (lower is better) |
| **Precision** | 0.244 | Fraction of generated images falling within the real data manifold |
| **Recall** | 0.263 | Fraction of real data manifold covered by generated images |

The raw values produced by this notebook are also persisted as plain text in `fid_inception score_results.txt` for quick reference outside of the notebook environment.

After metric computation, the notebook splits the dataset into a CIFAR-10-style structure:

- **Train:** 50,000 images (5,000 per class)
- **Test:** 10,000 images (1,000 per class)

The structured dataset is then zipped for distribution.

---

### Stage 4: CIFAR-10 Model Initialization and Baseline Benchmark

**Notebook:** `cifar10_model_init.ipynb`

Loads and evaluates **17 CNN and vision transformer architectures** on the standard CIFAR-10 test set. Models are initialized from the best available CIFAR-10 checkpoints.

**Weight sources (in priority order):**

1. **chenyaofo/pytorch-cifar-models**: CIFAR-10-native weights (VGG, MobileNetV2).
2. **huyvnphan/PyTorch_CIFAR10**: official repo checkpoints (GoogLeNet, InceptionV3, DenseNets, ResNets, VGG).
3. **Hugging Face Hub**: explicit CIFAR-10 checkpoints (ViT-Base/16, Swin-Base).
4. **Controlled transfer-learning fallback**: for architectures without public CIFAR-10 checkpoints, the notebook performs a head-only warm-up (2 epochs) followed by full fine-tuning (up to 8 epochs) with AdamW, LR scheduling, label smoothing, gradient clipping, and early stopping on validation macro-F1.

**Metrics reported:** Accuracy, Balanced Accuracy, Precision, Recall, F1 (macro/weighted), ROC-AUC, PR-AUC, MCC, Cohen's Kappa, Specificity, FPR, FNR, NPV, per-class metrics, and confusion matrices.

---

### Stage 5: Four-Stage Cross-Domain Evaluation

**Notebook:** `cifar10_benchmark_eval_with_synth10_four_stage_with_results (final).ipynb`

Evaluates all 17 models under four scientifically relevant settings to measure domain shift and adaptation:

| Stage | Evaluation Set | Description |
|-------|---------------|-------------|
| **1. Pre-FT on CIFAR-10** | CIFAR-10 test | Baseline performance on real data |
| **2. Pre-FT on Synth10** | Synth10 test | Zero-shot transfer to synthetic data |
| **3. Post-Synth10-FT on CIFAR-10** | CIFAR-10 test | Retention/forgetting after synthetic fine-tuning |
| **4. Post-Synth10-FT on Synth10** | Synth10 test | Adaptation gains from synthetic fine-tuning |

**Synth10 fine-tuning protocol:**

- Stratified train/validation split from `synth10/train`
- Classifier-head warm-up followed by full-network fine-tuning
- AdamW, label smoothing, gradient clipping, learning-rate scheduling
- Early stopping and best-checkpoint selection using validation macro-F1

**Generated artifacts:**

- `all_results_four_stage_comparison.csv`: full metrics for all models across all stages
- `delta_synth10_after_finetuning.csv`: per-model adaptation gains
- `delta_cifar10_after_synth10_finetuning.csv`: per-model retention/forgetting
- Heatmaps, confusion matrices, per-class F1 comparisons, and adaptation/retention bar charts

---

### Stage 6: Binary Real-vs-Synthetic Classification

**Notebook:** `SynthCIFAR_binary_classification_with_results (final) (1).ipynb`

All 17 models are adapted for **binary classification** on the **SynthCIFAR** dataset:

```
SynthCIFAR/
├── train/
│   ├── real/       # real CIFAR-10 images
│   └── synthetic/  # Synth10 images
└── test/
    ├── real/
    └── synthetic/
```

Models are first loaded from their CIFAR-10 checkpoints, then the final classification head is replaced with a 2-class head and fine-tuned on the SynthCIFAR train split.

**Primary positive class:** `synthetic`

**Evaluation metrics:** Accuracy, Balanced Accuracy, Precision, Recall, Specificity, F1, ROC AUC, PR AUC, MCC, Cohen's Kappa, Log Loss, Brier Score, calibration curves, ROC/PR curves, and confusion matrices.

---

### Stage 7: Single-Model (SD Large) Generalization Studies

These two notebooks isolate **SD 3.5 Large** as the only generator and probe how well models trained or evaluated on the multi-model **Synth10** dataset transfer to a single-generator regime.

**Notebook:** `SDCIFAR_binary_classification_with_results.ipynb`

A direct counterpart to the Stage 6 SynthCIFAR benchmark, but the synthetic half of the dataset comes **exclusively from SD** (`SdxlCIFAR/`) instead of the full multi-model Synth10 mix. All 17 architectures are loaded from CIFAR-10 checkpoints, adapted to a 2-class head, and fine-tuned to discriminate **real vs. SD-only** images. Useful for measuring how generator-specific real-vs-synthetic detection is, and how performance changes when the synthetic distribution is narrower.

**Notebook:** `sdxl_dataset_synth10_eval.ipynb`

Evaluates the **Synth10 fine-tuned model checkpoints** (produced in Stage 5) against a custom **SD-only test dataset** (`sdxl10/`) downloaded from Google Drive. Every `.pth` checkpoint in `model_cache` is loaded, its architecture reconstructed, and inference is run on 100% of the test set (no train/val split) using the same full classification metric suite as the main benchmark. This quantifies how well multi-model Synth10 fine-tuning generalizes to a single-generator (SDXL) test distribution and produces a ranked comparison table across all 17 architectures.

---

## Models Benchmarked

All evaluation and benchmark notebooks use the same set of 17 architectures spanning classic CNNs, efficient networks, and modern transformers:

| Family | Architecture |
|--------|-------------|
| VGG | VGG19-BN |
| ResNet | ResNet-50, ResNet-152 |
| DenseNet | DenseNet-121, DenseNet-169, DenseNet-201 |
| GoogLeNet / Inception | GoogLeNet, InceptionV3 |
| MobileNet | MobileNetV2 |
| MNASNet | MNASNet-1.0 |
| EfficientNet | EfficientNet-B0, EfficientNetV2-S |
| RegNet | RegNetY-8GF |
| Vision Transformer | ViT-Base/16 |
| Swin Transformer | Swin-Base |
| ConvNeXt | ConvNeXt-Base, ConvNeXtV2-Base |

---

## Key Results

### Synth10 Distributional Quality (vs. CIFAR-10)

| Metric | Value |
|--------|-------|
| Inception Score | 8.77 ± 0.11 |
| FID | 35.17 |
| KID | 0.0183 ± 0.0010 |
| Precision | 0.244 |
| Recall | 0.263 |

### Four-Stage Classification (Best Model per Stage)

| Stage | Best Model | Accuracy | Balanced Acc. | F1 Macro | ROC AUC | MCC |
|-------|-----------|----------|---------------|----------|---------|-----|
| Pre-FT on CIFAR-10 | Swin-Base | 98.83% | 98.83% | 98.83% | 99.98% | 0.9870 |
| Pre-FT on Synth10 (zero-shot) | Swin-Base | 96.86% | 96.86% | 96.83% | 99.84% | 0.9656 |
| Post-Synth10-FT on CIFAR-10 | ViT-Base/16 | 95.12% | 95.12% | 95.13% | n/a | n/a |
| Post-Synth10-FT on Synth10 | ViT-Base/16 | 99.85% | 99.85% | 99.85% | 100.00% | 0.9983 |

**Key findings:**

- **Largest Synth10 F1 gain after fine-tuning:** MNASNet-1.0 (+17.80 points)
- CIFAR-10-trained models achieve **96.86% zero-shot accuracy** on Synth10, indicating high distributional alignment
- After fine-tuning on Synth10, models reach **99.85% accuracy** on the synthetic test set while retaining strong CIFAR-10 performance

---

## Prompt Grammar

### Slot Definitions

| Slot | Description | Example Values |
|------|-------------|----------------|
| **CLASSES** | 10 CIFAR-10 categories | `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck` |
| **INSTANCES** | Per-class subtypes | `passenger jet`, `sedan`, `sparrow`, `tabby cat`, `stag with antlers`, `labrador`, `tree frog`, `racehorse`, `cargo ship`, `pickup truck` |
| **SCENES** | Background settings | `a grassy field`, `an urban street`, `a lakeside shore`, `a parking lot` |
| **VIEWS_CLEAN** | Standard camera views | `side profile`, `3/4 view`, `front view`, `close-up`, `wide shot` |
| **VIEWS_HARD** | Clean plus harder views | adds `cropped at the edge of the frame`, `distant subject in the background` |
| **LIGHTING** | Lighting conditions | `sunny afternoon`, `golden hour`, `foggy morning`, `rainy with wet reflections` |
| **ACTIONS** | Per-class pose/activity | `in flight` (bird), `parked` (automobile), `grazing` (deer) |
| **HARD_MODIFIERS** | Difficulty phrases | `partially occluded by a foreground object`, `slight motion blur`, `in the distance` |
| **PROMPT_OPENERS** | Opening phrase variety | `a photorealistic color photo`, `an ultra-realistic rendering`, `a professional photo` |

### Fixed Constraints (appended to every prompt)

```
single main subject, photorealistic color photo, natural colors, realistic texture,
sharp focus on subject, no people, no text, no watermark, no logo, no border,
no collage, no CGI, no illustration
```

### Negative Prompt (for models that support it)

```
text, watermark, logo, caption, signature, border, frame, collage, multiple panels,
cartoon, illustration, CGI, 3d render, low-poly, painting, anime, surreal,
deformed, duplicate objects, extra limbs, blurry, out of focus, noise, oversaturated,
posterized, artifacts, jpeg artifacts
```

### Example Prompts

**Clean tier:**

```
a photorealistic color photo of a stag with antlers deer standing in a sandy beach,
low angle, foggy morning. single main subject, photorealistic color photo, natural colors,
realistic texture, sharp focus on subject, no people, no text, no watermark, no logo,
no border, no collage, no CGI, no illustration.
```

**Hard tier:**

```
a finely detailed photo of a bullfrog near the ground in a gravel lot, side profile,
nighttime under streetlights. partially occluded by a foreground object. single main
subject, photorealistic color photo, natural colors, realistic texture, sharp focus on
subject, no people, no text, no watermark, no logo, no border, no collage, no CGI,
no illustration.
```

### Prompt Row Schema (JSONL/CSV)

| Field | Description |
|-------|-------------|
| `id` | Unique integer |
| `class_label` | One of the 10 CIFAR-10 classes |
| `instance` | Subtype within the class |
| `action` | Pose or activity |
| `scene` | Background setting |
| `view` | Camera angle |
| `lighting` | Illumination condition |
| `tier` | `"clean"` or `"hard"` |
| `prompt_opener` | Opening phrase |
| `prompt` | Full assembled prompt |
| `negative_prompt` | Global negative prompt string |

---

## Setup and Dependencies

### Environment

- **Python** 3.10+
- **CUDA-capable GPU** required for image generation and model benchmarking (Minimum 96 GB VRAM)

### Core Dependencies

```
torch
torchvision
transformers>=4.40.0
diffusers>=0.30.0
huggingface_hub>=0.24.0
accelerate
safetensors
pillow
tqdm
```

### Image Generation (additional)

```
bitsandbytes          # 4-bit quantization (FLUX.2, HunyuanImage-3)
einops                # HunyuanImage-3
sentencepiece
protobuf
```

### Evaluation and Benchmarking (additional)

```
torch-fidelity        # FID, IS, KID metrics
timm>=0.9.0
scikit-learn
pandas
matplotlib
seaborn
torchinfo
fvcore
gdown                 # dataset download
```

### Hugging Face Authentication

Some models are gated. Authenticate before running image generation:

```bash
huggingface-cli login
```

Or in a notebook:

```python
from huggingface_hub import login
login()  # paste your HF token when prompted
```

---

## Usage

### 1. Generate Prompts

Open `Prompt_Generation.ipynb` and run all cells. The notebook generates prompts in-process using `argparse` with explicit args:

```python
args = p.parse_args(['--out', 'prompts_10k.jsonl', '--total', '10000'])
```

Pre-generated prompt files (`prompts_10k.jsonl`, `prompts_10k.csv`, `per_class_prompts/`) are included in the repo.

### 2. Generate Images

1. Unzip per-class prompts if needed: `unzip per_class_prompts.zip`
2. Log in to Hugging Face in the notebook.
3. Open one of the `Image_Generation_*.ipynb` notebooks.
4. Update `GenConfig`:
   - `jsonl_paths`: list of per-class prompt files
   - `out_root`: output directory
   - `samples_per_class`, `batch_size`, `seed`, etc.
5. Run the notebook. Images are saved as PNGs in per-class folders.

### 3. Evaluate Dataset Quality

Open `synth10_fid_inception_metrics_eval.ipynb`. Ensure `SYNTH_ROOT` points to the `cifar32/` folder containing all 60,000 images in class subfolders. The notebook computes FID, IS, KID, Precision, and Recall, then restructures the dataset into train/test splits.

### 4. Run Classification Benchmarks

- **CIFAR-10 baseline:** `cifar10_model_init.ipynb`
- **Four-stage cross-domain study:** `cifar10_benchmark_eval_with_synth10_four_stage_with_results (final).ipynb`
- **Binary real-vs-synthetic detection (Synth10):** `SynthCIFAR_binary_classification_with_results (final) (1).ipynb`
- **Binary real-vs-SDXL detection (single-model):** `SDCIFAR_binary_classification_with_results.ipynb`
- **Synth10 fine-tuned models on SDXL-only test set:** `sdxl_dataset_synth10_eval.ipynb`

The benchmark notebooks download model checkpoints and datasets (SynthCIFAR, SdxlCIFAR, Synth10, sdxl10) from Google Drive automatically via `gdown`.

---

## Hardware Used

| Task | GPU | VRAM |
|------|-----|------|
| Image generation (SD 3.5, FLUX.2) | NVIDIA RTX PRO 6000 Blackwell | 96 GB |
| Image generation (Qwen-Image) | NVIDIA RTX PRO 6000 Blackwell | 96 GB |
| Image generation (HunyuanImage-3) | NVIDIA RTX PRO 6000 Blackwell | 96 GB |
| Evaluation & benchmarks | NVIDIA RTX PRO 6000 Blackwell | 96 GB |

The 4-bit quantized notebooks (FLUX.2, HunyuanImage-3) are designed to reduce VRAM requirements. Reduce `batch_size` or resolution if running on lower-VRAM hardware.

All notebooks are set up to run on cloud GPU environments such as runpod, vast.ai, and others (paths use `!pip`, `!unzip`, etc.). Adjust paths if running locally.

---

## License

See the repository license, if any. Model weights (Stable Diffusion 3.5, FLUX.2, Qwen-Image, HunyuanImage-3, etc.) follow their respective Hugging Face model cards and licenses.
