# EPIC-KITCHENS-100 Multi-Instance Retrieval Challenge 2026

A production-ready, GPU-accelerated research codebase for the [EPIC-KITCHENS-100 Multi-Instance Retrieval Challenge](https://www.codabench.org/competitions/12008/) using PyTorch and modern MLOps practices.

## 1. Challenge Overview

**Task**: Video-text Multi-Instance Retrieval on EPIC-KITCHENS-100.  
Given N video clips and M text captions, retrieve the best matching caption for each video (video→text) and vice versa (text→video). The challenge provides a **soft-label correlation matrix** where each entry encodes semantic similarity between clip-caption pairs.

**Evaluation Metrics**:
- **mAP** (Mean Average Precision) — measures precision at each rank position
- **nDCG** (Normalized Discounted Cumulative Gain) — rewards ranking highly-relevant results at the top
- Both metrics are computed for video→text and text→video, then averaged

**Dataset**: EPIC-KITCHENS-100 — 100 hours, 700 videos, 90K actions, 45 kitchen environments

**Leaderboard**: [Codabench Competition](https://www.codabench.org/competitions/12008/)

## 2. Environment Setup

### Prerequisites
- Python 3.11+
- CUDA 12.1+ compatible GPU (tested on RTX 3090 24GB)
- [UV](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/isis4825-imagenes-vision-202610/epic-kitchens-2026.git
cd epic-kitchens-2026

# Install with UV (recommended)
uv sync

# Install dev dependencies
uv sync --group dev
```

### Core Dependencies
- PyTorch, torchvision, torchaudio (CUDA 12.1)
- transformers, timm, einops
- hydra-core, omegaconf, pydantic
- wandb, tensorboard
- matplotlib, seaborn, plotly

## 3. Data Download

### Step 1: Download Annotations (~50MB)

```bash
python data/download/download_annotations.py
```

### Step 2: Download Videos and RGB frames (~740GB full, subset available)

```bash
# Full dataset (all participants, ~740 GB)
python data/download/download_ek100.py

# Subset (specific participants, much smaller)
python data/download/download_ek100.py --participants P01 P02 P03

# Only RGB frames (skip original videos)
python data/download/download_ek100.py --rgb-frames-only
```

> **Note**: The downloader script automatically clones the official
> [epic-kitchens-download-scripts](https://github.com/epic-kitchens/epic-kitchens-download-scripts)
> repository and calls it on your behalf.  Partial downloads are safe to
> resume by re-running the same command.

### Step 3: Download Pretrained Weights

```bash
python data/download/download_pretrained.py
```

### Step 4: Verify Dataset

```bash
python data/verify.py --root data/raw/EK100
```

### Expected Layout

```
data/raw/EK100/
├── videos/                         # .MP4 files
├── rgb_frames/                     # Pre-extracted RGB frames
└── epic-kitchens-100-annotations/
    ├── EPIC_100_retrieval_train.csv
    ├── EPIC_100_retrieval_test.csv
    ├── EPIC_100_retrieval_train_sentence.csv
    ├── EPIC_100_retrieval_test_sentence.csv
    └── retrieval_annotations/relevancy/
        ├── caption_relevancy_EPIC_100_retrieval_train.pkl
        └── caption_relevancy_EPIC_100_retrieval_test.pkl
```

## 4. Pretrained Weights

```bash
python data/download/download_pretrained.py
```

Download and place weights in `checkpoints/pretrained/`:
| Model | Source | File |
|-------|--------|------|
| AVION ViT-L/14 | [AVION Releases](https://github.com/zhaoyue-zephyrus/AVION/releases) | `avion_vitl14.pt` |
| SMS-Loss (2024) | [SMS-Loss Repo](https://github.com/xqwang14/SMS-Loss) | `sms_loss_ek100.pt` |
| CR-CLIP (2025) | [CR-CLIP Repo](https://github.com/delCayr/ContextRefine-Clip) | `cr_clip_ek100.pt` |

## 5. Training

### Single GPU

```bash
python scripts/train.py model=cr_clip experiment=baseline_crclip
```

### Multi-GPU (4 GPUs)

```bash
torchrun --nproc_per_node=4 scripts/train.py model=cr_clip experiment=baseline_crclip
```

### Key Flags

```bash
python scripts/train.py \
    training.epochs=10 \
    dataset.batch_size=40 \
    optimizer.lr=1.8e-5 \
    loss.tau=0.05 \
    training.amp=true \
    logging.wandb=true
```

### Resume Training

```bash
python scripts/train.py +checkpoint.resume=checkpoints/run_xyz/epoch_5.pt
```

## 6. Evaluation

```bash
# Evaluate on validation split
python scripts/evaluate.py --checkpoint checkpoints/best.pt --split val

# Evaluate on test split with TTA
python scripts/evaluate.py --checkpoint checkpoints/best.pt --split test --tta
```

Output includes:
- Console table with per-metric results (mAP, nDCG at various K)
- JSON results file saved to `experiments/{run}/eval_{split}.json`

## 7. Submission

Generate a Codabench-compatible submission:

```bash
python scripts/export_submission.py --checkpoint checkpoints/best.pt
```

This produces a `submission_{timestamp}.zip` containing `results.pkl` with:
- `v2t`: Video-to-text similarity matrix (N×M, float32)
- `t2v`: Text-to-video similarity matrix (M×N, float32)

Upload to [Codabench](https://www.codabench.org/competitions/12008/).

## 8. Experiment Management

### Configuration System

Uses [Hydra](https://hydra.cc/) with composable YAML configs:

```
configs/
├── base.yaml              # Shared defaults
├── model/                 # Model architectures
│   ├── cr_clip.yaml       # CR-CLIP (2025 winner)
│   ├── sms_loss.yaml      # SMS-Loss (2024 winner)
│   ├── avion.yaml         # AVION backbone
│   └── custom.yaml        # Custom model
├── dataset/
│   └── ek100.yaml
├── optimizer/
│   ├── adamw.yaml
│   └── lars.yaml
├── scheduler/
│   ├── cosine.yaml
│   └── warmup_cosine.yaml
└── experiment/
    ├── baseline_crclip.yaml
    ├── ablation_loss.yaml
    ├── ablation_tta.yaml
    └── sweep_lr.yaml
```

### Hyperparameter Sweeps

```bash
python scripts/train.py --multirun optimizer.lr=1e-5,1.8e-5,3e-5 loss.tau=0.01,0.05,0.1
```

## 9. Baselines

| Model | Year | mAP (%) | nDCG (%) | Paper |
|-------|------|---------|----------|-------|
| EgoVLP | 2022 | — | — | Video-Language Pretraining on Ego4D |
| AVION+LaViLa | 2023 | — | — | Efficient CLIP Pretraining |
| SMS-Loss | 2024 | 63.76 | 74.25 | Symmetric Multi-Similarity Loss |
| **CR-CLIP** | **2025** | **66.78** | **82.08** | Cross-Modal Contextual Refinement |

## 10. Adding a New Model

1. **Implement** `BaseRetriever` in `src/ekmr/models/your_model.py`:
   ```python
   from ekmr.models.base import BaseRetriever

   class YourModel(BaseRetriever):
       def encode_video(self, video: torch.Tensor) -> torch.Tensor: ...
       def encode_text(self, text: dict[str, torch.Tensor]) -> torch.Tensor: ...
   ```

2. **Add config** in `configs/model/your_model.yaml`:
   ```yaml
   _target_: ekmr.models.your_model.YourModel
   embed_dim: 768
   ```

3. **Register** in `src/ekmr/models/__init__.py`:
   ```python
   from ekmr.models.your_model import YourModel
   ```

4. **Run**:
   ```bash
   python scripts/train.py model=your_model
   ```

## 11. Troubleshooting

### Out of Memory (OOM)
- Reduce `dataset.batch_size` (try 20 or 16)
- Enable gradient checkpointing: `model.gradient_checkpointing_enable()`
- Use `bf16` instead of `fp16` on Ampere+ GPUs

### DDP Issues
- Ensure all processes can see the same data directory
- Use `torchrun` instead of `torch.distributed.launch`
- Set `NCCL_DEBUG=INFO` for debugging communication issues

### Slow Data Loading
- Increase `dataset.num_workers` (try 16)
- Enable `dataset.pin_memory=true`
- Pre-extract features: `python data/preprocess/extract_features.py`

## Project Structure

```
├── configs/                  # Hydra configuration files
├── data/                     # Download and preprocessing scripts
├── src/ekmr/                 # Main package
│   ├── datasets/             # Dataset loaders and transforms
│   ├── losses/               # Loss functions (SMS, MIL-NCE, etc.)
│   ├── metrics/              # mAP, nDCG, TTA
│   ├── models/               # Retrieval models and backbones
│   ├── modules/              # Reusable neural network modules
│   ├── training/             # Trainer, evaluator, callbacks
│   ├── utils/                # Checkpoints, logging, seeding
│   └── visualization/        # Embedding, attention, metrics viz
├── scripts/                  # Training, evaluation, export scripts
├── tests/                    # Unit and integration tests
├── experiments/              # Auto-generated run directories
└── checkpoints/              # Saved model weights
```

## Running Tests

```bash
# Run all tests
PYTHONPATH=src pytest tests/ -v

# Run with coverage
PYTHONPATH=src pytest tests/ --cov=src/ekmr --cov-report=term-missing
```

## Key References

- [AVION](https://github.com/zhaoyue-zephyrus/AVION) — Primary backbone
- [CR-CLIP](https://github.com/delCayr/ContextRefine-Clip) — 2025 winner
- [SMS-Loss](https://github.com/xqwang14/SMS-Loss) — 2024 winner
- [EK100 Annotations](https://github.com/epic-kitchens/epic-kitchens-100-annotations)
- [C5 Challenge](https://github.com/epic-kitchens/C5-Multi-Instance-Retrieval)
- [LaViLa](https://github.com/facebookresearch/lavila)
- [EK100 Official Site](https://epic-kitchens.github.io/2025)

## License

This project is for research and educational purposes as part of the ISIS4825 course.
