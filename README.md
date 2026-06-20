# [ECCV 2026] 3DZip: Spatial-Aware Feature Diversity-Guided Token Compression for 3D Question Answering

**[Changwoo Baek](https://sites.google.com/view/changwoobaek00/%ED%99%88), [Kyeongbo Kong](https://www.pnu-cvsp.com/prof)†** 

🌐 [Project page](https://higokri.github.io/3dzip/) &nbsp;·&nbsp; 📄 arXiv: *coming soon* 

> *We hope 3DZip serves as a useful base codebase for token pruning in 3D VLMs.*

## 📰 News

- **2026-06-20** — 🚀 Code & project page released!
- **2026-06-18** — 🎉 3DZip is accepted to **ECCV 2026**!

## ✨ Highlight

https://github.com/user-attachments/assets/5776e4df-a16b-47c9-ad09-e0f909ecd2c8

---

3DZip is a **training-free**, three-stage token compression framework for projection-based 3D vision-language models.
It removes the redundancy introduced by multi-view aggregation while preserving geometric coherence, retaining
**94.7%** of the original performance with only **128** tokens and a **1.92× faster** inference speed.

> Recent 3D VLMs construct geometry-aware tokens by projecting 2D visual features into world coordinates, generating
> thousands of tokens per scene. Existing 2D token-compression methods rely on semantic relevance or attention, which
> overlook the structured spatial nature of 3D tokens, and object-level imbalance persists even after spatial
> aggregation. 3DZip first applies coarse **voxelization** to remove point-level redundancy, then selects anchor
> tokens by **feature-space diversity** via a Determinantal Point Process (DPP), and finally **merges** the remaining
> tokens under spatial constraints to preserve geometric structure.

## 🧠 3DZip algorithm

The core algorithm is implemented in
[`llava/model/multimodal_encoder/video_encoder.py`](llava/model/multimodal_encoder/video_encoder.py) — see the
`3dzip` pooling branch. It consists of three stages:

1. **Voxelization** — group visual tokens by a 3D voxel grid using `scatter_mean`.
2. **DPP selection** — greedy MAP inference via Cholesky decomposition on a cosine-similarity kernel; selects the `k`
   most diverse anchor tokens.
3. **Cosine merge** — assign remaining voxels to the nearest anchor (cosine distance) and merge by uniform averaging
   with a spatial cutoff.

This code is built on top of [LLaVA-3D](https://github.com/ZCMax/LLaVA-3D). Please refer to the original repository
for full documentation, training scripts, and model details.

## 🛠️ Setup

### Prerequisites

ScanNet data access is required. Please sign the
[ScanNet Terms of Use](http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf) and follow the instructions at
[ScanNet](https://github.com/ScanNet/ScanNet) to obtain the data.

### Environment

```bash
conda create -n 3dzip python=3.10 -y
conda activate 3dzip

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -e .

# torch-scatter (required for voxelization)
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu118.html
```

For the full environment setup, refer to the
[LLaVA-3D installation guide](https://github.com/ZCMax/LLaVA-3D#installation).

### Data

The evaluation benchmarks (ScanQA, SQA3D) are 3D question-answering tasks based on ScanNet scenes. Scene data should
be placed under `playground/data/LLaVA-3D-Pretrain/scannet/`, with each scene folder containing `color/`, `depth/`,
`pose/`, and `intrinsic/` subdirectories. Evaluation annotations should be placed under
`playground/data/annotations/`, derived from
[SQA3D](https://github.com/SilongYong/SQA3D) and [ScanQA](https://github.com/ATR-DBI/ScanQA).

## 🚀 Usage

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POOLING` | Pooling method: `voxelize` (baseline) or `3dzip` (ours) | `voxelize` |
| `VOXEL_SIZE` | Voxel grid size in meters | `0.2` |
| `ADAPTIVE_RATIO` | Target token count (e.g., 32, 64, 128). `0` = use fixed `VOXEL_SIZE` | `0` |
| `MERGE_CUTOFF` | Max voxel-grid distance for merge | `5` |

### Evaluation

Each eval script takes three arguments: `<pooling> <voxel_size> <target_tokens>`.

```bash
# Baseline (LLaVA-3D native)
bash scripts/eval/eval_sqa3d.sh native

# 3DZip with 32 / 64 / 128 target tokens
bash scripts/eval/eval_sqa3d.sh 3dzip 0.2 32
bash scripts/eval/eval_sqa3d.sh 3dzip 0.2 64
bash scripts/eval/eval_sqa3d.sh 3dzip 0.2 128
```

Available benchmarks:

```bash
bash scripts/eval/eval_sqa3d.sh   <pooling> <voxel_size> <target_tokens>   # SQA3D
bash scripts/eval/eval_scanqa.sh  <pooling> <voxel_size> <target_tokens>   # ScanQA
```

## 📁 Repository structure

```
.
├── index.html              # Project page (served at higokri.github.io/3dzip)
├── static/                 # Project-page assets (figures, teaser video)
├── llava/                  # LLaVA-3D + 3DZip implementation
│   └── model/multimodal_encoder/video_encoder.py   # core 3DZip algorithm
├── playground/data/        # Evaluation annotations (ScanQA, SQA3D)
├── scripts/eval/           # Evaluation scripts
└── pyproject.toml
```

## 🙏 Acknowledgements

This code is built upon [LLaVA-3D](https://github.com/ZCMax/LLaVA-3D); we thank the authors for their excellent work.
The project page uses the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template),
adapted from [Nerfies](https://nerfies.github.io).

## 📄 License

This project follows the license of [LLaVA-3D](https://github.com/ZCMax/LLaVA-3D).
