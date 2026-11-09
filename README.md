# Efficient ResNet under 5M Parameters for CIFAR-10

A ResNet-18-style network with **4,941,898 trainable parameters** (limit: 5,000,000), trained from scratch on the 50,000 CIFAR-10 training images.

| Result | Value |
| Trainable parameters | 4,941,898 |
| Validation accuracy (baseline recipe, 5k held-out images, flip TTA) | 97.16% |
| Final submission, Kaggle private leaderboard | **87.1%** (rank 38 of 81) |
| Final submission, Kaggle public leaderboard | 89.1% |

The competition test set (2,000 images) is not the standard CIFAR-10 test split. The gap between validation and test accuracy is discussed in the report.

## Repository contents

| Path | What it is |
| `train.py` | Complete pipeline: data loading from `data.zip`, model, training, evaluation, submission writing. Prints the parameter count and refuses any submittable model above 5M. |
| `predict.py` | Loads a checkpoint and writes the 18-view test-time-augmentation submission. Deterministic given the checkpoint. |
| `run_colab.ipynb` | The Colab notebook with its execution output (parameter count, per-epoch logs, curves, final accuracies). |
| `checkpoints/final_model.pt` | Trained weights of the final model. |
| `runs/<run>/` | `config.json` (all flags), `history.csv` (per-epoch log), `summary.json` (final metrics), `curves.png` for each run. |

The competition data (`data.zip`: `train_images/`, `train_labels.csv`, `test_images/`, `test.csv`) is not included; download it from the Kaggle competition page.

## Architecture

Basic residual block: `out = ReLU(S(x) + F(x))`, with `F = conv3x3 → BN → ReLU → conv3x3 → BN` and `S` = identity or 1×1 conv + BN.

| Layer | Output | Channels | Blocks | Parameters |
| Stem: 3×3 conv, BN, ReLU | 32×32 | 64 | – | 1,856 |
| Stage 1 | 32×32 | 64 | 2 | 147,968 |
| Stage 2 | 16×16 | 128 | 2 | 525,568 |
| Stage 3 | 8×8 | 256 | 2 | 2,099,712 |
| Stage 4 | 4×4 | 368 | 1 | 2,163,104 |
| Global avg pool + FC | 1×1 | 10 | – | 3,690 |
| **Total** | | | **7** | **4,941,898** |

## Reproduce

Requirements: Python 3.10+, PyTorch ≥ 2.1, torchvision ≥ 0.16, numpy, Pillow, matplotlib. A CUDA GPU is strongly recommended (all runs used a Colab A100).

**Exact final submission from the saved weights:**

```bash
python predict.py --model checkpoints/final_model.pt --data data.zip --out submission_tta.csv
```

**Retrain the final model from scratch** (≈43 min on an A100):

```bash
python train.py --data data.zip --val_size 0 --epochs 400 \
    --gray 0.1 --jitter 0.3 --cutmix 0.5 \
    --out runs/full_e400_color_cutmix
```

**Pipeline check** (≈1 min, also runs on CPU): `python train.py --data data.zip --smoke`

All other settings are the script defaults (seed 0, SGD + Nesterov, batch 256, peak LR 0.2, cosine schedule with 5-epoch warm-up, weight decay 5e-4, label smoothing 0.1, EMA 0.999, TrivialAugment, Cutout 16). GPU training is not bit-for-bit deterministic, so retraining gives an equivalent model, not identical predictions.

## All runs

| # | Run folder | Command flags (besides `--data data.zip --val_size 0`) | Public | Private |
| – | `val_e200` | `--val_size 5000 --epochs 200` (validation run) | – | – |
| 1 | `full_e200` | `--epochs 200` | 87.2 | 85.0 |
| 2 | `full_e400` | `--epochs 400` | 88.1 | 86.4 |
| 3 | `full_e200_color` | `--epochs 200 --gray 0.1 --jitter 0.3` | 88.7 | 86.1 |
| 4 | `full_e400_color_cutmix` | `--epochs 400 --gray 0.1 --jitter 0.3 --cutmix 0.5` | 88.8 | 86.5 |
| 5 | `kd_e400` | run 4 flags + `--teachers <4 model.pt files>` (teacher: `--arch wrn --teacher_only`) | 88.5 | 87.1 |
| 6 | run 4 + `predict.py` | 18-view TTA on run 4's model (**final**) | 89.1 | 87.1 |
| 7 | `full_e600_color2_cutmix` | `--epochs 600 --gray 0.2 --jitter 0.4 --cutmix 0.5` | 88.5 | 88.0 |

The WideResNet-28-10 teacher (36.5M parameters) is used only for distillation and is never submitted; `train.py --teacher_only` never writes a submission file.

## References

He et al., Deep residual learning, CVPR 2016 · Zagoruyko & Komodakis, Wide residual networks, BMVC 2016 · Müller & Hutter, TrivialAugment, ICCV 2021 · DeVries & Taylor, Cutout, 2017 · Yun et al., CutMix, ICCV 2019 · Hinton et al., Distilling the knowledge in a neural network, 2015 · Beyer et al., Knowledge distillation: a good teacher is patient and consistent, CVPR 2022. Full list in the report.
