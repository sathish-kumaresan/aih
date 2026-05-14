# Breast Cancer Detection from Histopathology Images

Patch-level binary classification (benign vs malignant) of H&E-stained breast
tissue microscopy images at 400× magnification, evaluated under a patient-aware
protocol on the BreaKHis 400× subset.

## Models

Three tiers are compared under identical preprocessing, splitting, and evaluation.
Each is driven by a YAML config in `configs/` and run over five seeds.

- **Tier 1** — PFTAS handcrafted texture features + RBF-kernel SVM. The `C` and
  `gamma` hyperparameters are selected by patient-grouped cross-validation. CPU.
- **Tier 2** — EfficientNetV2-S, ImageNet-pretrained, fine-tuned end-to-end. GPU.
- **Tier 3** — EfficientNetV2-S + CBAM (channel + spatial attention). The backbone
  is identical to Tier 2; a single CBAM module applied to the final feature map
  is the only architectural difference, isolating the effect of attention.

## Evaluation protocols

Each model is evaluated under two protocols:

- **honest** — a patient-disjoint train/test split; no patient appears on both
  sides. Tier 2 and Tier 3 additionally carve a patient-grouped validation set
  from the training split for early stopping.
- **provided** — the dataset's original train/test folders, which share patients
  across the split. Reported alongside the honest protocol to quantify the
  optimism introduced by patient leakage.

Sensitivity, specificity, precision, F1, AUROC, balanced accuracy, and the
confusion matrix are reported at both image and patient level, as mean ± std
over five seeds.

## Running

```
python train.py --tier 1   # PFTAS + RBF SVM
python train.py --tier 2   # EfficientNetV2-S
python train.py --tier 3   # EfficientNetV2-S + CBAM
```

Per-seed predictions, metrics, and checkpoints are written to
`results/runs/tier{N}/`; aggregated summaries to `results/reports/`.

## Data

BreaKHis (Spanhol et al., 2016), 400× subset only, under `data/raw/400X/`.
