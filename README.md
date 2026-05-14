# Breast Cancer Detection from Histopathology Images

Patch-level binary classification (benign vs malignant) of H&E-stained breast tissue microscopy images at 400× magnification.

Three-tier model comparison under patient-aware evaluation:
- Classical baseline: handcrafted texture features (PFTAS) + linear SVM
- Modern CNN: EfficientNetV2-S, fine-tuned from ImageNet
- Attention-augmented CNN: EfficientNetV2-S + CBAM

Dataset: BreaKHis (Spanhol et al., 2016), 400× subset only.
