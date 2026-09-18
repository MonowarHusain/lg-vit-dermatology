# Colab Prototyping & Early Ablation Study

This folder contains the initial dual-model exploratory notebook developed on Google Colab before migrating the compute pipeline to Kaggle and local GPU runtimes.

---

### Notebook Information
* **File:** `colab-lgvit-vs-vanillavit-mpox-ham10000.ipynb`
* **Environment:** Google Colab (Free-Tier T4 / High-RAM)
* **Architectures Evaluated:** Vanilla ViT-Base (`vit_b_16`) vs. LG-ViT (Laplacian Dual-Stream)
* **Datasets Covered:** MPox-Vision & HAM10000

---

### Phase Summary & Key Results

#### 1. LG-ViT on MPox-Vision (Initial Run)
* **Validation Accuracy:** 95.00% (later evaluated at 99.00%)
* **Macro F1-Score:** 0.9900
* **Setup:** 80/20 train/validation split
* **Outcome:** Validated initial dual-stream feasibility on viral lesion images.

#### 2. Vanilla ViT Baseline on MPox-Vision (Ablation Verification)
* **Validation Accuracy:** 100.00%
* **Macro F1-Score:** 1.0000
* **Finding:** Attention auditing confirmed **shortcut learning**; the unconstrained model memorized background skin pigmentation and illumination borders rather than morphological features.

#### 3. LG-ViT on MPox-Vision (Stratified 70/15/15 Split)
* **Test Accuracy:** 97.50%
* **Macro F1-Score:** 0.9749
* **Macro ROC-AUC:** 0.9867
* **Outcome:** Reached 1.0000 precision on Monkeypox and Measles; deterministic Laplacian contour guidance successfully anchored attention to lesion edges.

#### 4. LG-ViT on HAM10000 (Stratified Split + Weighting)
* **Test Accuracy:** 83.10%
* **Macro F1-Score:** 0.7211
* **Macro ROC-AUC:** 0.9674
* **Weighted F1-Score:** 0.8400
* **Outcome:** Square-root frequency-weighted loss boosted sensitivity for critical neoplastic malignancies (Melanoma recall: 0.69, Actinic Keratoses recall: 0.67) while maintaining 1.00 precision on vascular lesions.
