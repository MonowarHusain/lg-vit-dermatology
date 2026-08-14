# LG-ViT: Lesion-Guided Vision Transformer

LG-ViT is a dual-stream Vision Transformer that integrates structural edge gradients (1st-order Sobel and 2nd-order Laplacian operators) with semantic RGB features via multi-head cross-attention.

## 📊 Experimental Results Across Benchmarks

### 1. HAM10000 Generalizability Benchmark (10,015 Images, 7 Classes)
*Evaluated under identical validation splits (Seed 42, 10 Epochs):*

| Architecture | Validation Accuracy | Macro F1 | Weighted F1 | BCC Recall | NV Recall |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vanilla ViT (Baseline)** | 86.00% | **0.78** | **0.86** | 0.78 | 0.94 |
| **LG-ViT (Proposed)** | **86.17%** | 0.72 | 0.85 | **0.94 (+16%)** | **0.95 (+1%)** |

### 2. MPox-Vision Benchmark (Bias Mitigation & Explainability)
* **Vanilla ViT Baseline:** 100.0% accuracy (exhibits shortcut learning on background/skin tones).
* **LG-ViT (Proposed):** 99.00% accuracy with attention maps isolated strictly to lesion contours.

## 🏗️ Architecture Overview
1. **Semantic Stream:** `vit_tiny_patch16_224` processing global color and context.
2. **Morphological Stream:** 2nd-order Laplacian differential operator extracting grayscale boundary topographies.
3. **Cross-Attention Fusion:** Edge tokens act as Keys ($K$) and Values ($V$) to guide RGB Queries ($Q$).

---
*Last updated: Saturday, August 15, 2026 at 02:03 AM BDT*
