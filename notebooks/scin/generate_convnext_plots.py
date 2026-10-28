import os
import ast
import json
import torch
import timm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

# Setup directory paths
DATA_ROOT = r"C:\workspace\SCIN"
DATASET_DIR = os.path.join(DATA_ROOT, "dataset")
CKPT_PATH = os.path.join(DATA_ROOT, "best_convnext_scin.pth")
FIG_DIR = os.path.join(DATA_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------------------------
# 1. Load Checkpoint and Rebuild Validation Split
# ------------------------------------------------------------------------------
print("Loading ConvNeXt-Base checkpoint...")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
top_10 = ckpt['classes']
class_to_idx = ckpt['class_to_idx']
n_classes = len(top_10)

# Instantiate ConvNeXt and load trained weights
model = timm.create_model("convnext_base", pretrained=False, num_classes=n_classes)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device).eval()

cases_df = pd.read_csv(os.path.join(DATASET_DIR, "scin_cases.csv"))
labels_df = pd.read_csv(os.path.join(DATASET_DIR, "scin_labels.csv"))
df = pd.merge(cases_df, labels_df, on="case_id").dropna(subset=['image_1_path', 'weighted_skin_condition_label'])

def extract_label(val):
    try:
        data = ast.literal_eval(val)
        if isinstance(data, dict) and len(data) > 0:
            return max(data, key=data.get)
    except Exception:
        return None
    return None

df['primary_label'] = df['weighted_skin_condition_label'].apply(extract_label)
df = df.dropna(subset=['primary_label'])
df = df[df['primary_label'].isin(top_10)].copy()
df['target'] = df['primary_label'].map(class_to_idx)

# Exact seed=42 holdout validation split (340 samples)
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index).reset_index(drop=True)

tform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------------------------------------------------------------------
# 2. Extract Validation Probabilities
# ------------------------------------------------------------------------------
print(f"Generating inference predictions for {len(val_df)} validation samples...")
y_true = []
y_probs = []

with torch.no_grad():
    for _, row in val_df.iterrows():
        img_path = os.path.join(DATA_ROOT, os.path.normpath(row['image_1_path']))
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (0, 0, 0))

        inp = tform(img).unsqueeze(0).to(device)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(inp)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        y_probs.append(probs)
        y_true.append(row['target'])

y_true = np.array(y_true)
y_probs = np.array(y_probs)
y_preds = np.argmax(y_probs, axis=1)
y_bin = label_binarize(y_true, classes=range(n_classes))

# ------------------------------------------------------------------------------
# 3. Multi-Class ROC & AUC Curves
# ------------------------------------------------------------------------------
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.figure(figsize=(10, 8), dpi=300)
plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-avg ROC (AUC = {roc_auc["micro"]:.2f})', color='deeppink', linestyle=':', linewidth=3)
plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-avg ROC (AUC = {roc_auc["macro"]:.2f})', color='navy', linestyle=':', linewidth=3)

colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=1.6, label=f'{top_10[i]} ({roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ConvNeXt-Base Multi-Class ROC Curves (SCIN 10-Class Benchmark)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
roc_out = os.path.join(FIG_DIR, "convnext_scin_roc_auc.png")
plt.savefig(roc_out)
plt.close()
print(f"--> Saved: {roc_out}")

# ------------------------------------------------------------------------------
# 4. Normalized Confusion Matrix
# ------------------------------------------------------------------------------
cm = confusion_matrix(y_true, y_preds, normalize='true')
plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Greens', xticklabels=top_10, yticklabels=top_10, cbar=True)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('Ground Truth Label', fontsize=12)
plt.title('Normalized Confusion Matrix (ConvNeXt-Base - SCIN)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
cm_out = os.path.join(FIG_DIR, "convnext_scin_confusion_matrix.png")
plt.savefig(cm_out)
plt.close()
print(f"--> Saved: {cm_out}")

# ------------------------------------------------------------------------------
# 5. Three-Model Comparative F1-Score Chart (ViT vs LG-ViT vs ConvNeXt)
# ------------------------------------------------------------------------------
vit_json = os.path.join(DATA_ROOT, "metrics_summary.json")
lgvit_json = os.path.join(DATA_ROOT, "lgvit_metrics_summary.json")
convnext_json = os.path.join(DATA_ROOT, "convnext_metrics_summary.json")

if os.path.exists(vit_json) and os.path.exists(lgvit_json) and os.path.exists(convnext_json):
    with open(vit_json) as f:
        vit_rep = json.load(f)["detailed_report"]
    with open(lgvit_json) as f:
        lgvit_rep = json.load(f)["detailed_report"]
    with open(convnext_json) as f:
        cnext_rep = json.load(f)["detailed_report"]

    comp_df = pd.DataFrame({
        "Vanilla ViT": [vit_rep[c]["f1-score"] for c in top_10],
        "LG-ViT": [lgvit_rep[c]["f1-score"] for c in top_10],
        "ConvNeXt-Base": [cnext_rep[c]["f1-score"] for c in top_10]
    }, index=top_10)

    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    comp_df.plot(kind="bar", ax=ax, width=0.8, colormap="viridis", edgecolor="black", alpha=0.9)
    plt.title("Comparative Per-Class F1-Score: Vanilla ViT vs. LG-ViT vs. ConvNeXt-Base", fontsize=14, fontweight="bold")
    plt.ylabel("F1-Score", fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    tri_out = os.path.join(FIG_DIR, "scin_3way_benchmark_comparison.png")
    plt.savefig(tri_out)
    plt.close()
    print(f"--> Saved: {tri_out}")

print(f"\nAll plots generated in {FIG_DIR}")