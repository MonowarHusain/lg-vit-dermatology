import os
import ast
import json
import torch
import torch.nn as nn
import timm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

# Setup paths
DATA_ROOT = r"C:\workspace\SCIN"
DATASET_DIR = os.path.join(DATA_ROOT, "dataset")
CKPT_PATH = os.path.join(DATA_ROOT, "best_lgvit_scin.pth")
FIG_DIR = os.path.join(DATA_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------------------------
# 1. Model Definition (Matches Training Exactly)
# ------------------------------------------------------------------------------
class LaplacianEdgeExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[0,  1, 0],
                               [1, -4, 1],
                               [0,  1, 0]], dtype=torch.float32)
        kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.laplacian = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False)
        self.laplacian.weight.data = kernel
        self.laplacian.weight.requires_grad = False
        self.proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)

    def forward(self, x):
        edges = self.laplacian(x)
        edges = self.proj(edges)
        edges = edges.flatten(2).transpose(1, 2)
        return edges

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, vit_features, edge_features):
        attn_out, _ = self.cross_attn(
            query=self.norm1(vit_features),
            key=self.norm2(edge_features),
            value=self.norm2(edge_features)
        )
        return vit_features + attn_out

class LGViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        self.edge_extractor = LaplacianEdgeExtractor()
        self.cross_attention = CrossAttentionBlock(dim=768, num_heads=8)
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        vit_out = self.vit.forward_features(x)
        vit_patches = vit_out[:, 1:, :]
        edge_patches = self.edge_extractor(x)
        fused_patches = self.cross_attention(vit_patches, edge_patches)
        pooled_features = fused_patches.mean(dim=1)
        return self.head(pooled_features)

# ------------------------------------------------------------------------------
# 2. Load Checkpoint and Reconstruct Validation Split
# ------------------------------------------------------------------------------
print("Loading LG-ViT checkpoint...")
ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
top_10 = ckpt['classes']
class_to_idx = ckpt['class_to_idx']
n_classes = len(top_10)

model = LGViT(num_classes=n_classes)
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

# Exact seed=42 80/20 split
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index).reset_index(drop=True)

tform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------------------------------------------------------------------
# 3. Validation Inference
# ------------------------------------------------------------------------------
print(f"Extracting inference probabilities for {len(val_df)} validation samples...")
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
# 4. Multi-Class ROC & AUC Curves
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
plt.title('LG-ViT Multi-Class ROC Curves (SCIN 10-Class Benchmark)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
roc_out = os.path.join(FIG_DIR, "lgvit_scin_roc_auc.png")
plt.savefig(roc_out)
plt.close()
print(f"--> Saved: {roc_out}")

# ------------------------------------------------------------------------------
# 5. Normalized Confusion Matrix
# ------------------------------------------------------------------------------
cm = confusion_matrix(y_true, y_preds, normalize='true')
plt.figure(figsize=(10, 8), dpi=300)
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=top_10, yticklabels=top_10, cbar=True)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('Ground Truth Label', fontsize=12)
plt.title('Normalized Confusion Matrix (LG-ViT - SCIN)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
cm_out = os.path.join(FIG_DIR, "lgvit_scin_confusion_matrix.png")
plt.savefig(cm_out)
plt.close()
print(f"--> Saved: {cm_out}")

# ------------------------------------------------------------------------------
# 6. Baseline vs LG-ViT Comparative F1-Score Chart
# ------------------------------------------------------------------------------
vanilla_json_path = os.path.join(DATA_ROOT, "metrics_summary.json")
lgvit_json_path = os.path.join(DATA_ROOT, "lgvit_metrics_summary.json")

if os.path.exists(vanilla_json_path) and os.path.exists(lgvit_json_path):
    with open(vanilla_json_path) as f:
        vanilla_data = json.load(f)["detailed_report"]
    with open(lgvit_json_path) as f:
        lgvit_data = json.load(f)["detailed_report"]

    comp_df = pd.DataFrame({
        "Condition": top_10,
        "Vanilla ViT F1": [vanilla_data[c]["f1-score"] for c in top_10],
        "LG-ViT F1": [lgvit_data[c]["f1-score"] for c in top_10]
    }).set_index("Condition")

    plt.figure(figsize=(12, 6), dpi=300)
    comp_df.plot(kind="bar", figsize=(12, 6), colormap="coolwarm", edgecolor="black", alpha=0.9)
    plt.title("Per-Condition F1 Score: Vanilla ViT-Base vs. LG-ViT (SCIN Benchmark)", fontsize=14, fontweight="bold")
    plt.ylabel("F1-Score", fontsize=12)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    comp_out = os.path.join(FIG_DIR, "scin_vit_vs_lgvit_f1.png")
    plt.savefig(comp_out)
    plt.close()
    print(f"--> Saved: {comp_out}")

# ------------------------------------------------------------------------------
# 7. Laplacian Edge Feature Guidance Visualization
# ------------------------------------------------------------------------------
sample_row = val_df.iloc[0]
sample_path = os.path.join(DATA_ROOT, os.path.normpath(sample_row['image_1_path']))
try:
    sample_img = Image.open(sample_path).convert('RGB').resize((224, 224))
    img_tensor = transforms.ToTensor()(sample_img).unsqueeze(0).to(device)
    extractor = LaplacianEdgeExtractor().to(device)
    
    with torch.no_grad():
        lap_features = extractor.laplacian(img_tensor).squeeze(0).cpu().numpy()
        lap_map = np.abs(lap_features).mean(axis=0)
        lap_norm = (lap_map - lap_map.min()) / (lap_map.max() - lap_map.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
    axes[0].imshow(sample_img)
    axes[0].set_title(f"Input ({sample_row['primary_label']})", fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(lap_norm, cmap='inferno')
    axes[1].set_title("Extracted Laplacian Edge Prior", fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    edge_out = os.path.join(FIG_DIR, "scin_laplacian_edge_sample.png")
    plt.savefig(edge_out)
    plt.close()
    print(f"--> Saved: {edge_out}")
except Exception as e:
    print(f"Edge visualizer skipped: {e}")

print(f"\nAll diagnostic plots successfully generated in {FIG_DIR}")