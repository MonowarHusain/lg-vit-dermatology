import os
import ast
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
import timm
from tqdm.auto import tqdm

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
CONFIG = {
    "seed": 42,
    "img_size": 224,
    "batch_size": 32,
    "grad_accum_steps": 2,        # Effective batch size = 64
    "num_epochs": 15,             # Matches Vanilla ViT 15-epoch benchmark
    "weight_decay": 1e-4,
    "num_workers": 2,             # Stable for Windows dataloader
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "data_dir": r"C:\workspace\SCIN",
    "checkpoint_path": r"C:\workspace\SCIN\best_lgvit_scin.pth",
    "metrics_path": r"C:\workspace\SCIN\lgvit_metrics_summary.json"
}

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(CONFIG["seed"])
print(f"⚡ Device: {CONFIG['device']} | GPUs: {torch.cuda.device_count()}")

# ==============================================================================
# 2. LG-ViT ARCHITECTURE
# ==============================================================================
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
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.edge_extractor = LaplacianEdgeExtractor()
        self.cross_attention = CrossAttentionBlock(dim=768, num_heads=8)
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        vit_out = self.vit.forward_features(x)
        vit_patches = vit_out[:, 1:, :] # Remove CLS token for patch-level edge fusion
        edge_patches = self.edge_extractor(x)
        fused_patches = self.cross_attention(vit_patches, edge_patches)
        pooled_features = fused_patches.mean(dim=1)
        return self.head(pooled_features)

# ==============================================================================
# 3. SCIN DATASET PIPELINE
# ==============================================================================
class SCINDataset(Dataset):
    def __init__(self, records, root_dir, data_root, transform=None):
        self.records = records.reset_index(drop=True)
        self.root_dir = root_dir
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        rel_path = os.path.normpath(row['image_1_path'])
        full_path = os.path.join(self.root_dir, rel_path)
        if not os.path.exists(full_path):
            full_path = os.path.join(self.data_root, rel_path)

        try:
            image = Image.open(full_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (CONFIG["img_size"], CONFIG["img_size"]), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        target = torch.tensor(row['target'], dtype=torch.long)
        return image, target

def prepare_scin_data(config):
    dataset_dir = os.path.join(config["data_dir"], "dataset")
    cases_df = pd.read_csv(os.path.join(dataset_dir, "scin_cases.csv"))
    labels_df = pd.read_csv(os.path.join(dataset_dir, "scin_labels.csv"))
    
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

    # Top-10 Prevalent Classes
    top_10 = df['primary_label'].value_counts().nlargest(10).index.tolist()
    class_to_idx = {name: idx for idx, name in enumerate(top_10)}
    df = df[df['primary_label'].isin(top_10)].copy()
    df['target'] = df['primary_label'].map(class_to_idx)

    print(f"🏷️ 10 SCIN Classes Selected: {top_10}")

    # Reproducible 80/20 train/val split
    train_df = df.sample(frac=0.8, random_state=config["seed"])
    val_df = df.drop(train_df.index)

    print(f"📊 Samples: {len(train_df)} Train | {len(val_df)} Val")

    # Inverse square-root class weights for imbalance mitigation
    class_counts = np.bincount(train_df['target'].values, minlength=len(top_10))
    class_weights = 1.0 / np.sqrt(np.maximum(class_counts, 1))
    class_weights = class_weights / np.sum(class_weights) * len(top_10)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(config["device"])

    train_transform = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((config["img_size"], config["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(
        SCINDataset(train_df, dataset_dir, config["data_dir"], train_transform),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        SCINDataset(val_df, dataset_dir, config["data_dir"], val_transform),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    return train_loader, val_loader, top_10, class_to_idx, class_weights

# ==============================================================================
# 4. TRAINING & EVALUATION LOOP
# ==============================================================================
def main():
    train_loader, val_loader, top_10, class_to_idx, class_weights = prepare_scin_data(CONFIG)
    model = LGViT(num_classes=len(top_10)).to(CONFIG["device"])

    # Differential Learning Rates
    pretrained_params = []
    custom_params = []
    for name, param in model.named_parameters():
        if "vit." in name:
            pretrained_params.append(param)
        else:
            custom_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': pretrained_params, 'lr': 1e-5},
        {'params': custom_params, 'lr': 3e-4}
    ], weight_decay=CONFIG["weight_decay"])

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"])
    scaler = torch.amp.GradScaler('cuda')

    best_macro_f1 = 0.0

    print("\n🚀 Starting LG-ViT Training on SCIN Dataset...")
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{CONFIG['num_epochs']} [Train]")
        
        for step, (images, targets) in enumerate(pbar):
            images = images.to(CONFIG["device"], non_blocking=True)
            targets = targets.to(CONFIG["device"], non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss = loss / CONFIG["grad_accum_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % CONFIG["grad_accum_steps"] == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * CONFIG["grad_accum_steps"] * images.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(targets.cpu().numpy())

            pbar.set_postfix({"Loss": f"{loss.item() * CONFIG['grad_accum_steps']:.4f}"})

        scheduler.step()

        # Validation Phase
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(CONFIG["device"], non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(targets.numpy())

        val_acc = (np.array(val_preds) == np.array(val_targets)).mean()
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)
        
        print(f"📈 Epoch {epoch:02d} | Train Acc: {(np.array(train_preds)==np.array(train_targets)).mean():.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

        # Save Best Model Checkpoint
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_macro_f1': val_f1,
                'classes': top_10,
                'class_to_idx': class_to_idx
            }, CONFIG["checkpoint_path"])
            print(f"  ⭐ Best Checkpoint Saved! (Val Macro F1: {best_macro_f1:.4f})")

    # ==============================================================================
    # 5. FINAL REPORT ON BEST CHECKPOINT
    # ==============================================================================
    print("\n🔬 Loading Best Model for Benchmark Classification Report...")
    best_ckpt = torch.load(CONFIG["checkpoint_path"], map_location=CONFIG["device"], weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()

    eval_preds, eval_targets = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(CONFIG["device"], non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(images)
            eval_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            eval_targets.extend(targets.numpy())

    print("\n" + "=" * 60)
    print("FINAL LG-ViT SCIN BENCHMARK CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(eval_targets, eval_preds, target_names=top_10, digits=4, zero_division=0))

    report_dict = classification_report(eval_targets, eval_preds, target_names=top_10, output_dict=True, zero_division=0)
    summary_data = {
        "model": "LG-ViT (Lesion-Guided Vision Transformer)",
        "best_epoch": best_ckpt['epoch'],
        "best_val_acc": float(best_ckpt['val_acc']),
        "best_macro_f1": float(best_ckpt['val_macro_f1']),
        "classes": top_10,
        "detailed_report": report_dict
    }

    with open(CONFIG["metrics_path"], "w") as f:
        json.dump(summary_data, f, indent=4)

    print(f"✅ Benchmark summary saved to: {CONFIG['metrics_path']}")

if __name__ == '__main__':
    main()