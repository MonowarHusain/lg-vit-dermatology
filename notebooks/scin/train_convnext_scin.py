import os
import ast
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
import timm
from tqdm.auto import tqdm

# ==============================================================================
# CONFIGURATION (C:\workspace\SCIN)
# ==============================================================================
CONFIG = {
    "seed": 42,
    "model_name": "convnext_base",
    "img_size": 224,
    "batch_size": 32,
    "grad_accum_steps": 2,          # Effective batch size = 64
    "num_epochs": 20,               # ~15-20 minutes total runtime on RTX 3070 Ti
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "num_workers": 2,               # Safe & reliable on Windows
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "data_dir": r"C:\workspace\SCIN",
    "checkpoint_path": r"C:\workspace\SCIN\best_convnext_scin.pth",
    "metrics_path": r"C:\workspace\SCIN\convnext_metrics_summary.json",
    "log_path": r"C:\workspace\SCIN\convnext_training.log"
}

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(CONFIG["seed"])

def log_print(msg):
    print(msg)
    with open(CONFIG["log_path"], "a", encoding="utf-8") as f:
        f.write(msg + "\n")

log_print(f"⚡ Device: {CONFIG['device']} | Model: {CONFIG['model_name']}")

# ==============================================================================
# DATASET
# ==============================================================================
class SCINDataset(Dataset):
    def __init__(self, records, root_dir, transform=None):
        self.records = records.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        rel_path = os.path.normpath(row['image_1_path'])
        full_path = os.path.join(self.root_dir, rel_path)
        if not os.path.exists(full_path):
            full_path = os.path.join(self.root_dir, "dataset", rel_path)

        try:
            image = Image.open(full_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (CONFIG["img_size"], CONFIG["img_size"]), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        target = torch.tensor(row['target'], dtype=torch.long)
        return image, target

def prepare_data(config):
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

    top_10 = df['primary_label'].value_counts().nlargest(10).index.tolist()
    class_to_idx = {name: idx for idx, name in enumerate(top_10)}
    df = df[df['primary_label'].isin(top_10)].copy()
    df['target'] = df['primary_label'].map(class_to_idx)

    # Identical 80/20 train/val split (seed=42)
    train_df = df.sample(frac=0.8, random_state=config["seed"])
    val_df = df.drop(train_df.index)

    log_print(f"📊 Partitions: {len(train_df)} Train | {len(val_df)} Val")

    # Inverse square-root class weights
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
        SCINDataset(train_df, config["data_dir"], train_transform),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        SCINDataset(val_df, config["data_dir"], val_transform),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True
    )

    return train_loader, val_loader, top_10, class_to_idx, class_weights

# ==============================================================================
# MAIN TRAINING
# ==============================================================================
def main():
    train_loader, val_loader, top_10, class_to_idx, class_weights = prepare_data(CONFIG)
    
    # Pretrained ConvNeXt backbone
    model = timm.create_model(CONFIG["model_name"], pretrained=True, num_classes=len(top_10))
    model = model.to(CONFIG["device"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"])
    scaler = torch.amp.GradScaler('cuda')

    best_macro_f1 = 0.0

    log_print("\n🚀 Starting ConvNeXt-Base Training on SCIN...")
    for epoch in range(1, CONFIG["num_epochs"] + 1):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{CONFIG['num_epochs']}")
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

        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(CONFIG["device"], non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(images)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_targets.extend(targets.numpy())

        val_acc = (np.array(val_preds) == np.array(val_targets)).mean()
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)

        log_print(f"📈 Epoch {epoch:02d} | Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

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
            log_print(f"  ⭐ Best Model Saved! (Macro F1: {best_macro_f1:.4f})")

    # Evaluate Best Checkpoint
    log_print("\n🔬 Loading Best Model Checkpoint for Final Benchmark Report...")
    ckpt = torch.load(CONFIG["checkpoint_path"], map_location=CONFIG["device"], weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    eval_preds, eval_targets = [], []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(CONFIG["device"], non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(images)
            eval_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            eval_targets.extend(targets.numpy())

    report_str = classification_report(eval_targets, eval_preds, target_names=top_10, digits=4, zero_division=0)
    log_print("\n" + "=" * 60)
    log_print("FINAL CONVNEXT-BASE SCIN BENCHMARK REPORT")
    log_print("=" * 60)
    log_print(report_str)

    report_dict = classification_report(eval_targets, eval_preds, target_names=top_10, output_dict=True, zero_division=0)
    with open(CONFIG["metrics_path"], "w") as f:
        json.dump({
            "model": "ConvNeXt-Base",
            "best_epoch": ckpt['epoch'],
            "best_val_acc": float(ckpt['val_acc']),
            "best_macro_f1": float(ckpt['val_macro_f1']),
            "classes": top_10,
            "detailed_report": report_dict
        }, f, indent=4)

    log_print(f"\n✅ Finished! Best weights saved to: {CONFIG['checkpoint_path']}")
    log_print(f"✅ Metrics summary saved to: {CONFIG['metrics_path']}")

if __name__ == '__main__':
    main()