import os
import ast
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
import timm
from sklearn.metrics import f1_score, classification_report

# ==========================================
# 1. Dataset Definition
# ==========================================
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
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        target = torch.tensor(row['target'], dtype=torch.long)
        mst = torch.tensor(row['mst'] if pd.notnull(row['mst']) else -1, dtype=torch.float)
        return image, target, mst


def main():
    # ==========================================
    # 2. Hardware & Configuration
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = r"C:\workspace\SCIN"
    dataset_dir = os.path.join(data_root, "dataset")
    checkpoint_path = os.path.join(data_root, "best_scin_nvme.pth")
    metrics_path = os.path.join(data_root, "metrics_summary.json")

    batch_size = 32
    accumulation_steps = 2   # Effective batch size = 64
    epochs = 15
    lr = 1e-4
    num_workers = 4

    print(f"--> Using compute device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"--> Dataset root path: {dataset_dir}")

    # ==========================================
    # 3. Metadata Loading & Processing
    # ==========================================
    cases_csv = os.path.join(dataset_dir, "scin_cases.csv")
    labels_csv = os.path.join(dataset_dir, "scin_labels.csv")

    if not os.path.exists(cases_csv) or not os.path.exists(labels_csv):
        raise FileNotFoundError(f"Missing CSVs in {dataset_dir}.")

    print("--> Loading SCIN metadata files...")
    cases_df = pd.read_csv(cases_csv)
    labels_df = pd.read_csv(labels_csv)

    df = pd.merge(cases_df, labels_df, on="case_id")
    df = df.dropna(subset=['image_1_path', 'weighted_skin_condition_label'])

    def extract_primary_condition(val):
        try:
            data = ast.literal_eval(val)
            if isinstance(data, dict) and len(data) > 0:
                return max(data, key=data.get)
        except Exception:
            return None
        return None

    df['primary_label'] = df['weighted_skin_condition_label'].apply(extract_primary_condition)
    df = df.dropna(subset=['primary_label'])

    top_10 = df['primary_label'].value_counts().nlargest(10).index.tolist()
    df = df[df['primary_label'].isin(top_10)].copy()

    class_to_idx = {cls_name: i for i, cls_name in enumerate(top_10)}
    df['target'] = df['primary_label'].map(class_to_idx)
    df['mst'] = df['monk_skin_tone_label_us'].fillna(df['monk_skin_tone_label_india'])

    print(f"--> Target classes ({len(top_10)}): {top_10}")
    print(f"--> Total usable samples: {len(df)}")

    # ==========================================
    # 4. Augmentations & DataLoaders
    # ==========================================
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    train_loader = DataLoader(
        SCINDataset(train_df, dataset_dir, data_root, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        SCINDataset(val_df, dataset_dir, data_root, val_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # ==========================================
    # 5. Model Setup
    # ==========================================
    print("--> Initializing ViT-Base model backbone...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=len(top_10))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    # ==========================================
    # 6. Training & Validation Loop
    # ==========================================
    best_val_f1 = 0.0

    print("\n" + "="*50)
    print(f"Beginning Training on NVMe: {epochs} Epochs ({len(train_loader)} batches/epoch)")
    print("="*50)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, (images, targets, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Train]")):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets) / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps

        scheduler.step()

        # Validation Phase
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets, _ in tqdm(val_loader, desc=f"Epoch {epoch:02d}/{epochs:02d} [Val]"):
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(images)
                    preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)

        val_acc = (all_preds == all_targets).mean() * 100
        val_macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        epoch_loss = running_loss / len(train_loader)

        print(f"\n[Result Epoch {epoch:02d}] Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}% | Macro F1: {val_macro_f1:.2f}%")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'classes': top_10,
                'class_to_idx': class_to_idx,
                'val_acc': val_acc,
                'val_macro_f1': val_macro_f1
            }, checkpoint_path)
            print(f"--> Saved best model checkpoint to {checkpoint_path}")

    # ==========================================
    # 7. Final Evaluation & Report Generation
    # ==========================================
    print("\n" + "="*50)
    print("Evaluating Best Model Checkpoint...")
    print("="*50)

    best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.eval()

    eval_preds = []
    eval_targets = []

    with torch.no_grad():
        for images, targets, _ in val_loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                preds = model(images).argmax(dim=1)
            eval_preds.extend(preds.cpu().numpy())
            eval_targets.extend(targets.numpy())

    report = classification_report(
        eval_targets,
        eval_preds,
        target_names=top_10,
        output_dict=True,
        zero_division=0
    )

    print(classification_report(eval_targets, eval_preds, target_names=top_10, zero_division=0))

    summary_data = {
        "best_epoch": best_ckpt['epoch'],
        "best_val_acc": float(best_ckpt['val_acc']),
        "best_macro_f1": float(best_ckpt['val_macro_f1']),
        "classes": top_10,
        "detailed_report": report
    }

    with open(metrics_path, "w") as f:
        json.dump(summary_data, f, indent=4)

    print(f"\n--> Checkpoint stored at: {checkpoint_path}")
    print(f"--> Metrics summary stored at: {metrics_path}")
    print("Benchmark complete!")


if __name__ == '__main__':
    main()