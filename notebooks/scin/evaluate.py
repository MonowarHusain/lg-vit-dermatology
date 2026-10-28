import os
import ast
import json
import torch
import timm
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

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
        return image, target

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_root = r"C:\workspace\SCIN"
    dataset_dir = os.path.join(data_root, "dataset")
    checkpoint_path = os.path.join(data_root, "best_scin_nvme.pth")
    metrics_path = os.path.join(data_root, "metrics_summary.json")

    print("Loading saved model checkpoint...")
    best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    top_10 = best_ckpt['classes']
    class_to_idx = best_ckpt['class_to_idx']

    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=len(top_10))
    model.load_state_dict(best_ckpt['model_state_dict'])
    model.to(device)
    model.eval()

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
    df = df[df['primary_label'].isin(top_10)].copy()
    df['target'] = df['primary_label'].map(class_to_idx)

    # Use the same 20% validation split from training
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_loader = DataLoader(
        SCINDataset(val_df, dataset_dir, data_root, val_transform),
        batch_size=32,
        shuffle=False,
        num_workers=2
    )

    print(f"Evaluating {len(val_df)} validation images on {device}...")
    eval_preds = []
    eval_targets = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                preds = model(images).argmax(dim=1)
            eval_preds.extend(preds.cpu().numpy())
            eval_targets.extend(targets.numpy())

    print("\n" + "="*60)
    print("FINAL BENCHMARK CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(eval_targets, eval_preds, target_names=top_10, zero_division=0))

    report = classification_report(eval_targets, eval_preds, target_names=top_10, output_dict=True, zero_division=0)
    summary_data = {
        "best_epoch": best_ckpt['epoch'],
        "best_val_acc": float(best_ckpt['val_acc']),
        "best_macro_f1": float(best_ckpt['val_macro_f1']),
        "classes": top_10,
        "detailed_report": report
    }

    with open(metrics_path, "w") as f:
        json.dump(summary_data, f, indent=4)

    print(f"\nReport saved to: {metrics_path}")

if __name__ == '__main__':
    main()