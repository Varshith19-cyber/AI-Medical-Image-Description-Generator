import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# Configuration
# =========================
ROUTER_MODEL_PATH = "app/models/router_model.pth"
ROUTER_VAL_DIR = "data/routing/val"

RADIOLOGY_MODEL_PATH = "app/models/radiology_model_multilabel.pth"
RADIOLOGY_CSV = "data/radiology/labels.csv"
RADIOLOGY_IMG_DIR = "data/radiology/images"

IMG_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0

TARGET_FINDINGS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax"
]

# =========================
# Dataset Class for Radiology
# =========================
class ChestXrayDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        # Keep only rows whose images actually exist
        self.df = self.df[self.df["Image Index"].apply(
            lambda x: os.path.exists(os.path.join(self.image_dir, x))
        )].reset_index(drop=True)

        self.labels = []
        for _, row in self.df.iterrows():
            finding_text = str(row["Finding Labels"])
            label_vector = [1 if finding in finding_text else 0 for finding in TARGET_FINDINGS]
            self.labels.append(label_vector)

        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]["Image Index"]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label

def evaluate_router(device):
    print("\n--- Evaluating Routing Model ---")
    if not os.path.exists(ROUTER_MODEL_PATH):
        print(f"Error: Router model not found at {ROUTER_MODEL_PATH}")
        return

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(ROUTER_VAL_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, len(val_dataset.classes))
    model.load_state_dict(torch.load(ROUTER_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    print("\nSample Predictions (Routing):")
    for i in range(min(5, len(all_preds))):
        true_cls = val_dataset.classes[all_labels[i]]
        pred_cls = val_dataset.classes[all_preds[i]]
        print(f"  Example {i+1}: True: {true_cls} | Pred: {pred_cls}")

def evaluate_radiology(device):
    print("\n--- Evaluating Radiology Model ---")
    if not os.path.exists(RADIOLOGY_MODEL_PATH):
        print(f"Error: Radiology model not found at {RADIOLOGY_MODEL_PATH}")
        return

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ChestXrayDataset(RADIOLOGY_CSV, RADIOLOGY_IMG_DIR, transform=transform)
    
    # Use the same seed as in train_model.py for consistency
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = models.resnet18()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(TARGET_FINDINGS))
    model.load_state_dict(torch.load(RADIOLOGY_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).int()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    print(f"Macro F1-score: {f1:.4f}")

    print("\nSample Predictions (Radiology - Multi-label):")
    for i in range(min(3, len(y_pred))):
        true_indices = np.where(y_true[i] == 1)[0]
        pred_indices = np.where(y_pred[i] == 1)[0]
        true_labs = [TARGET_FINDINGS[idx] for idx in true_indices]
        pred_labs = [TARGET_FINDINGS[idx] for idx in pred_indices]
        print(f"  Example {i+1}:")
        print(f"    True: {true_labs if true_labs else 'No Finding'}")
        print(f"    Pred: {pred_labs if pred_labs else 'No Finding'}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    evaluate_router(device)
    evaluate_radiology(device)
