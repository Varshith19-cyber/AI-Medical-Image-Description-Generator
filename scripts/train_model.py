import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

from sklearn.metrics import precision_score, recall_score, f1_score


# =========================
# Configuration
# =========================
DATA_CSV = "data/radiology/labels.csv"
IMAGE_DIR = "data/radiology/images"
MODEL_SAVE_PATH = "app/models/radiology_model_multilabel.pth"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.0001
VAL_SPLIT = 0.2
NUM_WORKERS = 0  # keep 0 for Windows safety

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
# Dataset
# =========================
class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        # Keep only rows whose images actually exist
        self.df = self.df[self.df["Image Index"].apply(
            lambda x: os.path.exists(os.path.join(self.image_dir, x))
        )].reset_index(drop=True)

        # Remove rows with missing labels
        self.df = self.df.dropna(subset=["Finding Labels"]).reset_index(drop=True)

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


# =========================
# Metrics
# =========================
def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return precision, recall, f1


# =========================
# Training
# =========================
def main():
    print("Starting multi-label chest X-ray training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = ChestXrayDataset(DATA_CSV, IMAGE_DIR, transform=transform)

    if len(dataset) == 0:
        print("Error: No valid images found in dataset.")
        return

    print(f"Total valid samples: {len(dataset)}")

    # Train-validation split
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size

    if train_size <= 0 or val_size <= 0:
        print("Error: Dataset too small for train/validation split.")
        return

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Compute pos_weight from training labels only
    train_indices = train_dataset.indices
    train_labels_np = dataset.labels[train_indices]

    positive_counts = train_labels_np.sum(axis=0)
    negative_counts = len(train_labels_np) - positive_counts

    # Avoid division by zero
    pos_weight = np.where(positive_counts > 0, negative_counts / positive_counts, 1.0)
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

    print("Positive counts per class:")
    for finding, count in zip(TARGET_FINDINGS, positive_counts):
        print(f"  {finding}: {int(count)}")

    print("Computed pos_weight:")
    for finding, weight in zip(TARGET_FINDINGS, pos_weight.cpu().numpy()):
        print(f"  {finding}: {weight:.4f}")

    # Model
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(TARGET_FINDINGS))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"Epoch {epoch + 1} | Train Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader, start=1):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).int()

                all_targets.append(labels.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch + 1} | Val Batch {batch_idx}/{len(val_loader)}")

        avg_val_loss = running_val_loss / len(val_loader)

        y_true = np.vstack(all_targets)
        y_pred = np.vstack(all_preds)

        precision, recall, f1 = calculate_metrics(y_true, y_pred)

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss           : {avg_train_loss:.4f}")
        print(f"Validation Loss      : {avg_val_loss:.4f}")
        print(f"Validation Precision : {precision:.4f}")
        print(f"Validation Recall    : {recall:.4f}")
        print(f"Validation F1        : {f1:.4f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Best model saved to: {MODEL_SAVE_PATH}")

    print("\nTraining completed.")
    print(f"Best Validation F1: {best_val_f1:.4f}")


if __name__ == "__main__":
    main()