import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

IMAGE_DIR = "data/radiology/images"
LABELS_CSV = "data/radiology/labels.csv"
SAVE_PATH = "app/models/radiology_densenet_multilabel.pth"

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

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001
MAX_IMAGES = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class XrayDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, row["Image Index"])

        image = Image.open(image_path).convert("RGB")

        labels_text = row["Finding Labels"]
        labels = [1 if disease in labels_text else 0 for disease in TARGET_FINDINGS]
        labels = torch.tensor(labels, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels


df = pd.read_csv(LABELS_CSV)
df["Finding Labels"] = df["Finding Labels"].fillna("")

available_images = set(os.listdir(IMAGE_DIR))
df = df[df["Image Index"].isin(available_images)]

df = df.head(MAX_IMAGES)

print("Total images used:", len(df))

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = XrayDataset(train_df, IMAGE_DIR, train_transform)
val_dataset = XrayDataset(val_df, IMAGE_DIR, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Class weights for imbalance
label_counts = np.zeros(len(TARGET_FINDINGS))

for labels_text in train_df["Finding Labels"]:
    for i, disease in enumerate(TARGET_FINDINGS):
        if disease in labels_text:
            label_counts[i] += 1

total = len(train_df)
pos_weight = (total - label_counts) / (label_counts + 1e-6)
pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

print("Class positive counts:", label_counts)
print("Positive weights:", pos_weight)

model = models.densenet121(weights="IMAGENET1K_V1")
model.classifier = nn.Linear(model.classifier.in_features, len(TARGET_FINDINGS))
model = model.to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_f1 = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            preds = (probs >= 0.5).int().cpu().numpy()
            true = labels.int().cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(true)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    hamming_accuracy = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    avg_loss = train_loss / len(train_loader)

    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
    print("Train Loss:", round(avg_loss, 4))
    print("Validation Hamming Accuracy:", round(hamming_accuracy * 100, 2), "%")
    print("Macro Precision:", round(precision * 100, 2), "%")
    print("Macro Recall:", round(recall * 100, 2), "%")
    print("Macro F1 Score:", round(f1 * 100, 2), "%")

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), SAVE_PATH)
        print("Best model saved:", SAVE_PATH)

print("\nTraining completed.")
print("Best Macro F1:", round(best_f1 * 100, 2), "%")