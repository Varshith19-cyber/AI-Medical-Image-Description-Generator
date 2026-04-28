import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# =========================
# Configuration
# =========================
TRAIN_DIR = "data/routing/train"
VAL_DIR = "data/routing/val"
MODEL_SAVE_PATH = "app/models/router_model.pth"

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.0001
NUM_WORKERS = 0  # Windows safe


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    print("Classes found:", train_dataset.classes)

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

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_accuracy = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        # Train
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            if batch_idx % 10 == 0:
                print(f"Train Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_loss = running_train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        # Validation
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_loader, start=1):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss = running_val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        # Calculate additional metrics
        y_true = []
        y_pred = []
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        print(f"Train Loss     : {train_loss:.4f}")
        print(f"Train Accuracy : {train_acc:.2f}%")
        print(f"Val Loss       : {val_loss:.4f}")
        print(f"Val Accuracy   : {val_acc:.2f}%")
        print(f"Val Precision  : {precision:.4f}")
        print(f"Val Recall     : {recall:.4f}")
        print(f"Val F1-score   : {f1:.4f}")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(best_model_wts, MODEL_SAVE_PATH)
            print(f"Best model saved to: {MODEL_SAVE_PATH}")

    print(f"\nBest Validation Accuracy: {best_val_accuracy:.2f}%")

    # save final best weights again to be safe
    torch.save(best_model_wts, MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()