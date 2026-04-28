import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

MODEL_PATH = "app/models/radiology_model_multilabel.pth"
IMAGE_DIR = "data/radiology/images"
LABELS_CSV = "data/radiology/labels.csv"

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(TARGET_FINDINGS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

df = pd.read_csv(LABELS_CSV)
df["Finding Labels"] = df["Finding Labels"].fillna("")

available_images = set(os.listdir(IMAGE_DIR))
df = df[df["Image Index"].isin(available_images)]

df = df.head(500)

y_true = []
all_probs = []

for _, row in df.iterrows():
    image_path = os.path.join(IMAGE_DIR, row["Image Index"])
    labels = row["Finding Labels"]

    true_labels = [1 if disease in labels else 0 for disease in TARGET_FINDINGS]

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    y_true.append(true_labels)
    all_probs.append(probs)

y_true = np.array(y_true)
all_probs = np.array(all_probs)

best_accuracy = 0
best_threshold = 0
best_precision = 0
best_recall = 0
best_f1 = 0

for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    y_pred = (all_probs >= threshold).astype(int)

    hamming_accuracy = (y_true == y_pred).mean()
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if hamming_accuracy > best_accuracy:
        best_accuracy = hamming_accuracy
        best_threshold = threshold
        best_precision = precision
        best_recall = recall
        best_f1 = f1

print("Images Tested:", len(y_true))
print("Best Threshold:", best_threshold)
print("Accuracy (Hamming Accuracy):", round(best_accuracy * 100, 2), "%")
print("Precision:", round(best_precision * 100, 2), "%")
print("Recall:", round(best_recall * 100, 2), "%")
print("F1 Score:", round(best_f1 * 100, 2), "%")