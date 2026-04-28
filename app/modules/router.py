import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "app/models/router_model.pth"
CLASS_NAMES = ["chest_xray", "document"]


class RouterModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(CLASS_NAMES))

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Router model file not found: {MODEL_PATH}")

        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()

        predicted_class = CLASS_NAMES[pred_idx]

        if predicted_class == "chest_xray":
            return "radiology"
        else:
            return "document"


router_model = RouterModel()


def detect_image_type(image_path):
    return router_model.predict(image_path)