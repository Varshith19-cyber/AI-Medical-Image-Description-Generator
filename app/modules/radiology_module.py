import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "app/models/radiology_model_multilabel.pth"

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

FINDING_EXPLANATIONS = {
    "Atelectasis": "Possible partial collapse or reduced expansion of a part of the lung is indicated.",
    "Cardiomegaly": "The cardiac silhouette may appear enlarged compared to normal expectations.",
    "Effusion": "There may be fluid accumulation in the pleural space around the lungs.",
    "Infiltration": "There may be abnormal opacities suggesting inflammatory or infectious-type changes.",
    "Mass": "A denser focal area may be present and should be interpreted carefully.",
    "Nodule": "A small rounded opacity may be present in the lung field.",
    "Pneumonia": "The image may show opacity patterns that can be associated with pneumonia-like change.",
    "Pneumothorax": "There may be air in the pleural space causing possible partial lung collapse."
}


class RadiologyModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.resnet18(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, len(TARGET_FINDINGS))

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

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

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        image = image.unsqueeze(0)
        return image.to(self.device)

    def predict(self, image_path):
        image_tensor = self.preprocess_image(image_path)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy()[0]

        indexed_probs = list(zip(TARGET_FINDINGS, probs))
        indexed_probs.sort(key=lambda x: x[1], reverse=True)

        top_finding, top_confidence = indexed_probs[0]
        second_finding, second_confidence = indexed_probs[1]
        third_finding, third_confidence = indexed_probs[2]

        top_findings = [
            {"finding": top_finding, "confidence": float(top_confidence)},
            {"finding": second_finding, "confidence": float(second_confidence)},
            {"finding": third_finding, "confidence": float(third_confidence)},
        ]

        if top_confidence >= 0.60:
            status = "High Confidence"
            description = (
                f"The radiology image shows a stronger indication of {top_finding}. "
                f"{FINDING_EXPLANATIONS[top_finding]} "
                f"Other possible findings include {second_finding} and {third_finding}, "
                f"but with lower confidence."
            )
        elif top_confidence >= 0.35:
            status = "Moderate Confidence"
            description = (
                f"The radiology image suggests {top_finding} as the most likely finding. "
                f"{FINDING_EXPLANATIONS[top_finding]} "
                f"The model also considered {second_finding} and {third_finding} as alternative possibilities."
            )
        else:
            status = "Low Confidence"
            description = (
                f"The radiology image could not be interpreted with strong confidence. "
                f"The most likely finding predicted was {top_finding}, but confidence is low. "
                f"Other possible findings include {second_finding} and {third_finding}. "
                f"This output should be treated only as an academic assistive interpretation."
            )

        return {
            "description": description,
            "top_finding": top_finding,
            "confidence": float(top_confidence),
            "status": status,
            "top_findings": top_findings
        }


radiology_model = RadiologyModel()


def process_radiology(image_path):
    return radiology_model.predict(image_path)