import os
from scripts.dataset import ChestXrayDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

images_dir = os.path.join(BASE_DIR, "data", "radiology", "images")
labels_file = os.path.join(BASE_DIR, "data", "radiology", "labels.csv")

dataset = ChestXrayDataset(images_dir, labels_file)

print("Total samples:", len(dataset))

img, label = dataset[0]

print("Sample label:", label)
print("Image size:", img.size)