import os
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    def __init__(self, images_dir, labels_file, transform=None):
        self.images_dir = images_dir
        self.df = pd.read_csv(labels_file)
        self.transform = transform

        # Only keep rows with available images
        self.image_files = set(os.listdir(images_dir))
        self.df = self.df[self.df["Image Index"].isin(self.image_files)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = row["Image Index"]
        label = row["Finding Labels"]

        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label