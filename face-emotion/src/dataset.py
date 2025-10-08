import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import kagglehub

import kagglehub

def download_fer2013():
    """
    Downloads the FER2013 dataset using kagglehub.
    Returns the path to the dataset files.
    """
    path = kagglehub.dataset_download("msambare/fer2013")
    print("Path to dataset files:", path)
    return path

class FERDataset(Dataset):
    def __init__(self, csv_path=None, image_dir=None, transform=None, usage=None):
        assert csv_path or image_dir, 'Provide csv_path or image_dir'
        self.transform = transform
        self.samples = []
        if csv_path:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"CSV file not found: {csv_path}\n"
                    "Please ensure the file exists at this path or specify the correct path using --csv argument."
                )
            df = pd.read_csv(csv_path)
            if usage:
                df = df[df['Usage'] == usage]
            for _, row in df.iterrows():
                emotion = int(row['emotion'])
                pixels = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8).reshape(48,48)
                self.samples.append(('fer_pixels', emotion, pixels))
        else:
            classes = sorted([d for d in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, d))])
            for cls_idx, cls in enumerate(classes):
                for fname in os.listdir(os.path.join(image_dir, cls)):
                    if fname.lower().endswith(('.png','.jpg','.jpeg')):
                        self.samples.append((os.path.join(image_dir, cls, fname), cls_idx, None))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path_tag, label, pixels = self.samples[idx]
        if path_tag == 'fer_pixels':
            img = Image.fromarray(pixels).convert('RGB')
            img = np.array(img)
        else:
            img = Image.open(path_tag).convert('RGB')
            img = np.array(img)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, int(label)

def get_transforms(train=True, size=224):
    if train:
        return A.Compose([
            A.Resize(size,size),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])
    return A.Compose([A.Resize(size,size), A.Normalize(), ToTensorV2()])
