import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(phase, img_size=224):
    if phase == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                               rotate_limit=30, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def apply_clahe(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img


class DRDataset(Dataset):
    DR_STAGES = {
        0: 'No DR', 1: 'Mild DR', 2: 'Moderate DR',
        3: 'Severe DR', 4: 'Proliferative DR'
    }

    def __init__(self, df, image_dir, transform=None,
                 apply_clahe_flag=False):
        self.df               = df.reset_index(drop=True)
        self.image_dir        = image_dir
        self.transform        = transform
        self.apply_clahe_flag = apply_clahe_flag

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']:
            candidate = os.path.join(self.image_dir,
                                     row['id_code'] + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            raise FileNotFoundError(
                f"No image found for {row['id_code']} in {self.image_dir}")

        if self.apply_clahe_flag:
            image = apply_clahe(img_path)
        else:
            image = np.array(Image.open(img_path).convert('RGB'))

        label = int(row['diagnosis'])

        if self.transform:
            image = self.transform(image=image)['image']

        return image, label


def prepare_dataloaders(csv_path, image_dir, batch_size=8,
                        img_size=224, val_size=0.15,
                        test_size=0.10, seed=42):
    df = pd.read_csv(csv_path)
    df.columns = ['id_code', 'diagnosis']
    df['id_code'] = df['id_code'].astype(str).str.strip()

    train_df, temp_df = train_test_split(
        df, test_size=val_size + test_size,
        stratify=df['diagnosis'], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        stratify=temp_df['diagnosis'], random_state=seed
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_set = DRDataset(train_df, image_dir,
                          get_transforms('train', img_size))
    val_set   = DRDataset(val_df,   image_dir,
                          get_transforms('val',   img_size))
    test_set  = DRDataset(test_df,  image_dir,
                          get_transforms('test',  img_size))

    loaders = {
        'train': DataLoader(train_set, batch_size=batch_size,
                            shuffle=True,  num_workers=0,
                            pin_memory=False),
        'val':   DataLoader(val_set,   batch_size=batch_size,
                            shuffle=False, num_workers=0,
                            pin_memory=False),
        'test':  DataLoader(test_set,  batch_size=batch_size,
                            shuffle=False, num_workers=0,
                            pin_memory=False),
    }
    return loaders, train_df, val_df, test_df