import os
from PIL import Image
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import config

train_ratio= config.train_ratio

class CelebASRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_images = sorted([f for f in os.listdir(hr_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        self.lr_images = sorted([f for f in os.listdir(lr_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        self.transform = transform

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_img = Image.open(os.path.join(self.hr_dir, self.hr_images[idx])).convert("L")
        lr_img = Image.open(os.path.join(self.lr_dir, self.lr_images[idx])).convert("L")

        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)

        return lr_img, hr_img


def get_datasets():
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = CelebASRDataset(config.HR_dir, config.LR_dir, transform=transform)

    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    return train_dataset, val_dataset


if __name__ == "__main__":
    train_dataset, val_dataset = get_datasets()
    print(f"Train size: {len(train_dataset)} | Validation size: {len(val_dataset)}")
