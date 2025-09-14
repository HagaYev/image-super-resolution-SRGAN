import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from model import train_srgan
import config
from dataset_maker import get_datasets


# Prepare directories
os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Load datasets
train_dataset, val_dataset = get_datasets()
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# Training SRGAN
if __name__ == "__main__":
    generator, discriminator = train_srgan(
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        lr_G=config.lr_G,
        lr_D=config.lr_D
    )

    # Evaluate on validation set (optional)
    generator.eval()
    with torch.no_grad():
        val_losses = []
        for lr_image, hr_image in val_loader:
            lr_image, hr_image = lr_image.to(generator.device), hr_image.to(generator.device)
            sr = generator(lr_image)
            val_losses.append(torch.mean(torch.abs(sr - hr_image)).item())
    if val_losses:
        print(f"Validation pixel loss: {sum(val_losses)/len(val_losses):.4f}")

    print("Training completed. SR images saved in 'outputs/' and checkpoints in 'checkpoints/'")
