import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import vgg19, VGG19_Weights
from torchvision.utils import save_image
import config
from dataset_maker import get_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VGG19 for perceptual loss
vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
for param in vgg.parameters():
    param.requires_grad = False

def perceptual_loss(sr, hr, vgg_model=vgg):
    sr_features = vgg_model(sr)
    hr_features = vgg_model(hr)
    return F.mse_loss(sr_features, hr_features)

# Generator
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * scale_factor ** 2, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class GeneratorSRGAN(nn.Module):
    def __init__(self, num_res_blocks=16):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 9, 1, 4)
        self.prelu = nn.PReLU()

        res_blocks = [ResidualBlock(64) for _ in range(num_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)

        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.upsample1 = UpsampleBlock(64, 2)
        self.upsample2 = UpsampleBlock(64, 2)

        self.conv3 = nn.Conv2d(64, 3, 9, 1, 4)

    def forward(self, x):
        out1 = self.prelu(self.conv1(x))
        out = self.res_blocks(out1)
        out = self.bn2(self.conv2(out))
        out = out + out1
        out = self.upsample1(out)
        out = self.upsample2(out)
        out = self.conv3(out)
        return out

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        def disc_block(in_f, out_f, stride):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 3, stride, 1),
                nn.BatchNorm2d(out_f),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            disc_block(64, 64, 2),
            disc_block(64, 128, 1),
            disc_block(128, 128, 2),
            disc_block(128, 256, 1),
            disc_block(256, 256, 2),
            disc_block(256, 512, 1),
            disc_block(512, 512, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Training setup
def train_srgan(num_epochs, batch_size, lr_G, lr_D):
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    train_dataset, val_dataset = get_datasets()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    generator = GeneratorSRGAN().to(device)
    discriminator = Discriminator().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D)

    adversarial_criterion = nn.BCELoss()
    pixel_criterion = nn.L1Loss()

    for epoch in range(num_epochs):
        for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            # Train Discriminator
            optimizer_D.zero_grad()
            sr_imgs = generator(lr_imgs)

            real_labels = torch.ones(lr_imgs.size(0), 1, device=device)
            fake_labels = torch.zeros(lr_imgs.size(0), 1, device=device)

            real_loss = adversarial_criterion(discriminator(hr_imgs), real_labels)
            fake_loss = adversarial_criterion(discriminator(sr_imgs.detach()), fake_labels)
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            sr_imgs = generator(lr_imgs)
            perceptual = perceptual_loss(sr_imgs, hr_imgs)
            pixel = pixel_criterion(sr_imgs, hr_imgs)
            g_adv = adversarial_criterion(discriminator(sr_imgs), real_labels)
            g_loss = pixel + lambda_perceptual * perceptual + adversarial weight * g_adv
            g_loss.backward()
            optimizer_G.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")


        # Save SR images and checkpoints each epoch
        sr_imgs = generator(lr_imgs)
        save_image(sr_imgs, f"outputs/sr_epoch{epoch+1}.png")
        torch.save(generator.state_dict(), f"checkpoints/generator_epoch{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"checkpoints/discriminator_epoch{epoch+1}.pth")
        print(f"Saved SR images and checkpoints for epoch {epoch+1}")

    return generator, discriminator

# Run training
if __name__ == "__main__":
    generator, discriminator = train_srgan(
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        lr_G=config.lr_G,
        lr_D=config.lr_D
    )
