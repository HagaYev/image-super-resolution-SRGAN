import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import SRCNN, device, pixel_criterion, lambda_perceptual, perceptual_loss, vgg
import config
from dataset_maker import get_datasets

lr= config.lr
num_epochs= config.num_epochs
train_dataset, val_dataset = get_datasets()

if __name__ == "__main__":
    model = SRCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for lr_image, hr_image in train_loader:
            lr_image, hr_image = lr_image.to(device), hr_image.to(device)
            optimizer.zero_grad()
            sr = model(lr_image, target_size=hr_image.shape[2:])
            loss = pixel_criterion(sr, hr_image) + lambda_perceptual * perceptual_loss(sr, hr_image, vgg)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for lr_image, hr_image in val_loader:
                lr_image, hr_image = lr_image.to(device), hr_image.to(device)
                sr = model(lr_image, target_size=hr_image.shape[2:])
                val_loss_total += (pixel_criterion(sr, hr_image) + lambda_perceptual * perceptual_loss(sr, hr_image,                                                                                            vgg)).item()
        val_loss = val_loss_total / len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "srcnn_model.pth")

    # Loss graph plot
    plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Train + Perceptual Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.savefig("graph.png")
    print("graph.png saved !!!")