import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import GeneratorSRGAN
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained Generator
generator = GeneratorSRGAN().to(device)
generator.load_state_dict(torch.load("checkpoints/generator_epoch6.pth", map_location=device))  # עדכן לאיזה epoch יש לך
generator.eval()


# Load images
hr_path = os.path.join(config.HR_dir, "000.png")
lr_path = os.path.join(config.LR_dir, "000.png")

H_img = Image.open(hr_path).convert("RGB")
L_img = Image.open(lr_path).convert("RGB")

transform = transforms.ToTensor()
input_tensor = transform(L_img).unsqueeze(0).to(device)  # 1x3xHxW


# Super-resolution
with torch.no_grad():
    sr = generator(input_tensor)

sr_img = transforms.ToPILImage()(sr.squeeze().cpu().clamp(0, 1))

# Save results
output_dir = os.path.join("image_res", "results")
os.makedirs(output_dir, exist_ok=True)

L_img.save(os.path.join(output_dir, "LR_image.png"))
sr_img.save(os.path.join(output_dir, "SRGAN_HR_image.png"))
H_img.save(os.path.join(output_dir, "Original_image.png"))

print("All images saved to", output_dir)

# Show side-by-side
plt.figure(figsize=(12,5))

plt.subplot(1, 3, 1)
plt.imshow(L_img)
plt.axis("off")
plt.title("LR image", fontsize=16)

plt.subplot(1, 3, 2)
plt.imshow(sr_img)
plt.axis("off")
plt.title("SRGAN HR image", fontsize=16)

plt.subplot(1, 3, 3)
plt.imshow(H_img)
plt.axis("off")
plt.title("Original HR image", fontsize=16)

plt.tight_layout()
plt.show()
