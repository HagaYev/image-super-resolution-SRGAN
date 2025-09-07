import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import SRCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model loading
model = SRCNN()
model.load_state_dict(torch.load("srcnn_model.pth", map_location=device))
model.to(device)
model.eval()  # מצב חיזוי

H_img = Image.open(r"normal_resolution_images\000.png").convert("L")
L_img = Image.open(r"low_resolution_images\000.png").convert("L")

transform = transforms.ToTensor()
input_tensor = transform(L_img).unsqueeze(0).to(device)  # 1x1xHxW


with torch.no_grad():
    sr = model(input_tensor, target_size=H_img.size[::-1])  # PIL: (W,H) -> torch: (H,W)

sr_img = transforms.ToPILImage()(sr.squeeze().cpu())

output_dir = r"image_res\results"
os.makedirs(output_dir, exist_ok=True)

L_img.save(os.path.join(output_dir, "LR_image.png"))
sr_img.save(os.path.join(output_dir, "SRCNN_HR_image.png"))
H_img.save(os.path.join(output_dir, "Original_image.png"))
print(H_img.size)
print("All images saved to", output_dir)

plt.figure(figsize=(10,5))

plt.subplot(1, 3, 1)
plt.imshow(L_img, cmap="gray")
plt.axis("off")
plt.title("LR image", fontsize=16)

plt.subplot(1, 3, 2)
plt.imshow(sr_img, cmap="gray")
plt.axis("off")
plt.title("SRCNN HR image", fontsize=16)

plt.subplot(1, 3, 3)
plt.imshow(H_img, cmap="gray")
plt.axis("off")
plt.title("Original image", fontsize=16)

plt.tight_layout()
plt.show()
