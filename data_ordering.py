from PIL import Image
import os
import matplotlib.pyplot as plt
from shutil import copyfile
import shutil
import config

HR_dir = config.HR_dir
LR_dir = config.LR_dir
data_dir = config.data_dir
number_of_img_in_data = config.number_of_img_in_data

# Initiate directories
for d in [HR_dir, LR_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

# Copy all HR images
all_files = sorted([
    f for f in os.listdir(data_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])[:number_of_img_in_data]

for fname in all_files:
    src_path = os.path.join(data_dir, fname)
    dst_path = os.path.join(HR_dir, fname)
    copyfile(src_path, dst_path)

print(f"Copied {len(all_files)} images to {HR_dir}")

# Downscale images function
def image_downscale(input_dir, output_dir, scale_factor):
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        hr_path = os.path.join(input_dir, fname)
        lr_path = os.path.join(output_dir, fname)

        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = hr_img.resize(
            (hr_img.width // scale_factor, hr_img.height // scale_factor),
            Image.BICUBIC
        )
        lr_img.save(lr_path)

image_downscale(HR_dir, LR_dir, scale_factor=4)
print(f"Downscaled images saved to {LR_dir}")

# Five example images
sample_images = sorted(os.listdir(HR_dir))[:5]
plt.figure(figsize=(12, 10))

for i, fname in enumerate(sample_images):
    hr_img = Image.open(os.path.join(HR_dir, fname)).convert("RGB")
    lr_img = Image.open(os.path.join(LR_dir, fname)).convert("RGB")

    plt.subplot(5, 2, 2 * i + 1)
    plt.imshow(hr_img)
    plt.title("HR")
    plt.axis("off")

    plt.subplot(5, 2, 2 * i + 2)
    plt.imshow(lr_img)
    plt.title("LR")
    plt.axis("off")

plt.tight_layout()
plt.savefig("sample_images.png")
print("Saved figure to sample_images.png")
