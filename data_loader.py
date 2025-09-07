from datasets import load_dataset
from PIL import Image
import os
from torch.utils.data import Dataset
import config


data_dir= config.data_dir
number_of_img_in_data= config.number_of_img_in_data

# Data loader
ds = load_dataset("eurecom-ds/celeba-hq-256", split="train")

os.makedirs(data_dir, exist_ok=True)

number_of_img_in_data = int(input("Choose how many images you want in your data: "))
number_of_img_in_data= int(number_of_img_in_data)

for i, example in enumerate(ds):
    if i >= number_of_img_in_data:
        break
    # Convert to grayscale
    gray_img = example['image'].convert("L")
    gray_img.save(os.path.join(data_dir, f"{i:03d}.png"))

print(f"Saved {i} images in {data_dir}")
