from datasets import load_dataset
from PIL import Image
import os
import sys
import config


data_dir= config.data_dir
number_of_img_in_data= config.number_of_img_in_data

def main() -> int:
    ds = load_dataset("eurecom-ds/celeba-hq-256", split="train")

    os.makedirs(data_dir, exist_ok=True)

    # Resolve how many images to download
    if number_of_img_in_data is None:
        try:
            number_of_img_in_data = int(input("Choose how many images you want in your data: "))
            number_of_img_in_data = int(number_of_img_in_data)
        except Exception:
            number_of_img_in_data = 200
    count = number_of_img_in_data

    saved = 0
    for i, example in enumerate(ds):
        if i >= count:
            break
        gray_img = example['image'].convert("L")
        gray_img.save(os.path.join(data_dir, f"{i:03d}.png"))
        saved += 1

    print(f"Saved {saved} images in {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
