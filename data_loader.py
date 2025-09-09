from datasets import load_dataset
from PIL import Image
import os
import config

data_dir = config.data_dir
number_of_img_in_data = config.number_of_img_in_data

def main() -> int:
    # Load dataset
    ds = load_dataset("eurecom-ds/celeba-hq-256", split="train")

    # Create target directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Determine how many images to download
    count = number_of_img_in_data
    if count is None:
        try:
            count = int(input("Choose how many images you want in your data: "))
        except Exception:
            count = 200  # default value

    saved = 0
    for i, example in enumerate(ds):
        if i >= count:
            break
        # Convert image to grayscale
        gray_img = example['image'].convert("L")
        # Save image as PNG
        gray_img.save(os.path.join(data_dir, f"{i:03d}.png"))
        saved += 1

    print(f"Saved {saved} images in {data_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
