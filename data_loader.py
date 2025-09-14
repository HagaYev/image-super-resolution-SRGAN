from datasets import load_dataset
from PIL import Image
import os
import shutil
import config

# Data directory
original_dir = "celeba-hq-256"
HR_dir = config.HR_dir
LR_dir = config.LR_dir

# Number of downloading images
number_of_img_in_data = config.number_of_img_in_data

def main() -> int:
    ds = load_dataset("eurecom-ds/celeba-hq-256", split="train")

    for d in [original_dir, HR_dir, LR_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    count = number_of_img_in_data

    if count is None:
        try:
            count = int(input("Choose how many images you want in your data: "))
        except Exception:
            count = 50

    saved = 0
    for i, example in enumerate(ds):
        if i >= count:
            break

        img = example['image'].convert("RGB")
        img.save(os.path.join(original_dir, f"{i:03d}.png"))
        img.save(os.path.join(HR_dir, f"{i:03d}.png"))
        lr_img = img.resize((img.width // config.upscale_factor, img.height // config.upscale_factor),Image.BICUBIC)
        lr_img.save(os.path.join(LR_dir, f"{i:03d}.png"))

        saved += 1

    print(f"Saved {saved} images in {original_dir}, {HR_dir}, and {LR_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
