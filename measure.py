import math
import os
import torch
from torchvision import transforms
from PIL import Image
import config

DEFAULT_RESULTS_DIR = os.path.join("image_res", "results")
sr_path = os.path.join(DEFAULT_RESULTS_DIR, "SRCNN_HR_image.png")
hr_path = os.path.join(DEFAULT_RESULTS_DIR, "Original_image.png")
def psnr_from_files(sr_path, hr_path):

    sr_img = Image.open(sr_path).convert("L")
    hr_img = Image.open(hr_path).convert("L")

    transform = transforms.ToTensor()
    sr_tensor = transform(sr_img).unsqueeze(0)  # 1x1xHxW
    hr_tensor = transform(hr_img).unsqueeze(0)

    mse = torch.mean((sr_tensor - hr_tensor) ** 2)
    if mse == 0:
        return float('inf'), "Identical images ✅"

    max_pixel = 1.0
    psnr_value = 20 * math.log10(max_pixel / math.sqrt(mse))

    if psnr_value > 30:
        status = "Excellent reconstruction ✅"
    elif psnr_value > 25:
        status = "Good reconstruction 👍"
    elif psnr_value > 20:
        status = "Fair reconstruction ⚠️"
    else:
        status = "Poor reconstruction ❌"

    return psnr_value, status



psnr_val, evaluation = psnr_from_files(sr_path, hr_path)
print(f"PSNR: {psnr_val:.2f} dB, Evaluation: {evaluation}")