import math
import torch
from torchvision import transforms
from PIL import Image

sr_path=r"C:\Users\hagay\PycharmProjects\image_res\results\SRCNN_HR_image.png"
hr_path=r"C:\Users\hagay\PycharmProjects\image_res\results\Original_image.png"
def psnr_from_files(sr_path, hr_path):
    """
    מחשב PSNR בין תמונת SR לבין תמונה מקורית (HR) מהדיסק
    מחזיר גם הערכה של הצלחת המודל
    """
    # טוענים תמונות וממירים ל-Tensor
    sr_img = Image.open(sr_path).convert("L")
    hr_img = Image.open(hr_path).convert("L")

    transform = transforms.ToTensor()
    sr_tensor = transform(sr_img).unsqueeze(0)  # 1x1xHxW
    hr_tensor = transform(hr_img).unsqueeze(0)

    # מחשבים MSE
    mse = torch.mean((sr_tensor - hr_tensor) ** 2)
    if mse == 0:
        return float('inf'), "Identical images ✅"

    max_pixel = 1.0  # ערכים מנורמלים בין 0 ל-1
    psnr_value = 20 * math.log10(max_pixel / math.sqrt(mse))

    # אינדקציה להצלחה
    if psnr_value > 30:
        status = "Excellent reconstruction ✅"
    elif psnr_value > 25:
        status = "Good reconstruction 👍"
    elif psnr_value > 20:
        status = "Fair reconstruction ⚠️"
    else:
        status = "Poor reconstruction ❌"

    return psnr_value, status


# דוגמה לשימוש:
sr_path = r"C:\Users\hagay\PycharmProjects\image_res\results\SRCNN_HR_image.png"
hr_path = r"C:\Users\hagay\PycharmProjects\image_res\results\Original_image.png"

psnr_val, evaluation = psnr_from_files(sr_path, hr_path)
print(f"PSNR: {psnr_val:.2f} dB, Evaluation: {evaluation}")