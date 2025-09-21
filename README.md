## Image Super-Resolution with SRGAN (PyTorch)
PyTorch implementation of single-image super-resolution using SRGAN, combining pixel-wise loss, perceptual loss (VGG19), and adversarial loss. The pipeline trains on RGB images and produces high-resolution reconstructions from low-resolution inputs.

### Key features
- SRGAN architecture with Residual Blocks for realistic super-resolution
- Perceptual loss (VGG19) + pixel loss + adversarial loss for sharper, photorealistic outputs
- Reproducible data preparation pipeline (LR/HR pairs)
- Training script with loss logging, checkpoint saving, and GAN stability techniques
- Inference script to generate and visualize results
---

# Results for 2000 images in dataset, 50 EPOCHS: (Of course, for better results, more examples should be downloaded from the image database)

<img width="1189" height="424" alt="הורדה (1)" src="https://github.com/user-attachments/assets/1686b939-e312-41ba-8d3d-e7479d52ac7d" />


# SRGAN Model – Architecture Overview
```
config.py                # Global configuration (paths, ratios, epochs, batch size...)
data_loader.py           # Downloads  CelebA-HQ-256 subset via Hugging Face Datasets
data_ordering.py         # Builds HR/LR pairs and visualizes samples
dataset_maker.py         # PyTorch Dataset + train/val split utilities
model.py                 # SRCNN model, VGG19 perceptual setup, loss functions
train.py                  # Training loop (saves model and loss plot)
implementation.py        # Inference and visualization on sample image
README.md                # You are here
```
## Generator – GeneratorSRGAN
**Flow:** Input → Initial Conv → Residual Blocks → Post-Residual Conv + skip → Upsample Blocks → Final Conv → Output

1. **Input:** Low-resolution image (LR)
2. **Initial Conv Layer:** `Conv2D` (kernel 9x9) → `PReLU`
3. **Residual Blocks:** sequence of blocks, each containing:
   - `Conv2D` → `BatchNorm` → `PReLU` → `Conv2D` → `BatchNorm`
   - Skip connection: input of the block added to output
4. **Post-Residual Conv:** `Conv2D` → `BatchNorm`  
   - Added skip connection from the output of the initial conv layer
5. **Upsample Blocks:** sequence of blocks, each containing:
   - `Conv2D` → `PixelShuffle` → `PReLU`
   - Increases spatial resolution
6. **Final Conv Layer:** `Conv2D` (kernel 9x9) → outputs high-resolution (HR) image

---

## Discriminator
**Flow:** Input → Conv Blocks → Downsampling → Global Pool → Linear → Sigmoid → Output

1. **Input:** Image (SR or HR)
2. **Conv Blocks:** sequence of convolutional blocks, each containing:
   - `Conv2D` → `BatchNorm` → `LeakyReLU`
   - Some blocks use `stride=2` for downsampling
3. **Global Pooling:** `AdaptiveAvgPool2d` → `Flatten`
4. **Fully Connected Layer:** `Linear` → `Sigmoid` → outputs probability (real/fake)

---

## Perceptual Loss
- Uses pre-trained `VGG19`
- Computes high-level feature differences between SR and HR images
- No gradients are updated in VGG19 during training

---

## Losses
- **Generator:** pixel-wise + perceptual + adversarial
- **Discriminator:** binary classification (real/fake)

---

## Quickstart
### 1) Download dataset
This script downloads `eurecom-ds/celeba-hq-256` via Hugging Face Datasets and saves locally.
```bash
python data_loader.py
```
At the prompt, enter how many images to download (e.g., `1000`). Files are saved to `config.data_dir`.

### 2) Build HR/LR pairs and preview samples
Creates `normal_resolution_images` (HR) and `low_resolution_images` (LR) and saves a small comparison figure.
```bash
python data_ordering.py
```
Outputs:
- Copies HR images into `HR_dir`
- Downscales and upsamples to create LR counterparts in `LR_dir`
- Saves a preview grid as `sample_images.png`

### 3) Train SRGAN
Trains SRGAN with **pixel-wise loss (L1) + perceptual loss (VGG19 features) + adversarial loss**.
```bash
python train.py
```

### 4) Run inference and visualize
Runs sample pair (`000.png`) and saves results.
```bash
python implementation.py
```
Outputs (default):
- `image_res/results/LR_image.png`
- `image_res/results/SRGAN_HR_image.png`
- `image_res/results/Original_image.png`
Also opens a side-by-side visualization window.

---

## How it works

- **Data preparation**: HR images are full-resolution originals. LR images are produced by bicubic downscaling (factor 4) and upsampling back to HR size to simulate low-quality inputs.

- **Model (`model.py`)**: SRGAN architecture with:
  - Generator: initial Conv → Residual Blocks → Post-Residual Conv + skip → Upsample Blocks → final Conv
  - Discriminator: convolutional blocks → downsampling → global pooling → fully connected → probability output

- **Losses**:
  - Pixel-wise loss: `L1Loss`
  - Perceptual loss: MSE between VGG19 feature maps of SR and HR; RGB images are used directly
  - Adversarial loss: binary classification loss from the Discriminator (real/fake)

- **Training (`train.py`)**: Standard PyTorch loop with Adam optimizer. Trains Generator and Discriminator jointly. Saves model weights and loss curves (pixel, perceptual, adversarial).

- **Inference (`implementation.py`)**: Loads trained SRGAN weights and produces super-resolved images from LR input, saving:
  - `LR_image.png` – low-resolution input
  - `SR_HR_image.png` – super-resolved output
  - `Original_image.png` – original HR image 

Also opens a side-by-side visualization window for comparison.
  
---
## References

- Dong et al., *Learning a Deep Convolutional Network for Image Super-Resolution*, ECCV 2014 – original SRCNN paper.
- Ledig et al., *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*, CVPR 2017 – SRGAN, the main inspiration for this project.

