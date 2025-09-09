## Image Super-Resolution with SRCNN (PyTorch)
PyTorch implementation of single-image super-resolution using SRCNN with an optional perceptual loss (VGG19). The pipeline trains on grayscale faces derived from the `eurecom-ds/celeba-hq-256` dataset and produces higher-resolution reconstructions from low-resolution inputs.

### Key features
- **SRCNN architecture** for super-resolution on grayscale images
- **Perceptual loss (VGG19)** combined with pixel loss for sharper details
- Simple, reproducible **data preparation** pipeline (LR/HR pairs)
- **Training script** with loss logging and checkpoint saving
- **Inference script** to generate and visualize results

---

## Project structure
```
config.py                # Global configuration (paths, ratios, epochs, batch size)
data_loader.py           # Downloads grayscale CelebA-HQ-256 subset via Hugging Face Datasets
data_ordering.py         # Builds HR/LR pairs and visualizes samples
dataset_maker.py         # PyTorch Dataset + train/val split utilities
main.py                  # Training loop (saves model and loss plot)
model.py                 # SRCNN model, VGG19 perceptual setup, loss functions
implementation.py        # Inference and visualization on sample images
measure.py               # (Optional) metrics utilities, if used
README.md                # You are here
srcnn_model.pth          # Saved model weights (created after training)
```

---

## Requirements
- Python 3.9+ recommended
- Windows, Linux, or macOS

Install Python packages:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # choose CUDA/CPU per your setup
pip install pillow matplotlib datasets torchvision
```

Notes:
- If you do not have a compatible GPU/CUDA, install CPU-only PyTorch from the official instructions.
- The first run will download VGG19 weights for perceptual loss and the CelebA-HQ-256 split; ensure internet access.

---

## Configuration
Edit `config.py` to control data directories and training hyperparameters:
- `data_dir`: where downloaded grayscale images are stored (default `celeba_hq_256_subset`)
- `HR_dir`: directory for high-resolution images (default `normal_resolution_images`)
- `LR_dir`: directory for low-resolution images (default `low_resolution_images`)
- `train_ratio`: train/val split ratio (default `0.9`)
- `lr`: optimizer learning rate (default `5e-4`)
- `batch_size`: dataloader batch size (default `16`)
- `num_epochs`: number of training epochs (default `6`)
- `number_of_img_in_data`: optional cap on how many images to use; `None` means use all available

---

## Quickstart
### 1) Download and gray-scale the dataset
This script downloads `eurecom-ds/celeba-hq-256` via Hugging Face Datasets and saves a grayscale subset locally.
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

### 3) Train SRCNN
Trains SRCNN with pixel loss (L1) + perceptual loss (VGG19 features).
```bash
python main.py
```
Outputs:
- `srcnn_model.pth`: trained weights
- `graph.png`: train vs validation loss plot
Console prints epoch-wise losses.

### 4) Run inference and visualize
Runs the saved model on a sample pair (`000.png`) and saves results.
```bash
python implementation.py
```
Outputs (default):
- `image_res/results/LR_image.png`
- `image_res/results/SRCNN_HR_image.png`
- `image_res/results/Original_image.png`
Also opens a side-by-side visualization window.

---

## How it works
- **Data preparation**: HR images are the grayscale originals. LR images are produced by bicubic downscale (factor 4) and upsample back to HR size to simulate low-quality inputs.
- **Model (`model.py`)**: A 4-layer SRCNN with ReLU activations operating on single-channel inputs. The forward pass optionally upsamples the input to the HR target size before convolutional refinement.
- **Loss**:
  - Pixel loss: `L1Loss`
  - Perceptual loss: MSE between VGG19 feature maps of SR and HR; grayscale images are repeated to 3 channels for VGG.
- **Training (`main.py`)**: Standard PyTorch loop with Adam optimizer. Saves weights and a loss curve image.
- **Inference (`implementation.py`)**: Loads weights and produces an SR image from a given LR input, saving visual comparisons.

---

## Tips and troubleshooting
- **VGG19 / dataset downloads**: Requires internet access. If behind a proxy, configure your environment accordingly.
- **GPU usage**: The code autodetects CUDA. Ensure your PyTorch install matches your CUDA driver version.
- **Out-of-memory**: Reduce `batch_size` or the number of training images in `config.py`.
- **Different data**: Replace the images in `HR_dir` and re-run `data_ordering.py` to regenerate LR pairs. Ensure matching filenames across HR/LR.

---

## References
- Dong et al., Learning a Deep Convolutional Network for Image Super-Resolution, ECCV 2014 (SRCNN)
- Ledig et al., Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, CVPR 2017 (SRGAN inspiration)

---

## License
This repository is for research and educational purposes. Check the licenses of datasets and third-party models you use (e.g., VGG19 weights, CelebA-HQ-256).
