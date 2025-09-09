# image-super-resolution-SRCNN
PyTorch implementation of Image Super-Resolution Using Deep Convolutional Networks

## Overview
This project investigates the **SRCNN** (Super-Resolution Convolutional Neural Network) architecture for **image super-resolution** using portrait images from the [eurecom-ds/celeba-hq-256] dataset.  

To simplify the task and reduce computational complexity, all images are converted to **grayscale**, allowing the model to focus on learning structural details rather than color information.

The main goal is to **reconstruct high-resolution images from low-resolution inputs**, providing a foundation for more advanced super-resolution methods.

## Notes
- This project serves as a **preliminary study** before implementing more complex GAN-based architectures.
- Grayscale preprocessing reduces computational cost and allows faster experimentation.
- The use of Perceptual Loss ensures that the network learns high-level structures and details rather than just minimizing pixel-wise differences.

## Why This Project is Important
- Provides a hands-on exploration of SRCNN on real portrait data.
- Demonstrates how **Transfer Learning** (via VGG19) can improve super-resolution performance.
- Offers a foundation for building **GAN-based super-resolution models**, as proposed in the paper:  
  > *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network*  
  > Christian Ledig, Lucas Theis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi

    <img width="591" height="269" alt="image" src="https://github.com/user-attachments/assets/d309b1e8-b6d8-4343-930e-9b21d29d6b2c" />

---

## Project Description
The project workflow consists of several key steps:

1. **Data Preparation**
   - Original images are downscaled to simulate low-resolution inputs.
   - Both low-resolution and original high-resolution images are used for training.
   - Conversion to grayscale simplifies training and speeds up convergence.

2. **Model Architecture**
   - The core model is **SRCNN**, a convolutional neural network designed for super-resolution.
   - The network learns to map low-resolution images to high-resolution reconstructions.

3. **Custom Loss Function**
   - In addition to the standard **Pixel Loss (MSE)**, the project leverages **Perceptual Loss**.
   - Perceptual Loss is computed using features extracted from a pre-trained **VGG19** network via **Transfer Learning**.
   - This combination ensures that the reconstructed images are not only close in pixel values but also **perceptually similar** to the original images, capturing textures and high-level details.

4. **Training**
   - The model is trained on pairs of low- and high-resolution images.
   - The custom loss guides the optimization to generate more realistic high-resolution images.
   - Evaluation is performed on a separate validation set to monitor progress.

5. **Evaluation and Visualization**
   - After training, low-resolution images are passed through the SRCNN model to produce super-resolved outputs.
   - Results are compared visually and quantitatively against the original high-resolution images.
   - Loss curves and reconstruction examples illustrate the improvement during training.
