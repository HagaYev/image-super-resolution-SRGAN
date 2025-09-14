number_of_img_in_data =None

# Dataset paths
HR_dir = "normal_resolution_images"
LR_dir = "low_resolution_images"
data_dir = "celeba-hq-256"

# Training params
train_ratio = 0.9
batch_size = 10
num_epochs = 5

# Learning rates
lr_G = 1e-4  # Generator
lr_D = 1e-4  # Discriminator

# Upscaling
upscale_factor = 4

# Losses
use_vgg_loss = True
lambda_perceptual = 0.01
adversarial weight= 1e-3

# Checkpoints
save_model_interval = 1
