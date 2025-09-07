import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import vgg19, VGG19_Weights

from dataset_maker import get_datasets



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VGG19 for perceptual loss
vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(device)
for param in vgg.parameters():
    param.requires_grad = False

train_dataset, val_dataset = get_datasets()

pixel_criterion = nn.L1Loss()
lambda_perceptual = 0.01

def perceptual_loss(sr, hr, vgg_model):
    if sr.shape[1] == 1:
        sr, hr = sr.repeat(1, 3, 1, 1), hr.repeat(1, 3, 1, 1)
    sr_features = vgg_model(sr)
    hr_features = vgg_model(hr)
    return F.mse_loss(sr_features, hr_features)

# SRCNN model
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, target_size=None):
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bicubic', align_corners=False)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x
