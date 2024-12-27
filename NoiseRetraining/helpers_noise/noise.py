import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.util import random_noise
from io import BytesIO
from PIL import Image as PILImage
import cv2

# Add salt and pepper noise using intensity parameter
def add_salt_and_pepper(x, var):
    mask1 = torch.rand_like(x[1]) < var/2
    mask2 = torch.rand_like(x[1]) < var/2
    for i in range(3):
        x[i][mask1] = 0
        x[i][mask2] = 1
    return x

# Add gaussian noise using intensity parameter
def add_gaussian_noise(x, var):
    noise = torch.randn_like(x) * var
    x = x + noise
    return x

# Add gaussian blur using intensity parameter
def add_gaussian_blur(x, var):
    x_np = x.permute(1, 2, 0).cpu().numpy()  
    channel_images = [x_np[:, :, i] for i in range(x_np.shape[2])]
    blurred_channels = [gaussian_filter(channel, sigma=var, mode='reflect') for channel in channel_images]
    x_blurred = np.stack(blurred_channels, axis=2)
    x_blurred = torch.tensor(x_blurred, dtype=x.dtype).permute(2, 0, 1) 
    x_blurred = torch.clamp(x_blurred, 0, 1)
    return x_blurred

# Add no noise 
def no_noise(x, var):
    return x