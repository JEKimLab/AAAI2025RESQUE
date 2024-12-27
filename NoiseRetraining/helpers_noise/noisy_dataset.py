import sys, os

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from noise import add_gaussian_noise, add_shot_noise
from noise import add_salt_and_pepper, add_impulse_noise
from noise import add_gaussian_blur, add_frost, add_frost_TIN

# Generate noisy train and test set using the noise type and intensity parameter
def generate_noisy_data(trainset, testset, noise_type, noise_level):
    convert_2_tensor = transforms.Compose([transforms.ToTensor()])
    convert_2_img = transforms.Compose([transforms.ToPILImage()])
    trainset_noise = []
    testset_noise = []

    gauss = ["gauss", "gaussian"]
    salt_pepper = ["sp", "salt_pepper", "saltpepper", "salt-pepper"]
    blur = ["blur", "gaussian_blur", "gaussian blur", "image_blur", "image blur"]

    noise_type = noise_type.lower()

    if noise_type in gauss:
        noise_function = add_gaussian_noise
    elif noise_type in salt_pepper:
        noise_function = add_salt_and_pepper
    elif noise_type in blur:
        noise_function = add_gaussian_blur
    else:
        print(f"Error! Invalid noise type: {noise_type}")
        exit()
        
    for x, y in trainset:
        x = convert_2_tensor(x) 
        x = noise_function(x, noise_level)
        x = convert_2_img(x)
        trainset_noise.append((x, y))

    for x, y in testset:
        x = convert_2_tensor(x) 
        x = noise_function(x, noise_level)
        x = convert_2_img(x)
        testset_noise.append((x, y))

    return trainset_noise, testset_noise