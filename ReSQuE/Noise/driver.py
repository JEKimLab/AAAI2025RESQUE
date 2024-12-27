import gc
import time
import copy
import pickle
import sys, os
import argparse
import warnings
import random, math
import json, pickle
import importlib, inspect
from datetime import datetime

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

import numpy as np
from scipy import spatial
from skimage import metrics

from extractactivations import ExtractLayerActivations

warnings.filterwarnings('ignore')

from copy_model import create_copy, remove_copy
from datasets_for_train import create_train_test_set, DataTransform
from save_train_misc import output_to_std_and_file, save_model, save_img

from noisy_dataset import generate_noisy_data
from noise import add_gaussian_noise, add_shot_noise
from noise import add_salt_and_pepper, add_impulse_noise
from noise import add_gaussian_blur, add_frost, add_frost_TIN

from VisT import ViT
from resnet import ResNet, resnet_cfg
from vgg import VGG, make_layers, vgg_cfg

print("imports done")

##################################################################################################

RESULTS_ROOT_PATH = "<PATH_TO_STORE_RESULTS>"
DATA_SPLITS_ROOT = "<PATH_TO_STORE_DATA_SPLITS>"

##################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-mp', type=str, help='Model path')
parser.add_argument('-rs', type=int, help='Random seed')
parser.add_argument('-save', type=str, help='Save model and data location')
parser.add_argument('-data', type=str, help='Dataset')
parser.add_argument('-n_tp', type=str, help='Noise type')
parser.add_argument('-n_lvl', type=float, help='Noise level')
parser.add_argument('-layer_name', type=str, help='Layer name to extract activations from')

args = parser.parse_args()

model_path = args.mp
random_seed = args.rs
save_to = os.path.join(RESULTS_ROOT_PATH, args.save)
save_temp1 = os.path.join(save_to, "temp1")
save_temp2 = os.path.join(save_to, "temp2")
data = args.data
noise_type = args.n_tp
noise_level = args.n_lvl
layer_name = args.layer_name

np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

std_string = f"Model path: {model_path}\n"
std_string += f"Random seed: {random_seed}\n"
std_string += f"Save results path: {save_to}\n"
std_string += f"Dataset name: {data}\n"
std_string += f"Noise type: {noise_type}\n"
std_string += f"Noise level: {noise_level}\n"
std_string += f"Layer name: {layer_name}\n"

output_to_std_and_file(save_to, "standard_output.txt", std_string)

os.makedirs(save_temp1, exist_ok=True)
os.makedirs(save_temp2, exist_ok=True)

##################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
std_string = f"\nDevice: {device}\n\n"
output_to_std_and_file(save_to, "standard_output.txt", std_string)

##################################################################################################

## Load train splits and test data

transform_toTensor_Normalize = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

split_root = os.path.join(DATA_SPLITS_ROOT, data, f"rs{random_seed}")
trainset_main = torch.load(os.path.join(split_root, "split_0.pt"))
trainset_overlap = torch.load(os.path.join(split_root, "split_2.pt"))
trainset_clean = torch.utils.data.ConcatDataset([trainset_main, trainset_overlap])

trainset_main = torch.load(os.path.join(split_root, "split_1.pt"))
trainset_overlap = torch.load(os.path.join(split_root, "split_2.pt"))
trainset_noise = torch.utils.data.ConcatDataset([trainset_main, trainset_overlap])

std_string = f"\n\nTrainset clean size: {len(trainset_clean)}"
std_string += f"\n\nTrainset noise size: {len(trainset_noise)}"

output_to_std_and_file(save_to, "standard_output.txt", std_string)

trainset_noise, trainset_noise = generate_noisy_data(trainset_noise, trainset_noise, noise_type, noise_level, TinyIMGNET=False)

trainset_clean = DataTransform(trainset_clean, transform=transform_toTensor_Normalize)
trainset_noise = DataTransform(trainset_noise, transform=transform_toTensor_Normalize)

##################################################################################################

## Load the model
model = pickle.load(open(model_path, "rb"))
model = model["model"]
model.to(device) 

output_to_std_and_file(save_to, "standard_output.txt", "\nFound model!!\n")

##################################################################################################

# Create dataloaders for each class
if data.lower() == "cifar10" or data.lower() == "svhn":
    num_classes = 10  
elif data.lower() == "cifar100":
    num_classes = 100

class_angles = []

# Iterate through each class and store the angles
for cls in range(num_classes):
    model_1 = create_copy(model, save_temp1)
    model_2 = create_copy(model, save_temp2)
    model_1.to(device)
    model_2.to(device)

    clean_indices = [i for i, (_, label) in enumerate(trainset_clean) if label == cls]
    noise_indices = [i for i, (_, label) in enumerate(trainset_noise) if label == cls]

    clean_loader = DataLoader(Subset(trainset_clean, clean_indices), batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    noise_loader = DataLoader(Subset(trainset_noise, noise_indices), batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # Extract activations
    clean_activations = ExtractLayerActivations(model_1, device, [layer_name], clean_loader)
    noise_activations = ExtractLayerActivations(model_2, device, [layer_name], noise_loader)

    clean_activations_dict = clean_activations.forward_pass()
    noise_activations_dict = noise_activations.forward_pass()

    # Retrieve activations from the dictionary
    clean_activations = clean_activations_dict[layer_name]
    noise_activations = noise_activations_dict[layer_name]

    # Sum activations across samples (axis 1) to get a vector of shape (output vector size, )
    clean_sum = np.sum(clean_activations, axis=1)
    noise_sum = np.sum(noise_activations, axis=1)

    # Normalize the summed activations
    clean_norm = clean_sum / np.linalg.norm(clean_sum)
    noise_norm = noise_sum / np.linalg.norm(noise_sum)

    # Calculate angle
    angle = np.arccos(np.clip(np.dot(clean_norm.flatten(), noise_norm.flatten()), -1.0, 1.0))
    angle = angle / np.pi
    class_angles.append(angle)

    remove_copy(save_temp1)
    remove_copy(save_temp2)

# Average angles across all classes
average_angle = sum(class_angles) / len(class_angles)

# Save the average angle as a JSON file
with open(os.path.join(save_to, 'average_angle.json'), 'w') as json_file:
    json.dump({'average_angle': average_angle.item()}, json_file)

std_string = f"\n\nAverage angle: {average_angle}\n"
output_to_std_and_file(save_to, "standard_output.txt", std_string)

##################################################################################################
