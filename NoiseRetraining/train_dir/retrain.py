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
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torchvision.transforms as transforms

import numpy as np
from scipy import spatial
from skimage import metrics

import eco2ai
from eco2ai import track
from codecarbon import track_emissions
from carbontracker.tracker import CarbonTracker
from carbontracker import parser as carbon_logs_parser

warnings.filterwarnings('ignore')

from train_test import train, test
from data_classes import class_count
from he_weights import he_weights_init

from create_model_object import create_model
from copy_model import create_copy, remove_copy

from gradnorm_paramchange import grad_norm, layer_param_change

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
CONFIGURATIONS_ROOT_PATH = "<PATH_TO_STORE_CONFIGURATIONS>"
DATA_SPLITS_ROOT = "<PATH_TO_STORE_DATASPLITS>"

##################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('-mp', type=str, help='Model path')
parser.add_argument('-acc', type=float, help='Validation acc. to reach')
parser.add_argument('-rs', type=int, help='Random seed')
parser.add_argument('-save', type=str, help='Save model and data location')
parser.add_argument('-data', type=str, help='Dataset')
parser.add_argument('-n_tp', type=str, help='Noise type')
parser.add_argument('-n_lvl', type=float, help='Noise level')
parser.add_argument('-L_T_C', type=str, help='Learning rate plan, transforms and cutoff')
parser.add_argument('-optim_batch', type=str, help='Optimizer and batch size', default="adam_32")
parser.add_argument('-sbatch', type=str, help='Slurm Batch mode', default="false")
parser.add_argument('-tepochs', type=int, help='Total epochs', default=200)

args = parser.parse_args()

model_path = args.mp
val_acc_to_reach = args.acc
random_seed = args.rs
save_to = os.path.join(RESULTS_ROOT_PATH, args.save)
data = args.data
noise_type = args.n_tp
noise_level = args.n_lvl
learning_transforms_cutoff = args.L_T_C
optimizer_batchsize = args.optim_batch
slurm_batch_mode = "t" in args.sbatch.lower()
total_epochs = args.tepochs

np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

std_string = f"Model path: {model_path}\n"
std_string += f"Validation accuracy to reach: {val_acc_to_reach}\n"
std_string += f"Random seed: {random_seed}\n"
std_string += f"Save results path: {save_to}\n"
std_string += f"Dataset name: {data}\n"
std_string += f"Noise type: {noise_type}\n"
std_string += f"Noise level: {noise_level}\n"
std_string += f"Learning rate, transforms and cutoff options: {learning_transforms_cutoff}\n"
std_string += f"Optimizer and batch size string: {optimizer_batchsize}\n"
std_string += f"Slurm batch mode: {slurm_batch_mode}\n"
std_string += f"Total epochs: {total_epochs}\n\n"

output_to_std_and_file(save_to, "standard_output.txt", std_string)

##################################################################################################

## Load all learning rate, transforms, and cutoffs

learning_transforms_cutoff = learning_transforms_cutoff.split("_")
learning_rate_path = os.path.join(CONFIGURATIONS_ROOT_PATH, "LRs", f"l_{learning_transforms_cutoff[0]}.json")

with open(learning_rate_path, "r") as file:
    lr_data = file.read()
lr_data = json.loads(lr_data)

lr = lr_data["lr"]
weight_decay = lr_data["weight_decay"]
lr_sched = lr_data["lr_schedule"]
lr_schedule  = {int(key): value for key, value in lr_sched.items()}

transforms_path = os.path.join(CONFIGURATIONS_ROOT_PATH, "TFs", f"t_{learning_transforms_cutoff[1]}.json")

with open(transforms_path, "r") as file:
    transforms_data = file.read()
transforms_data = json.loads(transforms_data)

train_tranforms, test_tranforms = transforms_data["transforms_train"], transforms_data["transforms_test"]
train_tranforms = [eval(transform_cur) for transform_cur in train_tranforms]
test_tranforms = [eval(transform_cur) for transform_cur in test_tranforms]

transform_train, transform_test = transforms.Compose(train_tranforms), transforms.Compose(test_tranforms)

cutoffs_path = os.path.join(CONFIGURATIONS_ROOT_PATH, "COs", f"c_{learning_transforms_cutoff[2]}.json")

with open(cutoffs_path, "r") as file:
    cutoffs_data = file.read()
cutoff_data = json.loads(cutoffs_data)

std_string = f"\n\nInitial learning rate: {lr}\n"
std_string += f"LR Schedule: {lr_schedule}\n"
std_string += f"Weight decay: {weight_decay}\n"
std_string += f"Training transforms: {transform_train}\n"
std_string += f"Testing transforms: {transform_test}\n\n"
std_string += f"Cutoff points: {cutoffs_data}\n\n"

output_to_std_and_file(save_to, "standard_output.txt", std_string)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
std_string = f"\nDevice: {device}\n\n"
output_to_std_and_file(save_to, "standard_output.txt", std_string)

##################################################################################################

## Load train splits and test data

trainset, testset, dataset = create_train_test_set(data)

split_root = os.path.join(DATA_SPLITS_ROOT, data, f"rs{random_seed}")
trainset_main = torch.load(os.path.join(split_root, "split_1.pt"))
trainset_overlap = torch.load(os.path.join(split_root, "split_2.pt"))
trainset = torch.utils.data.ConcatDataset([trainset_main, trainset_overlap])

if data == "CIFAR10":
    testset = torchvision.datasets.CIFAR10(root='<PATH_TO_DATA_FOR_TEST>', train=False, download=True)
elif data == "CIFAR100":
    testset = torchvision.datasets.CIFAR100(root='<PATH_TO_DATA_FOR_TEST>', train=False, download=True)
elif data == "SVHN":
    testset = torchvision.datasets.SVHN(root='<PATH_TO_DATA_FOR_TEST>', split='test', download=True)

std_string = f"\n\nTrainset size: {len(trainset)}"
std_string += f"\nTestset size: {len(testset)}\n\n"
output_to_std_and_file(save_to, "standard_output.txt", std_string)

index_positions = [int(len(trainset) * (i + 1) / 6) for i in range(5)]
save_img(trainset, save_to, "clean_imgs", index_positions)

## Add noise

trainset_noise, testset_noise = generate_noisy_data(trainset, testset, noise_type, noise_level)
save_img(trainset_noise, save_to, "noise_imgs", index_positions)

trainset_transformed = DataTransform(trainset_noise, transform=transform_train)
testset_transformed = DataTransform(testset_noise, transform=transform_test)

_, classes = class_count(trainset_noise)

std_string = f"\nDataset: {dataset} loaded\n"
std_string += f"Added noise: {noise_type}\n"
std_string += f"With noise level: {noise_level}\n"
std_string += f"Class count: {classes}\n"
std_string += f"Train size: {len(trainset_noise)}\n"
std_string += f"Test size: {len(testset_noise)}\n\n"

output_to_std_and_file(save_to, "standard_output.txt", std_string)
testset_transformed_clean = DataTransform(testset, transform=transform_test)

##################################################################################################

## Load the model
model = pickle.load(open(model_path, "rb"))
model = model["model"]
model.to(device) 
output_to_std_and_file(save_to, "standard_output.txt", "\nFound model!!\n")

##################################################################################################

# Set up optimizers and dataloaders
optimizer_batchsize = optimizer_batchsize.split("_")
optimizer = optimizer_batchsize[0]
batch_size = int(optimizer_batchsize[1])

criterion = nn.CrossEntropyLoss()

if optimizer.lower() == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
elif optimizer.lower() == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

trainloader = DataLoader(trainset_transformed, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
testloader = DataLoader(testset_transformed, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

testloader_clean = DataLoader(testset_transformed_clean, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

##################################################################################################

test_correct_clean, test_loss_clean = test(model, device, testloader_clean, criterion)
test_acc_clean = test_correct_clean/len(testset)
output_to_std_and_file(save_to, "standard_output.txt", f"\nClean data test accuracy: {test_acc_clean*100}%\n")

test_correct_noise, test_loss_noise = test(model, device, testloader, criterion)
test_acc_noise = test_correct_noise/len(testset)

output_to_std_and_file(save_to, "standard_output.txt", f"Noise data test accuracy: {test_acc_noise*100}%\n")

##################################################################################################

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
test_loss_clean_list = []
test_acc_clean_list = []
train_times = []
grad_norms = []
layer_distance_2_init = {}
layer_distance_2_prev = {}
batch_time = ""

model_original = create_copy(model, save_to)
model_original.to(device)

model_prev = create_copy(model, save_to)
model_prev.to(device)

counter1, counter2 = 0, 0
flag1, flag2  = False, False
cutoff_accdiff_1, cutoff_epochs_1 = cutoff_data["cutoff1"]["acc_diff"], cutoff_data["cutoff1"]["epochs"]
cutoff_accdiff_2, cutoff_epochs_2 = cutoff_data["cutoff2"]["acc_diff"], cutoff_data["cutoff2"]["epochs"]
flagTrue = False

##################################################################################################

# Train function which also tracks carbon usage
@track_emissions(project_name="CodeCarbon", output_file=f"{save_to}/codecarbon.csv")
def train(model, device, trainloader, criterion, optimizer, model_prev_cur, slurm_batch_mode):
    model.train()
    train_correct = 0
    train_loss = 0.0
    total_model_loss = 0

    for i, (x, y) in enumerate(trainloader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()
                
        _, predicted_train = outputs.max(1)
        train_correct += predicted_train.eq(y).sum().item()
        
        if not slurm_batch_mode:
            print(f"\rBatch: {i+1}/{len(trainloader)}", end="")

    return train_correct, train_loss

##################################################################################################

# Training iteration loops
for epoch in range(total_epochs):
    if epoch in lr_schedule:
        new_lr = lr_schedule[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    train_correct, train_loss = train(model, device, trainloader, criterion, optimizer, model_prev, slurm_batch_mode)
    train_acc = train_correct/len(trainset)

    #Compute global L2 grad norm
    grad_norms.append(grad_norm(model))

    #Compute L2 parameter change of each layer to original model and previous epoch model
    layer_param_change(model, model_original, layer_distance_2_init)
    layer_param_change(model, model_prev, layer_distance_2_prev)
    model_prev = create_copy(model, save_to)

    test_correct, test_loss = test(model, device, testloader, criterion)
    test_acc = test_correct/len(testset)

    test_correct_clean, test_loss_clean = test(model, device, testloader_clean, criterion)
    test_acc_clean = test_correct_clean/len(testset)

    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    test_acc_list.append(test_acc)
    test_loss_list.append(test_loss)
    test_acc_clean_list.append(test_acc_clean)
    test_loss_clean_list.append(test_loss_clean)

    current_time = datetime.now()
    batch_time = "|time:" + current_time.strftime('%H:%M:%S')

    std_string = f"\rEpoch: {epoch+1}|Train Loss: {train_loss:.2f}|Train Acc: {train_acc*100:.3f}|Test Loss: {test_loss:.2f}|Test Accuracy: {test_acc*100:.3f}|Test Accuracy clean: {test_acc_clean*100:.3f}|lr: {optimizer.param_groups[0]['lr']}{batch_time}"
    output_to_std_and_file(save_to, "standard_output.txt", std_string)

    model_prev = create_copy(model, save_to)
    model_prev.to(device)

    # Stop training if accuracy is reached
    if test_acc >= val_acc_to_reach:
        std_string = "\nValidation accuracy reached. Training ended"
        output_to_std_and_file(save_to, "PASS.txt", "PASS TRUE")
        break

    # Check cutoffs
    if test_acc >= (val_acc_to_reach - cutoff_accdiff_1):
        flag1 = True
    if flag1:
        counter1 += 1
        if counter1 > cutoff_epochs_1:
            std_string = f"\nValidation accuracy reached within {cutoff_accdiff_1} and did not converge after {cutoff_epochs_1} epochs. Training ended"
            output_to_std_and_file(save_to, "PASS_1.txt", "PASS 1")
            flagTrue = True
            save_model(std_string, model, save_to, cutoff_epochs_1)
            break

    if test_acc >= (val_acc_to_reach - cutoff_accdiff_2):
        flag2 = True
    if flag2:
        counter2 += 1
        if counter2 > cutoff_epochs_2:
            std_string = f"\nValidation accuracy reached within {cutoff_accdiff_2} and did not converge after {cutoff_epochs_2} epochs. Training ended"
            output_to_std_and_file(save_to, "PASS_2.txt", "PASS 2")
            flagTrue = True
            save_model(std_string, model, save_to, cutoff_epochs_2)
            break

##################################################################################################

training_hp = {
    "epoch": epoch+1,
    "lr" : lr,
    "weight_decay" : weight_decay,
    "lr_schedule" : lr_schedule,
}

model_stats = {
    "training_hp_info" : training_hp,
    "train_loss" : train_loss_list,
    "train_acc" : train_acc_list,
    "test_loss" : test_loss_list,
    "test_acc" : test_acc_list,
    "test_loss_clean" : test_loss_clean_list,
    "test_acc_clean" : test_acc_clean_list,
    "global_grad_norms" : grad_norms,
    "layer_euc_dist_init" : layer_distance_2_init,
    "layer_euc_dist_prev" : layer_distance_2_prev
}

save_model_details = os.path.join(save_to, "model_stats.pkl")
with open(save_model_details, "wb") as fp:   
    pickle.dump(model_stats, fp)

remove_copy(save_to)
output_to_std_and_file(save_to, "standard_output.txt", "\nDone!\n")

##################################################################################################