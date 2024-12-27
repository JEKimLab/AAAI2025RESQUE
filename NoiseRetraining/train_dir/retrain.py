import sys, os
import json, pickle
import pickle
import argparse
import warnings
import gc
import random, math
import copy
from datetime import datetime
import time

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from scipy import spatial

warnings.filterwarnings('ignore')
sys.path.append("../helpers")
sys.path.append("../models")

from vgg import VGG, make_layers
from resnet import ResNet, BasicBlock, Bottleneck
from googlenet import GoogleNet
from mobilenet_v2 import MobileNetV2 
from densenet import DenseNet, BottleneckDense
from VisT import ViT

from save_images import save_img
import time
from datasets_for_train import create_train_test_set
from data_classes import class_count, count_layers

from noise import add_gaussian_noise, add_salt_and_pepper, add_gaussian_blur
from noise import add_shot_noise, add_impulse_noise, add_frost

RESULTS_PATH_ROOT = "<RESULTS_PATH>/results_noise"
DATA_SPLITS_ROOT = "<DATA_PATH>/split_dataset"
MODEL_PATH_ROOT = "<MODELS_PATH>/results_clean"

def get_weights_copy(model, save_to):
    weights_path = 'weights_temp.pt'
    path = os.path.join(save_to, weights_path)
    torch.save(model, path)
    return torch.load(path)

def output_to_std_and_file(directory, file_name, data):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name)
    print(data)
    with open(file_path, "a") as file:
        file.write(data)

def he_weights_init(weights):
    if isinstance(weights, nn.Conv2d):
        nn.init.kaiming_normal_(weights.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(weights, nn.Linear):
        nn.init.kaiming_normal_(weights.weight)

class DataTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)


def layer_cosine_sim_l2norm(model, init_model, layer_distance_2_final):
    for (name, param), (name_init, param_init) in zip(model.named_parameters(), init_model.named_parameters()):
        if name not in layer_distance_2_final.keys():
            layer_distance_2_final[name] = []
        if True: #param.grad is not None:
            param_all = torch.flatten(param).cpu().detach().numpy()
            param_all_init = torch.flatten(param_init).cpu().detach().numpy()
            layer_distance_2_final[name].append((np.linalg.norm(param_all - param_all_init))/math.sqrt(param_all.shape[0]))


def save_model(std_string, model, save_to, note=None):
    output_to_std_and_file(save_to, "standard_output.txt", std_string)

    save_model = {
        "model" : model,
        "note" : note
    }
    save_model_details = os.path.join(save_to, "model.pkl")
    with open(save_model_details, "wb") as fp:   
        #pickle.dump(save_model, fp)
        pass

def global_l2_norm(model):
    # Compute the Global L2-norm of gradient
    epoch_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'), norm_type=2.0)
    return epoch_grad_norm.item()

parser = argparse.ArgumentParser()

parser.add_argument('-mp', type=str, help='Model path')
parser.add_argument('-save', type=str, help='Save model and data location')
parser.add_argument('-acc', type=float, help='Validation acc. to rach')
parser.add_argument('-rs', type=int, help='Random seed')
parser.add_argument('-tlc', type=str, help='Transforms, LRs and Cutoffs option')
parser.add_argument('-batch', type=str, help='Slurm Batch mode', default="false")
parser.add_argument('-n_tp', type=str, help='Noise type', default='False')
parser.add_argument('-n_lvl', type=float, help='Noise level', default='None')
parser.add_argument('-opt_bat', type=str, help='Optimizer and batch size', default="adam_32")

args = parser.parse_args()

model_path = os.path.join(MODEL_PATH_ROOT, args.mp, "model.pkl")
save_to = os.path.join(RESULTS_PATH_ROOT, args.save)
std_string = f"Model path: {model_path}\n"
std_string += f"Save results to: {save_to}\n"

val_acc_to_reach = args.acc
random_seed = args.rs
std_string += f"Val accuracy to reach: {val_acc_to_reach}\n"
std_string += f"Random seed: {random_seed}\n"

tlc_option = args.tlc
std_string += f"Transforms, LRs, Cutoffs option: {tlc_option}\n"

slurm_batch_mode = "t" in args.batch.lower()
optimizer_batchsize = args.opt_bat.lower()
std_string += f"Optimizer and batch size: {optimizer_batchsize}\n"

noise_type = args.n_tp
noise_level = args.n_lvl
std_string += f"Noise type: {noise_type}\n"
std_string += f"Noise level: {noise_level}\n"


np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False


output_to_std_and_file(save_to, "standard_output.txt", std_string)

# Options setting

transforms_root = "<PATH>/transforms"
lr_scheds_root = "<PATH>/lr_scheds"
cutoffs_root = "<PATH>/cutoffs"

tlc_option = tlc_option.split("_")

transforms_path = os.path.join(transforms_root, f"t{tlc_option[0]}.json")
lr_scheds_path = os.path.join(lr_scheds_root, f"lr{tlc_option[1]}.json")
cutoffs_path = os.path.join(cutoffs_root, f"cf{tlc_option[2]}.json")


with open(transforms_path, "r") as file:
    transforms_data = file.read()
transforms = json.loads(transforms_data)

with open(lr_scheds_path, "r") as file:
    lr_data = file.read()
lr_data = json.loads(lr_data)

with open(cutoffs_path, "r") as file:
    cutoff_data = file.read()
cutoff_data = json.loads(cutoff_data)

std_string = f"\n\nOptions loaded: \n"
std_string += f"\nTrain transforms:\n{transforms}"
std_string += f"\nLR schdule:\n{lr_data}"
std_string += f"\nCutoff schdule:\n{cutoff_data}"

output_to_std_and_file(save_to, "standard_output.txt", std_string)

## Define transforms 
train_tranforms = transforms["transforms_train"]
train_tranforms = [eval(transform_val) for transform_val in train_tranforms]
transform_train = torchvision.transforms.Compose(train_tranforms)

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32,32)),
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

lr = lr_data["lr"]
weight_decay = lr_data["weight_decay"]
lr_sched = lr_data["lr_schedule"]
lr_schedule  = {int(key): value for key, value in lr_sched.items()}

optimizer_batchsize = optimizer_batchsize.split("_")
optimizer_name = optimizer_batchsize[0]
batch_size = int(optimizer_batchsize[1])

std_string = f"\n\nTraining transforms: {transform_train}\n"
std_string += f"Test transforms: {transform_test}\n"
std_string += f"\nOptimizer: {optimizer_name}\n"
std_string += f"Learning rate: {lr}\n"
std_string += f"Weight decay: {weight_decay}\n"
std_string += f"LR Schedule: {lr_schedule}\n"
std_string += f"Batch size: {batch_size}\n"
std_string += "\n"
output_to_std_and_file(save_to, "standard_output.txt", std_string)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
std_string = f"\nDevice: {device}\n"
output_to_std_and_file(save_to, "standard_output.txt", std_string)

split_root = os.path.join(DATA_SPLITS_ROOT, f"rs{random_seed}")

trainset_main = torch.load(os.path.join(split_root, "split_1.pt"))
trainset_overlap = torch.load(os.path.join(split_root, "split_2.pt"))

trainset = torch.utils.data.ConcatDataset([trainset_main, trainset_overlap])
testset = "<TEST DATA>"

std_string = f"\n\nTrainset size: {len(trainset)}"
std_string += f"\nTestset size: {len(testset)}\n\n"

class_labels, classes = class_count(trainset)
index_positions = [int(len(trainset) * (i + 1) / 6) for i in range(5)]

save_img(trainset, os.path.join(save_to, "clean_imgs"), index_positions)

transform_toTensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
transform_toPIL = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
transform_toTensor_Normalize = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

noise_type = noise_type.lower()
if "blur" in noise_type:
    std_string += "\nGaussian blur"
    noise_function = add_gaussian_blur
elif "sp" in noise_type or "salt_pepper" in noise_type:
    std_string += "\nSalt-and-pepper noise"
    noise_function = add_salt_and_pepper
elif "gauss" in noise_type or "gaussian" in noise_type:
    std_string += "\nGaussian noise"
    noise_function = add_gaussian_noise

trainset_tensor =  DataTransform(trainset, transform=transform_toTensor)
testset_tensor =  DataTransform(testset, transform=transform_toTensor)

std_string += f"\nAdded {noise_type} noise\n\n"
output_to_std_and_file(save_to, "standard_output.txt", std_string)

trainset_noise = [(noise_function(x, noise_level), y) for x, y in trainset_tensor]
testset_noise = [(noise_function(x, noise_level), y) for x, y in testset_tensor]

#To save samples
trainset_PIL = DataTransform(trainset_noise, transform=transform_toPIL)
testset_PIL = DataTransform(testset_noise, transform=transform_toPIL)

trainset_copy = copy.deepcopy(trainset_PIL)
save_img(trainset_copy, os.path.join(save_to, "noise_imgs"), index_positions)

trainset = DataTransform(trainset_PIL, transform=transform_train)
testset = DataTransform(testset_PIL, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

model = pickle.load(open(model_path, "rb"))
model = model["model"]
model.to(device) 
std_string = f"\nFound model !\n"
output_to_std_and_file(save_to, "standard_output.txt", std_string)

criterion = nn.CrossEntropyLoss()

if optimizer_name.lower() == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
train_times = []
grad_norms = []
layer_distance_2_init = {}
layer_distance_2_prev = {}

batch_time = ""

model.eval()
correct_test = 0
with torch.no_grad():
    for x_val, y_val in testloader:
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        val_outputs = model(x_val)
            
        _, predicted_test = val_outputs.max(1)
        correct_test += predicted_test.eq(y_val).sum().item()
raw_test_acc = correct_test/len(testset)

std_string = f"\nRaw test accuracy: {raw_test_acc}\n"
output_to_std_and_file(save_to, "standard_output.txt", std_string)


counter1, counter2 = 0, 0
flag1, flag2  = False, False
cutoff_accdiff_1, cutoff_epochs_1 = cutoff_data["cutoff1"]["acc_diff"], cutoff_data["cutoff1"]["epochs"]
cutoff_accdiff_2, cutoff_epochs_2 = cutoff_data["cutoff2"]["acc_diff"], cutoff_data["cutoff2"]["epochs"]
flagTrue = False
peak_test_acc = 0
peak_test_acc_epoch = 0

model_init = get_weights_copy(model, save_to) 
model_prev = get_weights_copy(model, save_to)

for epoch in range(250):
    #Train
    model.train()
    train_loss = 0
    correct = 0

    if epoch in lr_schedule:
        new_lr = lr_schedule[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    st = time.time()
    for i, (x, y) in enumerate(trainloader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        _, predicted_train = outputs.max(1)
        correct += predicted_train.eq(y).sum().item()
        
        if not slurm_batch_mode:
            print(f"\rBatch: {i+1}/{len(trainloader)}", end="")
    ft = time.time()
    
    train_times.append(ft-st)
            
    train_acc = correct/len(trainset)

    #Global(entire model level) L2-norm of gradient
    grad_norms.append(global_l2_norm(model))

    layer_cosine_sim_l2norm(model, model_init, layer_distance_2_init)
    layer_cosine_sim_l2norm(model, model_prev, layer_distance_2_prev)
    model_prev = get_weights_copy(model, save_to) #["model"]

    #Test
    curr_test_loss = 0.0
    correct_test = 0
    model.eval()
    with torch.no_grad():
        for x_val, y_val in testloader:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            val_outputs = model(x_val)
            curr_test_loss += criterion(val_outputs, y_val).item()
            
            _, predicted_test = val_outputs.max(1)
            correct_test += predicted_test.eq(y_val).sum().item()

    test_acc = correct_test/len(testset)
    
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    test_acc_list.append(test_acc)
    test_loss_list.append(curr_test_loss)

    
    current_time = datetime.now()
    batch_time = "|time:" + current_time.strftime('%H:%M:%S')

    std_string = f"\rEpoch: {epoch+1}|Train Loss: {train_loss:.4f}|Train Acc: {train_acc*100:.3f}|Test Loss: {curr_test_loss:.4f}|Test Accuracy: {test_acc*100:.3f}|lr: {optimizer.param_groups[0]['lr']}{batch_time}"
    output_to_std_and_file(save_to, "standard_output.txt", std_string)

    if test_acc > peak_test_acc:
        peak_test_acc = test_acc
        peak_test_acc_epoch = epoch + 1

    if test_acc >= val_acc_to_reach:
        std_string = "\nValidation accuracy reached. Training ended"
        output_to_std_and_file(save_to, "PASS.txt", "PASS TRUE")
        flagTrue = True
        save_model(std_string, model, save_to)
        break

    if test_acc >= (val_acc_to_reach - cutoff_accdiff_1):
        flag1 = True
    if flag1:
        counter1 += 1
        if counter1 > cutoff_epochs_1:
            std_string = f"\nValidation accuracy reached within {cutoff_accdiff_1} and did not converge after {cutoff_epochs_1} epochs. Training ended. Peak test accuracy: {peak_test_acc}"
            output_to_std_and_file(save_to, "PASS_1.txt", "PASS 1")
            flagTrue = True
            save_model(std_string, model, save_to, cutoff_epochs_1)
            break

    if test_acc >= (val_acc_to_reach - cutoff_accdiff_2):
        flag2 = True
    if flag2:
        counter2 += 1
        if counter2 > cutoff_epochs_2:
            std_string = f"\nValidation accuracy reached within {cutoff_accdiff_2} and did not converge after {cutoff_epochs_2} epochs. Training ended. Peak test accuracy: {peak_test_acc}"
            output_to_std_and_file(save_to, "PASS_2.txt", "PASS 2")
            flagTrue = True
            save_model(std_string, model, save_to, cutoff_epochs_2)
            break


training_hp = {
    "epoch": epoch+1,
    "lr" : lr,
    "weight_decay" : weight_decay,
    "lr_schedule" : lr_schedule,
    "random_seed" : random_seed
}

if flagTrue == False:
    std_string = f"\nModel has not converged. Peak test accuracy: {peak_test_acc}"
    save_model(std_string, model, save_to)

model_stats = {
    "training_hp_info" : training_hp,
    "train_loss" : train_loss_list,
    "train_acc" : train_acc_list,
    "test_loss" : test_loss_list,
    "test_acc" : test_acc_list,
    "train_times" : train_times,
    "peak_test_acc" : peak_test_acc,
    "peak_test_acc_epoch" : peak_test_acc_epoch,
    "raw_test_acc" : raw_test_acc,
    "global_grad_norms" : grad_norms,
    "layer_euc_dist_init" : layer_distance_2_init,
    "layer_euc_dist_prev" : layer_distance_2_prev
}

save_model_details = os.path.join(save_to, "model_stats.pkl")
with open(save_model_details, "wb") as fp:   
    pickle.dump(model_stats, fp)

weights_path = 'weights_temp.pt'
path = os.path.join(save_to, weights_path)
os.unlink(path)
print("\nDone !")