import sys, os
import json
import pickle
import argparse
import random, math
import copy
from datetime import datetime
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Subset
from codecarbon import track_emissions

import warnings
sys.path.append("../helpers")
sys.path.append("../models")

from resnet import ResNet, BasicBlock, Bottleneck
from mobilenet_v2 import MobileNetV2 

from layer_freeze import resize_out_features, freeze_layers
from data_classes import class_count
from VisT import ViT
from datasets_for_train import create_train_test_set


###################################################################################################################
###################################################################################################################


MODEL_PATH_ROOT = "PATH_TO_MODELS_DIR"
RESULTS_PATH_ROOT = "PATH_TO_SAVE_RESULTS"
BEEGFS_SAVE_PATH = "PATH_TO_BEE GFS"
transforms_root = "PATH TO transforms"
lr_scheds_root = "PATH TO lr_scheds"

transform_toTensor_Normalize = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_gray = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=3), torchvision.transforms.Resize((32,32)), 
    torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

#If using ViT B/16, use transforms:
"""
transform_toTensor_Normalize = torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)), 
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
transform_gray = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=3), torchvision.transforms.Resize((224,224)), 
    torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
"""

###################################################################################################################
###################################################################################################################


def output_to_std_and_file(directory, file_name, data):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name)
    print(data)
    with open(file_path, "a") as file:
        file.write(data)

def save_model(model, save_to, note=None):
    save_model = {
        "model" : model,
        "note" : note
    }

    if note is not None:
        save_model_as = f"ep_{note}_model.pkl"
    else:
        save_model_as = f"model.pkl"

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    save_model_details = os.path.join(save_to, save_model_as)
    with open(save_model_details, "wb") as fp:   
        pickle.dump(save_model, fp)

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

def global_l2_norm(model):
    # Compute the Global L2-norm of gradient
    epoch_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'), norm_type=2.0)
    return epoch_grad_norm.item()

def layer_cosine_sim_l2norm(model, init_model, layer_distance_2_final):
    for (name, param), (name_init, param_init) in zip(model.named_parameters(), init_model.named_parameters()):
        if name not in layer_distance_2_final.keys():
            layer_distance_2_final[name] = []
        if True: #param.grad is not None:
            param_all = torch.flatten(param).cpu().detach().numpy()
            param_all_init = torch.flatten(param_init).cpu().detach().numpy()
            layer_distance_2_final[name].append((np.linalg.norm(param_all - param_all_init))/math.sqrt(param_all.shape[0]))


###################################################################################################################
###################################################################################################################


parser = argparse.ArgumentParser()

parser.add_argument('-mp', type=str, help='Model path')
parser.add_argument('-save', type=str, help='Save model and data location')
parser.add_argument('-rs', type=int, help='Random seed')
parser.add_argument('-tl', type=str, help='Transforms and LRsoption')
parser.add_argument('-batch', type=str, help='Slurm Batch mode', default="false")
parser.add_argument('-d', type=str, help='Dataset')
parser.add_argument('-acc', type=float, help='Test accuracy to reach')

args = parser.parse_args()

random_seed = args.rs
model_path = os.path.join(MODEL_PATH_ROOT, args.mp)
save_to = os.path.join(RESULTS_PATH_ROOT, args.save)
save_beegfs = os.path.join(BEEGFS_SAVE_PATH, args.save)
std_string = f"Model path: {model_path}\n"
std_string += f"Save results to: {save_to}\n"
std_string += f"Random seed: {random_seed}\n"

tl_option = args.tl
std_string += f"Transforms, LRs option: {tl_option}\n"

slurm_batch_mode = "t" in args.batch.lower()

data = args.d
test_acc_reach = args.acc
std_string += f"Dataset: {data}\n"
std_string += f"Accuracy to reach: {test_acc_reach}\n"

np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

output_to_std_and_file(save_to, "standard_output.txt", std_string)


###################################################################################################################
###################################################################################################################


tl_option = tl_option.split("_")

transforms_path = os.path.join(transforms_root, f"t{tl_option[0]}.json")
lr_scheds_path = os.path.join(lr_scheds_root, f"lr{tl_option[1]}.json")


with open(transforms_path, "r") as file:
    transforms_data = file.read()
transforms = json.loads(transforms_data)

with open(lr_scheds_path, "r") as file:
    lr_data = file.read()
lr_data = json.loads(lr_data)

std_string = f"\n\nOptions loaded: \n"
std_string += f"\nTrain transforms:\n{transforms}"
std_string += f"\nLR schdule:\n{lr_data}"

output_to_std_and_file(save_to, "standard_output.txt", std_string)


###################################################################################################################
###################################################################################################################


train_tranforms = transforms["transforms_train"]
train_tranforms = [eval(transform_val) for transform_val in train_tranforms]
transform_train = torchvision.transforms.Compose(train_tranforms)

lr = lr_data["lr"]
weight_decay = lr_data["weight_decay"]
total_epochs = lr_data["total_epochs"]
lr_sched = lr_data["lr_schedule"]
lr_schedule  = {int(key): value for key, value in lr_sched.items()}
optimizer_name = "Adam"
batch_size = 32

std_string = f"\n\nTraining transforms: {transform_train}\n"
std_string += f"Model path: {model_path}\n"
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


###################################################################################################################
###################################################################################################################


trainset, testset, dataset = create_train_test_set(data)

labels, classes = class_count(trainset)

dataset += f"\nTrain size: {len(trainset)}"
dataset += f"\nTest size: {len(testset)}"
dataset += f"\nClasses: {classes}"

output_to_std_and_file(save_to, "standard_output.txt", dataset)

trainset_copy = copy.deepcopy(trainset)

trainset_copy = DataTransform(trainset_copy, transform=transform_toTensor_Normalize)

trainset = DataTransform(trainset, transform=transform_train)
testset = DataTransform(testset, transform=transform_toTensor_Normalize)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
trainloader_copy = DataLoader(trainset_copy, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


###################################################################################################################
###################################################################################################################


model = pickle.load(open(model_path, "rb"))
model = model["model"]
resize_out_features(model, classes)

model.to(device) 

std_string = ""
for name, param in model.named_parameters():
    std_string += f"\n{name}, Grad: {param.requires_grad}"

if hasattr(model, "linear"):
    std_string += f"\n\n{model.linear[-1]}"
elif hasattr(model, "classifier"):
    std_string += f"\n\n{model.classifier[-1]}"

std_string += f"\n\nFound model !\n"
output_to_std_and_file(save_to, "standard_output.txt", std_string)


##########################################################################################################
##########################################################################################################


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
grad_norms = []
layer_distance_2_init = {}
layer_distance_2_prev = {}
peak_test_acc = 0
peak_test_acc_epoch = 0

batch_time = ""

model_init = copy.deepcopy(model)
model_prev = copy.deepcopy(model)

@track_emissions(project_name=name_csv, output_file=f"PATH_TO_CO2_RESULTS_DIR/co2_csvs/{name_csv}.csv")
def train(model, criterion, optimizer, trainloader):
    correct = 0
    train_loss = 0
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
    return train_loss, correct

for epoch in range(total_epochs):
    #Train
    model.train()
    train_loss = 0
    correct = 0

    if epoch == 1:
        save_model(model, save_beegfs, note=epoch)

    if epoch in lr_schedule:
        new_lr = lr_schedule[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    train_loss, correct = train(model, criterion, optimizer, trainloader)

    layer_cosine_sim_l2norm(model, model_init, layer_distance_2_init)
    layer_cosine_sim_l2norm(model, model_prev, layer_distance_2_prev)
    model_prev = copy.deepcopy(model)
            
    train_acc = correct/len(trainset)

    grad_norms.append(global_l2_norm(model))

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

    if test_acc >= test_acc_reach:
        std_string = f"Test accuracy: {test_acc_reach} reached. Training ended"
        output_to_std_and_file(save_to, "standard_output.txt", std_string)
        break


##########################################################################################################
##########################################################################################################


training_hp = {
    "epoch": epoch+1,
    "lr" : lr,
    "weight_decay" : weight_decay,
    "lr_schedule" : lr_schedule,
    "random_seed" : random_seed
}

model_stats = {
    "training_hp_info" : training_hp,
    "train_loss" : train_loss_list,
    "train_acc" : train_acc_list,
    "test_loss" : test_loss_list,
    "test_acc" : test_acc_list,
    "peak_test_acc" : peak_test_acc,
    "peak_test_acc_epoch" : peak_test_acc_epoch,
    "global_grad_norms" : grad_norms,
    "layer_euc_dist_init" : layer_distance_2_init,
    "layer_euc_dist_prev" : layer_distance_2_prev
}

save_model_details = os.path.join(save_to, "model_stats.pkl")
with open(save_model_details, "wb") as fp:   
    pickle.dump(model_stats, fp)

save_model(model, save_beegfs, note=epoch)

print("\nDone !")


##########################################################################################################
##########################################################################################################
