import os
import gc
import sys
import json
import copy

import pickle
import random
import argparse
import warnings

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Subset
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from scipy.special import rel_entr
from scipy.stats import wasserstein_distance

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt

sys.path.append("../models")
sys.path.append("../helpers")

from vgg import VGG, make_layers
from resnet import ResNet, BasicBlock, Bottleneck

from extract_activations_task import ExtractAct
from save_images import save_img
from data_classes import class_count

warnings.filterwarnings('ignore')

MODEL_PATH_ROOT = "PATH_TO_MODELS_DIR"
RESULTS_PATH_ROOT = "PATH_TO_SAVE_RESULTS"
batch_size = 1


###################################################################################################################
###################################################################################################################


transform_toTensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
transform_toPIL = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
transform_toTensor_Normalize = torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
transform_gray = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=3), torchvision.transforms.Resize((32,32)), 
    torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])



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


def create_dataframe(groundtruth, features):
    data = {'GroundTruth' : groundtruth}

    for i in range(features.shape[1]):
        col_name = f"Value_{i}"
        data[col_name] = features[:, i]

    df = pd.DataFrame(data)
    cluster_centroids = df.groupby('GroundTruth').mean().reset_index()

    return df, cluster_centroids


def output_to_std_and_file(directory, file_name, data):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, file_name)
    print(data)
    with open(file_path, "a") as file:
        file.write(data)


def cluster_w_centroids(dataframe, centroids_np, classes):
    dataframe_features = dataframe.iloc[:, 1:]
    kmeans = KMeans(n_clusters=classes, init=centroids_np, n_init=1, random_state=0)
    cluster_labels = kmeans.fit_predict(dataframe_features)
    dataframe.insert(loc=dataframe.columns.get_loc("GroundTruth") + 1, column="ClusterLabels", value=cluster_labels)

    ground_truth = dataframe["GroundTruth"]
    predicted_labels = dataframe["ClusterLabels"]
    ari_score = adjusted_rand_score(ground_truth, predicted_labels)

    return ari_score


def cluster_kmeans_plusplus(dataframe, classes):
    dataframe.drop('ClusterLabels', axis=1, inplace=True)
    dataframe_features = dataframe.iloc[:, 1:]
    kmeans = KMeans(n_clusters=classes, init='k-means++', n_init=1, random_state=0)
    cluster_labels = kmeans.fit_predict(dataframe_features)
    dataframe.insert(loc=dataframe.columns.get_loc("GroundTruth") + 1, column="ClusterLabels", value=cluster_labels)

    ground_truth = dataframe["GroundTruth"]
    predicted_labels = dataframe["ClusterLabels"]
    ari_score = adjusted_rand_score(ground_truth, predicted_labels)

    return ari_score


def tsne_plot(dataframe, save_loc, save_name):
    dataframe_features = dataframe.iloc[:, 1:]
    labels = dataframe['GroundTruth']

    tsne = TSNE(n_components=2, random_state=1)
    tsne_result = tsne.fit_transform(dataframe_features)

    tsne_df = pd.DataFrame(tsne_result, columns=['Dimension 1', 'Dimension 2'])
    tsne_df['GroundTruth'] = labels

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_df['Dimension 1'], tsne_df['Dimension 2'], c=tsne_df['GroundTruth'])
    plt.title('t-SNE Plot')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend(handles=scatter.legend_elements()[0], title='GroundTruth')

    plt.savefig(f'{save_loc}/{save_name}.png')


###################################################################################################################
###################################################################################################################


parser = argparse.ArgumentParser()
parser.add_argument('-m', type=str, help='path to model')
parser.add_argument('-s', type=str, help='Save model and data location')
parser.add_argument('-d', type=str, help='Dataset')
parser.add_argument('-rs', type=int, help='Random seed')
parser.add_argument('-cl', type=str, help='Final conv or embedding layer')
parser.add_argument('-tsne', type=str, help='Plot t-SNE ?', default="False")
parser.add_argument('-mt', type=str, help='Model type')

args = parser.parse_args()

model_path = os.path.join(MODEL_PATH_ROOT, args.m)
save_to = os.path.join(RESULTS_PATH_ROOT, args.s)
data = args.d
random_seed = args.rs
final_conv_layer = args.cl
tsne_flag = "t" in args.tsne.lower()
model_type = args.mt

std_string = f"Model path: {model_path}\n"
std_string += f"Save results to: {save_to}\n"
std_string += f"Dataset: {data}\n"
std_string += f"Random seed: {random_seed}\n"
std_string += f"Final embedding layer name: {final_conv_layer}\n"
std_string += f"Plot t-SNE: {tsne_flag}\n"
std_string += f"Model Type: {model_type}\n"

output_to_std_and_file(save_to, "standard_output.txt", std_string)


np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False


###################################################################################################################
###################################################################################################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_to_std_and_file(save_to, "standard_output.txt", f"\n{device}")

model = pickle.load(open(model_path, "rb"))
model = model["model"]
model.to(device) 


if data.lower() == "cifar10":
    dataset = "CIFAR10"
    trainset = torchvision.datasets.CIFAR10(root='PATH_TO_CIFAR10', train=True, download=True)
elif data.lower() == "cifar100":
    dataset = "CIFAR100"
    trainset = torchvision.datasets.CIFAR100(root='PATH_TO_CIFAR100', train=True, download=True)
elif data.lower() == "gtsrb":
    dataset = "GTSRB"
    trainset = torchvision.datasets.GTSRB(root='PATH_TO_GTRSB', split='train', download=True)
elif data.lower() == "stl10":
    dataset = "STL10"
    trainset = torchvision.datasets.STL10(root='PATH_TO_STL10', split='train', download=True)
elif data.lower() == "mnist":
    dataset = "MNIST"
    trainset = torchvision.datasets.MNIST(root='PATH_TO_MNIST', train=True, download=True)
    transform_toTensor_Normalize = transform_gray
elif data.lower() == "emnist":
    dataset = "EMNIST"
    trainset = torchvision.datasets.EMNIST(root='PATH_TO_EMNIST', split='balanced', train=True, download=True)
    transform_toTensor_Normalize = transform_gray
elif data.lower() == "fashionmnist" or data.lower() == "fashmnist":
    dataset = "FashionMNIST"
    trainset = torchvision.datasets.FashionMNIST(root='PATH_TO_FashionMNIST', train=True, download=True)
    transform_toTensor_Normalize = transform_gray
elif data.lower() == "svhn":
    dataset = "SVHN"
    trainset = torchvision.datasets.SVHN(root='PATH_TO_SVHN', split='train', download=True)
elif data.lower() == "food101":
    dataset = "Food101"
    trainset = torchvision.datasets.Food101(root='PATH_TO_Food101', split='train', download=True)


labels, classes = class_count(trainset)
index_positions = [int(len(trainset) * (i + 1) / 6) for i in range(5)]
batch_size_options = [32, 20, 16, 15, 10, 8, 5, 4, 2]

for each_size in batch_size_options:
    if len(trainset)%each_size == 0:
        batch_size = each_size
        break

std_string = f"\n\nDataset selected: {dataset}\n"
std_string += f"Dataset size: {len(trainset)}\n"
std_string += f"Sample positions: {index_positions}\n"
std_string += f"Batch size: {batch_size}\n"

output_to_std_and_file(save_to, "standard_output.txt", std_string)


###################################################################################################################
###################################################################################################################


## Clean data pass

save_img(trainset, os.path.join(save_to, "clean_imgs"), index_positions)

trainset_norm_tensor = DataTransform(trainset, transform=transform_toTensor_Normalize)
trainloader = DataLoader(trainset_norm_tensor, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


output_to_std_and_file(save_to, "standard_output.txt", "\nStarting forward pass")
extractObject = ExtractAct(model, device, final_conv_layer, classes, trainloader, model_type)
final_layer_embedding_avg, final_layer_embedding, groundtruths = extractObject.forward_pass()

output_to_std_and_file(save_to, "standard_output.txt", "\nCreating Dataframes")
df_1, centroids_1 = create_dataframe(groundtruths, final_layer_embedding)
centroids_1_np = centroids_1.iloc[:, 1:].values

print(centroids_1_np[1])

df_2, centroids_2 = create_dataframe(groundtruths, final_layer_embedding_avg)
centroids_2_np = centroids_2.iloc[:, 1:].values

print(f"Centroid 1 original shape: {centroids_1.shape}")
print(f"Centroid 1 numpy shape: {centroids_1_np.shape}")

print(f"Centroid 2 original shape: {centroids_2.shape}")
print(f"Centroid 2 numpy shape: {centroids_2_np.shape}")

model = None
output_dict_clean = None
final_layer_embedding_clean = None
groundtruths_clean = None
del model
del output_dict_clean
del final_layer_embedding_clean
del groundtruths_clean
gc.collect()

output_to_std_and_file(save_to, "standard_output.txt", "\nCompleted clean forward pass")


if tsne_flag:
    tsne_plot(df1, save_to, "clean")


ari_score_df1_Centroids = cluster_w_centroids(df_1, centroids_1_np, classes)
ari_score_df1_Kpp = cluster_kmeans_plusplus(df_1, classes)
output_to_std_and_file(save_to, "standard_output.txt", "\nDone with clustering 1")
df_1 = None

ari_score_df2_Centroids = cluster_w_centroids(df_2, centroids_2_np, classes)
ari_score_df2_Kpp = cluster_kmeans_plusplus(df_2, classes)
output_to_std_and_file(save_to, "standard_output.txt", "\nDone with clustering 2")
df_2 = None
del df_2
gc.collect()

std_string = f"\n\nARI Score, explicits centroids:"
std_string += f"\nARI Score DF 1 (non-avg): {ari_score_df1_Centroids}"
std_string += f"\nARI Score DF 2 (avg): {ari_score_df2_Centroids}"

std_string += f"\n\nARI Score, Kmeans++:"
std_string += f"\nARI Score DF 1 (non-avg): {ari_score_df1_Kpp}"
std_string += f"\nARI Score DF 2 (avg): {ari_score_df2_Kpp}"

output_to_std_and_file(save_to, "standard_output.txt", std_string)

final_results = {}

final_results["explicit_centroids"] = {
    "df1" : 1 - ari_score_df1_Centroids,
    "df2" : 1 - ari_score_df2_Centroids
    
}

final_results["Kmeans_pluplus"] = {
    "df1" : 1 - ari_score_df1_Kpp,
    "df2" : 1 - ari_score_df2_Kpp
}


with open(os.path.join(save_to, "all_measures.json"), "w") as f:
    json.dump(final_results, f)


###################################################################################################################
###################################################################################################################