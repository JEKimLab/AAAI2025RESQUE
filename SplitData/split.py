import os
import torch
import torchvision
from torch.utils.data import Subset, DataLoader
import numpy as np
from collections import Counter

# Split the train dataset into 3 splits, one for clean training, one for noise retraining, anda  common overlap set
def split_dataset(dataset_name, split_size1, split_size2, split_size3, random_seed, save_location):
    assert split_size1 + split_size2 + split_size3 == 100, "Split sizes must sum to 100"
    
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    
    if dataset_name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='<DataLocation>/CIFAR10', train=True, download=True)
        targets = trainset.targets
    elif dataset_name == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='<DataLocation>/CIFAR100', train=True, download=True)
        targets = trainset.targets
    elif dataset_name == 'svhn':
        trainset = torchvision.datasets.SVHN(root='<DataLocation>/SVHN', split='train', download=True)
        targets = trainset.labels
    else:
        raise ValueError("Unsupported dataset. Choose from 'cifar10', 'cifar100', 'svhn'.")
    
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    num_classes = len(np.unique(targets))
    class_indices = [[] for _ in range(num_classes)]
    
    for idx, label in enumerate(targets):
        class_indices[label].append(idx)
    
    split_ratios = [split_size1 / 100, split_size2 / 100, split_size3 / 100]
    split_indices = [[] for _ in range(3)]
    
    for indices in class_indices:
        np.random.shuffle(indices)
        split_points = [int(len(indices) * ratio) for ratio in split_ratios]
        split_points = np.cumsum(split_points).tolist()
        
        split_indices[0].extend(indices[:split_points[0]])
        split_indices[1].extend(indices[split_points[0]:split_points[1]])
        split_indices[2].extend(indices[split_points[1]:])
    
    splits = [Subset(trainset, indices) for indices in split_indices]
    for i, split in enumerate(splits):
        split_file = os.path.join(save_location, f'split_{i}.pt')
        torch.save(split, split_file)
    
    log_file_path = os.path.join(save_location, 'split_log.txt')
    total_samples = 0
    with open(log_file_path, 'w') as log_file:
        for i, split in enumerate(splits):
            class_counter = Counter()
            split_samples = 0
            for index in split.indices:
                label = targets[index]
                class_counter.update([label])
                split_samples += 1
            total_samples += split_samples
            log_file.write(f'Split {i} class distribution:\n')
            for cls, count in class_counter.items():
                log_file.write(f'Class {cls}: {count}\n')
            log_file.write(f'Total samples in Split {i}: {split_samples}\n\n')
        log_file.write(f'Total samples across all splits: {total_samples}\n')

split_dataset('datasetname', 50, 30, 20, 1, 'SaveLocation')
