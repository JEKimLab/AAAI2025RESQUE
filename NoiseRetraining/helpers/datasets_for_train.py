import torchvision
import torch
from torchvision import datasets

def create_train_test_set(data):
    if data.lower() == "cifar10":
        dataset = "CIFAR10"
        trainset = torchvision.datasets.CIFAR10(root='<DataPath>/CIFAR10', train=True, download=True)
        testset = torchvision.datasets.CIFAR10(root='<DataPath>/CIFAR10', train=False, download=True)
    elif data.lower() == "cifar100":
        dataset = "CIFAR100"
        trainset = torchvision.datasets.CIFAR100(root='<DataPath>/CIFAR100', train=True, download=True)
        testset = torchvision.datasets.CIFAR100(root='<DataPath>/CIFAR100', train=False, download=True)
    elif data.lower() == "gtsrb":
        dataset = "GTSRB"
        trainset = torchvision.datasets.GTSRB(root='<DataPath>/GTRSB', split='train', download=True)
        testset = torchvision.datasets.GTSRB(root='<DataPath>/GTRSB', split='test', download=True)
    elif data.lower() == "stl10":
        dataset = "STL10"
        trainset = torchvision.datasets.STL10(root='<DataPath>/STL10', split='train', download=True)
        testset = torchvision.datasets.STL10(root='<DataPath>/STL10', split='test', download=True)
    elif data.lower() == "mnist":
        dataset = "MNIST"
        trainset = torchvision.datasets.MNIST(root='<DataPath>/MNIST', train=True, download=True)
        testset = torchvision.datasets.MNIST(root='<DataPath>/MNIST', train=False, download=True)
    elif data.lower() == "emnist":
        dataset = "EMNIST"
        trainset = torchvision.datasets.EMNIST(root='<DataPath>/EMNIST', split='balanced', train=True, download=True)
        testset = torchvision.datasets.EMNIST(root='<DataPath>/EMNIST', split='balanced', train=False, download=True)
    elif data.lower() == "fashionmnist" or data.lower() == "fashmnist":
        dataset = "FashionMNIST"
        trainset = torchvision.datasets.FashionMNIST(root='<DataPath>/FashionMNIST', train=True, download=True)
        testset = torchvision.datasets.FashionMNIST(root='<DataPath>/FashionMNIST', train=False, download=True)
    elif data.lower() == "svhn":
        dataset = "SVHN"
        trainset = torchvision.datasets.SVHN(root='<DataPath>/SVHN', split='train', download=True)
        testset = torchvision.datasets.SVHN(root='<DataPath>/SVHN', split='test', download=True)
    elif data.lower() == "food101":
        dataset = "Food101"
        trainset = torchvision.datasets.Food101(root='<DataPath>/Food101', split='train', download=True)
        testset = torchvision.datasets.Food101(root='<DataPath>/Food101', split='test', download=True)
    elif data.lower() == "tinyin" or data.lower() == "tinyimagenet":
        dataset = "TinyImageNet"
        trainset = datasets.ImageFolder('<DataPath>')
        testset = datasets.ImageFolder('<DataPath>')

    return trainset, testset, dataset

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