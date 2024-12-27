import random
import warnings
import copy
import torch
import gc
import pickle
import sys
import time
sys.path.append("../models")

import torch
import torch.nn as nn
import torchvision
import numpy as np
import scipy as sp

from scipy.special import rel_entr
from scipy.stats import wasserstein_distance

warnings.filterwarnings('ignore')


class ExtractAct:
    def __init__(self, model, device, final_conv_layer, classes, trainloader, model_type):
        self.model = model
        self.device = device

        #Named as "final_conv", but it is dense layer for ViT
        self.final_conv_layer = final_conv_layer #To identify and extract final layer embeddings
        self.classes = classes #To identify final output layer
        self.trainloader = trainloader

        self.output_dict = {} #Holds data as key-value pairs. Keys: Layer name, Value: Activation values of filter/neuron
        self.final_layer_embedding = [] #Holds final layer embeddings for each batch. Non-averaged filter activation outputs
        self.final_layer_embedding_averaged = [] #Holds final layer embeddings for each batch. Non-averaged filter activation outputs

        self.groundtruths = [] #Holds ground truth label for each sample
        self.model_type = model_type #CNN or ViT

        self.model.to(self.device)
           
        for name, module in self.model.named_modules():
            #Iterate through all named modules in the network
            
            if hasattr(module, "out_features") or hasattr(module, "out_channels"):
                if name == final_conv_layer:
                    self.hook_driver(name, self.final_conv_layer, module)
                    

    def hook_driver(self, name, final_conv, module=None):
        #Driver function to set up forward hooks

        if name not in self.output_dict.keys() and len(name) > 1:
            #If layer name is not present in the main dictionary and has a name, add key to dictionary
            self.output_dict[name] = []
            module.register_forward_hook(self.hook(name, final_conv)) #Register the forward hook for current layer
            print(f"Forward hook for {name} set")


    def hook(self, layer_name, final_conv):
        #Functions to setup forward hook
        def hook_function(module, input, output):
            if self.model_type.lower() == "cnn":
                activation_data = output.cpu().detach().numpy()

                #If activation data has 4 dimensions, it is the output of a convolutional layer
                #The four dimensions -> (batch_size, number_of_filters, activation_output_height, activation_output_weight)
                if len(activation_data.shape) == 4:
                    activation_data = activation_data.reshape(*activation_data.shape[:2], -1)
                    if layer_name == final_conv:
                        #If layer is final convolution layer, extract activation data without averaging
                        self.final_layer_embedding.append(activation_data)
                    activation_data = activation_data.mean(axis=-1, keepdims=True)   
            elif self.model_type.lower() == "vit":
                activation_data = output.cpu().detach().numpy()

                #If activation data has 4 dimensions, it is the output of a convolutional layer
                #The four dimensions -> (batch_size, number_of_filters, activation_output_height, activation_output_weight)
                #if len(activation_data.shape) == 4:
                activation_data = activation_data.reshape(*activation_data.shape[:2], -1)
                if layer_name == final_conv: #Named as "final_conv", but it is dense layer for ViT
                    #If layer is final convolution layer, extract activation data without averaging
                    self.final_layer_embedding.append(activation_data)
                activation_data = activation_data.mean(axis=-1, keepdims=True)   
                self.final_layer_embedding_averaged.append(activation_data)
            else:
                print("Invalid model type")
                exit()

            self.final_layer_embedding_averaged.append(activation_data)

        return hook_function


    def forward_pass(self):
        with torch.no_grad():
            self.model.eval()
            for i, (x, y) in enumerate(self.trainloader):
                self.groundtruths.extend(y)

                x = x.to(self.device)
                out = self.model(x)               
                print(f"\r{i+1}/{len(self.trainloader)}",end="")
                
        self.final_layer_embedding = np.array(self.final_layer_embedding)

        self.final_layer_embedding = self.final_layer_embedding.reshape(-1, *self.final_layer_embedding.shape[2:])
        self.final_layer_embedding = self.final_layer_embedding.reshape(self.final_layer_embedding.shape[0], -1)

        self.final_layer_embedding_averaged = np.array(self.final_layer_embedding_averaged)

        self.final_layer_embedding_averaged = self.final_layer_embedding_averaged.reshape(-1, *self.final_layer_embedding_averaged.shape[2:])
        self.final_layer_embedding_averaged = self.final_layer_embedding_averaged.reshape(self.final_layer_embedding_averaged.shape[0], -1)

        self.groundtruths = np.array(self.groundtruths)

        return self.final_layer_embedding_averaged, self.final_layer_embedding, self.groundtruths