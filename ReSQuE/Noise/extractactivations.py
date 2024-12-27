import random
import warnings
import copy
import torch
import gc
import pickle
import sys
import time

import torch
import torch.nn as nn
import torchvision
import numpy as np
import scipy as sp

warnings.filterwarnings('ignore')

# Extract activations of the givern laeyr for the entire data
class ExtractLayerActivations:
    def __init__(self, model, device, layer_list, trainloader):
        self.model = model
        self.device = device

        self.layer_list = layer_list #To identify and extract final layer embeddings
        self.trainloader = trainloader

        self.output_dict = {} #Holds data as key-value pairs. Keys: Layer name, Value: Activation values of filter/neuron
        self.final_layer_embedding = [] #Holds final layer embeddings for each batch. Non-averaged filter activation outputs
        self.final_layer_embedding_averaged = [] #Holds final layer embeddings for each batch. Non-averaged filter activation outputs

        self.model.to(self.device)
           
        for name, module in self.model.named_modules():
            #Iterate through all named modules in the network
            if hasattr(module, "out_features") or hasattr(module, "out_channels"):
                if name in self.layer_list:
                    self.hook_driver(name, module)
                    

    def hook_driver(self, name, module=None):
        #Driver function to set up forward hooks

        if name not in self.output_dict.keys() and len(name) > 1:
            #If layer name is not present in the main dictionary and has a name, add key to dictionary
            self.output_dict[name] = []
            module.register_forward_hook(self.hook(name)) #Register the forward hook for current layer
            print(f"Forward hook for {name} set")


    def hook(self, layer_name):
        #Functions to setup forward hook
        def hook_function(module, input, output):
            activation_data = output.cpu().detach().numpy()
            #print(layer_name, activation_data.shape)
            activation_data = activation_data.reshape(*activation_data.shape[:2], -1)
            if layer_name in self.layer_list:
                #If layer is final convolution layer, extract activation data without averaging
                ### self.final_layer_embedding.append(activation_data)
                activation_data = activation_data.mean(axis=-1, keepdims=True) 
                self.output_dict[layer_name].append(activation_data)  
        return hook_function


    #Perform a forward pass reshape the data of the layer
    def forward_pass(self):
        with torch.no_grad():
            self.model.eval()
            for i, (x, y) in enumerate(self.trainloader):

                x = x.to(self.device)
                out = self.model(x)               
                print(f"\r{i+1}/{len(self.trainloader)}",end="")

        for each_layer, each_list in self.output_dict.items():
            self.output_dict[each_layer] = np.array(each_list)
            self.output_dict[each_layer] = self.output_dict[each_layer].reshape(-1, *self.output_dict[each_layer].shape[2:])
            self.output_dict[each_layer] = self.output_dict[each_layer].reshape(self.output_dict[each_layer].shape[0], -1)
            self.output_dict[each_layer] = self.output_dict[each_layer].transpose()

        return self.output_dict