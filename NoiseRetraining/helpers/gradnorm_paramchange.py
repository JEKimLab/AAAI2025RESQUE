import math
import numpy as np
import torch
import torch.nn as nn

def grad_norm(model):
    epoch_grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'), norm_type=2.0)
    return epoch_grad_norm.item()

def layer_param_change(model, init_model, layer_distance_struct):
    for (name, param), (name_init, param_init) in zip(model.named_parameters(), init_model.named_parameters()):
        if name not in layer_distance_struct.keys():
            layer_distance_struct[name] = []
        if True: #In-place update
            param_all = torch.flatten(param).cpu().detach().numpy()
            param_all_init = torch.flatten(param_init).cpu().detach().numpy()
            layer_distance_struct[name].append((np.linalg.norm(param_all - param_all_init))/math.sqrt(param_all.shape[0]))