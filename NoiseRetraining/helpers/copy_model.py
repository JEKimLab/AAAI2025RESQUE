import os
import torch

def create_copy(model, save_to):
    weights = 'weights_temp.pt'
    path = os.path.join(save_to, weights)
    torch.save(model, path)
    return torch.load(path)

def remove_copy(save_to):
    weights = 'weights_temp.pt'
    path = os.path.join(save_to, weights)
    try:
        os.unlink(path)
    except:
        pass