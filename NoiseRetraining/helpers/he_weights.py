import torch.nn as nn

def he_weights_init(weights):
    if isinstance(weights, nn.Conv2d):
        nn.init.kaiming_normal_(weights.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(weights, nn.Linear):
        nn.init.kaiming_normal_(weights.weight)