import torch.nn as nn
import torch

def parse_function(function_name):
    if function_name is None:
        return None, False

    f = function_name.lower().strip()
    needs_features = False
    function = None

    if f == 'relu':
        function = nn.ReLU

    elif f == 'tanh':
        function = nn.Tanh

    elif f == 'softmax' or f == 'soft max':
        function = nn.Softmax

    elif f == 'id' or f == 'identity':
        function = nn.Identity

    elif f == 'sigmoid' or f == 'logit' or f == 'logits':
        function = nn.Sigmoid

    elif f == 'bn' or f == 'batch_norm' or f == 'batchnorm' or f == 'batch norm':
        function = nn.BatchNorm1d
        needs_features = True

    return function, needs_features