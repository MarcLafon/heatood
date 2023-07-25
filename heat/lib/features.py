from typing import List

import torch
import torch.nn as nn
from torch import Tensor


def _get_features(model: nn.Module, data: Tensor, layer_names: List[str] = ("layer1", "layer2", "layer3", "layer4")) -> Tensor:

    features = {'avg': [], 'std': [], 'avg_std': [], 'max': []}
    handles = []
    for i, layer_name in enumerate(layer_names):
        def fhook(module: nn.Module, inp: Tensor, outp: Tensor) -> Tensor:
            avg_pool = outp.mean(dim=(2, 3))
            std_pool = outp.std(dim=(2, 3))
            max_pool = outp.amax(dim=(2, 3))
            features['avg'].append(avg_pool)
            features['std'].append(std_pool)
            features['avg_std'].append(torch.cat([avg_pool, std_pool], dim=1))
            features['max'].append(max_pool)
        handles.append(model.__getattr__(layer_name).register_forward_hook(fhook))

    _ = model(data)

    for i, l_name in enumerate(layer_names):
        handles[i].remove()
    return features


@torch.no_grad()
def get_features(model: nn.Module, data: Tensor, layer_names: List[str] = ("layer1", "layer2", "layer3", "layer4")) -> Tensor:
    return _get_features(model, data, layer_names)


def get_features_with_grad(model: nn.Module, data: Tensor, layer_names: List[str] = ("layer1", "layer2", "layer3", "layer4")) -> Tensor:
    return _get_features(model, data, layer_names)


def get_feature_dim(model: nn.Module, layer_name: List[str]) -> int:
    try:
        return model.__getattr__(layer_name)[-1].conv3.out_channels
    except AttributeError:
        return model.__getattr__(layer_name)[-1].conv2.out_channels
