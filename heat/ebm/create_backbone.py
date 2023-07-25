from typing import List
import os
import pathlib
from functools import partial
from collections import OrderedDict

import timm
import torch
import torch.nn as nn
import torchvision
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


import heat.lib as lib


def adapt_checkpoint(state_dict: OrderedDict, replace_dict: dict = {'module.': ''}) -> OrderedDict:
    """
    This function renames keys in a state_dict.
    The default function is helpfull when a NN has been used with parallelism.
    """
    new_dict = OrderedDict()
    for key, weight in state_dict.items():
        new_key = key
        for remove, replace in replace_dict.items():
            new_key = new_key.replace(remove, replace)
        new_dict[new_key] = weight
    return new_dict


NUM_CLASSES = {
    'cifar10': 10,
    'cifar100': 100,
    'imagenet': 1000,
}

LOOK = {
    'cifar10-resnet34-multisteplr_randomresizedcrop_epoch_199.ckpt': lambda sd: sd['model_state'],
    'cifar100-resnet34-multisteplr_randomresizedcrop_epoch_199.ckpt': lambda sd: sd['model_state'],
}

ADAPT = {}
MISSING = {}
UNEXPECTED = {}


def create_backbone(model_id: str, layer_names: List[str] = ["layer4"]) -> nn.Module:
    dts, arch, name = model_id.split('-')

    if (dts == 'imagenet') and (name == '__pretrained__'):
        imagenet_pretrained = True
    else:
        imagenet_pretrained = False
        file = os.path.join(pathlib.Path(__file__).parent.parent.parent.resolve(), 'weights', dts, arch, name)  # A bit hacky...
        assert os.path.exists(file), f"The weight file does not exists: {file}"

    net = timm.create_model(arch, num_classes=NUM_CLASSES[dts])
    net.feature_dims = [lib.get_feature_dim(net, layer_name) for layer_name in layer_names]
    net.model_id = model_id
    net.num_classes = NUM_CLASSES[dts]
    if dts in ['cifar10', 'cifar100']:
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()

    if not imagenet_pretrained:
        state_dict = torch.load(file, map_location='cpu')
        state_dict = LOOK.get(model_id, lambda x: x)(state_dict)
        state_dict = ADAPT.get(model_id, lambda x: x)(state_dict)

        key_issue = net.load_state_dict(state_dict, strict=False)
        if key_issue.unexpected_keys:
            print(key_issue.unexpected_keys)
            assert key_issue.unexpected_keys == UNEXPECTED[model_id]
            print("Unexpected keys:", key_issue.unexpected_keys)
        if key_issue.missing_keys:
            assert key_issue.missing_keys == MISSING[model_id]
            print("Missing keys:", key_issue.missing_keys)
    else:
        # state_dict = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1").state_dict()
        state_dict = load_state_dict_from_url(torchvision.models.resnet.model_urls[arch], progress=True)
        _ = net.load_state_dict(state_dict, strict=True)

    print("Backbone has:", lib.num_parameters(net), "parameters")
    return net


if __name__ == '__main__':
    create_backbone('imagenet-resnet34-__pretrained__')
    create_backbone('imagenet-resnet50-__pretrained__')
