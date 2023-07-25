import torch.nn as nn

NoneType = type(None)


def requires_grad(model: nn.Module, flag: bool = True) -> NoneType:
    for p in model.parameters():
        p.requires_grad = flag


def num_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
