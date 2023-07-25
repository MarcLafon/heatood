from typing import Union, List, Tuple, Mapping, get_args

import torch

ToTypes = Union[torch.Tensor, torch.nn.Module]
AcceptedTypes = Union[ToTypes, List, Tuple, Mapping]


def _to(tens: ToTypes) -> ToTypes:
    return tens.to('cuda', non_blocking=True)


def to_device(tens: AcceptedTypes) -> AcceptedTypes:
    assert isinstance(tens, get_args(AcceptedTypes))

    if isinstance(tens, get_args(ToTypes)):
        return _to(tens)

    if isinstance(tens, (list, tuple)):
        return type(tens)([to_device(t) for t in tens])

    if isinstance(tens, dict):
        return {key: to_device(t) for key, t in tens.items()}
