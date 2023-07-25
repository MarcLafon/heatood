from typing import Any, Mapping, Union, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

import heat.lib as lib
from heat.scorers.abstract_scorer import AbastractOODScorer

NoneType = type(None)
KwargType = Mapping[str, Any]


class EnergyLogitsScorer(AbastractOODScorer):
    def __init__(self, pooling: str = 'avg', **kwargs: KwargType) -> NoneType:
        assert pooling in ['avg', 'token']
        super(EnergyLogitsScorer, self).__init__(**kwargs)
        self.name = "EnergyLogits"
        self.is_fitted = False
        self.classifier_head = None
        self.pooling = pooling

    @torch.no_grad()
    def _score_batch(self, model: nn.Module, data: Union[Mapping[str, Tensor], Tensor], features_dataset: bool = False) -> Tensor:
        data = lib.to_device(data)

        if features_dataset:
            z = data[self.pooling].float()
        else:
            z = lib.get_features(model, data, layer_names=[self.layer_name])[self.pooling][-1]

        if self.use_react:
            z = z.clip(max=self.react_threshold)

        return self.energy(z)

    def _fit(self, model: nn.Module, train_loader: DataLoader) -> NoneType:
        self.classifier_head = model.fc

    def energy(self, z: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        logits = self.classifier_head(z)
        return - torch.logsumexp(logits, dim=1)

    @property
    def has_sample(self) -> bool:
        return False
