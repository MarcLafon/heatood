from typing import Any, Mapping, Union, Optional
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

import heat.lib as lib
from heat.scorers.abstract_scorer import AbastractOODScorer

NoneType = type(None)
KwargType = Mapping[str, Any]


class DiceScorer(AbastractOODScorer):
    def __init__(self, pooling: str = 'avg', p_dice: float = 9, clip_th: Optional[float] = None, **kwargs: KwargType) -> NoneType:
        assert pooling in ['avg', 'token']
        super().__init__(**kwargs)
        self.name = "DICE"
        self.is_fitted = False
        self.classifier_head = None
        self.pooling = pooling
        self.p_dice = p_dice
        self.clip_th = clip_th

    @torch.no_grad()
    def _score_batch(self, model: nn.Module, data: Union[Mapping[str, Tensor], Tensor], features_dataset: bool = False) -> Tensor:
        data = lib.to_device(data)

        if features_dataset:
            z = data[self.pooling].float()
        else:
            z = lib.get_features(model, data, layer_names=[self.layer_name])[self.pooling][-1]

        if self.clip_th is not None:
            z = z.clip(max=self.clip_th)

        if self.use_react:
            z = z.clip(max=self.react_threshold)

        return self.energy(z)

    def _fit(self, model: nn.Module, train_loader: DataLoader) -> NoneType:
        self.classifier_head = deepcopy(model.fc)

        model.eval()
        count = 0
        mean = torch.empty(1, self.classifier_head.weight.size(1), device=self.classifier_head.weight.device)
        for batch, _ in tqdm(train_loader, desc='Fitting DICE', disable=os.getenv('TQDM_DISABLE')):
            batch = lib.to_device(batch)
            if self.features_dataset:
                z = batch[self.pooling].float()
            else:
                z = lib.get_features(model, batch, layer_names=[self.layer_name])[self.pooling][-1]

            if self.clip_th is not None:
                z = z.clip(max=self.clip_th)

            mean = ((mean * count) + z.sum(dim=0, keepdim=True)) / (count + z.size(0))
            count += z.size(0)

        self.contrib = self.classifier_head.weight.data.cpu().numpy() * mean.cpu().numpy()  # this to be as close as possible to the original implementation
        self.thresh = np.percentile(self.contrib, self.p_dice)
        mask = torch.tensor(self.contrib > self.thresh).float()
        self.masked_w = (self.classifier_head.weight.data.squeeze().cpu() * mask).to(self.classifier_head.weight.device)

        self.classifier_head.weight.data = self.masked_w

        self.is_fitted = True

    def energy(self, z: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        logits = self.classifier_head(z)
        return - torch.logsumexp(logits, dim=1)
