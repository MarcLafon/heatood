from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import heat.lib as lib
from heat.scorers.abstract_scorer import AbastractOODScorer

NoneType = type(None)
KwargsType = Mapping[str, Any]


class EBMScorer(AbastractOODScorer):
    def __init__(
        self,
        ebm: nn.Module,
        layer_name: str = "layer4",
        **kwargs: KwargsType,
    ) -> NoneType:
        super(EBMScorer, self).__init__(**kwargs)
        self.name = "EBMScorer"
        self.ebm = ebm
        self.is_fitted = True
        self.layer_name = layer_name

    def _score_batch(
        self,
        model: torch.nn.Module,
        data: torch.Tensor,
        features_dataset: bool = False,
        with_grad: bool = False,
    ) -> torch.Tensor:
        self.ebm.eval()
        data = lib.to_device(data)
        if features_dataset:
            z = data[self.ebm.base_dist.pooling].float()
        else:
            if with_grad:
                z = lib.get_features_with_grad(model, data, layer_names=[self.layer_name])[self.ebm.base_dist.pooling][-1]
            else:
                z = lib.get_features(model, data, layer_names=[self.layer_name])[self.ebm.base_dist.pooling][-1]

        if self.ebm.base_dist.normalize:
            z = F.normalize(z, dim=1)

        if self.ebm.base_dist.use_pca:
            z = self.ebm.base_dist.pca.transform(z)

        if self.use_react:
            z = z.clip(max=self.react_threshold)

        score, score_nn, score_g = self.ebm(z)
        return score, score_nn, score_g
