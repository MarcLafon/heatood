from typing import Any, Mapping, Union
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

import heat.lib as lib
from heat.scorers.abstract_scorer import AbastractOODScorer

NoneType = type(None)
KwargsType = Mapping[str, Any]


class MeanScorer(AbastractOODScorer):
    def __init__(
        self,
        num_classes: int,
        reduce: str = 'max',
        normalize_before: bool = False,
        **kwargs: KwargsType,
    ) -> NoneType:
        super().__init__(**kwargs)
        self.name = "MeanLogits"
        self.num_classes = num_classes
        self.reduce = reduce
        self.normalize_before = normalize_before
        self.pooling = 'avg'

    @torch.no_grad()
    def _fit(self, model: nn.Module, train_loader: DataLoader) -> NoneType:
        model.eval()
        lib.LOGGER.info("Fitting MeanScorer")

        if not self.features_dataset:
            self.class_mean = None
            class_count = torch.zeros(self.num_classes, device="cuda")
            for data, target in tqdm(train_loader, desc="Fitting MeanScorer", disable=os.getenv('DISABLE_TQDM')):
                data = data.to('cuda', non_blocking=True)
                target = target.to('cuda', non_blocking=True)

                z = lib.get_features(model, data, layer_names=[self.layer_name])[self.pooling][-1]
                if self.normalize:
                    z = F.normalize(z, dim=1)

                if self.class_mean is None:
                    self.class_mean = torch.zeros(self.num_classes, z.size(-1), device='cuda')

                self.class_mean.scatter_add_(0, target.view(-1, 1).repeat(1, z.size(-1)), z)
                class_count += target.float().histc(self.num_classes, max=self.num_classes)

            self.class_mean /= class_count.view(-1, 1)
        else:
            # https://github.com/deeplearning-wisc/knn-ood/blob/master/run_imagenet.py
            features = train_loader.dataset.features
            labels = torch.from_numpy(np.ascontiguousarray(train_loader.dataset.targets)).long()
            if self.normalize_before:  # degraded performances on ImageNet
                features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-10)
            z = torch.from_numpy(np.ascontiguousarray(features.astype(np.float32)))
            self.class_mean = torch.zeros(self.num_classes, z.size(-1))
            self.class_mean.scatter_add_(0, labels.view(-1, 1).repeat(1, z.size(-1)), z)
            _, class_count = labels.unique(return_counts=True)
            self.class_mean /= class_count.view(-1, 1)

        self.class_mean = self.class_mean.to('cuda', non_blocking=True)
        if self.normalize:
            self.class_mean = F.normalize(self.class_mean, dim=1)

        lib.LOGGER.info("Fitting MeanScorer done")
        self.is_fitted = True

    @torch.no_grad()
    def score_batch(self, model: nn.Module, data: Union[Mapping[str, Tensor], Tensor], features_dataset: bool = False) -> Tensor:
        data = lib.to_device(data)
        if features_dataset:
            z = data[self.pooling].float()
        else:
            z = lib.get_features(model, data, layer_names=[self.layer_name])[self.pooling][-1]

        if self.normalize:
            z = F.normalize(z, dim=1)

        logits = z @ self.class_mean.T
        if self.reduce == 'logsumexp':
            return - torch.logsumexp(logits, dim=1)
        if self.reduce == 'max':
            return - logits.max(dim=1)[0]
