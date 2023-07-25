from typing import List, Any, Mapping, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from heat.scorers.abstract_scorer import AbastractOODScorer
from heat.scorers.ssd_scorer import SSDScorer

NoneType = type(None)
KwargsType = Mapping[str, Any]


class CombineScorer(AbastractOODScorer):
    def __init__(
        self,
        scorers: List,
        beta: float = 0,
        features_dataset: bool = False,
        **kwargs: KwargsType,
    ) -> NoneType:
        super(CombineScorer, self).__init__(**kwargs)
        self.scorers = scorers

        if beta == 0:
            self.mode, self.beta = "sum", 1
        elif beta == float("inf"):
            self.mode, self.beta = "max", 1
        elif beta == -float("inf"):
            self.mode, self.beta = "min", 1
        else:
            self.mode, self.beta = "logsumexp", beta

        self.features_dataset = features_dataset
        self.coefs = [1] * len(scorers)
        self.means = [0] * len(scorers)
        self.std = [1] * len(scorers)
        assert len(self.means) == len(scorers)
        assert self.mode in ["sum", "logsumexp", "min", "max"]

    def _fit(self, model: nn.Module, train_loader: DataLoader) -> NoneType:
        for scorer in self.scorers:
            if not scorer.is_fitted:
                scorer.fit(model, train_loader)
            else:
                print(f"scorer {scorer.name} already fitted")

        batch, _ = next(iter(train_loader))
        for i, scorer in enumerate(self.scorers):
            scores = scorer.score_batch(model, batch, features_dataset=self.features_dataset)
            if isinstance(scores, tuple):
                scores = scores[0]

            self.means[i] = scores.mean()
            self.std[i] = scores.std()

    def score_batch(self, model: nn.Module, data: Union[Mapping[str, Tensor], Tensor], features_dataset: bool = False) -> Tensor:
        scores = []
        for i, scorer in enumerate(self.scorers):
            score = scorer.score_batch(model, data, features_dataset=features_dataset)
            if isinstance(score, tuple):
                score = score[0]

            scores.append(self.beta * self.coefs[i] * (score - self.means[i]) / self.std[i])

        scores = torch.stack(scores, dim=1)

        if self.mode == "sum":
            return scores.sum(dim=1)
        elif self.mode == "logsumexp":
            return torch.logsumexp(scores, dim=1) / self.beta
        elif self.mode == "max":
            return scores.max(dim=1)[0]
        elif self.mode == "min":
            return scores.min(dim=1)[0]
        else:
            raise NotImplementedError
