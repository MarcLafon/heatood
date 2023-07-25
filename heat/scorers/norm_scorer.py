import torch
import torch.nn.functional as F

import heat.lib as lib
from heat.scorers.abstract_scorer import AbastractOODScorer


class NormScorer(AbastractOODScorer):
    def __init__(self, **kwargs):
        super(NormScorer, self).__init__(**kwargs)
        self.name = "Norm"
        self.is_fitted = True

    @torch.no_grad()
    def score_batch(self, model, data, features_dataset=False):
        data = lib.to_device(data)
        if features_dataset:
            z = data['avg'].float()
        else:
            features = model.forward_features(data)
            z = model.forward_head(features, pre_logits=True).detach()

        if self.normalize:
            z = F.normalize(z, dim=1)

        if self.use_pca:
            z = self.pca.transform(z)

        return torch.norm(z, dim=1)
