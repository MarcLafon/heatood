import torch

import heat.lib as lib
from heat.scorers.abstract_scorer import AbastractOODScorer

class MaxLogitScorer(AbastractOODScorer):
    def __init__(self,  pooling='avg', **kwargs):
        assert pooling in ['avg', 'token']
        super(MaxLogitScorer, self).__init__(**kwargs)
        self.name = "MaxLogit"
        self.is_fitted = False
        self.classifier_head = None
        self.pooling = pooling

    def _fit(self, model, train_loader):
        self.classifier_head = model.fc

    def energy(self, z, labels=None):
        logits = self.classifier_head(z)
        return - logits.max(dim=1)[0]

    @property
    def has_sample(self):
        return False

    @torch.no_grad()
    def _score_batch(self, model, data, features_dataset=False):
        data = lib.to_device(data)

        if features_dataset:
            z = data[self.pooling].float()
        else:
            z = lib.get_features(model, data, layer_names=[self.layer_name])[self.pooling][-1]

        if self.use_react:
            z = z.clip(max=self.react_threshold)

        return self.energy(z)
