import torch
import torch.nn.functional as F

import heat.lib as lib
from heat.scorers.abstract_scorer import AbastractOODScorer


class MSPScorer(AbastractOODScorer):
    def __init__(self, pooling='avg', **kwargs):
        assert pooling in ['avg', 'token']
        super(MSPScorer, self).__init__(**kwargs)
        self.name = "MSP"
        self.is_fitted = False
        self.classifier_head = None
        self.pooling = pooling

    def _fit(self, model, train_loader):
        self.classifier_head = model.fc

    def energy(self, z, labels=None):
        logits = self.classifier_head(z)
        probs = F.softmax(logits, dim=1)
        return - probs.max(dim=1)[0]

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
