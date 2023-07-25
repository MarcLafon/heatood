import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import heat.lib as lib
from heat.scorers.abstract_scorer import AbastractOODScorer
from torch.distributions import Categorical

class KLMScorer(AbastractOODScorer):
    def __init__(self,
                 num_classes,
                 pooling='avg',
                 **kwargs):
        assert pooling in ['avg', 'token']
        super(KLMScorer, self).__init__(**kwargs)
        self.name = "KLMatching"
        self.is_fitted = False
        self.num_classes = num_classes
        self.pooling = pooling

        self.classifier_head = None
        self.posterior_templates = [torch.zeros(self.num_classes, device="cuda") for _ in range(self.num_classes)]

    def _fit(self, model, train_loader):
        self.classifier_head = model.fc

        if self.features_dataset:
            assert self.layer_name == "layer4", "Features dataset not yet implemented for different layers"

        class_count = torch.zeros(self.num_classes, device="cuda")

        for i, (data, target) in enumerate(tqdm(train_loader, 'Fitting KL Matching templates', disable=os.getenv('TQDM_DISABLE'))):
            data = lib.to_device(data)

            if self.features_dataset:
                z = data[self.pooling].float()
            else:
                z = lib.get_features(model, data, layer_names=[self.layer_name])[self.pooling][-1]

            logits = self.classifier_head(z)
            probs = F.softmax(logits, dim=1)
            y_pred = probs.max(dim=1)[1]
            class_count += y_pred.float().histc(self.num_classes, max=self.num_classes)

            for label in range(self.num_classes):
                if probs[y_pred == label].shape[0] > 0:
                    self.posterior_templates[label] += probs[y_pred == label].sum(dim=0)

            if i > self.max_fit_iter:
                break

        for label in range(self.num_classes):
            self.posterior_templates[label] /= class_count[label]


    def energy(self, z, labels=None):
        logits = self.classifier_head(z)
        probs = F.softmax(logits, dim=1)

        kl_divs = []
        for label in range(self.num_classes):
            kl_divs.append((probs * (probs.log() - self.posterior_templates[label].log().view(1,-1))).sum(dim=1))

        kl_divs = torch.stack(kl_divs, dim=1)
        return kl_divs.min(dim=1)[0]

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

        return self.energy(z)
