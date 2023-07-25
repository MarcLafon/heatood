import torch
import torch.nn.functional as F

import heat.lib as lib
from heat.scorers.abstract_scorer import AbastractOODScorer


class ODINScorer(AbastractOODScorer):
    def __init__(self, T=1000, eps=0.0014, **kwargs):
        super(ODINScorer, self).__init__(**kwargs)
        self.T = T
        self.eps = eps
        self.name = "ODIN"
        self.is_fitted = True

    def add_perturbation(self, model, data):
        x = data.clone()
        x.requires_grad = True
        out = model(x)
        labels = F.softmax(out, dim=1).max(dim=1)[1]

        s_x = F.softmax(out / self.T, dim=1).gather(1, labels[:, None])
        grad_x = torch.autograd.grad(torch.log(s_x).mean(), [x])[0]
        x.data = x.data - self.eps * torch.sign(-grad_x)
        return x.detach()

    def score_batch(self, model, data, features_dataset=False):
        data = lib.to_device(data)
        if features_dataset:
            raise NotImplementedError('ODINScorer is not implemented for features datasets')
        x = self.add_perturbation(model, data)
        with torch.no_grad():
            probs = F.softmax(model(x) / self.T, dim=1)

        return - probs.max(dim=1)[0]
