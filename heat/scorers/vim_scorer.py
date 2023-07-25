import torch
import numpy as np
from tqdm import tqdm
import os

import heat.lib as lib
from heat.scorers.abstract_scorer import AbastractOODScorer

# step 1 : Compute o =  - W_inv @ b
# step 2 : Compute covariance of translated features z - o : Sigma
# step 3 : Compute SVD of Sigma (eig_vals, eig_vectors)
# step 4 : Compute NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T) ??


class VIMScorer(AbastractOODScorer):
    def __init__(
            self,
            num_classes: int,
            principal_dim: int = 128,
            name: str = "VIM",
            max_fit_iter: int = np.inf,
            **kwargs
    ):
        super(VIMScorer, self).__init__(**kwargs)
        self.name = name
        self.num_classes = num_classes
        self.max_fit_iter = max_fit_iter
        self.R = None
        self.alpha = 1
        self.principal_dim = principal_dim
        self.u = None

    def _fit(self, model, train_loader):
        model.eval()
        self.u = - torch.pinverse(model.fc.weight).matmul(model.fc.bias)
        features = []
        logits = []
        for i, (data, _) in enumerate(tqdm(train_loader, f'Fitting covariance matrix at {self.layer_name}', disable=os.getenv('TQDM_DISABLE'))):
            data = lib.to_device(data)
            if self.features_dataset:
                z = data['avg'].float()
            else:
                z = lib.get_features(model, data, layer_names=[self.layer_name])['avg'][-1]

            logits.append(model.fc.forward(z).detach())
            features.append(z-self.u)

            if i > self.max_fit_iter:
                break

        Z = torch.cat(features)
        logits = torch.cat(logits)
        cov = np.cov(Z.T.cpu().numpy(), bias=True)
        # solved singular values pb in cholesky (see https://scicomp.stackexchange.com/questions/30631/how-to-find-the-nearest-a-near-positive-definite-from-a-given-matrix)
        cov += 1e-12 * np.eye(cov.shape[0])
        cov = torch.from_numpy(cov).float().to('cuda', non_blocking=True)

        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        eig_sorted = torch.argsort(eigenvalues, descending=True)[self.principal_dim:]
        self.R = eigenvectors.T[eig_sorted]
        self.R = self.R.T.contiguous()

        vlogits = torch.norm(Z.matmul(self.R), dim=-1)
        self.alpha = logits.max(dim=-1)[0].mean() / vlogits.mean()
        del features, logits, vlogits
        torch.cuda.empty_cache()

    @torch.no_grad()
    def score_batch(self, model, data, features_dataset=False):
        data = lib.to_device(data)
        if features_dataset:
            z = data['avg'].float()
        else:
            z = lib.get_features(model, data, layer_names=[self.layer_name])['avg'][-1]

        z = z - self.u
        logits = model.fc.forward(z)
        vlogits = torch.norm(z.matmul(self.R), dim=-1)
        energy = - torch.logsumexp(logits, dim=1)
        return self.alpha * vlogits + energy
