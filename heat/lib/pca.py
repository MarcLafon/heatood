import torch
import numpy as np


def _to_numpy(tens, dtype=np.float32):
    return tens.cpu().numpy().astype(dtype)


class PCA:
    def __init__(
            self,
            n_principal_components,
            whiten=True,
            n_last_components=0,
    ):
        super().__init__()
        self.n_principal_components = n_principal_components
        self.whiten = whiten
        self.n_last_components = n_last_components
        self.U, self.S, self.V = None, None, None
        self.n_samples = None
        self.mean = None
        self.components_ = None
        self.explained_variance_ = None

    def fit(
            self,
            features,
    ):
        self.mean = features.mean(dim=0)
        self.n_samples = features.size(0)
        X = features - self.mean
        self.U, self.S, self.V = torch.linalg.svd(X.float(), full_matrices=False)
        max_abs_cols = self.U.abs().argmax(0)
        signs = torch.sign(self.U[max_abs_cols, range(self.U.size(1))])
        self.U *= signs
        self.V *= signs.unsqueeze(1)

        self.components_ = self.V[:self.n_principal_components]
        self.explained_variance_ = ((self.S ** 2) / (self.n_samples - 1))[:self.n_principal_components]

    def transform(self, X):
        is_numpy = isinstance(X, (np.ndarray, np.memmap))
        mean = _to_numpy(self.mean, X.dtype) if is_numpy else self.mean
        components_ = _to_numpy(self.components_, X.dtype) if is_numpy else self.components_
        if self.whiten:
            exvar = torch.sqrt(self.explained_variance_)
            exvar = _to_numpy(exvar, X.dtype) if is_numpy else exvar

        X = (X - mean) @ components_.T
        if self.whiten:
            X /= exvar
        return X
