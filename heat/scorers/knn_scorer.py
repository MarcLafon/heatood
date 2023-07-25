from typing import Any, Mapping, Union
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
import faiss
from tqdm import tqdm

import heat.lib as lib
from heat.scorers.abstract_scorer import AbastractOODScorer

NoneType = type(None)
KwargsType = Mapping[str, Any]


class KNNScorer(AbastractOODScorer):
    def __init__(
        self,
        k: int,
        d: int,
        name: str = "knn",
        use_gpu: bool = True,
        distributed: bool = False,
        max_fit_iter: int = np.inf,
        pooling: str = 'avg',
        **kwargs: KwargsType,
    ) -> NoneType:
        super(KNNScorer, self).__init__(**kwargs)
        self.name = name
        self.k = k
        self.d = d
        self.use_gpu = use_gpu
        self.distributed = distributed
        self.max_fit_iter = max_fit_iter
        self.pooling = pooling

        if self.use_pca:
            self.d = self.pca_n_principal_components + self.pca_n_last_components

        self.index = faiss.IndexFlatL2(self.d, )
        if self.use_gpu:
            # if there are some GPU memory issue set use_gpu to False
            if self.distributed:
                # co = faiss.GpuMultipleClonerOptions()
                # co.shards = True
                # self.index = faiss.index_cpu_to_all_gpus(self.index, co)
                self.index = faiss.index_cpu_to_all_gpus(self.index,)
            else:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    @torch.no_grad()
    def _fit(self, model: nn.Module, train_loader: DataLoader) -> NoneType:
        model.eval()
        Z = []
        if not self.features_dataset:
            for data, target in tqdm(train_loader, 'Fitting KNN', disable=os.getenv('TQDM_DISABLE')):
                data = lib.to_device(data)
                z = lib.get_features(model, data, layer_names=[self.layer_name])[self.pooling][-1]

                if self.normalize:
                    z = F.normalize(z, dim=1)

                if self.use_pca:
                    z = self.pca.transform(z)

                Z.append(z)
            Z = torch.cat(Z).cpu().numpy()
        else:
            # https://github.com/deeplearning-wisc/knn-ood/blob/master/run_imagenet.py
            if self.pooling == 'avg':
                features = train_loader.dataset.features.astype(np.float32)
            elif self.pooling == 'token':
                features = train_loader.dataset.tokens.astype(np.float32)

            if self.normalize:
                features = features / (np.linalg.norm(features, axis=-1, keepdims=True) + 1e-10)

            if self.use_pca:
                features = self.pca.transform(features)

            Z = np.ascontiguousarray(features.astype(np.float32))

        lib.LOGGER.info('Adding index')
        self.index.add(Z)
        self.is_fitted = True

    @torch.no_grad()
    def _score_batch(self, model: nn.Module, data: Union[Mapping[str, Tensor], Tensor], features_dataset: bool = False) -> Tensor:
        data = lib.to_device(data)
        if features_dataset:
            z = data[self.pooling].float()
        else:
            z = lib.get_features(model, data, layer_names=[self.layer_name])[self.pooling][-1]

        if self.normalize:
            z = F.normalize(z)

        if self.use_pca:
            z = self.pca.transform(z)

        D, _ = self.index.search(z.cpu().numpy(), self.k)

        return torch.from_numpy(D[:, -1]).to('cuda', non_blocking=True)
