from typing import Any, Mapping, Optional, Tuple
import os
import pathlib

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from heat.scorers.abstract_scorer import AbastractOODScorer
import heat.lib as lib
from heat.lib.distributions import Categorical, MultivariateNormal, MixtureSameFamily

NoneType = type(None)
KwargsType = Mapping[str, Any]


class SSDScorer(AbastractOODScorer):
    def __init__(
        self,
        num_classes: int,
        cache_base_dist: bool = False,
        force_fit_base_dist: bool = False,
        diag_coefficient_only: bool = False,
        use_simplified_mahalanobis_score: bool = False,
        use_kmeans: bool = False,
        d: int = 512,
        use_gpu: bool = True,
        name: str = "SSD",
        **kwargs: KwargsType,
    ) -> NoneType:
        super(SSDScorer, self).__init__(**kwargs)
        self.name = name
        self.num_classes = num_classes
        self.cache_base_dist = cache_base_dist
        self.force_fit_base_dist = force_fit_base_dist
        self.diag_coefficient_only = diag_coefficient_only
        self.use_simplified_mahalanobis_score = use_simplified_mahalanobis_score
        self.use_kmeans = use_kmeans
        self.d = d
        self.use_gpu = use_gpu

        # Internal attributes
        self.global_mean = 0
        self.global_std = 1
        self.means = []
        self.dist = None
        self.comp = None
        self.mix = None
        self.precision = None

        if self.use_kmeans:
            self.kmeans = faiss.Kmeans(d=self.d, k=self.num_classes, niter=1000, update_index=True, nredo=5,
                                       verbose=True, spherical=False, gpu=self.use_gpu)
            lib.LOGGER.info("Using Kmeans clustering, true labels will be ignored")

    def reset(self) -> NoneType:
        self.global_mean = 0
        self.global_std = 1
        self.means = []
        self.dist = None
        self.comp = None
        self.mix = None
        self.precision = None

    @property
    def cache_name(self) -> str:
        str_cache = ""
        str_normalize = "_normalized" if self.normalize else ""
        str_diag_only = "_diagonly" if self.diag_coefficient_only else ""
        str_pooling = "_" + self.pooling + "_pooling"

        str_pca = ""
        if self.use_pca:
            str_pca = f"_pca_n_principal_comp{self.pca_n_principal_components}" if self.use_pca else ""
            str_pca += f"_pca_n_last_comp{self.pca_n_last_components}" if self.use_pca else ""
            str_pca += "_whiten" if self.pca_whiten else ""

        str_kmeans = ""
        if self.use_kmeans:
            str_kmeans = f"_kmeans{self.num_classes}"

        return str_cache + str_normalize + str_pooling + str_diag_only + str_pca + str_kmeans

    @property
    def has_sample(self) -> bool:
        return True

    def sample(self, sample_shape: int, target: Optional[Tensor] = None, temp: Optional[float] = None) -> Tuple[Tensor]:
        rand_labels, rand_imgs = self.dist.sample(sample_shape, target, temp)
        rand_labels = rand_labels if self.use_simplified_mahalanobis_score else None
        return rand_labels, rand_imgs

    def fit_kmeans(self, model: nn.Module, train_loader: DataLoader) -> NoneType:
        model.eval()
        features = []

        for i, (data, target) in enumerate(tqdm(train_loader, f'Fitting kmeans with K={self.num_classes}', disable=os.getenv('TQDM_DISABLE'))):
            data = lib.to_device(data)

            if self.features_dataset:
                z = data[self.pooling].float()
            else:
                z = lib.get_features(model, data, layer_names=[self.layer_name])[self.pooling][-1]

            # if self.normalize:
            #     z = F.normalize(z, dim=1)

            features.append(z)

        features = torch.cat(features).cpu().numpy().astype(np.float32)

        self.kmeans.train(features)

        del features
        torch.cuda.empty_cache()

    def _fit(self, model: nn.Module, train_loader: DataLoader) -> NoneType:
        self.reset()

        if self.features_dataset:
            assert self.layer_name == "layer4", "Features dataset not yet implemented for different layers"

        model.eval()

        if self.cache_base_dist:
            cache_dir = os.path.join(pathlib.Path(__file__).parent.parent.parent.resolve(), 'cache_gmm')
            os.makedirs(cache_dir, exist_ok=True)
            file = os.path.join(cache_dir, model.model_id + f"_{self.layer_name}" + self.cache_name)

            if os.path.isfile(file) and (not self.force_fit_base_dist):
                cache = torch.load(file, map_location='cuda')
                self.mix = Categorical(cache['probs'].to('cuda', non_blocking=True))
                self.comp = MultivariateNormal(
                    loc=cache['loc'].to('cuda', non_blocking=True),
                    scale_tril=cache['scale_tril'].to('cuda', non_blocking=True),
                    validate_args=False,
                )
                self.dist = MixtureSameFamily(self.mix, self.comp, validate_args=False)
                lib.LOGGER.info("Dist loaded from cache")
                return

        if self.use_kmeans:
            self.fit_kmeans(model, train_loader)

        class_features = [[] for _ in range(self.num_classes)]
        class_count = torch.zeros(self.num_classes, device="cuda")

        for i, (data, target) in enumerate(tqdm(train_loader, 'Fitting SSD dist', disable=os.getenv('TQDM_DISABLE'))):
            data = lib.to_device(data)
            # import ipdb; ipdb.set_trace()

            if self.features_dataset:
                z = data[self.pooling].float()
            else:
                z = lib.get_features(model, data, layer_names=[self.layer_name])[self.pooling][-1]

            if self.use_kmeans:
                _, target = self.kmeans.index.search(z.cpu().numpy().astype(np.float32), 1)
                target = torch.from_numpy(target).view(-1)
            target = target.to('cuda', non_blocking=True)
            class_count += target.float().histc(self.num_classes, max=self.num_classes)

            if self.use_react:
                z = z.clip(max=self.react_threshold)

            if self.normalize:
                z = F.normalize(z, dim=1)

            if self.use_pca:
                z = self.pca.transform(z)

            if self.num_classes == 1:
                class_features[0].append(z)
            else:
                for c in range(self.num_classes):
                    class_features[c].append(z[target == c])

            if i > self.max_fit_iter:
                break

        for c in range(self.num_classes):
            class_features[c] = torch.cat(class_features[c])
            self.means.append(class_features[c].mean(dim=0, keepdim=True))

        Z = torch.cat([class_features[c] - self.means[c] for c in range(self.num_classes)])
        del class_features
        cov = np.cov(Z.T.cpu().numpy(), bias=True)
        # solved singular values pb in cholesky (see https://scicomp.stackexchange.com/questions/30631/how-to-find-the-nearest-a-near-positive-definite-from-a-given-matrix)
        cov += 1e-12 * np.eye(cov.shape[0])
        cov = torch.from_numpy(cov).float().to('cuda', non_blocking=True)
        if self.diag_coefficient_only:
            cov *= torch.eye(cov.size(0), device='cuda')
        L = torch.linalg.cholesky(cov)
        del cov
        self.mix = Categorical(class_count)
        self.comp = MultivariateNormal(loc=torch.cat(self.means), scale_tril=L.float(), validate_args=False)
        self.dist = MixtureSameFamily(self.mix, self.comp, validate_args=False)
        torch.cuda.empty_cache()

        if self.cache_base_dist:
            lib.LOGGER.info("Base dist dumped in cache")
            torch.save(
                {'probs': self.mix.probs, 'loc': self.comp.loc, 'scale_tril': self.comp._unbroadcasted_scale_tril},
                file)  # cache _unbroadcasted_scale_tril because it is the real scale_tril

    def _maha_score(self, z: Tensor) -> Tensor:
        return lib._batch_mahalanobis(self.comp._unbroadcasted_scale_tril, z.unsqueeze(1) - self.comp.loc).min(dim=1).values

    def energy(self, z: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        return - self.dist.log_prob(z, labels=labels)

    def _score_batch(
            self,
            model: torch.nn.Module,
            data: torch.Tensor,
            features_dataset: bool = False,
            with_grad: bool = False,
    ) -> torch.Tensor:
        data = lib.to_device(data)
        if features_dataset:
            z = data[self.pooling].float()
        else:
            if with_grad:
                z = lib.get_features_with_grad(model, data, layer_names=[self.layer_name])[self.pooling][-1]
            else:
                z = lib.get_features(model, data, layer_names=[self.layer_name])[self.pooling][-1]

        if self.use_react:
            z = z.clip(max=self.react_threshold)

        if self.normalize:
            z = F.normalize(z, dim=1)

        if self.use_pca:
            z = self.pca.transform(z)

        # if self.input_preprocessing:
        #     gaussian_score = 0.5 * lib._batch_mahalanobis(self.comp._unbroadcasted_scale_tril, z.unsqueeze(1) - self.comp.loc)
        #     pure_gau = gaussian_score.min(dim=1)[0]
        #     loss = pure_gau.mean()
        #     loss.backward()
        #
        #     gradient = torch.sign(data.grad.data)
        #     tempInputs = data - self.eps * gradient
        #
        #     z = lib.get_features(model, tempInputs, layer_names=[self.layer_name])[self.pooling][-1]
        #
        #     if self.normalize:
        #         z = F.normalize(z, dim=1)
        #
        #     if self.use_pca:
        #         z = self.pca.transform(z)

        score = self._maha_score(z)
        assert len(score) == len(z)
        return score
