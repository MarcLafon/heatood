from typing import List, Mapping, Any, Optional
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

import heat.lib as lib
from heat.lib import ood_metrics as om

NoneType = type(None)
KwargsType = Mapping[str, Any]


class AbastractOODScorer(nn.Module):
    def __init__(
        self,
        name: str = "",
        layer_name: str = "layer4",
        normalize: bool = False,
        pooling: str = 'avg',
        features_dataset: bool = False,
        use_pca: bool = False,
        pca_n_principal_components: int = 128,
        pca_n_last_components: int = 0,
        pca_whiten: bool = False,
        max_fit_iter: float = np.inf,
        use_react: bool = False,
        input_preprocessing: bool = False,
        eps: float = 0.005,
        react_p: float = 0.95,
        **kwargs: KwargsType,
    ) -> NoneType:
        super().__init__()
        assert (pooling in ['avg', 'std', 'avg_std', 'token'])

        self.is_fitted = False
        self.name = name
        self.normalize = normalize
        self.layer_name = layer_name
        self.pooling = pooling
        self.features_dataset = features_dataset
        self.use_pca = use_pca
        self.pca_n_principal_components = pca_n_principal_components
        self.pca_n_last_components = pca_n_last_components
        self.pca_whiten = pca_whiten if self.use_pca else False
        self.use_react = use_react
        self.input_preprocessing = input_preprocessing
        self.eps = eps
        self.react_p = react_p
        self.react_threshold = None
        self.max_fit_iter = max_fit_iter
        self.max_fit_iter_react = 20

        if self.use_pca:
            self.pca = lib.PCA(self.pca_n_principal_components, self.pca_whiten, self.pca_n_last_components)

    def fit_react(self, model: nn.Module, train_loader: DataLoader) -> NoneType:
        model.eval()
        features = []

        for i, (data, target) in enumerate(tqdm(train_loader, f'Fitting React at {self.layer_name}', disable=os.getenv('TQDM_DISABLE'))):
            data = lib.to_device(data)

            if self.features_dataset:
                z = data[self.pooling].float()
            else:
                z = lib.get_features(model, data, layer_names=[self.layer_name])[self.pooling][-1]

            features.append(z)

            if i > self.max_fit_iter_react:
                break

        features = torch.cat(features)
        self.react_threshold = torch.quantile(features, self.react_p).item()

        del features
        torch.cuda.empty_cache()

    def fit_pca(self, model: nn.Module, train_loader: DataLoader) -> NoneType:
        model.eval()
        features = []

        for i, (data, target) in enumerate(tqdm(train_loader, f'Fitting PCA@{self.pca_n_principal_components} at {self.layer_name}', disable=os.getenv('TQDM_DISABLE'))):
            data = lib.to_device(data)
            if self.features_dataset:
                z = data[self.pooling].float()
            else:
                z = lib.get_features(model, data, layer_names=[self.layer_name])[self.pooling][-1]

            if self.normalize:
                z = F.normalize(z)
            features.append(z)

            if i > self.max_fit_iter:
                break

        features = torch.cat(features)
        self.pca.fit(features)
        del features
        torch.cuda.empty_cache()

    def _fit(
            self,
            model: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
    ) -> NoneType:
        pass

    def fit(
            self,
            model: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
    ) -> NoneType:
        model.eval()

        if self.use_react:
            self.fit_react(model, train_loader)

        if self.use_pca:
            self.fit_pca(model, train_loader)

        self._fit(model, train_loader)
        self.is_fitted = True

    def energy(self, z: Tensor, labels: Optional[Tensor] = None) -> torch.Tensor:
        return torch.zeros((z.shape[0],), device=z.device)

    @property
    def has_sample(self) -> bool:
        return False

    def _score_batch(
            self,
            model: torch.nn.Module,
            data: torch.Tensor,
            features_dataset: bool = False,
            with_grad: bool = False,
    ) -> torch.Tensor:
        pass

    def score_batch(
            self,
            model: torch.nn.Module,
            data: torch.Tensor,
            features_dataset: bool = False,
    ) -> torch.Tensor:

        if self.input_preprocessing and not features_dataset:
            is_score_batch_tuple = isinstance(self._score_batch(model, data, features_dataset=features_dataset), tuple)
            data.requires_grad = True
            score_data = self._score_batch(model, data, features_dataset, with_grad=True)
            if is_score_batch_tuple:
                score_data = score_data[0]
            loss = score_data.mean()
            loss.backward()
            gradient = torch.sign(data.grad.data)
            data.data = data - self.eps * gradient
            data = data.detach()

        return self._score_batch(model, data, features_dataset=features_dataset)

    def get_scores(
            self,
            model: torch.nn.Module,
            ood_loader: torch.utils.data.DataLoader,
            max_iter: float = np.inf,
            force_tqdm_disable: bool = True,
    ) -> torch.Tensor:
        tqdm_disable = os.getenv('TQDM_DISABLE') and force_tqdm_disable
        model.eval()
        data, _ = next(iter(ood_loader))
        is_score_batch_tuple = isinstance(self._score_batch(model, data), tuple)
        scores = []
        norms = []
        if is_score_batch_tuple:
            scores_nn = []
            scores_g = []

        for i, (data, _) in enumerate(tqdm(ood_loader, disable=tqdm_disable)):
            data = data.to('cuda', non_blocking=True)
            _score = self.score_batch(model, data)

            # *** This part is only to log the norm..  *** #
            z = lib.get_features(model, data, layer_names=[self.layer_name])['avg'][-1]

            if self.use_react:
                z = z.clip(max=self.react_threshold)

            if self.use_pca:
                if self.normalize:
                    z = F.normalize(z, dim=1)
                z = self.pca.transform(z)

            norms.append(z.norm(2, 1).mean().cpu().detach())
            # ****** *** *** *** *** *** *** *** ***  **** #

            if is_score_batch_tuple:
                scores.append(_score[0].detach().squeeze().cpu())
                scores_nn.append(_score[1].detach().squeeze().cpu())
                scores_g.append(_score[2].detach().squeeze().cpu())

            else:
                _s = _score.detach().squeeze().cpu()
                if _s.dim() == 0:
                    _s = _s.unsqueeze(0)
                scores.append(_s)

            if i >= max_iter:
                break

        scores = torch.cat(scores)
        norms = torch.stack(norms)
        if is_score_batch_tuple:
            return scores.numpy(), torch.cat(scores_nn).numpy(), torch.cat(scores_g).numpy(), norms.numpy()
        else:
            return scores.numpy(), torch.zeros_like(scores).numpy(), torch.zeros_like(scores).numpy(), norms.numpy()

    @staticmethod
    def get_metrics(dtest: Tensor, dood: Tensor) -> Tensor:
        fpr95 = om.get_fpr(dtest, dood)
        auroc = om.get_auroc(dtest, dood)
        dtacc = om.get_det_accuracy(dtest, dood)
        aupr_in = om.get_aupr_in(dtest, dood)
        aupr_out = om.get_aupr_out(dtest, dood)
        return fpr95, auroc, dtacc, aupr_in, aupr_out

    def ood_results(
            self,
            model: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader,
            ood_loaders: torch.utils.data.DataLoader,
            max_iter: float = np.inf,
            exclude_avg: List = [],
            print_res: bool = True,
            print_energy_values: bool = True,
            force_tqdm_disable: bool = True,
            open_ood: bool = False,
    ) -> pd.DataFrame:
        name_in = test_loader.dataset.__class__.__name__

        model.eval()
        data, _ = next(iter(test_loader))
        is_score_batch_tuple = isinstance(self.score_batch(model, data), tuple)

        table_names = [name_ood for name_ood in ood_loaders.keys() if name_ood not in exclude_avg]
        df_res = pd.DataFrame([[' / &'] * len(table_names)], columns=table_names)
        df_energy_values = pd.DataFrame([[' / &'] * len(["Test"] + table_names)], columns=["Test"] + table_names)
        df_energy_values.set_axis(["Energy"], axis=0, inplace=True)

        df_norms = pd.DataFrame([[' / &'] * len(["Test"] + table_names)], columns=["Test"] + table_names)
        df_norms.set_axis(["Norm"], axis=0, inplace=True)

        if is_score_batch_tuple:
            df_energy_nn_values = pd.DataFrame([[' / &'] * len(["Test"] + table_names)], columns=["Test"] + table_names)
            df_energy_g_values = pd.DataFrame([[' / &'] * len(["Test"] + table_names)], columns=["Test"] + table_names)
            df_energy_nn_values.set_axis(["Energy NN"], axis=0, inplace=True)
            df_energy_g_values.set_axis(["Energy Prior"], axis=0, inplace=True)

        _test_scores = self.get_scores(model, test_loader, force_tqdm_disable=force_tqdm_disable)
        test_scores = _test_scores[0]
        test_norms = _test_scores[3]
        df_energy_values["Test"] = [f" {np.around(np.mean(test_scores), 3):2.3f} ± {np.around(np.std(test_scores), 3):2.3f}"]
        df_norms["Test"] = [f" {np.around(np.mean(test_norms), 3):2.3f} ± {np.around(np.std(test_norms), 3):2.3f}"]
        if is_score_batch_tuple:
            test_scores_nn = _test_scores[1]
            test_scores_g = _test_scores[2]
            df_energy_nn_values["Test"] = [f" {np.around(np.mean(test_scores_nn), 3):2.3f} ± {np.around(np.std(test_scores_nn), 3):2.3f}"]
            df_energy_g_values["Test"] = [f" {np.around(np.mean(test_scores_g), 3):2.3f} ± {np.around(np.std(test_scores_g), 3):2.3f}"]

        fpr95_avg, auroc_avg, aupr_in_avg = 0, 0, 0
        for ood_name, ood_loader in ood_loaders.items():
            _ood_scores = self.get_scores(model, ood_loader, max_iter=max_iter, force_tqdm_disable=force_tqdm_disable)
            ood_scores = _ood_scores[0]
            ood_norms = _ood_scores[3]
            if is_score_batch_tuple:
                ood_scores_nn = _ood_scores[1]
                ood_scores_g = _ood_scores[2]

            fpr95, auroc, dtacc, aupr_in, aupr_out = self.get_metrics(test_scores, ood_scores)

            if ood_name not in exclude_avg:
                fpr95_avg += fpr95
                auroc_avg += auroc
                aupr_in_avg += aupr_in

                if open_ood:
                    df_res[ood_name] = [f"{np.around(100 * fpr95, 1):2.1f} / {np.around(100 * auroc, 1):2.1f} / {np.around(100 * aupr_in, 1):2.1f}&"]
                else:
                    df_res[ood_name] = [f"{np.around(100 * fpr95, 1):2.1f} / {np.around(100 * auroc, 1):2.1f}&"]

                df_energy_values[ood_name] = [f" {np.around(np.mean(ood_scores), 3):2.3f} ± {np.around(np.std(ood_scores), 3):2.3f}"]
                df_norms[ood_name] = [f" {np.around(np.mean(ood_norms), 3):2.3f} ± {np.around(np.std(ood_norms), 3):2.3f}"]
                if is_score_batch_tuple:
                    df_energy_nn_values[ood_name] = [f" {np.around(np.mean(ood_scores_nn), 3):2.3f} ± {np.around(np.std(ood_scores_nn), 3):2.3f}"]
                    df_energy_g_values[ood_name] = [f" {np.around(np.mean(ood_scores_g), 3):2.3f} ± {np.around(np.std(ood_scores_g), 3):2.3f}"]

        N = len(table_names)
        if open_ood:
            df_res["Average"] = [f" {np.around(100 * fpr95_avg / N, 1):2.1f} / {np.around(100 * auroc_avg / N, 1):2.1f} / {np.around(100 * aupr_in_avg  / N, 1):2.1f}&"]
        else:
            df_res["Average"] = [f" {np.around(100 * fpr95_avg / N, 1):2.1f} / {np.around(100 * auroc_avg / N, 1):2.1f}"]

        if print_res:
            res = df_res.to_string(index=True)
            lib.LOGGER.info(f"*** Method {self.name}, IN = {name_in} (FPR95 / AUC):")
            lib.LOGGER.info("*** =============")
            for r in res.split('\n'):
                lib.LOGGER.info("*** " + r)

        if print_energy_values:
            if is_score_batch_tuple:
                energy_res = pd.concat([df_energy_values, df_energy_nn_values, df_energy_g_values, df_norms]).to_string(index=True)
            else:
                energy_res = pd.concat([df_energy_values, df_norms]).to_string(index=True)

            lib.LOGGER.info("*** =============")
            lib.LOGGER.info("*** Energy values:")
            for r in energy_res.split('\n'):
                lib.LOGGER.info("*** " + r)
            lib.LOGGER.info("*** =============")
        return df_res
