from typing import Union
import os
import sys
import copy
import pathlib
import logging
import argparse
from functools import partial

import torch
import pandas as pd
from omegaconf import OmegaConf, errors, open_dict, DictConfig
from hydra.utils import instantiate

import heat.lib as lib
from heat import scorers

NoneType = type(None)

OmegaConf.register_new_resolver("join", lambda *pth: os.path.join(*pth))
OmegaConf.register_new_resolver("len", lambda x: 1 if isinstance(x, str) else len(x))
OmegaConf.register_new_resolver("mult", lambda a, b: a * b)
OmegaConf.register_new_resolver("sum", lambda a, b: a + b)
OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)
OmegaConf.register_new_resolver("ifisnone", lambda a, b: a if (a is not None) else b)

CLASS_SCORERS = {
    'knn': scorers.KNNScorer,
    'ssd': partial(scorers.SSDScorer, name="ssd"),
    'ssd_std': partial(scorers.SSDScorer, pooling='std', name="ssd_std"),
    'ssd_react': partial(scorers.SSDScorer, use_react=True, name="ssd_react"),
    'ssd_kmeans': partial(scorers.SSDScorer, name="ssd_kmeans"),
    'el': scorers.EnergyLogitsScorer,
    'dice': scorers.DiceScorer,
    'msp': scorers.MSPScorer,
    'vim': scorers.VIMScorer,
    'maha': partial(scorers.SSDScorer, normalize=False, name="maha"),

    'heat_gmm': partial(scorers.EBMScorer, name="heat_gmm"),
    'heat_gmm_std': partial(scorers.EBMScorer, pooling='std', name="heat_gmm_std"),
    'heat_el': partial(scorers.EBMScorer, name="heat_el"),
}

HP_SCORERS = {
    'CIFAR10': {
        "knn": {
            "k": 50,
            "normalize": True,
        },
        "ssd": {
            "num_classes": 10,
            "layer_name": "layer4",
            "pooling": 'avg',
            "normalize": True,
        },
        "ssd_std": {
            "num_classes": 10,
            "layer_name": "layer4",
            "input_preprocessing": False,
            "normalize": True,
            "pooling": 'std',
        },
        "el": {},
        "dice": {
            "p_dice": 90,
            "clip_th": 1e5,
        },
        "msp": {},
        "odin": {
            "T": 1000,
            "eps": 0.0014,
        },
        "vim": {
            "num_classes": 10,
            "principal_dim": 256,
        },
    },
    'CIFAR100': {
        "knn": {
            "k": 200,
            "normalize": True,
            "use_pca": False,
            "pca_n_principal_components": 64,
            "pca_n_last_components": 64,
            "pca_whiten": False,
        },
        "ssd": {
            "num_classes": 100,
            "layer_name": "layer4",
            "pooling": 'avg',
            "use_pca": False,
            "pca_n_principal_components": 128,
            "pca_n_last_components": 0,
            "pca_whiten": False,
            "normalize": True,
            "use_react": False,
        },
        "ssd_std": {
            "num_classes": 100,
            "layer_name": "layer4",
            "normalize": True,
            "pooling": 'std',
            "force_fit_base_dist": True,

        },
        "el": {},
        "dice": {
            "p_dice": 5,
            "clip_th": 1e5,
        },
        "msp": {},
        "vim": {
            "num_classes": 100,
            "principal_dim": 400,
        },
        "heat": {
            "num_classes": 100,
        },
    },
    'Imagenet': {
        "knn": {
            "k": 1000,  # k is scaled accordingly to 1000 * alpha with alpha sampling ratio
            "d": 2048,
            "normalize": True,
            "features_dataset": True,
            "distributed": False,
        },
        "ssd": {
            "num_classes": 1000,
            "max_fit_iter": 1000,
            "layer_name": "layer4",
            "features_dataset": True,
            "force_fit_base_dist": False,
            "normalize": True,
            "d": 2048,
        },
        "ssd_std": {
            "num_classes": 1000,
            "max_fit_iter": 1000,
            "layer_name": "layer4",
            "normalize": True,
            "pooling": 'std',
            "force_fit_base_dist": False,
            "features_dataset": True,
            "d": 2048,
        },
        "el": {
            "features_dataset": True,
        },
        "dice": {
            "p_dice": 70,
            "clip_th": 1e5,
            "features_dataset": True,
        },
        "msp": {},
        "vim": {
            "num_classes": 1000,
            "principal_dim": 256,
            "features_dataset": True,
            "max_fit_iter": 2500,
        },
    },
}


def override_config(config: DictConfig) -> DictConfig:
    dataset_name = f"{config.dataset.ID_NAME.lower()}{'_features' if config.dataset.features_dataset else ''}"
    path_current_conf = os.path.join(pathlib.Path(__file__).parent.resolve(), f"config/dataset/{dataset_name}.yaml")
    config_tmp = OmegaConf.load(path_current_conf)
    if config.backbone.arch in ["resnet50"]:
        config.backbone.dim = 2048
    else:
        config.backbone.dim = 512

    if config.dataset.features_dataset:
        config_tmp.ID.train.root = os.path.join(config.dataset.data_root, "Imagenet_precompute", config.backbone.net.model_id)
    config.dataset = config_tmp
    try:
        config.ebm.model.base_dist.force_fit_base_dist = False
    except errors.ConfigAttributeError:
        pass
    return config


def correct_path(config: DictConfig) -> DictConfig:
    config.dataset.ID.train.root = lib.expand_path(config.dataset.ID.train.root)
    config.dataset.ID.test.root = lib.expand_path(config.dataset.ID.test.root)
    for ood in config.dataset.OOD:
        getattr(config.dataset.OOD, ood).root = lib.expand_path(getattr(config.dataset.OOD, ood).root)

    return config


def main(args: argparse.Namespace) -> NoneType:
    config = OmegaConf.load(os.path.join(args.dir, args.config_file))
    config = override_config(config, args.open_ood, args.vim_vit)
    config = correct_path(config)

    train_transform = instantiate(config.transform)

    train_dataset = instantiate(config.dataset.ID.train, transform=train_transform)
    test_dataset = instantiate(config.dataset.ID.test)
    ood_datasets = instantiate(config.dataset.OOD)

    train_loader = instantiate(config.dataloader.test, train_dataset)
    test_loader = instantiate(config.dataloader.test, test_dataset)
    ood_loaders = {dts_name: instantiate(config.dataloader.test, dts) for dts_name, dts in ood_datasets.items()}

    model = instantiate(config.backbone.net)
    model = model.to('cuda', non_blocking=True)
    model.eval()
    model.requires_grad_(False)

    res = []
    scorers_to_combine = []
    for scorer_name in args.scorers:
        if scorer_name in ["heat_gmm", "heat_gmm_std", "heat_el"]:
            state = lib.load_checkpoint(os.path.join(args.dir, args.ckpts.pop(0)))
            ebm_config = OmegaConf.create(state["config"])
            if config.dataset.ID_NAME.lower() == "imagenet":
                with open_dict(ebm_config):
                    ebm_config.ebm.model.reduce_width = True

            ebm = instantiate(ebm_config.ebm.model, input_dim=model.feature_dims[-1], num_classes=model.num_classes)

            try:
                if ebm_config.ebm.model.base_dist.use_std_pooling:
                    lib.LOGGER.info('changing to std')
                    ebm.base_dist.pooling = "std"
            except errors.ConfigAttributeError:
                pass
            ebm = ebm.to('cuda', non_blocking=True)

            ebm, _, _ = lib.load_from_checkpoint(state, ebm, model=model, train_loader=train_loader)
            ebm.eval()
            hp_scorer = {"ebm": ebm,
                         "features_dataset": config.dataset.features_dataset}
            try:
                if ebm_config.ebm.model.base_dist.use_std_pooling:
                    lib.LOGGER.info('changing to std')
                    hp_scorer['pooling'] = 'std'
            except errors.ConfigAttributeError:
                pass
        else:
            hp_scorer = HP_SCORERS[config.dataset.ID_NAME][scorer_name]
            if scorer_name == "knn":
                hp_scorer.update({"d": model.feature_dims[-1]})
                if sbst is not None:
                    hp_scorer.update({"k": max(1, int(hp_scorer['k'] * sbst))})

        scorer = CLASS_SCORERS[scorer_name](**hp_scorer)
        scorer = lib.to_device(scorer)

        if not args.combine_only:
            lib.LOGGER.info(f"Evaluating {scorer.name} scorer")
            scorer.fit(model, train_loader)
            df = scorer.ood_results(model, test_loader, ood_loaders, print_res=True, open_ood=args.open_ood)

        if args.combine:
            if scorer_name in args.combine:
                scorers_to_combine.append(scorer)
        else:
            del scorer

        if not args.combine_only:
            df.set_axis([f"{scorer_name}"], axis=0, inplace=True)
            res.append(df)

    # Combining scorers
    if args.combine:
        for beta in args.beta:
            comb_scorer = scorers.CombineScorer(scorers=scorers_to_combine, beta=beta, features_dataset=config.dataset.features_dataset)
            comb_scorer.fit(model, train_loader)
            df = comb_scorer.ood_results(model, test_loader, ood_loaders, print_res=True, open_ood=args.open_ood)
            comb_scorer_name = "_".join(args.combine) + "_beta=" + str(beta)
            del comb_scorer
            df.set_axis([f"{comb_scorer_name} Scorer"], axis=0, inplace=True)
            res.append(df)

    lib.LOGGER.info("*** OOD detection results ***")
    res = pd.concat(res).to_string(index=True)
    for r in res.split('\n'):
        lib.LOGGER.info(r)


if __name__ == "__main__":
    def _read_beta(beta: Union[str, float]) -> float:
        if beta == 'neginf':
            return -float('inf')
        return float(beta)


    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

    parser = argparse.ArgumentParser("Evaulate OOD detection")
    parser.add_argument("--dir", required=True, type=str)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--ckpts", nargs="*")
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--scorers", required=True, nargs="*", default=['knn', 'ssd', 'heat_gmm'],
                        choices=list(CLASS_SCORERS.keys()) + ['all'])
    parser.add_argument("--combine", nargs="*", choices=list(CLASS_SCORERS.keys()))
    parser.add_argument("--combine_only", default=False, action='store_true')
    parser.add_argument("--beta", type=_read_beta, default=[0], nargs='*')

    args = parser.parse_args()

    if args.scorers == ['all']:
        args.scorers = list(CLASS_SCORERS.keys())

    main(args)
