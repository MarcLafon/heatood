from typing import List, Mapping, Any, Callable
import os
import time
import builtins
import subprocess

import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
from hydra.utils import instantiate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from torch.utils.tensorboard import SummaryWriter

import heat.lib as lib
from heat.scorers import EBMScorer

NoneType = type(None)

OmegaConf.register_new_resolver("join", lambda *pth: os.path.join(*pth))
OmegaConf.register_new_resolver("len", lambda x: 1 if isinstance(x, str) else len(x))
OmegaConf.register_new_resolver("mult", lambda a, b: a * b)
OmegaConf.register_new_resolver("sum", lambda a, b: a + b)
OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)
OmegaConf.register_new_resolver("ifisnone", lambda a, b: a if (a is not None) else b)
OmegaConf.register_new_resolver("cwd", lambda x: hydra.utils.get_original_cwd())

def train_cd(
    ebm_ddp: nn.Module,
    ebm: nn.Module,
    model: nn.Module,
    train_loader: DataLoader,
    cd_loss: Callable,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    epoch: int,
    config: DictConfig,
    writer: SummaryWriter,
    is_master: bool,
) -> lib.DictAverage:
    model.eval()
    ebm_ddp.train()
    _ebm = ebm_ddp.module if isinstance(ebm_ddp, torch.nn.parallel.DistributedDataParallel) else ebm_ddp

    optimizer.zero_grad(set_to_none=True)

    meter = lib.DictAverage()
    progress = lib.ProgressMeter(len(train_loader), meter, prefix=f"Epoch: [{epoch}]")
    end = time.time()

    for i, (data, target) in enumerate(train_loader):
        meter['data_time'].update(time.time() - end)
        data, target = lib.to_device([data, target])

        if config.dataset.features_dataset:
            real_z = data[_ebm.base_dist.pooling].float()
        else:
            real_z = lib.get_features(model, data, layer_names=config.layer_names)[_ebm.base_dist.pooling][-1]

        if _ebm.base_dist.normalize:
            real_z = F.normalize(real_z, dim=1)

        if _ebm.base_dist.use_pca:
            real_z = _ebm.base_dist.pca.transform(real_z)

        loss, logs, gen_z = cd_loss(ebm_ddp, real_z, target)
        loss = loss / config.accum_iter
        meter.update(logs, real_z.size(0))

        loss.backward()

        if ((i + 1) % config.accum_iter == 0) or (i + 1 == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        meter['batch_time'].update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            if is_master:
                progress.display(i)

        if i > _ebm.train_max_iter:
            break

    if is_master:
        progress.display_summary()
    scheduler.step()

    return meter


@hydra.main(config_path="config", config_name='default', version_base="1.1")
def main(config: DictConfig) -> None:
    config.layer_names = [config.layer_names] if isinstance(config.layer_names, str) else config.layer_names
    if config.is_cluster and config.distributed:
        # local rank on the current node / global rank
        local_rank = int(os.environ['SLURM_LOCALID'])
        global_rank = int(os.environ['SLURM_PROCID'])
        # number of processes / GPUs per node
        world_size = int(os.environ['SLURM_NTASKS'])
        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        master_addr = hostnames.split()[0].decode('utf-8')
        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = str(29500)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(global_rank)
        os.environ['LOCAL_RANK'] = str(local_rank)

    if config.distributed:
        torch.distributed.init_process_group(backend='nccl')
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        is_master = rank == 0
        if not is_master:
            os.environ['TQDM_DISABLE'] = 'yes'

            def print_pass(*args: List[Any], **kwargs: Mapping[str, Any]) -> NoneType:
                pass

            builtins.print = print_pass
    else:
        world_size = 1
        rank = 0
        local_rank = 0
        is_master = True

    torch.cuda.set_device(local_rank)

    train_transform = instantiate(config.transform)
    train_dataset = instantiate(config.dataset.ID.train, transform=train_transform)
    train_dataset = lib.create_subset_dataset(
        train_dataset, proportion=config.dataset.subset.proportion, number_labels=config.dataset.subset.number_labels, random_state=config.dataset.subset.random_state)
    test_dataset = instantiate(config.dataset.ID.test)
    test_dataset = lib.create_subset_dataset(test_dataset, number_labels=config.dataset.subset.number_labels, random_state=config.dataset.subset.random_state)
    ood_datasets = instantiate(config.dataset.OOD)

    assert config.dataloader.train.batch_size % world_size == 0, "Batch size should be a multiple of the world size"
    config.dataloader.train.batch_size = config.dataloader.train.batch_size // world_size
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if config.distributed else None
    shuffle = False if config.distributed else True
    train_loader = instantiate(config.dataloader.train, train_dataset, sampler=train_sampler, shuffle=shuffle)
    test_loader = instantiate(config.dataloader.test, test_dataset)
    ood_loaders = {dts_name: instantiate(config.dataloader.test, dts) for dts_name, dts in ood_datasets.items()}

    model = instantiate(config.backbone.net)
    model = model.to('cuda', non_blocking=True)
    model.eval()
    model.requires_grad_(False)

    assert config.n_layers == 1

    ebm = instantiate(config.ebm.model, input_dim=model.feature_dims[-1], num_classes=model.num_classes)
    ebm = ebm.to('cuda', non_blocking=True)

    if config.load_from_checkpoint:
        state = lib.load_checkpoint(config.ckpt_path)
        ebm, _, _ = lib.load_from_checkpoint(state, ebm, model=model, train_loader=train_loader,
                                             features_dataset=config.dataset.features_dataset)
    ebm_ddp = torch.nn.parallel.DistributedDataParallel(ebm, device_ids=[local_rank],
                                                        output_device=local_rank) if config.distributed else ebm
    cd_loss = instantiate(config.ebm.loss)
    cd_loss = cd_loss.to('cuda', non_blocking=True)

    optimizer = instantiate(config.optimizer.opt, ebm.parameters())
    scheduler = instantiate(config.optimizer.sch, optimizer)

    train_loader_ = instantiate(config.dataloader.test, train_dataset, persistent_workers=False)

    # Setup tensorboard
    if is_master:
        writer = SummaryWriter("tensorboard")
    else:
        writer = None

    # From now on there will always be a base distribution, it is just an AbastractScorer for a standard EBM
    # and the fit will only perform the PCA if needed.
    if is_master:
        lib.LOGGER.info("Fitting base distribution")
    ebm.base_dist.fit(model, train_loader_,)

    if is_master:
        lib.LOGGER.info("Starting Training")

    if config.eval_prior:
        lib.LOGGER.info("OOD detection results with prior scorer")
        _ = ebm.base_dist.ood_results(
            model, test_loader, ood_loaders, max_iter=config.eval_max_iter,
            print_res=True, force_tqdm_disable=False, open_ood=config.is_openood
        )

    for epoch in range(config.n_epochs):
        meter = train_cd(ebm_ddp, ebm, model, train_loader, cd_loss, optimizer, scheduler, epoch, config, writer, is_master)

        if config.distributed:
            torch.distributed.barrier()

        if is_master:
            if ((epoch % config.save_freq == 0) and (epoch > 0)) or (epoch + 1 == config.n_epochs):
                lib.save_checkpoint('weights', epoch, ebm, optimizer, scheduler, meter, config)

            if ((epoch % config.eval_freq == 0) and (epoch > 0)) or (epoch + 1 == config.n_epochs):
                lib.LOGGER.info("OOD detection results")
                ebm.eval()
                _ = EBMScorer(ebm).ood_results(model, test_loader, ood_loaders, max_iter=config.eval_max_iter,
                                               print_res=True, force_tqdm_disable=False, open_ood=config.is_openood)

        if config.distributed:
            torch.distributed.barrier()


if __name__ == "__main__":
    main()
