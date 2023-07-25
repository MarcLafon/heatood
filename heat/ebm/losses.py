from typing import Tuple, Mapping

import torch
from torch import nn
import torch.nn.functional as F


NoneType = type(None)


class ContrastiveDivergenceLoss(nn.Module):
    def __init__(
            self,
            l2_coef: float = 1e-1,
            eps_data: float = 1e-3,
            verbose: bool = True,
    ) -> NoneType:
        super().__init__()
        self.l2_coef = l2_coef
        self.eps_data = eps_data
        self.verbose = verbose

    def forward(
            self,
            ebm: nn.Module,
            real_z: torch.Tensor,
            target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:

        sample_size = real_z.shape[0]

        _ebm = ebm.module if isinstance(ebm, nn.parallel.DistributedDataParallel) else ebm

        init_labels, init_samples = _ebm.proposal_samples(sample_size, target=target, real_samples=real_z)
        gen_z = _ebm.negative_samples(init_samples, init_labels)

        real_energy, real_energy_nn, real_energy_g = ebm(real_z + torch.randn_like(real_z) * self.eps_data, nn_only=not self.verbose)
        gen_energy, gen_energy_nn, gen_energy_g = ebm(gen_z + torch.randn_like(gen_z) * self.eps_data, nn_only=not self.verbose)

        cdiv_loss = real_energy_nn.mean() - gen_energy_nn.mean()
        l2_reg = real_energy_nn.pow(2).mean() + gen_energy_nn.pow(2).mean()
        loss = cdiv_loss + self.l2_coef * l2_reg

        logs = {
            'energy IN': real_energy.mean().detach(),
            'engergy Gen': gen_energy.mean().detach(),
            'energy_nn IN': real_energy_nn.mean().detach(),
            'energy_nn Gen': gen_energy_nn.mean().detach(),
            'energy_prior IN': real_energy_g.mean().detach(),
            'energy_prior Gen': gen_energy_g.mean().detach(),
            "loss": loss.detach(),
            "cd_loss": cdiv_loss.detach(),
            "l2_loss": (self.l2_coef * l2_reg).detach(),
        }

        return loss, logs, gen_z


class NoiseContrastiveEstimation(nn.Module):
    def __init__(
            self,
            l2_coef: float = 1e-1,
            eps_data: float = 1e-5,
    ) -> NoneType:
        super().__init__()
        self.l2_coef = l2_coef
        self.eps_data = eps_data

    def forward(
            self,
            ebm: nn.Module,
            real_z: torch.Tensor,
            target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Mapping[str, torch.Tensor]]:

        sample_size = real_z.shape[0]

        _ebm = ebm.module if isinstance(ebm, nn.parallel.DistributedDataParallel) else ebm

        gen_z = _ebm.sgld_samples(sample_size, steps=0)

        real_energy, real_energy_nn, real_energy_g = ebm(real_z + torch.randn_like(real_z) * self.eps_data)
        gen_energy, gen_energy_nn, gen_energy_g = ebm(gen_z + torch.randn_like(gen_z) * self.eps_data)

        l2_reg = real_energy_nn.pow(2).mean() + gen_energy_nn.pow(2).mean()

        nce_loss = -(torch.log(F.sigmoid(-real_energy_nn)).mean() + torch.log(F.sigmoid(gen_energy_nn)).mean())
        loss = nce_loss + self.l2_coef * l2_reg

        logs = {
            'energy IN': real_energy.mean().detach(),
            'engergy Gen': gen_energy.mean().detach(),
            'energy_nn IN': real_energy_nn.mean().detach(),
            'energy_nn Gen': gen_energy_nn.mean().detach(),
            'energy_g IN': real_energy_g.mean().detach(),
            'energy_g Gen': gen_energy_g.mean().detach(),
            "loss": loss.detach(),
            "nce_loss": nce_loss.detach(),
        }

        return loss, logs
