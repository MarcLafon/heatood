from typing import Optional, Tuple, OrderedDict
from functools import partial

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

import heat.lib as lib
from heat.layers import spectral_norm_fc
from heat.scorers import AbastractOODScorer

NoneType = type(None)


def _identity(x: nn.Module) -> nn.Module:
    return x


class HybridEnergyModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        n_hidden_layers: int = 2,
        temperature: float = 2e-2,
        temperature_prior: float = 1e3,
        proposal_type: str = "random_normal",
        base_dist: Optional[AbastractOODScorer] = None,
        use_base_dist: bool = False,
        sample_from_batch_statistics: bool = False,
        steps: int = 200,
        step_size_start: float = 1e-0,
        step_size_end: float = 1e-2,
        eps_start: float = 1e-1,
        eps_end: float = 1e-3,
        sgld_relu: bool = True,
        use_sgld: bool = True,
        use_svgd: bool = False,
        use_pcd: bool = False,
        buffer_size: int = 40000,
        restart_prob: float = 0.1,
        lr: float = 1e-5,
        num_classes: int = 10,
        use_spectral_norm: bool = False,
        train_max_iter: Optional[int] = np.inf,
        reduce_width: bool = False,
    ) -> NoneType:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.temperature_prior = temperature_prior
        self.proposal_type = proposal_type
        self.base_dist = AbastractOODScorer() if base_dist is None else base_dist
        self.use_base_dist = use_base_dist
        self.sample_from_batch_statistics = sample_from_batch_statistics
        self.steps = steps
        self.step_size_start = step_size_start
        self.step_size_end = step_size_end
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.sgld_relu = sgld_relu
        self.use_sgld = use_sgld
        self.use_svgd = use_svgd
        self.use_pcd = use_pcd
        self.buffer_size = buffer_size
        self.restart_prob = restart_prob
        self.lr = lr # remove
        self.num_classes = num_classes # remove
        self.use_spectral_norm = use_spectral_norm
        self.train_max_iter = train_max_iter
        self.reduce_width = reduce_width

        if self.use_svgd:
            self.K = lib.RBF()

        self.buffer = lib.ReplayBuffer(buffer_size=self.buffer_size)

        if self.use_spectral_norm:
            snorm = partial(spectral_norm_fc, coeff=10)
        else:
            snorm = _identity

        input_dim = self.base_dist.pca_n_principal_components + self.base_dist.pca_n_last_components if self.base_dist.use_pca else input_dim

        if not self.reduce_width:
            self.mlp = nn.Sequential(
                snorm(nn.Linear(input_dim, hidden_dim)),
                nn.LeakyReLU(0.2),
                *([snorm(nn.Linear(hidden_dim, hidden_dim)), nn.LeakyReLU(0.2)] * n_hidden_layers),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.mlp = nn.Sequential(
                snorm(nn.Linear(input_dim, hidden_dim)),
                nn.LeakyReLU(0.2),
                *([nn.Sequential(
                    snorm(nn.Linear(hidden_dim // (2 ** i), hidden_dim // (2 ** i))),
                    nn.LeakyReLU(0.2),
                    snorm(nn.Linear(hidden_dim // (2 ** i), hidden_dim // (2 ** i))),
                    nn.LeakyReLU(0.2),
                    snorm(nn.Linear(hidden_dim // (2 ** i), hidden_dim // (2 ** (i + 1)))),
                    nn.LeakyReLU(0.2)
                ) for i in range(n_hidden_layers)]),
                nn.Linear(hidden_dim // (2 ** n_hidden_layers), 1),
            )

        print("EBM model has:", lib.num_parameters(self.mlp), "parameters")

    def energy_nn(self, z: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        return self.mlp(z).view(-1) / self.temperature

    def energy_prior(self, z: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        # prior_scores = - self.base_dist.log_prob(z, labels=labels)
        prior_scores = self.base_dist.energy(z, labels=labels)
        return prior_scores / self.temperature_prior

    def forward(self, z: Tensor, labels: Optional[Tensor] = None, nn_only: bool = False) -> Tuple[Tensor]:
        energy_nn = self.energy_nn(z)  # .view(-1) placed inside energy_nn !

        energy_g = torch.zeros_like(energy_nn)
        if self.use_base_dist and (not nn_only):
            energy_g = self.energy_prior(z, labels=labels)

        assert energy_nn.shape == energy_g.shape

        energy = energy_nn + energy_g

        assert energy.ndim == 1 or (energy.ndim == 2 and energy.shape[1] == 1)

        return energy, energy_nn, energy_g

    def energy(self, z: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        return self(z, labels=labels)[0]

    def log_prob(self, z: Tensor, labels: Optional[Tensor] = None) -> Tensor:
        return - self(z, labels=labels)[0]

    @staticmethod
    def polynomial(t: int, T: int, init_val: float, end_val: float, power: float = 2.) -> float:
        return (init_val - end_val) * ((1 - t / T) ** power) + end_val

    def proposal_samples(self, sample_size: float, target: Optional[Tensor] = None, real_samples: Optional[Tensor] = None) -> Tuple[Tensor]:
        init_labels = None
        target = target if self.sample_from_batch_statistics else None
        dim = self.base_dist.pca_n_principal_components + self.base_dist.pca_n_last_components if self.base_dist.use_pca else self.input_dim

        buffer_labels = None
        buffer_samples = None
        # drawing samples from buffer
        if self.use_pcd and self.buffer.current_size >= self.buffer_size // 2:
            buffer_labels, buffer_samples = self.buffer.sample(int((1 - self.restart_prob) * sample_size))
            sample_size = int(self.restart_prob * sample_size)
            buffer_samples = lib.to_device(buffer_samples)

        # drawing fresh samples
        if self.proposal_type == "random_normal":
            init_samples = torch.randn((sample_size, dim))
        elif self.proposal_type == "random_uniform":
            init_samples = 2 * torch.rand((sample_size, dim)) - 4
        elif self.proposal_type == "base_dist":
            assert self.base_dist.has_sample, f"base dist {self.base_dist.name} has not a sample method."
            init_labels, init_samples = self.base_dist.sample((sample_size,), target)
        elif self.proposal_type == "base_dist_temp":
            assert self.base_dist.has_sample, f"base dist {self.base_dist.name} has not a sample method."
            temp_scale = 100 * torch.rand((1,)).item()
            init_labels, init_samples = self.base_dist.sample((sample_size,), target, temp_scale)
        elif self.proposal_type == "data":
            init_labels = target
            init_samples = real_samples
        else:
            raise NotImplementedError

        # push to device
        init_samples = lib.to_device(init_samples)

        # concatenating fresh and buffer samples
        init_samples = torch.cat([init_samples, buffer_samples]) if buffer_samples is not None else init_samples
        init_labels = torch.cat([init_labels, buffer_labels]) if buffer_labels is not None else init_labels

        return init_labels, init_samples

    def negative_samples(self, init_samples: Optional[Tensor] = None, init_labels: Optional[Tensor] = None, steps: Optional[int] = None) -> Tensor:
        if self.use_sgld:
            gen_z = self.sgld_samples(init_samples=init_samples, init_labels=init_labels, steps=steps)
        elif self.use_svgd:
            gen_z = self.svgd_samples(init_samples=init_samples, init_labels=init_labels, steps=steps)
        else:
            raise NotImplementedError

        return gen_z

    def load_state_dict(
        self,
        state_dict: OrderedDict[str, Tensor],
        model: nn.Module,
        train_loader: DataLoader,
    ) -> NoneType:
        super().load_state_dict(state_dict, strict=False)
        # super().load_state_dict(state_dict)
        self.base_dist.fit(model, train_loader)

    def sgld_samples(self, init_samples: Tensor, init_labels: Optional[Tensor] = None, steps: Optional[int] = None) -> Tensor:
        steps = self.steps if steps is None else steps
        is_training = self.training
        self.eval()
        lib.requires_grad(self, False)

        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        if not self.base_dist.use_pca:
            if self.sgld_relu:
                init_samples = F.relu(init_samples)

            if self.base_dist.normalize:
                init_samples = F.normalize(init_samples, dim=1)

        z = init_samples.clone()
        z = lib.to_device(z)

        noise = torch.randn(z.shape, device=z.device)
        random_temperature = lib.to_device(torch.FloatTensor(z.shape[0]).uniform_(1, 1))
        random_noise = lib.to_device(torch.FloatTensor(z.shape[0]).uniform_(1, 1))

        for t in range(steps):
            z = z.detach().requires_grad_(True)
            noise.normal_(0, self.polynomial(t, steps, self.eps_start, self.eps_end))

            energy = self.energy(z, labels=init_labels)
            grad_Ez = torch.autograd.grad(energy.sum(), [z])[0]

            lr = self.polynomial(t, steps, self.step_size_start, self.step_size_end) * random_temperature

            z.data -= torch.diag(lr).matmul(grad_Ez) + torch.diag(random_noise).matmul(noise)

            if not self.base_dist.use_pca:
                # Projection
                if self.sgld_relu:
                    z = F.relu(z)

                if self.base_dist.normalize:
                    z = F.normalize(z, dim=1)

        lib.requires_grad(self, True)
        self.train(is_training)
        torch.set_grad_enabled(had_gradients_enabled)

        # appending generated samples to buffer if needed
        if self.use_pcd:
            self.buffer.append(z.detach(), init_labels)

        return torch.cat([z.detach(), init_samples.detach()])

    def svgd_samples(self, init_samples: Tensor, init_labels: Optional[Tensor] = None, steps: Optional[int] = None) -> Tensor:
        steps = self.steps if steps is None else steps
        is_training = self.training
        self.eval()
        lib.requires_grad(self, False)

        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        if not self.base_dist.use_pca:
            if self.sgld_relu:
                init_samples = F.relu(init_samples)

            if self.base_dist.normalize:
                init_samples = F.normalize(init_samples, dim=1)

        z = init_samples.clone()
        z = lib.to_device(z)

        noise = torch.randn(z.shape, device=z.device)
        random_temperature = lib.to_device(torch.FloatTensor(z.shape[0]).uniform_(1, 1))

        for t in range(steps):
            z = z.detach().requires_grad_(True)
            noise.normal_(0, self.polynomial(t, steps, self.eps_start, self.eps_end))

            log_prob = self.log_prob(z, labels=init_labels)
            grad_log_prob_z = torch.autograd.grad(log_prob.sum(), [z])[0]
            K_zz = self.K(z, z.detach())
            grad_K = torch.autograd.grad(K_zz.sum(), z)[0]

            grad = (K_zz.detach().matmul(grad_log_prob_z) + grad_K) / z.size(0)

            lr = self.polynomial(t, steps, self.step_size_start, self.step_size_end) * random_temperature

            z.data += torch.diag(lr).matmul(grad) + noise

            if not self.base_dist.use_pca:
                # Projection
                if self.sgld_relu:
                    z = F.relu(z)

                if self.base_dist.normalize:
                    z = F.normalize(z, dim=1)

        lib.requires_grad(self, True)
        self.train(is_training)
        torch.set_grad_enabled(had_gradients_enabled)

        if self.use_pcd:
            self.buffer.append(z.detach(), init_labels)

        return z.detach()
