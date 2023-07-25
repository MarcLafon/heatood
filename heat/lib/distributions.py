import math
import numpy as np

import torch
from torch.distributions.multivariate_normal import _batch_mahalanobis, _batch_mv, _standard_normal

class Categorical(torch.distributions.Categorical):

    def sample(self, sample_shape=torch.Size(), target=None):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)

        if target is not None:
            probs_2d = torch.zeros_like(probs_2d)
            index, tmp_probs = target.unique(return_counts=True)
            tmp_probs = tmp_probs.float() / target.size(0)
            probs_2d.scatter_(1, index.view(1, -1), tmp_probs.view(1, -1))

        samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
        return samples_2d.reshape(self._extended_shape(sample_shape))


class MultivariateNormal(torch.distributions.MultivariateNormal):
    """
    https://pytorch.org/docs/1.11/_modules/torch/distributions/multivariate_normal.html#MultivariateNormal
    """

    def log_prob(self, value, labels=None):
        """
        This is modified to compute the Mahalanobis distance of a sample to its
        reference gaussian
        """
        if labels is None:
            # this is the default behaviour
            return super().log_prob(value)

        diff = value - self.loc[labels].unsqueeze(1)
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, diff)
        half_log_det = self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return -0.5 * (self._event_shape[0] * math.log(2 * math.pi) + M) - half_log_det

    def sample(self, sample_shape, temp_scale=None):
        with torch.no_grad():
            return self.rsample(sample_shape, temp_scale)


    def rsample(self, sample_shape=torch.Size(), temp_scale=None):
        if temp_scale:
            shape = self._extended_shape(sample_shape)
            eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
            # temp = temp_scale * torch.rand((eps.shape[0], 1, 1), device=eps.device)
            temp = torch.Tensor([[[np.random.choice([temp_scale / 100, temp_scale / 10, temp_scale, 10 * temp_scale, 100 * temp_scale])]] for i in range(eps.shape[0])]).to(eps.device)
            return self.loc + _batch_mv(self._unbroadcasted_scale_tril, temp * eps)
        else:
            return super().rsample(sample_shape)


class MixtureSameFamily(torch.distributions.MixtureSameFamily):
    """
    https://pytorch.org/docs/1.11/_modules/torch/distributions/mixture_same_family.html#MixtureSameFamily
    """

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size(), target=None, temp=None):
        """
        This is modified to return the index of the gaussian selected for each sample
        """
        sample_len = len(sample_shape)
        batch_len = len(self.batch_shape)
        gather_dim = sample_len + batch_len
        es = self.event_shape

        # mixture samples [n, B]
        mix_sample = self.mixture_distribution.sample(sample_shape, target)
        mix_shape = mix_sample.shape

        # component samples [n, B, k, E]
        comp_samples = self.component_distribution.sample(sample_shape, temp)

        # Gather along the k dimension
        mix_sample_r = mix_sample.reshape(
            mix_shape + torch.Size([1] * (len(es) + 1)))
        mix_sample_r = mix_sample_r.repeat(
            torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es)

        samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
        return mix_sample, samples.squeeze(gather_dim)

    def log_prob(self, x, labels=None, reduce='logsumexp'):
        """
        This is modified to compute the Mahalanobis distance of a sample to its
        reference gaussian
        """
        if self._validate_args:
            self._validate_sample(x)
        assert reduce in ['logsumexp', 'min'], f"Got reduce={reduce}, expected one of ['logsumexp', 'min']"

        x = self._pad(x)
        log_prob_x = self.component_distribution.log_prob(x, labels=labels)  # [S, B, k]

        if reduce == 'logsumexp':
            if labels is None:
                log_mix_prob = torch.log_softmax(self.mixture_distribution.logits, dim=-1)  # [B, k]
            else:
                log_mix_prob = torch.log_softmax(self.mixture_distribution.logits[labels], dim=-1)  # [B, k]

            return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]

        elif reduce == 'min':
            return torch.min(log_prob_x, dim=-1).values  # [S, B]
