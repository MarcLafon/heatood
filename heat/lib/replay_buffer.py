import torch
import numpy as np

class ReplayBuffer:
    def __init__(
            self,
            buffer_size=10000,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.memory_samples = []
        self.memory_labels = []

    def append(
            self,
            samples: torch.Tensor,
            labels: torch.Tensor,
    ):
        if samples.shape[0] + self.current_size >= self.buffer_size:
            del self.memory_samples[:samples.shape[0]]
            del self.memory_labels[:samples.shape[0]]

        self.memory_samples.extend(list(torch.unbind(samples)))
        if labels is not None:
            self.memory_labels.extend(list(torch.unbind(labels)))
        else:
            self.memory_labels.extend(samples.shape[0]*[None])

    def sample(
            self,
            n: int,
    ):

        indices = np.random.choice(range(len(self.memory_samples)),size=n)

        samples = torch.stack([self.memory_samples[i] for i in indices], dim=0)
        if self.memory_labels[0] is not None:
            labels = torch.stack([self.memory_labels[i] for i in indices], dim=0)
        else:
            labels = None

        return labels, samples

    @property
    def is_full(self):
        return len(self.memory_samples) >= self.buffer_size

    @property
    def current_size(self):
        return len(self.memory_samples)
