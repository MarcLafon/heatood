import torch


class TorchLoad(torch.nn.Module):

    def __call__(self, file: str) -> torch.Tensor:
        return torch.load(file).cpu().detach()
