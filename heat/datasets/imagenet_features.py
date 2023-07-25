from typing import Optional, Tuple
import os

import numpy as np
import torch

import heat.lib as lib

NoneType = type(None)

_LEN = 1281167


class ImagenetFeatures(torch.utils.data.Dataset):

    def __init__(
        self,
        root: str,
        dim: int,
        correct_root: bool = False,
        transform: Optional[torch.nn.Module] = None,
    ) -> NoneType:
        super().__init__()
        if correct_root:
            root = root.split('.')[0]
        if transform is not None:
            print("Transform is ignored for ImagenetFeatures dataset")

        self.root = lib.expand_path(root)
        self.dim = dim
        self.features = np.memmap(os.path.join(self.root, "feat.mmap"), dtype=np.float64, mode='r', shape=(_LEN, dim))
        self.std = np.memmap(os.path.join(self.root, "std.mmap"), dtype=np.float64, mode='r', shape=(_LEN, dim))
        self.targets = np.memmap(os.path.join(self.root, "label.mmap"), dtype=np.float64, mode='r', shape=(_LEN,))
        self.tokens = None
        if os.path.exists(os.path.join(self.root, "token.mmap")):
            self.tokens = np.memmap(os.path.join(self.root, "token.mmap"), dtype=np.float64, mode='r', shape=(_LEN, dim))

    def __len__(self,) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        feat = torch.tensor(self.features[idx].astype(np.float32))
        std = torch.tensor(self.std[idx].astype(np.float32))
        lb = torch.tensor(self.targets[idx].astype(np.float32)).long()
        out = {'avg': feat, 'std': std}
        if self.tokens is not None:
            out['token'] = torch.tensor(self.tokens[idx].astype(np.float32))
        return out, lb
