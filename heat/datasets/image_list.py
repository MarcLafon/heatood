from typing import Optional, Tuple
import os

import torch
from PIL import Image

import heat.lib as lib

NoneType = type(None)


class ImageList(torch.utils.data.Dataset):

    def __init__(self, root: str, image_list: str, transform: Optional[torch.nn.Module] = None) -> NoneType:
        super().__init__()

        self.root = lib.expand_path(root)
        self.transform = transform

        with open(image_list) as f:
            lines = f.read().splitlines()
        paths, target = [x.split(' ')[0] for x in lines], [int(x.split(' ')[1]) for x in lines]

        self.paths = [os.path.join(self.root, pth) for pth in paths]
        self.targets = target

    def __len__(self,) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:

        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        target = torch.tensor(self.targets[idx])

        if self.transform is not None:
            img = self.transform(img)

        return img, target
