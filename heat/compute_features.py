from typing import Tuple, List, Mapping, Any
import os
import argparse
import builtins

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from heat.ebm import create_backbone

NoneType = type(None)


class ImageFolder(torchvision.datasets.ImageFolder):

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = super().__getitem__(idx)
        return img, label, idx


@torch.no_grad()
def compute_features(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    path: str,
    amp: bool = False,
) -> NoneType:
    model.eval()

    if not os.path.exists(path):
        os.makedirs(path)
    feat_log = np.memmap(f"{path}/feat.mmap", dtype=float, mode='w+', shape=(len(loader.dataset), model.feature_dims[-1]))
    std_log = np.memmap(f"{path}/std.mmap", dtype=float, mode='w+', shape=(len(loader.dataset), model.feature_dims[-1]))
    label_log = np.memmap(f"{path}/label.mmap", dtype=float, mode='w+', shape=(len(loader.dataset),))

    for i, (image, label, index) in enumerate(tqdm(loader)):
        with torch.cuda.amp.autocast(enabled=amp):
            X = model.forward_features(image.to('cuda', non_blocking=True))
            avg_pool = X.mean(dim=(2, 3))
            std_pool = X.std(dim=(2, 3))

        feat_log[index.numpy()] = avg_pool.float().cpu().numpy()
        std_log[index.numpy()] = std_pool.float().cpu().numpy()
        label_log[index.numpy()] = label.cpu().numpy()


def main(args: argparse.Namespace) -> NoneType:

    model = create_backbone(args.model_id)
    model.eval()
    model.requires_grad_(False)
    model.to('cuda', non_blocking=True)

    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    dts = ImageFolder(args.data_dir, transform=transform)
    loader = torch.utils.data.DataLoader(
        dts,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=10,
        sampler=None,
    )

    path = os.path.join(args.out_dir, args.model_id.split(".")[0])
    _ = compute_features(
        model,
        loader,
        path,
        args.amp,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help='Data directory')
    parser.add_argument("--out_dir", type=str, required=True, help='Output directory')
    parser.add_argument("--model_id", type=str, required=True, help='Model id for create_backbone')
    parser.add_argument("--batch_size", type=int, default=512, help='Batch size (per worker)')
    parser.add_argument("--amp", default=False, action='store_true', help='Use AMP')
    args = parser.parse_args()

    main(args)
