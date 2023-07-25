from typing import Iterable, Tuple, Optional
import random

import torch
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit

NoneType = type(None)


class TargetSubset(Dataset):

    def __init__(self, dts: Dataset, targets: Iterable) -> NoneType:
        super().__init__()
        self.dts = dts
        targets = set(targets)
        dts_targets = dts.targets
        self.keep_indices = []
        for idx, lb in enumerate(dts_targets):
            if lb in targets:
                self.keep_indices.append(idx)

        self.map_targets = {tgt: new_tgt for new_tgt, tgt in enumerate(targets)}
        self.targets = [self.map_targets[tgt] for tgt in [dts.targets[x] for x in self.keep_indices]]

    def __len__(self,) -> int:
        return len(self.keep_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        img, lb = self.dts[self.keep_indices[idx]]
        if isinstance(lb, torch.Tensor):
            lb = lb.item()
        lb = torch.tensor(self.map_targets[lb])
        return img, lb


def create_label_subset_dataset(dts: Dataset, number_labels: int, random_state: int = 0) -> Dataset:
    assert isinstance(number_labels, int)
    labels = dts.targets
    unique_labels = set(labels)

    random.seed(random_state)
    label_subset = set(random.sample(unique_labels, number_labels))

    return TargetSubset(dts, label_subset)


def create_proportion_subset_dataset(dts: Dataset, proportion: float, random_state: int = 0) -> Dataset:
    assert isinstance(proportion, float)
    labels = dts.targets

    splits = StratifiedShuffleSplit(n_splits=1, train_size=proportion, random_state=random_state)
    indices, _ = next(splits.split(labels, labels))
    _ = indices.sort()

    return Subset(dts, indices)


def create_subset_dataset(dts: Dataset, proportion: Optional[float] = None, number_labels: Optional[int] = None, random_state: int = 0) -> Dataset:
    if number_labels is not None:
        dts = create_label_subset_dataset(dts, number_labels=number_labels, random_state=random_state)

    if proportion is not None:
        dts = create_proportion_subset_dataset(dts, proportion=proportion, random_state=random_state)

    return dts
