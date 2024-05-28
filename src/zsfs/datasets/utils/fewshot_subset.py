import numpy
import torch
from torch.utils.data import Dataset, Subset
from random import Random
from typing import Optional
from collections.abc import Iterable


def get_targets(dataset: Dataset):
    # get targets from dataset
    targets = dataset.__dict__.get('targets')
    if not targets:
        targets = dataset.__dict__.get('labels')
    if not targets:
        targets = dataset.__dict__.get('_labels')
    if not targets:
        data = dataset.__dict__.get('data')
        if isinstance(data, Iterable):
            if all(isinstance(x, Iterable) and len(x) == 2 for x in data):
                targets = [y for _, y in data]

    # convert targets to list[int]
    if isinstance(targets, torch.Tensor):
        targets = dataset.targets.tolist()
    elif isinstance(targets, numpy.ndarray):
        targets = dataset.targets.tolist()

    # return if targets is list[int]
    if isinstance(targets, Iterable):
        if all(isinstance(x, int) for x in targets):
            return targets

    # else, default method
    return [y for _, y in dataset]



class FewShotSubset(Subset):
    def __init__(
        self,
        dataset: Dataset,
        k_shots: int,
        random_seed: Optional[int] = None,
        oversampling: bool = False,
    ):
        self.k_shots = k_shots
        self.random = Random(random_seed)
        self.oversampling = oversampling
        targets = get_targets(dataset)
        index_lists = [[] for _ in range(max(targets) + 1)]
        [index_lists[y].append(i) for i, y in enumerate(targets)]
        index_lists = [self.sample_k_shots(x) for x in index_lists]
        super().__init__(dataset, sum(index_lists, []))
        self.targets = [targets[i] for i in self.indices]

    def sample_k_shots(self, x: list[int]):
        if len(x) < self.k_shots and self.oversampling:
            m = self.k_shots // len(x)
            r = self.k_shots % len(x)
            return x * m + self.random.sample(x, r)
        return self.random.sample(x, self.k_shots)
