import numpy
import torch
from torch.utils.data import Dataset, Subset
from random import Random
from typing import Optional
from collections.abc import Iterable


def get_targets(dataset: Dataset):
    targets = dataset.__dict__.get('targets')

    if not targets:
        targets = dataset.__dict__.get('labels')

    if isinstance(targets, torch.Tensor):
        targets = dataset.targets.tolist()
    elif isinstance(targets, numpy.ndarray):
        targets = dataset.targets.tolist()

    data = dataset.__dict__.get('data')

    if isinstance(data, Iterable) and all(
        isinstance(x, Iterable) and len(x) == 2 for x in data
    ):
        targets = [y for _, y in data]

    if isinstance(targets, Iterable) and all(
        isinstance(x, int) for x in targets
    ):
        return targets

    return [y for _, y in dataset]



class FewShotSubset(Subset):
    def __init__(
        self,
        dataset: Dataset,
        k_shots: int,
        random_seed: Optional[int] = None,
    ):
        targets = get_targets(dataset)
        index_lists = [[] for _ in range(max(targets) + 1)]
        [index_lists[y].append(i) for i, y in enumerate(targets)]
        rand = Random(random_seed)
        index_lists = [rand.sample(x, k_shots) for x in index_lists]
        super().__init__(dataset, sum(index_lists, []))
