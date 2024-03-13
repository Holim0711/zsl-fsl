from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageNet, CIFAR10, CIFAR100, MNIST, SVHN
from random import Random
from typing import Optional
from .from_coop import SimpleDataset


class FewShotSubset(Subset):
    def __init__(
        self,
        dataset: Dataset,
        k_shots: int,
        random_seed: Optional[int] = None,
    ):
        if isinstance(dataset, (ImageNet, CIFAR10, CIFAR100)):
            targets = dataset.targets
        elif isinstance(dataset, (MNIST, SVHN)):
            targets = dataset.targets.tolist()
        elif isinstance(dataset, SimpleDataset):
            targets = [y for _, y in dataset.data]
        else:
            targets = [y for _, y in dataset]
        index_lists = [[] for _ in range(max(targets) + 1)]
        [index_lists[y].append(i) for i, y in enumerate(targets)]
        rand = Random(random_seed)
        index_lists = [rand.sample(x, k_shots) for x in index_lists]
        super().__init__(dataset, sum(index_lists, []))
