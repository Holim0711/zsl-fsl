import os
import json
import random
from collections import defaultdict
from typing import Optional, Callable
from importlib import resources as impresources
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageNet, FGVCAircraft
from ..utils import get_targets


DATAPATHS = {
    "Caltech101": {
        "images": "caltech101/101_ObjectCategories",
        "splits": "split_zhou_Caltech101.json",
    },
    "DTD": {
        "images": "dtd/dtd/images",
        "splits": "split_zhou_DescribableTextures.json",
    },
    "EuroSAT": {
        "images": "eurosat/2750",
        "splits": "split_zhou_EuroSAT.json",
    },
    "Flowers102": {
        "images": "flowers-102/jpg",
        "splits": "split_zhou_OxfordFlowers.json",
    },
    "Food101": {
        "images": "food-101/images",
        "splits": "split_zhou_Food101.json",
    },
    "OxfordIIITPet": {
        "images": "oxford-iiit-pet/images",
        "splits": "split_zhou_OxfordPets.json",
    },
    "StanfordCars": {
        "images": "stanford_cars",
        "splits": "split_zhou_StanfordCars.json",
    },
    "SUN397": {
        "images": "SUN397",
        "splits": "split_zhou_SUN397.json",
    },
    "UCF101": {
        "images": "UCF-101-midframes",
        "splits": "split_zhou_UCF101.json",
    },
}


class CoOpDataset(Dataset):
    def __init__(
        self,
        root: str,
        data: list[tuple],
        transform: Optional[Callable] = None,
    ):
        self.root = root
        self.data = [(x[0], int(x[1])) for x in data]
        self.transform = transform
        assert all(os.path.isfile(os.path.join(root, x[0])) for x in data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = Image.open(os.path.join(self.root, x))
        if self.transform:
            x = self.transform(x)
        return x, y


def read_coop_splits(filenmae: str):
    from . import splits
    resource = impresources.files(splits)
    resource = resource / filenmae
    return json.load(resource.open('rt'))


def get_datasets(
    name: str,
    train_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
):
    root = os.path.join(os.environ['TORCHVISION_DATASETS'], name)

    if name == 'ImageNet':
        return {
            'train': ImageNet(root, 'train', transform=train_transform),
            'val': ImageNet(root, 'val', transform=test_transform),
            'test': ImageNet(root, 'val', transform=test_transform),
        }
    elif name == 'FGVCAircraft':
        return {
            'train': FGVCAircraft(root, 'train', transform=train_transform),
            'val': FGVCAircraft(root, 'val', transform=test_transform),
            'test': FGVCAircraft(root, 'test', transform=test_transform),
        }
    else:
        path = DATAPATHS[name]
        root = os.path.join(root, path['images'])
        splits = read_coop_splits(path['splits'])
        return {
            'train': CoOpDataset(root, splits['train'], train_transform),
            'val': CoOpDataset(root, splits['val'], test_transform),
            'test': CoOpDataset(root, splits['test'], test_transform),
        }


def sample_indices(dataset: Dataset, shot: int, rand: random.Random):
    indices = defaultdict(list)
    for i, y in enumerate(get_targets(dataset)):
        indices[y].append(i)
    return [i for v in indices.values() for i in rand.sample(v, shot)]


def sample_fewshot_indices(train: Dataset, val: Dataset, shot: int, seed: int):
    rand = random.Random(seed)
    train_indices = sample_indices(train, shot, rand)
    val_indices = sample_indices(val, min(shot, 4), rand)
    return {'train': train_indices, 'val': val_indices}


def get_fewshot_datasets(
    name: str,
    train_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
    shot: int = 1,
    seed: int = 1,
):
    datasets = get_datasets(name, train_transform, test_transform)
    indices = sample_fewshot_indices(datasets['train'], datasets['val'], shot, seed)
    return {
        'train': Subset(datasets['train'], indices['train']),
        'val': Subset(datasets['val'], indices['val']),
        'test': datasets['test'],
    }
