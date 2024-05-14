import os
import json
from typing import Optional, Callable
from importlib import resources as impresources
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet, FGVCAircraft


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
