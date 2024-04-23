import os
import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets import ImageNet


class SimpleDataset(Dataset):
    def __init__(self, root: str, data: list, transform=None):
        self.data = [(os.path.join(root, x[0]), int(x[1])) for x in data]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = Image.open(x)
        if self.transform:
            x = self.transform(x)
        return x, y


class CoOpDatasets:
    DATA_DIR: str
    IMAGE_DIR: str
    SPLIT_FILE: str

    @classmethod
    def get_datasets(
        cls,
        root: str,
        train_transform=None,
        test_transform=None,
    ):
        # check data
        data_dir = os.path.join(root, cls.DATA_DIR)
        image_dir = os.path.join(data_dir, cls.IMAGE_DIR)
        split_file = os.path.join(data_dir, cls.SPLIT_FILE)
        assert os.path.isdir(data_dir), data_dir
        assert os.path.isdir(image_dir), image_dir
        assert os.path.isfile(split_file), split_file

        # read split_file [(image_path, class_index, class_name)]
        split = json.load(open(split_file))
        train, val, test = split['train'], split['val'], split['test']

        # build datasets
        return {
            'train': SimpleDataset(image_dir, train, train_transform),
            'val': SimpleDataset(image_dir, val, test_transform),
            'test': SimpleDataset(image_dir, test, test_transform),
        }


class Caltech101(CoOpDatasets):
    DATA_DIR = 'caltech101'
    IMAGE_DIR = '101_ObjectCategories'
    SPLIT_FILE = 'split_zhou_Caltech101.json'


class DTD(CoOpDatasets):
    DATA_DIR = 'dtd'
    IMAGE_DIR = 'dtd/images'
    SPLIT_FILE = 'split_zhou_DescribableTextures.json'


class EuroSAT(CoOpDatasets):
    DATA_DIR = 'eurosat'
    IMAGE_DIR = '2750'
    SPLIT_FILE = 'split_zhou_EuroSAT.json'


class FGVCAircraft(CoOpDatasets):
    DATA_DIR = 'fgvc-aircraft-2013b'
    IMAGE_DIR = 'data/images'
    SPLIT_FILE = 'split_zhou_FGVCAircraft.json'


class Flowers102(CoOpDatasets):
    DATA_DIR = 'flowers-102'
    IMAGE_DIR = 'jpg'
    SPLIT_FILE = 'split_zhou_OxfordFlowers.json'


class Food101(CoOpDatasets):
    DATA_DIR = 'food-101'
    IMAGE_DIR = 'images'
    SPLIT_FILE = 'split_zhou_Food101.json'


class OxfordPets(CoOpDatasets):
    DATA_DIR = 'oxford-iiit-pet'
    IMAGE_DIR = 'images'
    SPLIT_FILE = 'split_zhou_OxfordPets.json'


class StanfordCars(CoOpDatasets):
    DATA_DIR = 'stanford_cars'
    IMAGE_DIR = ''
    SPLIT_FILE = 'split_zhou_StanfordCars.json'


class SUN397(CoOpDatasets):
    DATA_DIR = 'SUN397'
    IMAGE_DIR = 'SUN397'
    SPLIT_FILE = 'split_zhou_SUN397.json'


class UCF101(CoOpDatasets):
    DATA_DIR = 'ucf-101'
    IMAGE_DIR = 'UCF-101-midframes'
    SPLIT_FILE = 'split_zhou_UCF101.json'


def build_datasets(
    name: str,
    root: str,
    train_transform=None,
    test_transform=None,
):
    if name == 'ImageNet':
        root = os.path.join(root, 'imagenet')
        return {
            'train': ImageNet(root, split='train', transform=train_transform),
            'val': ImageNet(root, split='val', transform=test_transform),
            'test': ImageNet(root, split='val', transform=test_transform),
        }
    cls = {
        "Caltech101": Caltech101,
        "DTD": DTD,
        "EuroSAT": EuroSAT,
        "FGVCAircraft": FGVCAircraft,
        "Flowers102": Flowers102,
        "Food101": Food101,
        "OxfordPets": OxfordPets,
        "StanfordCars": StanfordCars,
        "SUN397": SUN397,
        "UCF101": UCF101,
    }[name]
    return cls.get_datasets(root, train_transform, test_transform)
