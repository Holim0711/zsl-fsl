import os
import json
from torch.utils.data import Dataset
from PIL import Image


class SimpleDataset(Dataset):
    def __init__(self, root, data, transform=None):
        self.root = root
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = os.path.join(self.root, x)
        x = Image.open(x).convert('RGB')
        if self.transform:
            x = self.transform(x)
        return x, y


class CoOpDatasets:
    DATA_DIR: str
    IMAGE_DIR: str
    SPLIT_FILE: str

    def __init__(self, root, num_shots=-1):
        self.data_dir = os.path.join(root, self.DATA_DIR)
        self.image_dir = os.path.join(self.data_dir, self.IMAGE_DIR)
        self.split_file = os.path.join(self.data_dir, self.SPLIT_FILE)

        # read split_file [(image_path, class_index, class_name)]
        split = json.load(open(self.split_file))
        train, val, test = split['train'], split['val'], split['test']

        # get classnames
        mapping = {idx: name for _, idx, name in train + val + test}
        self.classes = [mapping[i] for i in range(len(mapping))]

        # build datasets
        self.train = SimpleDataset(self.image_dir, [x[:2] for x in train])
        self.val = SimpleDataset(self.image_dir, [x[:2] for x in val])
        self.test = SimpleDataset(self.image_dir, [x[:2] for x in test])

        if num_shots > 0:
            # TODO: train = self.generate_fewshot_dataset(train, num_shots=num_shots)
            pass


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


def build_dataset(dataset, root_path, shots):
    return {
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
    }[dataset](root_path, shots)
