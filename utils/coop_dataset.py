import os
import random
from collections import defaultdict
import json
from torch.utils.data import Dataset
from PIL import Image


class DatasetBase:
    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self.train_x = train_x  # labeled training data
        self.train_u = train_u  # unlabeled training data (optional)
        self.val = val          # validation data (optional)
        self.test = test        # test data
        self.num_classes = self.get_num_classes(train_x)
        self.lab2cname, self.classnames = self.get_lab2cname(train_x)

    def generate_fewshot_dataset(
        self, *data_sources, num_shots=-1, repeat=True
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed.
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        # print(f'Creating a {num_shots}-shot dataset')

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []

            for label, items in tracker.items():
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output


class DatasetWrapper(Dataset):
    def __init__(self, root, data, transform=None):
        self.root = root
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        x = Image.open(os.path.join(self.root, x)).convert('RGB')
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
        self.train = DatasetWrapper(self.image_dir, [x[:2] for x in train])
        self.val = DatasetWrapper(self.image_dir, [x[:2] for x in val])
        self.test = DatasetWrapper(self.image_dir, [x[:2] for x in test])

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
