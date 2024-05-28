from .torchvision import get_datasets as get_torchvision_datasets
from .coop import get_datasets as get_coop_datasets
from .coop import get_fewshot_datasets


def get_datasets(name, train_transform=None, test_transform=None):
    return get_coop_datasets(name, train_transform, test_transform)
