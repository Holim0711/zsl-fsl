from .torchvision import get_datasets as get_torchvision_datasets
from .coop.coop import get_datasets as get_coop_datasets


def get_datasets(
    module: str,
    name: str,
    train_transform=None,
    test_transform=None,
):
    if module.lower() == 'torchvision':
        return get_torchvision_datasets(name, train_transform, test_transform)
    elif module.lower() == 'coop':
        return get_coop_datasets(name, train_transform, test_transform)
    else:
        raise ValueError(f'Unknown module: {module}')
