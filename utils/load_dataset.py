import os
import sys
from torchvision.datasets import (
    ImageNet,
    Flowers102,
    Caltech101,
    OxfordIIITPet,
    EuroSAT,
    Food101,
    UCF101,
    StanfordCars,
    SUN397,
    DTD,
    FGVCAircraft,
)


def load_test_dataset(name, transform=None):
    try:
        root = os.environ['DATA_DIR']
    except KeyError:
        print("export DATA_DIR=", file=sys.stderr)
        raise
    if name == 'imagenet':
        return ImageNet(os.path.join(root, 'imagenet'), split='val', transform=transform)
    if name == 'caltech101':
        return Caltech101(root, transform=transform)
    if name == 'dtd':
        return DTD(root, split='test', transform=transform)
    if name == 'eurosat':
        return EuroSAT(root, transform=transform)
    if name == 'fgvcaircraft':
        return FGVCAircraft(root, split='test', transform=transform)
    if name == 'flowers102':
        return Flowers102(root, split='test', transform=transform)
    if name == 'food101':
        return Food101(root, split='test', transform=transform)
    if name == 'oxfordpets':
        return OxfordIIITPet(root, split='test', transform=transform)
    if name == 'stanfordcars':
        return StanfordCars(root, split='test', transform=transform)
    if name == 'sun397':
        return SUN397(root, transform=transform)
    if name == 'ucf101':
        # Next
        return UCF101(root, )
    raise NotImplementedError(name)
