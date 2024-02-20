import os
import sys
from typing import Dict, Tuple
import torch
from torchvision.datasets import (
    ImageNet,
    Caltech101,
    DTD,
    EuroSAT,
    FGVCAircraft,
    Flowers102,
    Food101,
    OxfordIIITPet,
    StanfordCars,
    SUN397,
    UCF101,
)
from torchvision.transforms import Compose, Lambda, ToPILImage
from PIL import Image


class ImageUCF101(UCF101):

    def __init__(self, root: str, transform=None, **kwargs) -> None:
        vid2img = Compose([Lambda(torch.squeeze), ToPILImage('RGB')])
        transform = Compose([vid2img, transform]) if transform else vid2img
        super().__init__(os.path.join(root, 'UCF-101'),
                         os.path.join(root, 'ucfTrainTestlist'),
                         frames_per_clip=1,
                         transform=transform,
                         num_workers=os.cpu_count(),
                         output_format='TCHW',
                         **kwargs)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        image, _, label = super().__getitem__(idx)
        return image, label


def load_test_dataset(name, transform=None):
    try:
        root = os.environ['DATA_DIR']
    except KeyError:
        print("export DATA_DIR=", file=sys.stderr)
        raise
    if name == 'imagenet':
        path = os.path.join(root, 'imagenet')
        return ImageNet(path, split='val', transform=transform)
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
        path = os.path.join(root, 'ucf-101')
        return ImageUCF101(path, train=False, transform=transform)
    raise NotImplementedError(name)
