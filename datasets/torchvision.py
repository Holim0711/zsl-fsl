import os
from typing import Tuple
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


def build_datasets(
    name: str,
    root: str,
    train_transform=None,
    test_transform=None,
) -> dict:
    cls = {
        "ImageNet": ImageNet,
        "Caltech101": Caltech101,
        "DTD": DTD,
        "EuroSAT": EuroSAT,
        "FGVCAircraft": FGVCAircraft,
        "Flowers102": Flowers102,
        "Food101": Food101,
        "OxfordPets": OxfordIIITPet,
        "StanfordCars": StanfordCars,
        "SUN397": SUN397,
        "UCF101": ImageUCF101,
    }[name]

    if name == 'ImageNet':
        root = os.path.join(root, 'imagenet')
        return {
            'train': cls(root, split='train', transform=train_transform),
            'val': cls(root, split='val', transform=test_transform),
            'test': cls(root, split='val', transform=test_transform),
        }
    if name == 'Caltech101':
        return {
            'test': cls(root, transform=test_transform),
        }
    if name == 'DTD':
        return {
            'train': cls(root, split='train', transform=train_transform),
            'val': cls(root, split='val', transform=test_transform),
            'test': cls(root, split='test', transform=test_transform),
        }
    if name == 'EuroSAT':
        return {
            'test': cls(root, transform=test_transform),
        }
    if name == 'FGVCAircraft':
        return {
            'train': cls(root, split='train', transform=train_transform),
            'val': cls(root, split='val', transform=test_transform),
            'test': cls(root, split='test', transform=test_transform),
        }
    if name == 'Flowers102':
        return {
            'train': cls(root, split='train', transform=train_transform),
            'val': cls(root, split='val', transform=test_transform),
            'test': cls(root, split='test', transform=test_transform),
        }
    if name == 'Food101':
        return {
            'train': cls(root, split='train', transform=train_transform),
            'test': cls(root, split='test', transform=test_transform),
        }
    if name == 'OxfordPets':
        return {
            'train': cls(root, split='trainval', transform=train_transform),
            'test': cls(root, split='test', transform=test_transform)
        }
    if name == 'StanfordCars':
        return {
            'train': cls(root, split='train', transform=train_transform),
            'test': cls(root, split='test', transform=test_transform),
        }
    if name == 'SUN397':
        root = os.path.join(root, 'SUN397')
        return {
            'test': cls(root, transform=test_transform),
        }
    if name == 'UCF101':
        root = os.path.join(root, 'ucf-101')
        return {
            'train': cls(root, train=True, transform=train_transform),
            'test': cls(root, train=False, transform=test_transform),
        }
    raise NotImplementedError(name)
