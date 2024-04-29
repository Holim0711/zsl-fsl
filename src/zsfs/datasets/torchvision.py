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


class UCF101MidFrames(UCF101):

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
        self.clip_len = [
            len(x) for x in self.video_clips.metadata['video_pts']]
        self.clip_start_index = [
            sum(self.clip_len[:i]) for i in range(len(self.clip_len))]

    def __len__(self):
        return len(self.clip_start_index)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        clip_start_index = self.clip_start_index[idx]
        clip_mid_index = clip_start_index + self.clip_len[idx] // 2
        image, _, label = super().__getitem__(clip_mid_index)
        return image, label


def build_datasets(
    name: str,
    train_transform=None,
    test_transform=None,
) -> dict:
    # get environment variable
    root = os.environ['TORCHVISION_DATASETS']

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
        "UCF101": UCF101MidFrames,
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
