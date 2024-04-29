import os
import sys
sys.path.append(os.getcwd())
from datasets import build_coop_datasets, FewShotSubset
import matplotlib.pyplot as plt
from math import ceil
import yaml
from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.transforms import InterpolationMode


prep = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
])

dataset_name = sys.argv[1]
dataset_root = os.environ['TORCHVISION_DATASETS']
datasets = build_coop_datasets(dataset_name, dataset_root, prep, prep)
samples = FewShotSubset(datasets['train'], 1)

n_classes = len(samples)
classnames = yaml.safe_load(open(f'data/CoOp/{dataset_name}/classes.yaml'))

assert len(samples) == len(classnames)



def plot_images_with_titles(images, titles):
    n = len(images)
    k = ceil(n ** 0.5)
    fig, axes = plt.subplots(k, k, figsize=(k * 4, k * 4))
    axes = axes.flatten()
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    return fig


fig = plot_images_with_titles([x for x, _ in samples], classnames)
fig.savefig(f'{dataset_name}.png')
