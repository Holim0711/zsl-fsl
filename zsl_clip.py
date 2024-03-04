import os
import sys
import torch
import torch.nn.functional as F
import clip
from tqdm import tqdm
from utils import prompt_ensembler
from datasets import build_coop_datasets
import yaml


@torch.no_grad()
def run_test(loader, model, zeroshot_weights):
    n_right, n_total = 0, 0
    for x, y in tqdm(loader):
        x, y = x.cuda(), y.cuda()
        v = F.normalize(model.encode_image(x))
        z = 100. * v @ zeroshot_weights
        n_right += (y == z.argmax(dim=1)).sum().item()
        n_total += x.size(0)
    return n_right / n_total


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'RN50'
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else 'ImageNet'
    dataset_root = os.environ['TORCHVISION_DATASETS']

    # load model & dataset
    model, preprocess = clip.load(model_name)
    model.name = model_name
    datasets = build_coop_datasets(dataset_name, dataset_root, preprocess, preprocess)
    loader = torch.utils.data.DataLoader(datasets['test'], batch_size=64, num_workers=os.cpu_count(), pin_memory=True)

    # load classes & templates
    class_path = os.path.join('data', 'CoOp', dataset_name)
    classes = yaml.safe_load(open(os.path.join(class_path, 'classes.yaml')))
    templates = yaml.safe_load(open(os.path.join(class_path, 'templates.yaml')))

    # build classifier
    zeroshot_weights = prompt_ensembler(model, classes, templates)

    # run test
    acc = run_test(loader, model, zeroshot_weights)
    print(model_name, dataset_name, len(datasets['test']), len(classes), len(templates), acc, sep='\t')
