import os
import sys
import torch
import torch.nn.functional as F
import clip
from tqdm import tqdm
from utils import encode_prompts, MeanEnsembler
from zsfs.datasets import get_datasets
from zsfs.prompts import get_prompts
import yaml
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


@torch.no_grad()
def run_test(loader, model, zeroshot_classifier):
    n_right, n_total = 0, 0
    for x, y in tqdm(loader):
        x, y = x.cuda(), y.cuda()
        v = F.normalize(model.encode_image(x))
        z = 100. * zeroshot_classifier(v)
        n_right += (y == z.argmax(dim=1)).sum().item()
        n_total += x.size(0)
    return n_right / n_total


@torch.no_grad()
def draw_confusion_matrix(loader, model, zeroshot_weights):
    Y, Ŷ = [], []
    for x, y in tqdm(loader):
        v = F.normalize(model.encode_image(x.cuda()))
        z = 100. * v @ zeroshot_weights
        ŷ = z.argmax(dim=1).cpu()
        Y.extend(y.tolist())
        Ŷ.extend(ŷ.tolist())
    return confusion_matrix(Y, Ŷ)


@torch.no_grad()
def draw_winner_matrix(loader, model, zeroshot_weights):
    n_classes = zeroshot_weights.shape[-1]
    M = torch.zeros((n_classes, n_classes), dtype=int)
    for x, y in tqdm(loader):
        v = F.normalize(model.encode_image(x.cuda()))
        z = 100. * v @ zeroshot_weights
        win = (z >= z[torch.arange(x.size(0)), y][:, None]).cpu()
        for y_i, win_i in zip(y, win):
            M[y_i] += win_i.cpu()
    return M.numpy()


def save_confusion_matrix(cm, filename, figsize=(20, 16), classes=None):
    if not classes:
        classes = [f'class {i}' for i in range(len(cm))]
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xticks(rotation=90)  # x축 라벨 회전
    plt.yticks(rotation=0)
    plt.ylabel('True Class')
    plt.xlabel('Pred Class')
    plt.title('Confusion Matrix')
    plt.savefig(filename)


if __name__ == "__main__":
    method_name = sys.argv[1] if len(sys.argv) > 1 else 'CLIP'
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'RN50'
    dataset_module = sys.argv[3] if len(sys.argv) > 3 else 'torchvision'
    dataset_name = sys.argv[4] if len(sys.argv) > 4 else 'ImageNet'

    # load model & dataset
    model, preprocess = clip.load(model_name)
    datasets = get_datasets(dataset_module, dataset_name, preprocess, preprocess)
    loader = torch.utils.data.DataLoader(datasets['test'], batch_size=64, num_workers=os.cpu_count(), pin_memory=True)

    # load classes & templates
    prompts = get_prompts(method_name, dataset_name)

    # special treatment for Caltech101
    # - torchvision: 101 classes ('Face', 'Face_easy', 'leopard', ...)
    # - coop: 100 classes ('Face', 'leopard', ...)
    if dataset_name == 'Caltech101':
        if dataset_module.lower() == 'torchvision' and len(prompts) == 100:
            prompts.insert(0, prompts[0])   # copy 'Face' to 'Face_easy'
        elif dataset_module.lower() == 'coop' and len(prompts) == 101:
            prompts.pop(1)                  # remove 'Face_easy'

    # build classifier
    zeroshot_classifier = MeanEnsembler(encode_prompts(model, prompts))

    # run test
    acc = run_test(loader, model, zeroshot_classifier)
    print(method_name, model_name, dataset_name, acc, sep='\t', flush=True)

    # cm = draw_confusion_matrix(loader, model, zeroshot_weights)
    # cm = draw_winner_matrix(loader, model, zeroshot_weights)
    # save_confusion_matrix(cm, 'qqq.winner.png', classes=classes)
    
    # print('', *classes, sep='\t')
    # for c, row in zip(classes, cm):
    #     print(c, *row, sep='\t')
