import os
import sys
import torch
import torch.nn.functional as F
import clip
from tqdm import tqdm
from utils import encode_prompts, MeanEnsembler
from zsfs.datasets import get_datasets
from zsfs.prompts import get_classes, get_prompts
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


@torch.no_grad()
def run_test(loader, model, zeroshot_classifier):
    n_right, n_total = 0, 0
    y_list, ŷ_list = [], []
    for x, y in tqdm(loader):
        v = F.normalize(model.encode_image(x.cuda()))
        z = 100. * zeroshot_classifier(v)
        ŷ = z.argmax(dim=1).cpu()
        n_right += (y == ŷ).sum().item()
        n_total += x.size(0)
        y_list.extend(y.tolist())
        ŷ_list.extend(ŷ.tolist())
    return {
        'acc': n_right / n_total,
        'cm': confusion_matrix(y_list, ŷ_list),
    }


def save_cm_csv(cm, filename, classes=None):
    if not classes:
        classes = [f'class {i}' for i in range(len(cm))]
    with open(filename, 'w') as file:
        print(',' + ','.join(classes), file=file)
        for i, row in enumerate(cm):
            print(classes[i] + ',' + ','.join(map(str, row)), file=file)


def save_cm_png(cm, filename, figsize=(20, 16), classes=None):
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
    dataset_name = sys.argv[3] if len(sys.argv) > 3 else 'ImageNet'

    # load model & dataset
    model, preprocess = clip.load(model_name)
    datasets = get_datasets(dataset_name, test_transform=preprocess)
    loader = torch.utils.data.DataLoader(datasets['test'], batch_size=64, num_workers=os.cpu_count(), pin_memory=True)

    # load classes & templates
    classes = get_classes(method_name, dataset_name)
    prompts = get_prompts(method_name, dataset_name)

    # build classifier
    zeroshot_classifier = MeanEnsembler(encode_prompts(model, prompts))

    # run test
    results = run_test(loader, model, zeroshot_classifier)

    print(method_name, model_name, dataset_name, results['acc'], sep='\t')
    filename = f'results/zsl/{method_name}.{model_name}.{dataset_name}'
    save_cm_csv(results['cm'], filename + '.csv', classes=classes)
    save_cm_png(results['cm'], filename + '.png', classes=classes)
