import os
import sys
import torch
from tqdm import tqdm
from utils import zeroshot_classifier, load_clip
from datasets import build_coop_datasets
import yaml


def count_correct(output, targets, topk=1):
    return (output.topk(topk, -1)[1].t() == targets).sum().item()


@torch.no_grad()
def test_inference(loader, model, zeroshot_weights):
    top1, top5, n = 0, 0, 0
    for images, targets in tqdm(loader):
        images = images.cuda()
        targets = targets.cuda()

        # predict
        image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = 100 * image_features @ zeroshot_weights

        # accuracy
        top1 += count_correct(logits, targets, 1)
        top5 += count_correct(logits, targets, 5)
        n += images.size(0)

    return {'top1': top1 / n, 'top5': top5 / n}


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'RN50'
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else 'ImageNet'
    dataset_root = os.environ['TORCHVISION_DATASETS']

    # load model & dataset
    model, preprocess = load_clip(model_name)
    datasets = build_coop_datasets(dataset_name, dataset_root, preprocess, preprocess)
    loader = torch.utils.data.DataLoader(datasets['test'], batch_size=32, num_workers=os.cpu_count())

    # load classes & templates
    class_path = os.path.join('data', 'CoOp', dataset_name)
    classes = yaml.safe_load(open(os.path.join(class_path, 'classes.yaml')))
    templates = yaml.safe_load(open(os.path.join(class_path, 'templates.yaml')))
    print(f"{len(classes)} classes, {len(templates)} templates")

    # build classifier
    zeroshot_weights = zeroshot_classifier(model, classes, templates)
    print(zeroshot_weights.shape)

    # run test
    metrics = test_inference(loader, model, zeroshot_weights)
    print(metrics)
