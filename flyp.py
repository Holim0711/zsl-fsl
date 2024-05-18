import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import clip
from clip.model import CLIP
from tqdm import tqdm
from utils import encode_prompts, MeanEnsembler, load_clip
from zsfs.datasets import get_datasets
from zsfs.prompts import get_prompts
import torch.nn as nn


@torch.no_grad()
def run_test(loader, model, zeroshot_classifier):
    model.eval()
    n_right, n_total = 0, 0
    for x, y in tqdm(loader):
        x, y = x.cuda(), y.cuda()
        v = F.normalize(model.encode_image(x))
        z = 100. * zeroshot_classifier(v)
        n_right += (y == z.argmax(dim=1)).sum().item()
        n_total += x.size(0)
    return n_right / n_total


def run_train(
    model: CLIP,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    total_epochs = 100
    total_iters = total_epochs = len(train_loader)
    warmup_iters = 500
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=(1 / warmup_iters), total_iters=warmup_iters)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_iters - warmup_iters))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_iters])

    print('start training')

    for epoch in range(total_epochs):
        model.train()
        for x, y in tqdm(train_loader):
            x, y = x.cuda(), y.cuda()
            logits_per_image, logits_per_text = model(x, y)
            labels = torch.arange(x.size(0)).cuda()
            loss = (F.cross_entropy(logits_per_image, labels) +
                    F.cross_entropy(logits_per_text, labels)) / 2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        zeroshot_classifier = MeanEnsembler(encode_prompts(model, prompts))
        acc = run_test(val_loader, model, zeroshot_classifier)
        print('val acc:', acc)
        break

    model.eval()
    return model


class FLYPDataset(Dataset):
    def __init__(self, dataset, prompts):
        self.dataset = dataset
        self.prompts = prompts
        assert all(len(x) == len(prompts[0]) for x in prompts)

    def __len__(self):
        return len(self.dataset) * len(self.prompts[0])

    def __getitem__(self, idx):
        image_index, prompt_index = divmod(idx, len(self.prompts[0]))
        x, y = self.dataset[image_index]
        z = self.prompts[y][prompt_index]
        return x, clip.tokenize(z)[0]


if __name__ == "__main__":
    method_name = sys.argv[1] if len(sys.argv) > 1 else 'CLIP'
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'RN50'
    dataset_name = sys.argv[3] if len(sys.argv) > 3 else 'Caltech101'
    batch_size = 64

    # load model & transforms
    model, train_preprocess, val_preprocess = load_clip(model_name, fp32=False)

    # load prompts
    prompts = get_prompts(method_name, dataset_name)

    # load datasets
    datasets = get_datasets(dataset_name, train_preprocess, val_preprocess)
    train_dataset = FLYPDataset(datasets['train'], prompts)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True, drop_last=True)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

    # run test
    zeroshot_classifier = MeanEnsembler(encode_prompts(model, prompts))
    acc = run_test(test_loader, model, zeroshot_classifier)
    print(method_name, model_name, dataset_name, acc, sep='\t', flush=True)
    print(model.logit_scale.exp().item())

    # train
    model = run_train(model, train_loader, val_loader)

    # run test
    zeroshot_classifier = MeanEnsembler(encode_prompts(model, prompts))
    acc = run_test(test_loader, model, zeroshot_classifier)
    print(method_name, model_name, dataset_name, acc, sep='\t', flush=True)
    print(model.logit_scale.exp().item())
