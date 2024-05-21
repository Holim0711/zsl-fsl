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
from zsfs.datasets.utils import FewShotSubset
from zsfs.prompts import get_prompts
import torch.nn as nn


@torch.no_grad()
def run_test(
    model: CLIP,
    loader: DataLoader,
    prompts: list[list[str]],
):
    zeroshot_classifier = MeanEnsembler(encode_prompts(model, prompts))
    zeroshot_classifier.eval()
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
    train_dataset: Dataset,
    val_dataset: Dataset,
    prompts: list[list[str]],
    total_epochs: int = 1,
    warmup_epochs: float = 0,
    lr: float = 1e-5,
    wd: float = 0.0,
    eps: float = 1e-8,
    batch_size: int = 1,
):
    save_path = f'best_model_{total_epochs}_{warmup_epochs}_{lr}_{wd}_{eps}_{batch_size}.pth'

    train_loader = DataLoader(
        FLYPDataset(train_dataset, prompts),
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True
    )

    num_iters = len(train_loader)
    total_iters = total_epochs * num_iters
    warmup_iters = int(warmup_epochs * num_iters)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, eps=eps)
    if warmup_iters > 0:
        scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=(1 / warmup_iters), total_iters=warmup_iters)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_iters - warmup_iters))
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_iters])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)
    print('start training')

    best_acc = 0.0
    best_epoch = 0

    for curr_epoch in range(1, total_epochs + 1):
        model.train()
        pbar_info = f'[acc] {best_acc}@{best_epoch}, [epoch] {curr_epoch}/{total_epochs}'
        for x, y in tqdm(train_loader, desc=pbar_info):
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
        acc = run_test(model, val_loader, prompts)
        if acc > best_acc:
            best_acc = acc
            best_epoch = curr_epoch
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path))
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
    total_epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    warmup_epochs = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
    lr = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-5
    wd = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
    random_seed = int(sys.argv[8]) if len(sys.argv) > 8 else None
    batch_size = 64

    # load model & transforms
    model, train_preprocess, val_preprocess = load_clip(model_name, fp32=True)

    # load prompts
    prompts = get_prompts(method_name, dataset_name)

    # load datasets
    datasets = get_datasets(dataset_name, train_preprocess, val_preprocess)
    datasets = {
        'train' : FewShotSubset(datasets['train'], 16, random_seed=random_seed),
        'val' : FewShotSubset(datasets['val'], 16, random_seed=random_seed, oversampling=True),
        'test' : datasets['test']
    }
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)

    # train
    model = run_train(
        model,
        datasets['train'],
        datasets['val'],
        prompts,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        lr=lr,
        wd=wd,
        batch_size=batch_size
    )

    # run test
    zeroshot_classifier = MeanEnsembler(encode_prompts(model, prompts))
    acc = run_test(test_loader, model, zeroshot_classifier)
    print(method_name, model_name, dataset_name, total_epochs, warmup_epochs, lr, wd, batch_size, acc, sep='\t', flush=True)
