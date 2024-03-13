import os
import sys
import torch
import torch.nn.functional as F
import clip
from tqdm import tqdm
from utils import encode_list_of_texts
from datasets import build_coop_datasets
import yaml

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random


def save_pca(vecs, labs, list_of_embeddings, name, cmap='viridis', fit='txt'):
    pca = PCA(n_components=2)

    if fit == 'all':
        all_data = pca.fit_transform(torch.cat([vecs, *list_of_embeddings]))
        t_vecs = all_data[:len(vecs)]
        t_embs = all_data[len(vecs):]
    elif fit == 'txt':
        pca.fit(torch.cat(list_of_embeddings))
        t_vecs = pca.transform(vecs)
        t_embs = pca.transform(torch.cat(list_of_embeddings))
    else:
        raise ValueError(fit)

    l_embs = [i for i, w in enumerate(list_of_embeddings) for _ in range(len(w))]
    l_embs = torch.tensor(l_embs)

    plt.figure(figsize=(8, 6)) # 새로운 그림 생성
    plt.scatter(t_vecs[:, 0], t_vecs[:, 1], c=labs, cmap=cmap, alpha=0.1)
    plt.scatter(t_embs[:, 0], t_embs[:, 1], c=l_embs, cmap=cmap, s=100)

    plt.xlabel('PCA feature 0')
    plt.ylabel('PCA feature 1')
    plt.title(f'PCA visualization {name}')
    plt.savefig('results/vecvis/' + name)
    plt.close()


def save_tsne(vecs, labs, list_of_embeddings, name, cmap='viridis'):
    tsne = TSNE(n_components=2 , random_state=0)
    all_data = torch.cat([vecs, *list_of_embeddings])
    all_data = tsne.fit_transform(all_data)
    t_vecs = all_data[:len(vecs)]
    t_embs = all_data[len(vecs):]
    l_embs = [i for i, w in enumerate(list_of_embeddings) for _ in range(len(w))]
    l_embs = torch.tensor(l_embs)

    plt.figure(figsize=(8, 6)) # 새로운 그림 생성
    plt.scatter(t_vecs[:, 0], t_vecs[:, 1], c=labs, cmap=cmap, alpha=0.1)
    plt.scatter(t_embs[:, 0], t_embs[:, 1], c=l_embs, cmap=cmap, s=100)

    plt.xlabel('t-SNE feature 0')
    plt.ylabel('t-SNE feature 1')
    plt.title(f't-SNE visualization {name}')
    plt.savefig('results/vecvis/' + name)
    plt.close()


@torch.no_grad()
def run_tsne(loader, model, list_of_embeddings):
    vecs = []
    labs = []
    for x, y in tqdm(loader):
        x = x.cuda()
        v = F.normalize(model.encode_image(x))
        vecs.append(v.cpu())
        labs.append(y)
    vecs = torch.cat(vecs)
    labs = torch.cat(labs)
    list_of_embeddings = [w.cpu() for w in list_of_embeddings]

    cmap = 'tab20'
    save_tsne(vecs, labs, list_of_embeddings, 'tsne.all.png', cmap=cmap)
    save_pca(vecs, labs, list_of_embeddings, 'pca.all.fit_all.png', cmap=cmap, fit='all')
    save_pca(vecs, labs, list_of_embeddings, 'pca.all.fit_txt.png', cmap=cmap, fit='txt')

    random_classes = random.sample(range(len(list_of_embeddings)), 10)
    selector = torch.isin(labs, torch.tensor(random_classes))
    vecs = vecs[selector]
    labs = labs[selector]
    list_of_embeddings = [list_of_embeddings[i] for i in random_classes]

    save_tsne(vecs, labs, list_of_embeddings, 'tsne.r10.png', cmap=cmap)
    save_pca(vecs, labs, list_of_embeddings, 'pca.r10.fit_all.png', cmap=cmap, fit='all')
    save_pca(vecs, labs, list_of_embeddings, 'pca.r10.fit_txt.png', cmap=cmap, fit='txt')


def get_list_of_texts(classes):
    list_of_texts = []
    for i, c in enumerate(classes):
        c = c.replace(' ', '_').replace('/', '_')
        filename = f'data/Desc/FGVCAircraft/description/{i:03d}.{c}.txt'
        texts = [x.strip() for x in open(filename)]
        texts = [x.split(':', maxsplit=1)[1].strip() for x in texts]
        texts = [x.replace(f'The {c}', 'This').replace(f'the {c}', 'this') for x in texts]
        list_of_texts.append(texts)
    return list_of_texts


def get_list_of_texts_CuPL(classes):
    import json
    import random
    data = json.load(open('data/CuPL/base/airplane_prompts_base.json'))
    # data = json.load(open('data/CuPL/full/airplane_prompts_full.json'))
    list_of_texts = [data[c] for c in classes]
    # state = random.Random(0)
    # list_of_texts = [state.sample(v, 10) for v in list_of_texts]
    return list_of_texts


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

    # load text definitions
    list_of_texts = get_list_of_texts(classes)
    # list_of_texts = get_list_of_texts_CuPL(classes)

    list_of_embeddings = encode_list_of_texts(model, list_of_texts, '.cache')
    list_of_embeddings = [torch.nn.functional.normalize(x) for x in list_of_embeddings]

    # run test
    run_tsne(loader, model, list_of_embeddings)
