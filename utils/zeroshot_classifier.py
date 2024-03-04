import os
from hashlib import md5
from tqdm import tqdm
import torch
import clip


@torch.no_grad
def _encode_list_of_texts(
    model: clip.model.CLIP,
    list_of_texts: list[list[str]],
):
    list_of_embeddings = []
    for texts in tqdm(list_of_texts):
        tokens = clip.tokenize(texts).cuda()
        embeddings = model.encode_text(tokens)
        list_of_embeddings.append(embeddings)
    return list_of_embeddings


def encode_list_of_texts(
    model: clip.model.CLIP,
    list_of_texts: list[list[str]],
    cache_dir: str,
):
    hash_str = str(model) + '\n' + str(list_of_texts)
    hash_val = md5(hash_str.encode()).hexdigest()
    cache_file = os.path.join(cache_dir, hash_val) + '.pt'
    if not os.path.isfile(cache_file):
        list_of_embeddings = _encode_list_of_texts(model, list_of_texts)
        torch.save(list_of_embeddings, cache_file)
    return torch.load(cache_file)


def prompt_ensembler(
    model: clip.model.CLIP,
    classes: list,
    templates: list,
    cache_dir: str = '.cache',
):
    list_of_texts = [[t.format(c) for t in templates] for c in classes]
    list_of_embeddings = encode_list_of_texts(model, list_of_texts, cache_dir)
    list_of_embeddings = [torch.nn.functional.normalize(x) for x in list_of_embeddings]
    embeddings = [x.mean(dim=0) for x in list_of_embeddings]
    embeddings = torch.stack(embeddings, dim=1)
    embeddings = torch.nn.functional.normalize(embeddings, dim=0)
    return embeddings   # (dim x n_classes)
