import os
import sys
import torch
import clip
import json
import random
import yaml
from utils import CustomCLIP


def get_list_of_texts(classes):
    list_of_texts = []
    for i, c in enumerate(classes):
        c = c.replace(' ', '_').replace('/', '_')
        filename = f'data/Desc/FGVCAircraft/description/{i:03d}.{c}.txt'
        texts = [x.strip() for x in open(filename)]
        texts = [x.split(':', maxsplit=1)[1].strip() for x in texts]
        list_of_texts.append(texts)
    return list_of_texts


def get_list_of_texts_CuPL(classes, method='base', rand_k=None):
    if method == 'base':
        data = json.load(open('data/CuPL/base/airplane_prompts_base.json'))
    elif method == 'full':
        data = json.load(open('data/CuPL/full/airplane_prompts_full.json'))
    list_of_texts = [data[c] for c in classes]
    if rand_k:
        state = random.Random(0)
        list_of_texts = [state.sample(v, rand_k) for v in list_of_texts]
    return list_of_texts


def show_attention(tokens, weights):
    TEMPLATE = '<span style="background-color:rgba(255,0,0,{color});">{text}</span>'
    min_w, max_w = min(weights), max(weights)
    weights = [(w - min_w) / (max_w - min_w) for w in weights]
    # weights = [w for w in weights]
    html = [TEMPLATE.format(color=w, text=t) for t, w in zip(tokens, weights)]
    return '<div>' + ''.join(html) + '</div>'


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'RN50'
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else 'ImageNet'
    dataset_root = os.environ['TORCHVISION_DATASETS']

    # load model & dataset
    model, preprocess = clip.load(model_name)
    model = CustomCLIP(model)

    # load classes & templates
    class_path = os.path.join('data', 'CoOp', dataset_name)
    classes = yaml.safe_load(open(os.path.join(class_path, 'classes.yaml')))
    templates = yaml.safe_load(open(os.path.join(class_path, 'templates.yaml')))

    # load text definitions
    list_of_texts = get_list_of_texts(classes)
    # # replace {class_name} → 'this'
    # list_of_texts = [['This is ' + templates[0].format(c) + ' ' + x for x in v] for c, v in zip(classes, list_of_texts)]
    # list_of_texts = [[x.replace(c, f'aircraft {c}') for x in v] for c, v in zip(classes, list_of_texts)]
    # list_of_texts = get_list_of_texts_CuPL(classes)

    decoder = clip.simple_tokenizer.SimpleTokenizer()

    with torch.no_grad():
        for texts in list_of_texts:
            tokens = clip.tokenize(texts).cuda()
            vecs, weights = model.encode_text(tokens)
            for i, (token, vec, weight) in enumerate(zip(tokens, vecs, weights)):
                print('-------------------')
                end = token.argmax()
                token = token[1:end]
                weight = weight[:, end, 1:end]
                token = [decoder.decode([t.item()]) for t in token]
                avg_weight = weight.mean(dim=0)
                
                html = show_attention(token, avg_weight)
                for w in weight:
                    html += show_attention(token, w)
                with open(f'results/amap/txt/qqq-{i}.html', 'w') as f:
                    print(html, file=f)
            exit()