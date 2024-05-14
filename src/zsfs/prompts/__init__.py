import yaml
from importlib import resources as impresources
from . import CLIP
from . import CoOp
from . import CuPL


def read_file_in_module(module, *path):
    resource = impresources.files(module)
    for p in path:
        resource = resource / p
    return resource.open('rt')


def read_yaml_in_module(module, *path):
    with read_file_in_module(module, *path) as f:
        return yaml.safe_load(f)


def read_json_in_module(module, *path):
    with read_file_in_module(module, *path) as f:
        return yaml.safe_load(f)


def read_cupl_prmopts(dataset: str, base_or_full: str):
    dataset = {
        'FGVCAircraft': 'airplane',
        'Birdsnap': 'bird',
        'Caltech101': 'cal',
        'StanfordCars': 'cars',
        'CIFAR10': 'cifar10',
        'CIFAR100': 'cifar100',
        'Flowers102': 'flower',
        'Food101': 'food',
        'ImageNet': 'imagenet',
        'Kinetics700': 'kinetics',
        'OxfordPets': 'pets',
        'RESISC45': 'res45',
        'SUN397': 'sun',
        'DTD': 'texture',
        'UCF101': 'ucf',
    }[dataset]
    filename = dataset + f'_prompts_{base_or_full}.json'
    return read_json_in_module(CuPL, 'prompts', base_or_full, filename)


def get_classes(method: str, dataset: str):
    if method == 'CLIP':
        return read_yaml_in_module(CLIP, dataset, 'classes.yaml')
    elif method == 'CoOp':
        return read_yaml_in_module(CoOp, dataset, 'classes.yaml')
    elif method == 'CuPL.base' or method == 'CuPL.full':
        return read_yaml_in_module(CuPL, 'classes', dataset + '.yaml')
    else:
        raise ValueError(f'Unknown method: {method}')


def get_templates(method: str, dataset: str):
    if method == 'CLIP':
        return read_yaml_in_module(CLIP, dataset, 'templates.yaml')
    elif method == 'CoOp':
        return read_yaml_in_module(CoOp, dataset, 'templates.yaml')
    else:
        raise ValueError(f'Unknown method: {method}')


def get_prompts(method: str, dataset_module: str, dataset_name: str):
    if method == 'CLIP':
        classes = get_classes(method, dataset_name)
        templates = get_templates(method, dataset_name)
        prompts = [[t.format(c) for t in templates] for c in classes]
    elif method == 'CoOp':
        classes = get_classes(method, dataset_name)
        templates = get_templates(method, dataset_name)
        prompts = [[t.format(c) for t in templates] for c in classes]
    elif method == 'CuPL.base':
        classes = get_classes(method, dataset_name)
        prompts = read_cupl_prmopts(dataset_name, 'base')
        prompts = [prompts[c] for c in classes]
    elif method == 'CuPL.full':
        classes = get_classes(method, dataset_name)
        prompts = read_cupl_prmopts(dataset_name, 'full')
        prompts = [prompts[c] for c in classes]
    else:
        raise ValueError(f'Unknown method: {method}')

    # special treatment for CoOp.Caltech101
    # - Original: 101 classes ('Face', 'Face_easy', ...)
    # - CoOp: 100 classes ('Face', ...)
    if dataset_name == 'Caltech101':
        if dataset_module.lower() == 'coop' and len(prompts) == 101:
            prompts.pop(1)                  # remove 'Face_easy'
        if dataset_module.lower() != 'coop' and len(prompts) == 100:
            prompts.insert(0, prompts[0])   # copy 'Face' to 'Face_easy'

    return prompts
