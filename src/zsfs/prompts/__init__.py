import yaml
from importlib import resources as impresources
from . import CLIP
from . import CoOp


def read_file_in_module(module, *path):
    resource = impresources.files(module)
    for p in path:
        resource = resource / p
    return resource.open('rt')


def read_yaml_in_module(module, *path):
    with read_file_in_module(module, *path) as f:
        return yaml.safe_load(f)


def get_classes(method: str, dataset: str):
    if method == 'CLIP':
        return read_yaml_in_module(CLIP, dataset, 'classes.yaml')
    elif method == 'CoOp':
        return read_yaml_in_module(CoOp, dataset, 'classes.yaml')
    else:
        raise ValueError(f'Unknown method: {method}')


def get_prompts(method: str, dataset: str):
    if method == 'CLIP':
        classes = get_classes(method, dataset)
        templates = read_yaml_in_module(CLIP, dataset, 'templates.yaml')
        return [[t.format(c) for t in templates] for c in classes]
    elif method == 'CoOp':
        classes = get_classes(method, dataset)
        templates = read_yaml_in_module(CoOp, dataset, 'templates.yaml')
        return [[t.format(c) for t in templates] for c in classes]
    else:
        raise ValueError(f'Unknown method: {method}')
