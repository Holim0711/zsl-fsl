import os
import sys
sys.path.append(os.getcwd())
from datasets import build_coop_datasets, build_torchvision_datasets

ROOT = os.environ['TORCHVISION_DATASETS']


names = [
    'Caltech101',
    'DTD',
    'EuroSAT',
    'FGVCAircraft',
    'Flowers102',
    'Food101',
    'OxfordPets',
    'StanfordCars',
    'SUN397',
    'UCF101',
]


for name in names:
    coops = build_coop_datasets(name, ROOT)
    tovis = build_torchvision_datasets(name, ROOT)

    print('-----', name, '-----')
    coops_len = [len(coops['train']), len(coops['val']), len(coops['test'])]
    tovis_len = [len(tovis.get('train', [])), len(tovis.get('val', [])), len(tovis.get('test', []))]
    print(sum(coops_len), coops_len)
    print(sum(tovis_len), tovis_len)

    if name == 'Caltech101':
        print('#.coops.classes:', 1 + max([x[1] for x in coops['test']]))
        print('#.tovis.classes:', 1 + max([x[1] for x in tovis['test']]))
        print('#.tovis.class.0:', len([x for x in tovis['test'] if x[1] == 0]))
