def get_size_of_torchvision_datasets(name):
    from zsfs.datasets.torchvision import get_datasets
    return {k: len(v) for k, v in get_datasets(name).items()}


def get_size_of_coop_datasets(name):
    from zsfs.datasets.coop import get_datasets
    return {k: len(v) for k, v in get_datasets(name).items()}


if __name__ == '__main__':
    names = [
        'ImageNet',
        'Caltech101',
        'DTD',
        'EuroSAT',
        'FGVCAircraft',
        'Flowers102',
        'Food101',
        'OxfordIIITPet',
        'StanfordCars',
        'SUN397',
        'UCF101',
    ]
    for name in names:
        print(name)
        tv = get_size_of_torchvision_datasets(name)
        co = get_size_of_coop_datasets(name)
        print('torchvision:', sum(tv.values()), tv)
        print('coop:', sum(co.values()), co)
        print()
