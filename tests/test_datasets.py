def test_torchvision_datasets():
    from zsfs.datasets.torchvision import build_datasets

    names = [
        'ImageNet',
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

    print('-',  'torchvision')
    for name in names:
        datasets = build_datasets(name)
        print('--', name)
        for k, v in datasets.items():
            print('---', k, ':', len(v))


def test_coop_datasets():
    from zsfs.datasets.coop import build_datasets

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

    print('-',  'CoOp')
    for name in names:
        datasets = build_datasets(name)
        print('--', name)
        for k, v in datasets.items():
            print('---', k, ':', len(v))


if __name__ == '__main__':
    test_torchvision_datasets()
    test_coop_datasets()
