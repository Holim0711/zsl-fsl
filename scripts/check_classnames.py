import os
import json

DATA = 'path/to/zhou/split/files'


for f in os.listdir(DATA):
    print(f)
    data = json.load(open(os.path.join(DATA, f)))
    train, val, test = data['train'], data['val'], data['test']

    # check multiple classnames
    mapping = {}
    for _, y, n in train + val + test:
        if (z := mapping.setdefault(y, n)) != n:
            raise ValueError(f'multiple names for class {y}: {z}, {n}')

    # check class indices
    classnames = [mapping[i] for i in range(len(mapping))]
