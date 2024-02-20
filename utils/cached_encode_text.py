import os
import torch
import clip
import hashlib


def encode_text(
    model: clip.model.CLIP,
    texts: list,
    cache: str = '.cache',
):
    htxt = model.name + ''.join([f'\n- {x}' for x in texts])
    hval = hashlib.md5(htxt.encode()).hexdigest()
    cache_file = os.path.join(cache, hval)
    if not os.path.isfile(cache_file):
        with torch.no_grad():
            t = clip.tokenize(texts).cuda()
            v = model.encode_text(t)
        torch.save(v, cache_file)
    return torch.load(cache_file)
