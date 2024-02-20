import torch
from tqdm import tqdm
import clip
from .cached_encode_text import encode_text


def zeroshot_classifier(
    model: clip.model.CLIP,
    classes: list,
    templates: list,
):
    with torch.no_grad():
        zeroshot_weights = []
        for c in tqdm(classes):
            texts = [template.format(c) for template in templates]
            class_embeddings = encode_text(model, texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights
