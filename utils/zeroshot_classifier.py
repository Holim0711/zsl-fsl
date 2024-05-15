import torch
from clip.model import CLIP
from clip import tokenize

@torch.no_grad()
def encode_texts(model: CLIP, texts: list[str], device: str = 'cuda'):
    x = tokenize(texts).to(device)
    x = model.encode_text(x)
    x = torch.nn.functional.normalize(x)
    return x


def encode_prompts(model: CLIP, prompts: list[list[str]], device: str = 'cuda'):
    return [encode_texts(model, x, device) for x in prompts]


class MeanEnsembler(torch.nn.Linear):
    def __init__(self, weights: list[torch.Tensor]):
        weights = torch.stack([x.mean(dim=0) for x in weights])
        weights = torch.nn.functional.normalize(weights)
        out_features, in_features = weights.shape
        super().__init__(
            in_features,
            out_features,
            bias=False,
            device=weights.device,
            dtype=weights.dtype,
        )
        self.weight.data = weights


class MaxEnsembler(torch.nn.Module):
    def __init__(self, weights: list[torch.Tensor]):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.cat(weights))
        self.indices = torch.tensor([
            i for i, x in enumerate(weights) for _ in range(len(x))
        ])
        self.register_buffer('indices', self.indices)
        self.n_classes = torch.tensor(len(weights))
        self.register_buffer('n_classes', self.n_classes)

    def forward(self, x):
        x = torch.nn.functional.normalize(x)
        x = x @ self.weights.t()
        z = torch.full((self.n_classes,), float('-inf'),
                       dtype=x.dtype, device=x.device)
        z = z.scatter_reduce_(0, self.indices, x, reduce='amax')
        return z
