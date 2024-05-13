import torch.nn as nn
import torchvision.transforms as vtrfm
import clip
from PIL import Image


def _convert_weights_to_fp32(l):
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        l.weight.data = l.weight.data.float()
        if l.bias is not None:
            l.bias.data = l.bias.data.float()

    if isinstance(l, nn.MultiheadAttention):
        for attr in ["q_proj_weight", "k_proj_weight", "v_proj_weight",
                     "in_proj_weight", "in_proj_bias", "bias_k", "bias_v"]:
            tensor = getattr(l, attr)
            if tensor is not None:
                tensor.data = tensor.data.float()

    for name in ["text_projection", "proj"]:
        if hasattr(l, name):
            attr = getattr(l, name)
            if attr is not None:
                attr.data = attr.data.float()    


def _convert_image_to_rgb(image: Image.Image):
    return image.convert("RGB")


def _train_transform(n_px: int):
    BICUBIC = vtrfm.InterpolationMode.BICUBIC
    AVG_COLOR = (0.48145466, 0.4578275, 0.40821073)
    STD_COLOR = (0.26862954, 0.26130258, 0.27577711)
    return vtrfm.Compose([
        vtrfm.RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=BICUBIC),
        _convert_image_to_rgb,
        vtrfm.ToTensor(),
        vtrfm.Normalize(AVG_COLOR, STD_COLOR),
    ])


def load_clip_fp32(name: str):
    model, val_preprocess = clip.load(name)
    train_preprocess = _train_transform(model.visual.input_resolution)
    model.apply(_convert_weights_to_fp32)
    return model, train_preprocess, val_preprocess
