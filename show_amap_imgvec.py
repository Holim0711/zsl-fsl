import os
import sys
import torch
import torch.nn.functional as F
import clip
import matplotlib.pyplot as plt
import math

from datasets import build_coop_datasets
from utils import CustomCLIP


def my_mha_just_for_check(
    query,
    key,
    value,
    num_heads,
    q_proj_weight,
    k_proj_weight,
    v_proj_weight,
    q_proj_bias,
    k_proj_bias,
    v_proj_bias,
    out_proj_weight,
    out_proj_bias,
    need_weights: bool,
    average_attn_weights: bool = True,
):
    # F.multi_head_attention_forward 함수에서 필요한 부분만 재구성한 것
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    head_dim = embed_dim // num_heads

    q = F.linear(query, q_proj_weight, q_proj_bias)
    k = F.linear(key, k_proj_weight, k_proj_bias)
    v = F.linear(value, v_proj_weight, v_proj_bias)

    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    if need_weights:
        _, _, E = q.shape
        q_scaled = q / math.sqrt(E)

        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        return attn_output, attn_output_weights
    else:
        # attn_mask can be either (L,S) or (N*num_heads, L, S)
        # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
        # in order to match the input for SDPA of (N, num_heads, L, S)
        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

        attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        return attn_output, None


def show_attention(image_tensor, attention_maps, title):
    image_tensor = image_tensor.cpu()
    attention_maps = attention_maps.cpu()
    # 이미지 텐서의 정규화 역변환
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    image_tensor = image_tensor * std + mean  # 정규화 역변환
    image_tensor = image_tensor.squeeze(0)  # 배치 차원 제거

    # 어텐션 맵 재배열
    attention_maps = attention_maps.view(-1, 7, 7)  # (32, 7, 7)로 재배열
    avg_attention_maps = attention_maps.mean(dim=0)
    attention_maps = torch.cat([avg_attention_maps.unsqueeze(0), attention_maps])

    # 시각화
    fig, ax = plt.subplots(4, 9, figsize=(20, 10))
    # 이미지 시각화
    ax[0, 0].imshow(image_tensor.permute(1, 2, 0).clamp(0, 1))
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis('off')

    # 어텐션 맵 시각화
    for i in range(len(attention_maps)):
        row, col = divmod(i+1, 9)
        ax[row, col].imshow(attention_maps[i], cmap='jet', interpolation='bicubic')
        ax[row, col].set_title("Avg" if i == 0 else f"Head {i}")
        ax[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(title)


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'RN50'
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else 'ImageNet'
    dataset_root = os.environ['TORCHVISION_DATASETS']

    # load model & dataset
    model, preprocess = clip.load(model_name)
    model = CustomCLIP(model)
    datasets = build_coop_datasets(dataset_name, dataset_root)

    with torch.no_grad():
        for i in [0, 1, 2, 3, 4]:
            x, y = datasets['test'][i]
            x = preprocess(x).unsqueeze(0).cuda()
            v, w = model.encode_image(x)
            w = w[0, :, 0, :]
            w[:, -1] = 0  # only for CLIP RN50
            show_attention(x, w[:, 1:], f'results/amap/img/{i:02d}.png')
