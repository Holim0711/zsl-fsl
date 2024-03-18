import os
import sys
import torch
import clip
import json
import random
import yaml
import matplotlib.pyplot as plt


class AttViewerResidualAttentionBlock(clip.model.ResidualAttentionBlock):
    def __init__(self, model: clip.model.ResidualAttentionBlock):
        d_model = model.attn.embed_dim
        n_head = model.attn.num_heads
        attn_mask = model.attn_mask
        super().__init__(d_model, n_head, attn_mask)
        clip.model.convert_weights(self)
        self.load_state_dict(model.state_dict())
        self.to(next(model.parameters()).device)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, average_attn_weights=False, attn_mask=self.attn_mask)

    def forward(self, x: torch.Tensor):
        a, w = self.attention(self.ln_1(x))
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, w


class AttViewerCLIP(clip.model.CLIP):
    def __init__(self, model: clip.model.CLIP):
        torch.nn.Module.__init__(self)
        self.context_length = model.context_length
        self.visual = model.visual
        self.transformer = model.transformer
        self.vocab_size = model.vocab_size
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.logit_scale = model.logit_scale
        
        last_layer = self.transformer.resblocks.pop(-1)
        last_layer = AttViewerResidualAttentionBlock(last_layer)
        self.transformer.resblocks.append(last_layer)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, w = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x, w


def get_list_of_texts(classes):
    list_of_texts = []
    for i, c in enumerate(classes):
        c = c.replace(' ', '_').replace('/', '_')
        filename = f'data/Desc/FGVCAircraft/description/{i:03d}.{c}.txt'
        texts = [x.strip() for x in open(filename)]
        texts = [x.split(':', maxsplit=1)[1].strip() for x in texts]
        list_of_texts.append(texts)
    return list_of_texts


def get_list_of_texts_CuPL(classes, method='base', rand_k=None):
    if method == 'base':
        data = json.load(open('data/CuPL/base/airplane_prompts_base.json'))
    elif method == 'full':
        data = json.load(open('data/CuPL/full/airplane_prompts_full.json'))
    list_of_texts = [data[c] for c in classes]
    if rand_k:
        state = random.Random(0)
        list_of_texts = [state.sample(v, rand_k) for v in list_of_texts]
    return list_of_texts


def show_attention(tokens, weights):
    TEMPLATE = '<span style="background-color:rgba(255,0,0,{color});">{text}</span>'
    min_w, max_w = min(weights), max(weights)
    weights = [(w - min_w) / (max_w - min_w) for w in weights]
    # weights = [w for w in weights]
    html = [TEMPLATE.format(color=w, text=t) for t, w in zip(tokens, weights)]
    return '<div>' + ''.join(html) + '</div>'


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else 'RN50'
    dataset_name = sys.argv[2] if len(sys.argv) > 2 else 'ImageNet'
    dataset_root = os.environ['TORCHVISION_DATASETS']

    # load model & dataset
    model, preprocess = clip.load(model_name)
    model = AttViewerCLIP(model)

    # load classes & templates
    class_path = os.path.join('data', 'CoOp', dataset_name)
    classes = yaml.safe_load(open(os.path.join(class_path, 'classes.yaml')))
    templates = yaml.safe_load(open(os.path.join(class_path, 'templates.yaml')))

    # load text definitions
    list_of_texts = get_list_of_texts(classes)
    # # replace {class_name} → 'this'
    # list_of_texts = [['This is ' + templates[0].format(c) + ' ' + x for x in v] for c, v in zip(classes, list_of_texts)]
    # list_of_texts = [[x.replace(c, f'aircraft {c}') for x in v] for c, v in zip(classes, list_of_texts)]
    # list_of_texts = get_list_of_texts_CuPL(classes)

    decoder = clip.simple_tokenizer.SimpleTokenizer()

    with torch.no_grad():
        for texts in list_of_texts:
            tokens = clip.tokenize(texts).cuda()
            vecs, weights = model.encode_text(tokens)
            for i, (token, vec, weight) in enumerate(zip(tokens, vecs, weights)):
                print('-------------------')
                end = token.argmax()
                token = token[1:end]
                weight = weight[:, end, 1:end]
                token = [decoder.decode([t.item()]) for t in token]
                avg_weight = weight.mean(dim=0)
                
                html = show_attention(token, avg_weight)
                for w in weight:
                    html += show_attention(token, w)
                with open(f'results/amap/txt/qqq-{i}.html', 'w') as f:
                    print(html, file=f)
            exit()