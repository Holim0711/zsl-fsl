import clip
import torch
import torch.nn.functional as F


class CustomAttentionPool2d(clip.model.AttentionPool2d):

    def __init__(self, model: clip.model.AttentionPool2d):
        torch.nn.Module.__init__(self)
        self.positional_embedding = model.positional_embedding
        self.k_proj = model.k_proj
        self.q_proj = model.q_proj
        self.v_proj = model.v_proj
        self.c_proj = model.c_proj
        self.num_heads = model.num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        z = x
        x, attention_weights = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            # need_weights=False,
            need_weights=True,
            average_attn_weights=False
        )
        z = self.v_proj(z)
        z = self.c_proj(z)
        return x, attention_weights, z


class CustomResidualAttentionBlock(clip.model.ResidualAttentionBlock):

    def __init__(self, model: clip.model.ResidualAttentionBlock):
        torch.nn.Module.__init__(self)
        self.attn = model.attn
        self.ln_1 = model.ln_1
        self.mlp = model.mlp
        self.ln_2 = model.ln_2
        self.attn_mask = model.attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, average_attn_weights=False, attn_mask=self.attn_mask)

    def forward(self, x: torch.Tensor):
        a, w = self.attention(self.ln_1(x))
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x, w


class CustomTransformer(clip.model.Transformer):

    def __init__(self, model: clip.model.Transformer):
        torch.nn.Module.__init__(self)
        self.width = model.width
        self.layers = model.layers
        self.resblocks = torch.nn.Sequential(*[
            CustomResidualAttentionBlock(block) for block in model.resblocks
        ])

    def forward(self, x: torch.Tensor):
        W = []
        for block in self.resblocks:
            x, w = block(x)
            W.append(w)
        return x, torch.stack(W, dim=1)


class CustomVisionTransformer(clip.model.VisionTransformer):
    def __init__(self, model: clip.model.VisionTransformer):
        torch.nn.Module.__init__(self)
        self.input_resolution = model.input_resolution
        self.output_dim = model.output_dim
        self.conv1 = model.conv1
        self.class_embedding = model.class_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_pre = model.ln_pre
        self.transformer = CustomTransformer(model.transformer)
        # self.transformer.resblocks.pop(-1)  ###########################################
        self.ln_post = model.ln_post
        self.proj = model.proj

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, W = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj

        return x, W


class CustomCLIP(clip.model.CLIP):

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

        # Visual Encoder
        if isinstance(self.visual, clip.model.ModifiedResNet):
            self.visual.attnpool = CustomAttentionPool2d(self.visual.attnpool)
        elif isinstance(self.visual, clip.model.VisionTransformer):
            self.visual = CustomVisionTransformer(self.visual)
        else:
            raise Exception("Unknown CLIP Visual Encoder Structure")

        # Language Encoder
        self.transformer = CustomTransformer(self.transformer)
        # last_layer = self.transformer.resblocks.pop(-1)
        # last_layer = CustomResidualAttentionBlock(last_layer)
        # self.transformer.resblocks.append(last_layer)

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