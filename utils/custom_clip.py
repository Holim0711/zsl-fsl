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
        return x, attention_weights


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


# class CustomTransformer(clip.model.Transformer):
# class CustomVisionTransformer(clip.model.VisionTransformer):


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
            raise NotImplementedError()
        else:
            raise Exception("Unknown CLIP Visual Encoder Structure")

        # Language Encoder
        last_layer = self.transformer.resblocks.pop(-1)
        last_layer = CustomResidualAttentionBlock(last_layer)
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
