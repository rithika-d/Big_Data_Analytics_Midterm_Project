"""
Use EVA-X series as your backbone. You could get
EVA-X representations simply with timm. Try them
with your own X-ray tasks.
Enjoy!

Reference:
    https://github.com/baaivision/EVA
    https://github.com/huggingface/pytorch-image-models
Thanks for their work!

by Jingfeng Yao
from HUST-VL
"""

import torch
from timm.layers import resample_abs_pos_embed, resample_patch_embed
from timm.models.eva import Eva


def checkpoint_filter_fn(
    state_dict,
    model,
    interpolation="bicubic",
    antialias=True,
):
    """Convert patch embedding weights from patchify+linear into conv kernels."""
    out_dict = {}
    state_dict = state_dict.get("model_ema", state_dict)
    state_dict = state_dict.get("model", state_dict)
    state_dict = state_dict.get("module", state_dict)
    state_dict = state_dict.get("state_dict", state_dict)
    if "visual.trunk.pos_embed" in state_dict:
        prefix = "visual.trunk."
    elif "visual.pos_embed" in state_dict:
        prefix = "visual."
    else:
        prefix = ""
    mim_weights = prefix + "mask_token" in state_dict
    no_qkv = prefix + "blocks.0.attn.q_proj.weight" in state_dict

    len_prefix = len(prefix)
    for key, value in state_dict.items():
        if prefix:
            if key.startswith(prefix):
                key = key[len_prefix:]
            else:
                continue

        if "rope" in key:
            continue

        if "patch_embed.proj.weight" in key:
            _, _, height, width = model.patch_embed.proj.weight.shape
            if value.shape[-1] != width or value.shape[-2] != height:
                value = resample_patch_embed(
                    value,
                    (height, width),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif key == "pos_embed" and value.shape[1] != model.pos_embed.shape[1]:
            num_prefix_tokens = (
                0
                if getattr(model, "no_embed_class", False)
                else getattr(model, "num_prefix_tokens", 1)
            )
            value = resample_abs_pos_embed(
                value,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )

        key = key.replace("mlp.ffn_ln", "mlp.norm")
        key = key.replace("attn.inner_attn_ln", "attn.norm")
        key = key.replace("mlp.w12", "mlp.fc1")
        key = key.replace("mlp.w1", "mlp.fc1_g")
        key = key.replace("mlp.w2", "mlp.fc1_x")
        key = key.replace("mlp.w3", "mlp.fc2")
        if no_qkv:
            key = key.replace("q_bias", "q_proj.bias")
            key = key.replace("v_bias", "v_proj.bias")

        if mim_weights and key in (
            "mask_token",
            "lm_head.weight",
            "lm_head.bias",
            "norm.weight",
            "norm.bias",
        ):
            if key == "norm.weight" or key == "norm.bias":
                key = key.replace("norm", "fc_norm")
            else:
                continue

        out_dict[key] = value

    return out_dict


class EVA_X(Eva):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        for block in self.blocks:
            x = block(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            x = (
                x[:, self.num_prefix_tokens :].mean(dim=1)
                if self.global_pool == "avg"
                else x[:, 0]
            )
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def eva_x_tiny_patch16(pretrained=False):
    model = EVA_X(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),
    )
    eva_ckpt = checkpoint_filter_fn(
        torch.load(pretrained, map_location="cpu", weights_only=False), model
    )
    msg = model.load_state_dict(eva_ckpt, strict=False)
    print(msg)
    return model


def eva_x_small_patch16(pretrained=False):
    model = EVA_X(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),
    )
    eva_ckpt = checkpoint_filter_fn(torch.load(pretrained, map_location="cpu"), model)
    msg = model.load_state_dict(eva_ckpt, strict=False)
    print(msg)
    return model


def eva_x_base_patch16(pretrained=False):
    model = EVA_X(
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),
    )
    eva_ckpt = checkpoint_filter_fn(torch.load(pretrained, map_location="cpu"), model)
    msg = model.load_state_dict(eva_ckpt, strict=False)
    print(msg)
    return model


if __name__ == "__main__":
    eva_x_ti_pt = (
        "/home/jingfengyao/code/medical/EVA-X/classification/pretrained/eva_x_ti_16.pt"
    )
    eva_x_s_pt = (
        "/home/jingfengyao/code/medical/EVA-X/classification/pretrained/eva_x_s_16.pt"
    )
    eva_x_b_pt = (
        "/home/jingfengyao/code/medical/EVA-X/classification/pretrained/eva_x_b_16.pt"
    )

    eva_x_tiny_patch16(pretrained=eva_x_ti_pt)
    eva_x_small_patch16(pretrained=eva_x_s_pt)
    eva_x_base_patch16(pretrained=eva_x_b_pt)
