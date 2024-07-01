# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, mmseg, setr, xcit and swin code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/fudan-zvg/SETR
# https://github.com/facebookresearch/xcit/
# https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from compressai.entropy_models import EntropyBottleneck
from compressai.models.base import CompressionModel
from einops import rearrange, repeat
from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Block, PatchEmbed

from ..builder import BACKBONES


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return "p={}".format(self.drop_prob)


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# Copyright contributors to the neural-embedding-compression project
@BACKBONES.register_module()
class ViT_Compress_Entropy(CompressionModel):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=80,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        use_checkpoint=False,
        use_abs_pos_emb=False,
        use_rel_pos_bias=False,
        out_indices=[11],
        pretrained=None,
        with_decoder=False,
        qk_scale=None,  # unused
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.bottleneck = EntropyBottleneck(embed_dim)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.with_decoder = with_decoder
        num_patches = self.patch_embed.num_patches

        self.out_indices = out_indices

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_checkpoint = use_checkpoint

        # MHSA after interval layers
        # WMHSA in other layers
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        if with_decoder:
            self.decoder_embed_dim = 512
            self.decoder_embed = nn.Linear(embed_dim, self.decoder_embed_dim, bias=True)
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, self.decoder_embed_dim),
                requires_grad=False,
            )

            self.decoder_blocks = nn.ModuleList(
                [
                    Block(
                        dim=self.decoder_embed_dim,
                        num_heads=16,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,
                    )
                ]
            )

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.norm = norm_layer(embed_dim)

        embed_dim = self.decoder_embed_dim if with_decoder else self.embed_dim
        self.first_conv = nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1)
        self.conv_blocks = nn.Sequential(*[ConvBlock(embed_dim // 2) for _ in range(6)])
        self.upscaling_blocks = nn.Sequential(
            UpscalingBlock(embed_dim // 2), UpscalingBlock(embed_dim // 2)
        )

        self.apply(self._init_weights)
        self.fix_init_weight()
        self.pretrained = pretrained

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
                pretrained (str, optional): Path to pre-trained weights.
                        Defaults to None.
        """
        pretrained = pretrained or self.pretrained

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            print(f"load from {pretrained}")
            load_checkpoint(self, pretrained, strict=False, logger=logger)

        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_embedding(self, x):
        B, C, H, W = x.shape
        Hp, Wp = (
            H // self.patch_embed.patch_size[0],
            W // self.patch_embed.patch_size[1],
        )
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.norm(x)
        latent = rearrange(x, "b s (e n) -> b e s n", n=1)
        quant_dequant_latent, _ = self.bottleneck(latent, training=False)
        x = rearrange(quant_dequant_latent, "b e s n -> b s (e n)", n=1)

        if self.with_decoder:
            x = self.decoder_embed(x)
            x = x + self.decoder_pos_embed[:, 1:, :]
            x = self.decoder_blocks[0](x)

        x = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp)
        x = self.first_conv(x)
        original = x.clone()
        x = self.conv_blocks(x)
        x = x + original

        x = self.upscaling_blocks(x)
        return [x]

    def forward(self, x):
        x = self.forward_embedding(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gelu1 = nn.GELU()
        self.norm = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        original = x.clone()
        x = self.gelu1(self.conv1(x))
        x = self.norm(x)
        x = self.gelu2(self.conv1(x))
        return original + x


class UpscalingBlock(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(
            self.channels, self.channels * 2 * 2, kernel_size=3, stride=1, padding=1
        )
        self.upscale = nn.PixelShuffle(2)
        self.bn = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        post_conv = self.conv(x)
        upscaled = self.upscale(post_conv)
        return self.relu(self.bn(upscaled))


@BACKBONES.register_module()
class ViT_Compress(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=80,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        use_checkpoint=False,
        use_abs_pos_emb=False,
        use_rel_pos_bias=False,
        out_indices=[11],
        pretrained=None,
        quant=None,
        with_decoder=False,
        qk_scale=None,  # unused
    ):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.with_decoder = with_decoder

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        self.out_indices = out_indices
        self.quant = quant

        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        if with_decoder:
            self.decoder_embed_dim = 512
            self.decoder_embed = nn.Linear(embed_dim, self.decoder_embed_dim, bias=True)
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, self.decoder_embed_dim),
                requires_grad=False,
            )

            self.decoder_blocks = nn.ModuleList(
                [
                    Block(
                        dim=self.decoder_embed_dim,
                        num_heads=16,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,
                    )
                ]
            )

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.norm = norm_layer(embed_dim)
        embed_dim = self.decoder_embed_dim if with_decoder else self.embed_dim
        self.first_conv = nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1)
        self.conv_blocks = nn.Sequential(*[ConvBlock(embed_dim // 2) for _ in range(6)])
        self.upscaling_blocks = nn.Sequential(
            UpscalingBlock(embed_dim // 2), UpscalingBlock(embed_dim // 2)
        )

        self.apply(self._init_weights)
        self.fix_init_weight()
        self.pretrained = pretrained

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
                pretrained (str, optional): Path to pre-trained weights.
                        Defaults to None.
        """
        pretrained = pretrained or self.pretrained

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            print(f"load from {pretrained}")

            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError("pretrained must be a str or None")

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        B, C, H, W = x.shape
        Hp, Wp = (
            H // self.patch_embed.patch_size[0],
            W // self.patch_embed.patch_size[1],
        )
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x = self.norm(x)
        if self.quant and self.quant != 32:
            if self.quant == 16:
                x = x.to(torch.float16).float()
            else:
                # min max quant
                latent_min = x.min()
                latent_max = x.max()
                scale = (2**self.quant) - 1
                scaled = ((x - latent_min) / (latent_max - latent_min)) * scale
                quantized = scaled.round()
                assert (quantized <= scale).all() and (quantized >= 0).all()
                latent = quantized / scale
                latent = latent * (latent_max - latent_min)
                x = latent + latent_min

        if self.with_decoder:
            x = self.decoder_embed(x)
            x = x + self.decoder_pos_embed[:, 1:, :]
            x = self.decoder_blocks[0](x)

        x = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp)

        x = self.first_conv(x)
        original = x.clone()
        x = self.conv_blocks(x)
        x = x + original

        x = self.upscaling_blocks(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        return [x]
