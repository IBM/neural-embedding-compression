# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import math
from functools import partial
from typing import Union

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.base import CompressionModel
from compressai.models.utils import conv, deconv
from einops import rearrange, reduce
from timm.layers import AttentionPoolLatent
from timm.models.vision_transformer import Block, PatchEmbed
from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
	"""Masked Autoencoder with VisionTransformer backbone"""

	def __init__(
		self,
		img_size=224,
		patch_size=16,
		in_chans=3,
		embed_dim=1024,
		depth=24,
		num_heads=16,
		decoder_embed_dim=512,
		decoder_depth=8,
		decoder_num_heads=16,
		mlp_ratio=4.0,
		norm_layer=nn.LayerNorm,
		norm_pix_loss=False,
	):
		super().__init__()

		# --------------------------------------------------------------------------
		# MAE encoder specifics
		self.patch_size = patch_size
		self.img_size = img_size
		self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
		num_patches = self.patch_embed.num_patches
		self.num_heads = num_heads
		self.mlp_ratio = mlp_ratio
		self.embed_dim = embed_dim
		self.decoder_embed_dim = decoder_embed_dim
		self.decoder_num_heads = decoder_num_heads

		self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
		self.pos_embed = nn.Parameter(
			torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
		)  # fixed sin-cos embedding

		self.blocks = nn.ModuleList(
			[
				Block(
					embed_dim,
					num_heads,
					mlp_ratio,
					qkv_bias=True,
					norm_layer=norm_layer,
				)
				for i in range(depth)
			]
		)
		self.norm_layer = norm_layer
		self.norm = norm_layer(embed_dim)
		# --------------------------------------------------------------------------

		# --------------------------------------------------------------------------
		# MAE decoder specifics
		self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

		self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

		self.decoder_pos_embed = nn.Parameter(
			torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
		)  # fixed sin-cos embedding

		self.decoder_blocks = nn.ModuleList(
			[
				Block(
					decoder_embed_dim,
					decoder_num_heads,
					mlp_ratio,
					qkv_bias=True,
					norm_layer=norm_layer,
				)
				for i in range(decoder_depth)
			]
		)

		self.decoder_norm = norm_layer(decoder_embed_dim)
		self.decoder_pred = nn.Linear(
			decoder_embed_dim, patch_size**2 * in_chans, bias=True
		)  # decoder to patch
		# --------------------------------------------------------------------------

		self.norm_pix_loss = norm_pix_loss

		self.initialize_weights()

	def initialize_weights(self):
		# initialization
		# initialize (and freeze) pos_embed by sin-cos embedding
		pos_embed = get_2d_sincos_pos_embed(
			self.pos_embed.shape[-1],
			int(self.patch_embed.num_patches**0.5),
			cls_token=True,
		)
		self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

		decoder_pos_embed = get_2d_sincos_pos_embed(
			self.decoder_pos_embed.shape[-1],
			int(self.patch_embed.num_patches**0.5),
			cls_token=True,
		)
		self.decoder_pos_embed.data.copy_(
			torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
		)

		# initialize patch_embed like nn.Linear (instead of nn.Conv2d)
		w = self.patch_embed.proj.weight.data
		torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

		# timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
		torch.nn.init.normal_(self.cls_token, std=0.02)
		torch.nn.init.normal_(self.mask_token, std=0.02)

		# initialize nn.Linear and nn.LayerNorm
		self.apply(self._init_weights)

	def _init_weights(self, m):
		if isinstance(m, nn.Linear):
			# we use xavier_uniform following official JAX ViT:
			torch.nn.init.xavier_uniform_(m.weight)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)

	def patchify(self, imgs):
		"""
		imgs: (N, 3, H, W)
		x: (N, L, patch_size**2 *3)
		"""
		p = self.patch_embed.patch_size[0]
		assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

		h = w = imgs.shape[2] // p
		x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
		x = torch.einsum("nchpwq->nhwpqc", x)
		x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
		return x

	def unpatchify(self, x):
		"""
		x: (N, L, patch_size**2 *3)
		imgs: (N, 3, H, W)
		"""
		p = self.patch_embed.patch_size[0]
		h = w = int(x.shape[1] ** 0.5)
		assert h * w == x.shape[1]

		x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
		x = torch.einsum("nhwpqc->nchpwq", x)
		imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
		return imgs

	def random_masking(self, x, mask_ratio):
		"""
		Perform per-sample random masking by per-sample shuffling.
		Per-sample shuffling is done by argsort random noise.
		x: [N, L, D], sequence
		"""
		N, L, D = x.shape  # batch, length, dim
		len_keep = int(L * (1 - mask_ratio))  # the remained token number

		noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

		# sort noise for each sample
		# ascend: small is keep, large is remove

		ids_shuffle = torch.argsort(noise, dim=1)
		ids_restore = torch.argsort(ids_shuffle, dim=1)

		# keep the first subset
		ids_keep = ids_shuffle[:, :len_keep]
		x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

		# generate the binary mask: 0 is keep, 1 is remove
		mask = torch.ones([N, L], device=x.device)
		mask[:, :len_keep] = 0
		# unshuffle to get the binary mask for computing loss
		mask = torch.gather(mask, dim=1, index=ids_restore)

		return x_masked, mask, ids_restore

	def forward_encoder(self, x, mask_ratio):
		# embed patches
		x = self.patch_embed(x)
		# add pos embed w/o cls token
		x = x + self.pos_embed[:, 1:, :]  # cls_token doesn't have positional encod.

		# masking: length -> length * mask_ratio
		if mask_ratio != 0:
			x, mask, ids_restore = self.random_masking(x, mask_ratio)

		# append cls token
		cls_token = self.cls_token + self.pos_embed[:, :1, :]
		cls_tokens = cls_token.expand(x.shape[0], -1, -1)
		x = torch.cat((cls_tokens, x), dim=1)

		# apply Transformer blocks
		for blk in self.blocks:
			x = blk(x)
		x = self.norm(x)

		if mask_ratio == 0:
			return x, None, None
		return x, mask, ids_restore

	def forward_decoder(self, x, ids_restore):
		# embed tokens
		x = self.decoder_embed(x)

		# append mask tokens to sequence

		# x: [N, L+1, D]
		mask_tokens = self.mask_token.repeat(
			x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
		)
		x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token

		# inversely random sequence
		x_ = torch.gather(
			x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
		)  # unshuffle
		x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

		# add pos embed
		x = x + self.decoder_pos_embed

		# apply Transformer blocks
		for blk in self.decoder_blocks:
			x = blk(x)
		x = self.decoder_norm(x)

		# predictor projection
		x = self.decoder_pred(x)

		# remove cls token
		x = x[:, 1:, :]

		return x

	def forward_loss(self, imgs, pred, mask):
		"""
		imgs: [N, 3, H, W]
		pred: [N, L, p*p*3]
		mask: [N, L], 0 is keep, 1 is remove,
		"""
		target = self.patchify(imgs)
		if self.norm_pix_loss:
			mean = target.mean(dim=-1, keepdim=True)
			var = target.var(dim=-1, keepdim=True)
			target = (target - mean) / (var + 1.0e-6) ** 0.5

		loss = (pred - target) ** 2
		loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

		loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
		return loss

	def forward(self, imgs, mask_ratio=0.75):
		latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
		pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
		loss = self.forward_loss(imgs, pred, mask)
		return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
	model = MaskedAutoencoderViT(
		patch_size=16,
		embed_dim=768,
		depth=12,
		num_heads=12,
		decoder_embed_dim=512,
		decoder_depth=8,
		decoder_num_heads=16,
		mlp_ratio=4,
		norm_layer=partial(nn.LayerNorm, eps=1e-6),
		**kwargs,
	)
	return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
	model = MaskedAutoencoderViT(
		patch_size=16,
		embed_dim=1024,
		depth=24,
		num_heads=16,
		decoder_embed_dim=512,
		decoder_depth=8,
		decoder_num_heads=16,
		mlp_ratio=4,
		norm_layer=partial(nn.LayerNorm, eps=1e-6),
		**kwargs,
	)
	return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
	model = MaskedAutoencoderViT(
		patch_size=14,
		embed_dim=1280,
		depth=32,
		num_heads=16,
		decoder_embed_dim=512,
		decoder_depth=8,
		decoder_num_heads=16,
		mlp_ratio=4,
		norm_layer=partial(nn.LayerNorm, eps=1e-6),
		**kwargs,
	)
	return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


# Copyright contributors to the neural-embedding-compression project

class MaskedAutoencoderViTCompressed(MaskedAutoencoderViT, CompressionModel):
	"""This class adds a compression bottleneck to the standard MAE VIT
	"""
	def __init__(
		self,
		ld,
		img_size=224,
		patch_size=16,
		in_chans=3,
		embed_dim=1024,
		depth=24,
		num_heads=16,
		decoder_embed_dim=512,
		decoder_depth=8,
		decoder_num_heads=16,
		mlp_ratio=4,
		norm_layer=nn.LayerNorm,
		norm_pix_loss=False,
	):
		super().__init__(
			img_size,
			patch_size,
			in_chans,
			embed_dim,
			depth,
			num_heads,
			decoder_embed_dim,
			decoder_depth,
			decoder_num_heads,
			mlp_ratio,
			norm_layer,
			norm_pix_loss,
		)
		self.ld = ld
		self.embed_dim = embed_dim
		# self.n_tokens = ( (img_size**2) / (patch_size**2) ) * (1 - 0.75) + 1 # n_tokens produced + cls token which is then thrown away
		self.bottleneck = EntropyBottleneck(embed_dim)
		self.loss = RateDistortionLoss(lmbda=ld)
		pos_embed = self.pos_embed.clone()
		decoder_pos_embed = self.decoder_pos_embed.clone()
		del self.pos_embed
		del self.decoder_pos_embed
		self.register_buffer("pos_embed", pos_embed)
		self.register_buffer("decoder_pos_embed", decoder_pos_embed)

	def forward(self, imgs, mask_ratio=0.75):
		latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
		latent = rearrange(latent, "b s (e n) -> b e s n", n=1)
		quant_dequant_latent, y_likelihoods = self.bottleneck(latent)
		quant_dequant_latent = rearrange(
			quant_dequant_latent, "b e s n -> b s (e n)", n=1
		)
		pred = self.forward_decoder(quant_dequant_latent, ids_restore)  # [N, L, p*p*3]
		recon_loss = self.forward_loss(imgs, pred, mask)
		loss = self.loss(recon_loss, [y_likelihoods])
		aux_loss = self.aux_loss()
		return {
			"loss": loss,
			"aux_loss": aux_loss,
			"pred": pred,
			"mask": mask,
			"likelihoods": {"y": y_likelihoods},
		}

	def compress(self, x):
		latent, mask, ids_restore = self.forward_encoder(x, 0)
		latent = rearrange(latent, "b s (e n) -> b e s n", n=1)
		y_strings = self.bottleneck.compress(latent)
		return {"strings": [y_strings], "shape": (latent.shape[-2], latent.shape[-1])}

	def decompress(self, strings, shape):
		assert isinstance(strings, list) and len(strings) == 1
		y_hat = self.bottleneck.decompress(strings[0], shape)
		recovered_reshaped = rearrange(
				y_hat, "b e s n -> b s (e n)", e=self.embed_dim, n=1
			)
		return recovered_reshaped

	def quantized_embedding(self, x):
		latent, mask, ids_restore = self.forward_encoder(x, 0)
		latent = rearrange(latent, "b s (e n) -> b e s n", n=1)
		quant_dequant_latent, y_likelihoods = self.bottleneck(latent, training=False)
		quant_dequant_latent = rearrange(
			quant_dequant_latent, "b e s n -> b s (e n)", n=1
		)
		return quant_dequant_latent


def mae_vit_compress(ld, **kwargs):
	model = MaskedAutoencoderViTCompressed(
		ld,
		patch_size=16,
		embed_dim=768,
		depth=12,
		num_heads=12,
		decoder_embed_dim=512,
		decoder_depth=8,
		decoder_num_heads=16,
		mlp_ratio=4,
		norm_layer=partial(nn.LayerNorm, eps=1e-6),
		**kwargs,
	)
	return model


class MaskedAutoencoderViTCompressedAdapter(MaskedAutoencoderViT, CompressionModel):
	"""This class adds a compression bottleneck to the standard MAE VIT, 
	tuning only a selected number of blocks and freezing the rest
	"""
	def __init__(
		self,
		ld,
		img_size=224,
		patch_size=16,
		in_chans=3,
		embed_dim=1024,
		depth=24,
		num_heads=16,
		decoder_embed_dim=512,
		decoder_depth=8,
		decoder_num_heads=16,
		mlp_ratio=4,
		norm_layer=nn.LayerNorm,
		norm_pix_loss=False,
		trainable_encoder_decoder_blocks=1
	):
		super().__init__(
			img_size,
			patch_size,
			in_chans,
			embed_dim,
			depth,
			num_heads,
			decoder_embed_dim,
			decoder_depth,
			decoder_num_heads,
			mlp_ratio,
			norm_layer,
			norm_pix_loss,
		)
		self.ld = ld
		self.embed_dim = embed_dim
		self.bottleneck = EntropyBottleneck(embed_dim)
		self.loss = RateDistortionLoss(lmbda=ld)
		self.trainable_blocks = trainable_encoder_decoder_blocks
		pos_embed = self.pos_embed.clone()
		decoder_pos_embed = self.decoder_pos_embed.clone()
		del self.pos_embed
		del self.decoder_pos_embed
		self.register_buffer("pos_embed", pos_embed)
		self.register_buffer("decoder_pos_embed", decoder_pos_embed)
		for block in self.blocks[:-self.trainable_blocks]:
			for parameter in block.parameters():
				parameter.requires_grad_(False)

		for block in self.decoder_blocks[self.trainable_blocks:]:
			for parameter in block.parameters():
				parameter.requires_grad_(False)
	
	def prepare_for_finetune(self):
		self.bottleneck.requires_grad_(False)
		self.update(force=True)
		self.bottleneck.eval()
	
	def freeze_encoder(self):
		for param in self.blocks.parameters():
			param.requires_grad_(False)
		for layer in self.blocks:
			layer.eval()
		if hasattr(self, "decoder_blocks"):
			for layer in self.decoder_blocks:
				layer.requires_grad_(False)
				layer.eval()
		self.patch_embed.requires_grad_(False)
		self.cls_token.requires_grad_(False)

	def forward(self, imgs, mask_ratio=0.75):
		latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
		# compress ai expects "image like" embedding of shape (B, C, H, W). It then models each C dimension
		# we simulate that by making the embedding dimension our C and appending an additional dummy dimension
		latent = rearrange(latent, "b s (e n) -> b e s n", n=1)
		quant_dequant_latent, y_likelihoods = self.bottleneck(latent)
		quant_dequant_latent = rearrange(
			quant_dequant_latent, "b e s n -> b s (e n)", n=1
		)
		pred = self.forward_decoder(quant_dequant_latent, ids_restore)  # [N, L, p*p*3]
		recon_loss = self.forward_loss(imgs, pred, mask)
		loss = self.loss(recon_loss, [y_likelihoods])
		aux_loss = self.aux_loss()
		return {
			"loss": loss,
			"aux_loss": aux_loss,
			"pred": pred,
			"mask": mask,
			"likelihoods": {"y": y_likelihoods},
		}

	def compress(self, x):
		latent, mask, ids_restore = self.forward_encoder(x, 0)
		latent = rearrange(latent, "b s (e n) -> b e s n", n=1)
		y_strings = self.bottleneck.compress(latent)
		return {"strings": [y_strings], "shape": (latent.shape[-2], latent.shape[-1])}

	def decompress(self, strings, shape):
		assert isinstance(strings, list) and len(strings) == 1
		y_hat = self.bottleneck.decompress(strings[0], shape)
		recovered_reshaped = rearrange(
				y_hat, "b e s n -> b s (e n)", e=self.embed_dim, n=1
			)
		return recovered_reshaped

	def quantized_embedding(self, x):
		latent, mask, ids_restore = self.forward_encoder(x, 0)
		latent = rearrange(latent, "b s (e n) -> b e s n", n=1)
		quant_dequant_latent, y_likelihoods = self.bottleneck(latent, training=False)
		quant_dequant_latent = rearrange(
			quant_dequant_latent, "b e s n -> b s (e n)", n=1
		)
		return quant_dequant_latent


def mae_vit_compress_adapter(ld, **kwargs):
	model = MaskedAutoencoderViTCompressedAdapter(
		ld,
		patch_size=16,
		embed_dim=768,
		depth=12,
		num_heads=12,
		decoder_embed_dim=512,
		decoder_depth=8,
		decoder_num_heads=16,
		mlp_ratio=4,
		norm_layer=partial(nn.LayerNorm, eps=1e-6),
		**kwargs,
	)
	return model

class MaskedAutoencoderViTCompressedHyperpriorAdapter(MaskedAutoencoderViT, CompressionModel):
	def __init__(
		self,
		ld,
		img_size=224,
		patch_size=16,
		in_chans=3,
		embed_dim=1024,
		depth=24,
		num_heads=16,
		decoder_embed_dim=512,
		decoder_depth=8,
		decoder_num_heads=16,
		mlp_ratio=4,
		norm_layer=nn.LayerNorm,
		norm_pix_loss=False,
		trainable_encoder_decoder_blocks=1
	):
		super().__init__(
			img_size,
			patch_size,
			in_chans,
			embed_dim,
			depth,
			num_heads,
			decoder_embed_dim,
			decoder_depth,
			decoder_num_heads,
			mlp_ratio,
			norm_layer,
			norm_pix_loss,
		)
		self.ld = ld
		self.embed_dim = embed_dim
		self.entropy_bottleneck = EntropyBottleneck(96)
		self.gaussian_conditional = GaussianConditional(None)
		self.trainable_blocks = trainable_encoder_decoder_blocks

		self.loss = RateDistortionLoss(lmbda=ld)
		pos_embed = self.pos_embed.clone()
		decoder_pos_embed = self.decoder_pos_embed.clone()
		del self.pos_embed
		del self.decoder_pos_embed
		self.register_buffer("pos_embed", pos_embed)
		self.register_buffer("decoder_pos_embed", decoder_pos_embed)
		for block in self.blocks[:-self.trainable_blocks]:
			for parameter in block.parameters():
				parameter.requires_grad_(False)

		for block in self.decoder_blocks[self.trainable_blocks:]:
			for parameter in block.parameters():
				parameter.requires_grad_(False)

		h_a = [
			nn.Linear(embed_dim, 96),
			nn.ReLU()] \
		+ [
			Block(
				96,
				4,
				mlp_ratio,
				qkv_bias=True,
				norm_layer=norm_layer,
			)
			for _ in range(2)
		]

		self.h_a = nn.Sequential(*h_a)

		h_s = [
			Block(
				96,
				4,
				mlp_ratio,
				qkv_bias=True,
				norm_layer=norm_layer,
			)
			for _ in range(2)
		] \
		+ [
			nn.Linear(96, embed_dim),
			nn.ReLU()
		]

		self.h_s = nn.Sequential(*h_s)
	
	def prepare_for_finetune(self):
		self.entropy_bottleneck.requires_grad_(False)
		self.gaussian_conditional.requires_grad_(False)
		self.update(force=True)
		self.entropy_bottleneck.eval()
		self.gaussian_conditional.eval()
	
	def freeze_encoder(self):
		for param in self.backbone.parameters():
			param.requires_grad_(False)
		for layer in self.backbone.blocks:
			layer.eval()
		if hasattr(self.backbone, "decoder_blocks"):
			for layer in self.backbone.decoder_blocks:
				layer.requires_grad_(False)
				layer.eval()
		self.backbone.patch_embed.requires_grad_(False)
		self.backbone.cls_token.requires_grad_(False)

	def forward(self, imgs, mask_ratio=0.75):
		latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
		
		z = self.h_a(torch.abs(latent))
		# compress ai expects "image like" embedding of shape (B, C, H, W). It then models each C dimension
		# we simulate that by making the embedding dimension our C and appending an additional dummy dimension
		z = rearrange(z, "b s (e n) -> b e s n", n=1) 
		z_hat, z_likelihoods = self.entropy_bottleneck(z)
		
		z_hat = rearrange(z_hat, "b e s n -> b s (e n)", n=1) 
		scales_hat = self.h_s(z_hat)

		scales_hat = rearrange(scales_hat, "b s (e n) -> b e s n", n=1)
		y = rearrange(latent, "b s (e n) -> b e s n", n=1) 
		quant_dequant_latent, y_likelihoods = self.gaussian_conditional(y, scales_hat)
		quant_dequant_latent = rearrange(
			quant_dequant_latent, "b e s n -> b s (e n)", n=1
		)
		pred = self.forward_decoder(quant_dequant_latent, ids_restore)  # [N, L, p*p*3]
		recon_loss = self.forward_loss(imgs, pred, mask)
		loss = self.loss(recon_loss, [y_likelihoods, z_likelihoods])
		aux_loss = self.aux_loss()
		return {
			"loss": loss,
			"aux_loss": aux_loss,
			"pred": pred,
			"mask": mask,
			"likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
		}

	# test
	def compress(self, x):
		latent, mask, ids_restore = self.forward_encoder(x, 0)
		z = self.h_a(torch.abs(latent))

		z = rearrange(z, "b s (e n) -> b e s n", n=1) 
		z_strings = self.entropy_bottleneck.compress(z)
		z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

		z_hat = rearrange(z_hat, "b e s n -> b s (e n)", n=1) 
		scales_hat = self.h_s(z_hat)
		scales_hat = rearrange(scales_hat, "b s (e n) -> b e s n", n=1)

		indexes = self.gaussian_conditional.build_indexes(scales_hat)

		y = rearrange(latent, "b s (e n) -> b e s n", n=1) 
		y_strings = self.gaussian_conditional.compress(y, indexes)
		return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
	
	def decompress(self, strings, shape):
		assert isinstance(strings, list) and len(strings) == 2
		z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
		z_hat = rearrange(
				z_hat, "b e s n -> b s (e n)", e=96, n=1
			)
		
		scales_hat = self.h_s(z_hat)
		scales_hat = rearrange(scales_hat, "b s (e n) -> b e s n", n=1)

		indexes = self.gaussian_conditional.build_indexes(scales_hat)
		decompressed = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
		recovered = rearrange(decompressed, "b e s n -> b s (e n)", n=1) 
		return recovered

	def quantized_embedding(self, x):
		latent, mask, ids_restore = self.forward_encoder(x, 0)
		
		z = self.h_a(torch.abs(latent))
		# compress ai expects "image like" embedding of shape (B, C, H, W). It then models each C dimension
		# we simulate that by making the embedding dimension our C and appending an additional dummy dimension
		z = rearrange(z, "b s (e n) -> b e s n", n=1) 
		z_hat, z_likelihoods = self.entropy_bottleneck(z, training=False)
		
		z_hat = rearrange(z_hat, "b e s n -> b s (e n)", n=1) 
		scales_hat = self.h_s(z_hat)

		scales_hat = rearrange(scales_hat, "b s (e n) -> b e s n", n=1)
		y = rearrange(latent, "b s (e n) -> b e s n", n=1) 
		quant_dequant_latent, y_likelihoods = self.gaussian_conditional(y, scales_hat, training=False)
		quant_dequant_latent = rearrange(
			quant_dequant_latent, "b e s n -> b s (e n)", n=1
		)
		return quant_dequant_latent


def mae_vit_compress_hyperprior_adapter(ld, **kwargs):
	model = MaskedAutoencoderViTCompressedHyperpriorAdapter(
		ld,
		patch_size=16,
		embed_dim=768,
		depth=12,
		num_heads=12,
		decoder_embed_dim=512,
		decoder_depth=8,
		decoder_num_heads=16,
		mlp_ratio=4,
		norm_layer=partial(nn.LayerNorm, eps=1e-6),
		**kwargs,
	)
	return model

class MaskedAutoencoderViTClassifier(MaskedAutoencoderViT):
	"""
	Classifier with VIT backbone using AttentionPoolLatent for pooling
	"""

	def __init__(
		self,
		img_size=224,
		patch_size=16,
		in_chans=3,
		embed_dim=1024,
		depth=24,
		num_heads=16,
		decoder_embed_dim=512,
		decoder_depth=8,
		decoder_num_heads=16,
		mlp_ratio=4,
		norm_layer=nn.LayerNorm,
		norm_pix_loss=False,
		num_classes=80,
		quant=None,
		with_decoder=False,
	):
		super().__init__(
			img_size,
			patch_size,
			in_chans,
			embed_dim,
			depth,
			num_heads,
			decoder_embed_dim,
			decoder_depth,
			decoder_num_heads,
			mlp_ratio,
			norm_layer,
			norm_pix_loss,
		)
		embedding_size = int(((img_size // patch_size) ** 2))
		if with_decoder:
			self.pool = AttentionPoolLatent(
				decoder_embed_dim,
				num_heads=decoder_num_heads,
				mlp_ratio=mlp_ratio,
				norm_layer=norm_layer,
			)
			self.head = (
				nn.Linear(decoder_embed_dim, num_classes)
				if num_classes > 0
				else nn.Identity()
			)
		else:
			self.pool = AttentionPoolLatent(
				embed_dim,
				num_heads=num_heads,
				mlp_ratio=mlp_ratio,
				norm_layer=norm_layer,
			)
			self.head = (
				nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
			)
			del self.decoder_blocks
			del self.decoder_embed
		self.quant = quant
		self.with_decoder = with_decoder
		del self.decoder_pred
		del self.decoder_norm

	
	def freeze_encoder(self):
		for param in self.parameters():
			param.requires_grad_(False)
		for layer in self.blocks:
			layer.eval()
		if self.with_decoder:
			for layer in self.decoder_blocks:
				layer.requires_grad_(False)
				layer.eval()
		self.patch_embed.requires_grad_(False)
		self.cls_token.requires_grad_(False)
		
	def unfreeze_head(self):
		self.pool.requires_grad_(True)
		self.head.requires_grad_(True)
		if self.with_decoder:
			self.decoder_blocks[0].requires_grad_(True)
			self.decoder_blocks[0].train()

	@torch.jit.ignore
	def no_weight_decay(self):
		return {"pos_embed", "cls_token"}

	def forward(self, imgs):
		latent, mask, ids_restore = self.forward_encoder(imgs, 0)
		# drop cls token
		x = latent[:, 1:, :]
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
				assert (scaled <= scale).all() and (scaled >= 0).all()
				latent = quantized / scale
				latent = latent * (latent_max - latent_min)
				x = latent + latent_min
		if self.with_decoder:
			x = self.decoder_embed(x)
			x = x + self.decoder_pos_embed[:, 1:, :]
			x = self.decoder_blocks[0](x)
		x = self.pool(x)
		return self.head(x)


class MaskedAutoencoderViTClassifierCompression(nn.Module):
	"""This class takes a backbone model and adds the compression head (and optionally decoder) to it.
	"""
	def __init__(
		self,
		backbone: Union[MaskedAutoencoderViTCompressedHyperpriorAdapter, MaskedAutoencoderViTCompressedAdapter],
		num_classes=80,
		with_decoder=False,
	):
		super().__init__()
		self.backbone = backbone
		if with_decoder:
			self.pool = AttentionPoolLatent(
				backbone.decoder_embed_dim,
				num_heads=backbone.decoder_num_heads,
				mlp_ratio=backbone.mlp_ratio,
				norm_layer=backbone.norm_layer,
			)
			self.head = (
				nn.Linear(backbone.decoder_embed_dim, num_classes)
				if num_classes > 0
				else nn.Identity()
			)
		else:
			self.pool = AttentionPoolLatent(
				backbone.embed_dim,
				num_heads=backbone.num_heads,
				mlp_ratio=backbone.mlp_ratio,
				norm_layer=backbone.norm_layer,
			)
			self.head = (
				nn.Linear(backbone.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
			)

			del self.backbone.decoder_blocks
			del self.backbone.decoder_embed
		self.with_decoder = with_decoder
		del self.backbone.decoder_pred
		del self.backbone.decoder_norm
	
	def freeze_encoder(self):
		self.backbone.freeze_encoder()

	def prepare_for_finetune(self):
		self.backbone.prepare_for_finetune()

	@property
	def patch_embed(self):
		return self.backbone.patch_embed

	@property
	def pos_embed(self):
		return self.backbone.pos_embed

	@property
	def decoder_pos_embed(self):
		return self.backbone.decoder_pos_embed
	
	@property
	def blocks(self):
		return self.backbone.blocks
	
	def unfreeze_head(self):
		self.pool.requires_grad_(True)
		self.head.requires_grad_(True)
		if self.with_decoder:
			self.backbone.decoder_blocks[0].requires_grad_(True)

	@torch.jit.ignore
	def no_weight_decay(self):
		return {"pos_embed", "cls_token"}

	def forward(self, x):
		quant_dequant_latent = self.backbone.quantized_embedding(x)
		quant_dequant_latent = quant_dequant_latent[:, 1:, :] # no cls token
		if self.with_decoder:
			x = self.backbone.decoder_embed(quant_dequant_latent)
			x = x + self.backbone.decoder_pos_embed[:, 1:, :]
			x = self.backbone.decoder_blocks[0](x)
			x = self.pool(x)
		else:
			x = self.pool(quant_dequant_latent)
		return self.head(x)


class RateDistortionLoss(nn.Module):
	"""Custom rate distortion loss with a Lagrangian parameter."""

	def __init__(self, lmbda=1e-2):
		super().__init__()
		self.lmbda = lmbda

	def forward(self, recon_loss, likelihoods):
		out = {}

		out["bits_loss"] = sum(
			(torch.log(likelihoods).sum() / -math.log(2)) for likelihoods in likelihoods
		)
		out["recon_loss"] = recon_loss
		out["loss"] = self.lmbda * out["recon_loss"] + out["bits_loss"]

		return out
