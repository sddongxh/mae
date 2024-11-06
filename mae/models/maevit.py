# Copyright (c) Meta Platforms, Inc. and affiliates.
# All Rights Reserved.
# References:
# - MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# - timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block

from .pos_embed import get_2d_sincos_pos_embed
from .masking import random_masking

class MAEViT(nn.Module):
    """Masked Autoencoder with Vision Transformer backbone"""

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        learnable_pos_embed: bool = False,
    ):
        """
        Args:
            img_size (int): Size of the input image (assumes square shape).
            in_chans (int): Number of input channels.
            patch_size (int): Size of each patch.
            embed_dim (int): Dimension of the embedding.
            depth (int): Number of transformer blocks in the encoder.
            num_heads (int): Number of attention heads in each transformer block.
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
            norm_layer (nn.Module): Normalization layer to use.
            learnable_pos_embed (bool): Whether the positional embeddings are learnable or fixed.
        """
        super().__init__()

        self.learnable_pos_embed = learnable_pos_embed

        # MAE Encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token and position embedding (fixed sin-cos embedding or learnable)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=learnable_pos_embed
        )

        # Transformer blocks for the encoder
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
            ) for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize the model's weights."""
        if not self.learnable_pos_embed:
            # Initialize (and freeze) positional embedding using 2D sin-cos embedding
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim=self.pos_embed.shape[-1],
                grid_size=int(self.patch_embed.num_patches ** 0.5),
                cls_token=True,
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            # Learnable positional embeddings are initialized as zeros
            nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize the patch embedding like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize the class token
        torch.nn.init.normal_(self.cls_token, std=0.02)

        # Initialize all linear and normalization layers
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Initialize the weights for linear and normalization layers."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, mask_ratio: float = 0) -> tuple:
        """
        Forward pass through the encoder with optional random masking.

        Args:
            x (torch.Tensor): Input image tensor of shape (N, C, H, W).
            mask_ratio (float): Proportion of patches to mask.

        Returns:
            tuple: (latent representation, mask, restore indices)
                If mask_ratio is 0, `mask` is all zeros and `ids_restore` is the identity mapping.
        """
        # Embed patches
        x = self.patch_embed(x)

        # Add position embedding without class token
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio > 0:
            # Perform random masking if mask_ratio > 0
            x, mask, ids_restore = random_masking(x, mask_ratio)
        else:
            # No masking: create default mask and restore indices
            N, L, _ = x.shape
            mask = torch.zeros(N, L, device=x.device)  # No patches are masked
            ids_restore = torch.arange(L, device=x.device).unsqueeze(0).repeat(N, 1)

        # Add class token to the sequence
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore


# Example usage
if __name__ == "__main__":
    # Example instantiation of MAEViT with learnable positional embeddings
    model = MAEViT()
    img = torch.randn(2, 3, 224, 224)  # Batch of 2 images of shape (3, 224, 224)
    encoded, mask, ids_restore = model(img, mask_ratio=0.75)
    print("Encoded shape:", encoded.shape)
