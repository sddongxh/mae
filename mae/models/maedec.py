# Copyright (c) Meta Platforms, Inc. and affiliates.
# All Rights Reserved.
#
# This source code is licensed under the LICENSE file in the root directory
# of this source tree.
#
# References:
# - MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# - timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# - DeiT: https://github.com/facebookresearch/deit

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from .pos_embed import get_2d_sincos_pos_embed


class MAEDecoder(nn.Module):
    """
    Masked Autoencoder Decoder using Vision Transformer architecture.

    Args:
        num_patches (int): Number of patches in the input.
        patch_size (int): Size of each patch (default: 16).
        in_chans (int): Number of input channels (default: 3).
        embed_dim (int): Dimension of the encoder embedding (default: 1024).
        decoder_embed_dim (int): Dimension of the decoder embedding (default: 512).
        decoder_depth (int): Number of transformer blocks in the decoder (default: 8).
        decoder_num_heads (int): Number of attention heads in each transformer block (default: 16).
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension (default: 4.0).
        norm_layer (nn.Module): Normalization layer to use (default: nn.LayerNorm).
        learnable_pos_embed (bool): Whether the positional embeddings are learnable or fixed (default: False).
    """

    def __init__(
        self,
        num_patches: int,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        learnable_pos_embed: bool = False,
    ):
        super().__init__()

        self.num_patches = num_patches
        self.learnable_pos_embed = learnable_pos_embed

        # MAE Decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # Positional embeddings (learnable or fixed)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=learnable_pos_embed
        )

        # Transformer blocks
        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
            ) for _ in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights for the decoder."""
        if not self.learnable_pos_embed:
            # Initialize (and freeze) positional embedding using sin-cos embedding
            decoder_pos_embed = get_2d_sincos_pos_embed(
                embed_dim=self.decoder_pos_embed.shape[-1],
                grid_size=int(self.num_patches ** 0.5),
                cls_token=True,
            )
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        else:
            # Learnable positional embeddings are initialized as zeros
            nn.init.normal_(self.decoder_pos_embed, std=0.02)

        # Initialize the mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # Initialize linear and normalization layers
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Initialize weights for layers."""
        if isinstance(m, nn.Linear):
            # Xavier uniform initialization for linear layers
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Latent representation from the encoder of shape (N, L, D).
            ids_restore (torch.Tensor): Indices to restore the original order of patches.

        Returns:
            torch.Tensor: Reconstructed patches of shape (N, L, patch_size**2 * in_chans).
        """
        # Embed tokens
        x = self.decoder_embed(x)

        # Append mask tokens to the sequence for masked positions
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_unmasked = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # Skip class token
        x_unmasked = torch.gather(
            x_unmasked, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # Unshuffle the tokens to their original order
        x = torch.cat([x[:, :1, :], x_unmasked], dim=1)  # Append class token

        # Add positional embeddings
        x = x + self.decoder_pos_embed

        # Apply transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Apply prediction layer
        x = self.decoder_pred(x)

        # Remove class token from the output
        x = x[:, 1:, :]

        return x

# Example usage
if __name__ == "__main__":
    # Assume num_patches comes from the encoder, patch_size of 16, and we have 3 channels (RGB images)
    num_patches = 196  # For example, if the image is divided into 14x14 patches
    embed_dim = 1024   # Encoder embedding dimension
    batch_size = 4     # Number of images in a batch

    # Initialize the MAEDecoder with learnable positional embeddings
    decoder = MAEDecoder(num_patches=num_patches, embed_dim=embed_dim, learnable_pos_embed=True)

    # Create dummy inputs
    latent_representation = torch.randn(batch_size, num_patches + 1, embed_dim)  # Latent representation
    ids_restore = torch.arange(num_patches).unsqueeze(0).repeat(batch_size, 1)   # Identity mapping for demonstration

    # Perform forward pass
    reconstructed_patches = decoder(latent_representation, ids_restore)

    print("Latent representation shape:", latent_representation.shape)
    print("Reconstructed patches shape:", reconstructed_patches.shape)
