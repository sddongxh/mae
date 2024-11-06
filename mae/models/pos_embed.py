# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    """
    Generate a 2D sine-cosine position embedding.

    Args:
        embed_dim (int): Dimension of the embedding.
        grid_size (int): Height and width of the grid.
        cls_token (bool): Whether to include a class token at the beginning.

    Returns:
        np.ndarray: Position embedding of shape [grid_size*grid_size, embed_dim]
                    or [1+grid_size*grid_size, embed_dim] if cls_token is True.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    """
    Generate a 2D sine-cosine position embedding from the given grid.

    Args:
        embed_dim (int): Dimension of the embedding.
        grid (np.ndarray): Input grid of shape (2, 1, grid_size, grid_size).

    Returns:
        np.ndarray: Position embedding of shape (grid_size*grid_size, embed_dim).
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even."

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    return np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)

def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    """
    Generate a 1D sine-cosine position embedding for the given positions.

    Args:
        embed_dim (int): Output dimension for each position.
        pos (np.ndarray): Positions to be encoded of shape (M,).

    Returns:
        np.ndarray: Position embedding of shape (M, D).
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even."

    omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

def interpolate_pos_embed(model, checkpoint_model):
    """
    Interpolate position embeddings for high-resolution inputs.

    Args:
        model (nn.Module): Model containing the current position embedding.
        checkpoint_model (dict): Checkpoint model containing the original position embedding.

    Returns:
        None: Modifies `checkpoint_model` in place to update 'pos_embed'.
    """
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        # Height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # Height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)

        # Class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            
            # Only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False
            )
            
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
