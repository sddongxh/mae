import torch

# Reference: https://github.com/facebookresearch/mae/blob/main/models_mae.py
def patchify(imgs: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Splits images into patches.

    Args:
        imgs (torch.Tensor): Tensor of shape (N, C, H, W) where N is the batch size,
                             C is the number of channels, and H, W are the height and width.
        patch_size (int): The size of each patch (patches are square).

    Returns:
        torch.Tensor: Patches reshaped into (N, L, patch_size**2 * C) where L is the total number of patches.
    """
    N, C, H, W = imgs.shape
    assert H == W, f"Image height and width must be equal, but got H={H}, W={W}."
    assert (
        H % patch_size == 0
    ), f"Image dimensions must be divisible by patch_size, but got H={H}, patch_size={patch_size}."

    num_patches = H // patch_size
    imgs_reshaped = imgs.reshape(N, C, num_patches, patch_size, num_patches, patch_size)
    patches = torch.einsum("nchpwq->nhwpqc", imgs_reshaped)
    patches = patches.reshape(N, num_patches * num_patches, patch_size * patch_size * C)

    return patches


def unpatchify(patches: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
    """
    Reconstructs images from their patches.

    Args:
        patches (torch.Tensor): Patches of shape (N, L, patch_size**2 * C) where L is the total number of patches.
        patch_size (int): The size of each patch (patches are square).

    Returns:
        torch.Tensor: Reconstructed images of shape (N, C, H, W).
    """
    N, L, patch_dim = patches.shape
    num_patches = int(L**0.5)
    assert (
        num_patches**2 == L
    ), f"The total number of patches (L) must be a perfect square, but got L={L}."

    C = patch_dim // (patch_size * patch_size)
    assert (
        patch_dim == patch_size * patch_size * C
    ), f"Patch dimensions do not match the expected size, got {patch_dim}."

    patches_reshaped = patches.reshape(
        N, num_patches, num_patches, patch_size, patch_size, C
    )
    imgs = torch.einsum("nhwpqc->nchpwq", patches_reshaped)
    imgs = imgs.reshape(N, C, num_patches * patch_size, num_patches * patch_size)

    return imgs


# Example usage:
if __name__ == "__main__":
    imgs = torch.randn(10, 3, 128, 128)  # Example batch of images
    patch_size = 16

    patches = patchify(imgs, patch_size)
    reconstructed_imgs = unpatchify(patches, patch_size)

    print("Original shape:", imgs.shape)
    print("Patches shape:", patches.shape)
    print("Reconstructed shape:", reconstructed_imgs.shape)
