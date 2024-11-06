import torch

# Reference: https://github.com/facebookresearch/mae/blob/main/models_mae.py
def random_masking(x: torch.Tensor, mask_ratio: float) -> tuple:
    """
    Perform per-sample random masking by shuffling each sample independently.
    The masking is performed by using random noise to determine which elements to keep.

    Args:
        x (torch.Tensor): Input tensor of shape (N, L, D), where
                          N is the batch size,
                          L is the sequence length,
                          D is the feature dimension.
        mask_ratio (float): The ratio of elements to mask (between 0 and 1).

    Returns:
        tuple: A tuple containing:
            - x_masked (torch.Tensor): Masked tensor of shape (N, len_keep, D).
            - mask (torch.Tensor): Binary mask of shape (N, L), where 0 indicates kept and 1 indicates masked.
            - ids_restore (torch.Tensor): Indices used to restore the original order, of shape (N, L).
    """
    N, L, D = x.shape  # batch size, sequence length, feature dimension
    len_keep = int(L * (1 - mask_ratio))

    # Generate random noise for shuffling
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # Sort noise to generate shuffled indices (ascending order)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Keep the first len_keep elements based on the sorted indices
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

    # Generate the binary mask: 0 is keep, 1 is masked
    mask = torch.ones((N, L), device=x.device)
    mask[:, :len_keep] = 0
    # Unshuffle the mask to match the original order
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore

# Example usage
if __name__ == "__main__":
    x = torch.randn(5, 10, 3)  # Example input tensor of shape (N=5, L=10, D=3)
    mask_ratio = 0.3

    x_masked, mask, ids_restore = random_masking(x, mask_ratio)

    print("Original shape:", x.shape)
    print("Masked shape:", x_masked.shape)
    print("Mask:", mask)
    print("Restore indices:", ids_restore)
