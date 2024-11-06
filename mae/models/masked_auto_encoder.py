import torch
import torch.nn as nn

from .maevit import MAEViT
from .maedec import MAEDecoder
from .patchify import patchify, unpatchify


class MaskedAutoEncoder(nn.Module):
    """
    Masked Autoencoder (MAE) model with Vision Transformer (ViT) encoder and decoder.

    Args:
        img_size (int): Size of the input image (assumed to be square).
        patch_size (int): Size of each patch (assumed to be square).
        in_chans (int): Number of input channels (e.g., 3 for RGB).
        embed_dim (int): Dimension of the embedding in the encoder.
        depth (int): Number of transformer blocks in the encoder.
        num_heads (int): Number of attention heads in the encoder.
        decoder_embed_dim (int): Dimension of the embedding in the decoder.
        decoder_depth (int): Number of transformer blocks in the decoder.
        decoder_num_heads (int): Number of attention heads in the decoder.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
        norm_layer (nn.Module): Normalization layer to use (e.g., LayerNorm).
        norm_pix_loss (bool): Whether to normalize pixel values during the loss computation.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: nn.Module = nn.LayerNorm,
        norm_pix_loss: bool = False,
    ):
        super().__init__()

        # Check image size compatibility with patch size
        assert img_size % patch_size == 0, "Image size must be divisible by patch size."

        # Encoder: Vision Transformer with masked autoencoder
        self.encoder = MAEViT(
            img_size=img_size,
            in_chans=in_chans,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            learnable_pos_embed=False,
        )

        # Decoder: Masked autoencoder decoder
        self.decoder = MAEDecoder(
            num_patches=(img_size // patch_size) ** 2,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            learnable_pos_embed=False,
        )

        # Additional configuration
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.in_chans = in_chans

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> tuple:
        """
        Compute the reconstruction loss for the autoencoder.

        Args:
            imgs (torch.Tensor): Original input images of shape (N, C, H, W).
            pred (torch.Tensor): Predicted patches from the decoder of shape (N, L, P*P*C).
            mask (torch.Tensor): Mask indicating the positions to reconstruct (N, L).

        Returns:
            tuple: Mean squared error loss on masked patches, and the predicted reconstruction.
        """
        # Generate target patches from original images
        target = patchify(imgs, patch_size=self.patch_size)

        # Apply normalization to the target and pred if required
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
            pred_compensated = pred * (var + 1.0e-6) ** 0.5 + mean
        else:
            pred_compensated = pred
        pred_corrected = unpatchify(pred_compensated, self.patch_size)
        # Calculate mean squared error between predicted and target patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean loss per patch (N, L)

        # Mean loss on removed patches only
        loss = (loss * mask).sum() / mask.sum()
        return loss, pred_corrected

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75) -> tuple:
        """
        Forward pass through the Masked Autoencoder.

        Args:
            imgs (torch.Tensor): Input images of shape (N, C, H, W).
            mask_ratio (float): Proportion of patches to mask.

        Returns:
            tuple: (loss, predictions, mask)
        """
        # Encode the images
        latent, mask, ids_restore = self.encoder(imgs, mask_ratio)

        # Decode the latent representation
        pred = self.decoder(latent, ids_restore)  # (N, L, patch_size*patch_size*in_chans)

        # Calculate the loss
        loss, pred_corrected = self.forward_loss(imgs, pred, mask)

        return loss, pred_corrected, mask


# Example usage
if __name__ == "__main__":
    # Instantiate the MaskedAutoEncoder
    model = MaskedAutoEncoder(img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12)

    # Generate a batch of dummy images
    imgs = torch.randn(4, 3, 224, 224)  # Batch size of 4, RGB images of 224x224

    # Forward pass through the model
    loss, pred, mask = model(imgs, mask_ratio=0.75)

    print(f"Loss: {loss.item()}")
    print(f"Predictions shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
