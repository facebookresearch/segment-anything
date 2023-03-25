import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):

    def __init__(
        self,
        backbone,
        backbone_out_channels,
        out_channels,
        pixel_mean,
        pixel_std,
    ):
        super(ImageEncoder, self).__init__()
        self.backbone = backbone
        self.neck = nn.Sequential(
            nn.Conv2d(
                backbone_out_channels, 
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm(out_channels),
        )
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    def preprocess(self, x):
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[2:]
        padh = self.backbone.img_size - h
        padw = self.backbone.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    @staticmethod
    def postprocess_masks(masks, input_image_h, input_image_w):
        """
        Reverses 'preprocess' on results masks. Here this just means
        stripping padding.
        """
        return masks[..., :input_image_h, :input_image_w]

    def forward(self, x):
        x = self.preprocess(x)
        x = self.backbone(x)
        x = self.neck(x)
        return x

class LayerNorm(nn.Module):
    """
    D2LayerNorm for BxCxHxW
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
