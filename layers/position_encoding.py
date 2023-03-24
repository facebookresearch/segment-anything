# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py  # noqa
"""
Various positional encodings for the transformer.
"""
import numpy as np
import torch
from torch import nn


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats=64, scale=None):
        """
        'memory_efficient' returns a batch_size=1 PE for images and expects
        broadcasting to add things correctly.
        """
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords):
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, x):
        bs, H, W = x.size(0), x.size(2), x.size(3)
        mask = torch.ones((bs, H, W), device=x.device, dtype=torch.bool)
        y_embed = mask.cumsum(1, dtype=torch.float32) - 0.5
        x_embed = mask.cumsum(2, dtype=torch.float32) - 0.5
        y_embed = y_embed / H
        x_embed = x_embed / W

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(0, 3, 1, 2)  # B x C x H x W

    def forward_with_coords(self, coords_input, image_size):
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
