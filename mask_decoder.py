# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F

from layers import MLP
from image_encoder import LayerNorm # FIXME: Put this in a better location


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        iou_prediction_head: nn.Module,
        num_multimask_outputs: int = 1,
        final_layer_hypernetwork_mlp: bool = False,
        dedicated_multiclick_slot: bool = False,
        activation: Type[nn.Module] = nn.GELU,
        num_extra_tokens: int = 0,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.iou_prediction_head = iou_prediction_head

        self.dedicated_multiclick_slot = dedicated_multiclick_slot
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = int(dedicated_multiclick_slot) + num_multimask_outputs
        self.mask_tokens = nn.Embedding(
            self.num_mask_tokens, transformer_dim
        )
        if num_extra_tokens > 0:
            self.extra_tokens = nn.Embedding(number_of_additional_tokens, transformer_dim)
        else:
            self.extra_tokens = None

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        if final_layer_hypernetwork_mlp:
            self.output_hypernetworks_mlps = nn.ModuleList(
                [
                    MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                    for i in range(self.num_mask_tokens)
                ]
            )
        else:
            self.output_hypernetworks_mlps = None


    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Dict[str, torch.Tensor]:

        output_tokens = torch.cat([
            self.iou_token.weight, self.mask_tokens.weight
        ], dim=0)
        if self.extra_tokens is not None:
            output_tokens = torch.cat([
                output_tokens, self.extra_tokens.weight
            ], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1+self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        outputs_seg_masks = self.output_upscaling(src)
        if self.output_hypernetworks_mlps is not None:
            hyper_in = []
            for i in range(self.num_mask_tokens):
                hyper_in.append(
                        self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                )
            hyper_in = torch.stack(hyper_in, dim=1)
        else:
            # TODO: Verify non-mlp model
            hyper_in = mask_tokens_out[:, :, :outputs_seg_masks.shape[1]]
        b, c, h, w = outputs_seg_masks.shape
        outputs_seg_masks = (
            hyper_in @ outputs_seg_masks.view(b, c, h * w)
        ).view(b, -1, h, w) 

        iou_pred = self.iou_prediction_head(iou_token_out)

        # Select the dedicated slot or other slots if
        # dedicated_multiclick_slot = True
        if self.dedicated_multiclick_slot and not multimask_output:
            channel_slice = slice(0, 1)
        elif self.dedicated_multiclick_slot and multimask_output:
            channel_slice = slice(1, None)
        else:
            channel_slice = slice(None, None)
        outputs_seg_masks = outputs_seg_masks[:, channel_slice, :, :]


        # Prepare output
        return outputs_seg_masks, iou_pred[:, channel_slice]
