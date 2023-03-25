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
        iou_prediction_head: Optional[nn.Module] = None,
        number_of_additional_tokens: Optional[int] = 5,
        num_outputs: int = 1,
        final_layer_hypernetwork_mlp: bool = False,
        dedicated_multiclick_slot: bool = False,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        # Activation function for upscaling
        self.activation = activation

        self.dedicated_multiclick_slot = dedicated_multiclick_slot
        if dedicated_multiclick_slot:
            num_outputs += 1
        self.num_outputs = num_outputs

        # Transformer
        self.transformer = transformer

        # Token embeddings (optim uses nn.Embedding detection to set WD to 0)
        self.output_embed = nn.Embedding(number_of_additional_tokens, transformer_dim)

        self.n_iou_token = int(iou_prediction_head is not None)
        assert number_of_additional_tokens >= num_outputs + self.n_iou_token, (
            "With final_layer_hypernetwork, must have enough output tokens "
            "to support each mask prediction, as well as IoU/slot prediction."
        )
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm(transformer_dim // 4),
            self.activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            self.activation(),
        )
        if final_layer_hypernetwork_mlp:
            self.output_hypernetworks_mlps = nn.ModuleList(
                [
                    MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                    for i in range(num_outputs)
                ]
            )
        else:
            self.output_hypernetworks_mlps = None

        self.transformer_dim = transformer_dim

        self.iou_prediction_head = iou_prediction_head

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Dict[str, torch.Tensor]:

        output_token = self.output_embed.weight.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        queries = torch.cat((output_token, sparse_prompt_embeddings), dim=1)
        queries = queries.to(torch.float)  # Preserved from old version, maybe unncessary

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, queries.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, queries.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, queries)

        src = src.transpose(1, 2).view(b, c, h, w)
        outputs_seg_masks = self.output_upscaling(src)
        if self.output_hypernetworks_mlps is not None:
            hyper_in = []
            for i in range(self.num_outputs):
                hyper_in.append(
                    self.output_hypernetworks_mlps[i](hs[:, i+self.n_iou_token, :])
                )
            hyper_in = torch.stack(hyper_in, dim=1)
        else:
            # TODO: Verify non-mlp model
            hyper_in = hs[
                :, 
                self.n_iou_token : (self.n_iou_token + self.num_outputs), 
                : outputs_seg_masks.shape[1]
            ]
        b, c, h, w = outputs_seg_masks.shape
        outputs_seg_masks = (
            hyper_in @ outputs_seg_masks.view(b, c, h * w)
        ).view(b, -1, h, w) 

        # Select the dedicated slot or other slots if
        # dedicated_multiclick_slot = True
        if self.dedicated_multiclick_slot and not multimask_output:
            channel_slice = slice(0, 1)
        elif self.dedicated_multiclick_slot and multimask_output:
            channel_slice = slice(1, None)
        else:
            channel_slice = slice(None, None)
        outputs_seg_masks = outputs_seg_masks[:, channel_slice, :, :]

        iou_pred = self.iou_prediction_head(hs[:, 0, :])

        # Prepare output
        return outputs_seg_masks, iou_pred[:, channel_slice]
