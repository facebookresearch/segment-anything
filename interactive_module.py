# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F

from layers import MLP, DynamicLinear, PositionEmbeddingRandom
from image_encoder import LayerNorm # FIXME: Put this in a better location


class InteractiveModule(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        mlp_hidden_dim: int,
        mask_dim: Optional[int],
        iou_prediction_head: Optional[nn.Module] = None,
        num_point_embeddings: int = 4,
        number_of_additional_tokens: Optional[int] = 5,
        add_mask_pred: bool = False,
        mask_pred_dim: int = 16,
        num_outputs: int = 1,
        final_layer_hypernetwork_mlp: bool = False,
        dedicated_multiclick_slot: bool = False,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        # Activation function for upscaling
        self.activation = activation

        # Position encoding generator
        N_steps = transformer_dim // 2
        self.pe_layer = PositionEmbeddingRandom(N_steps)
        self.dedicated_multiclick_slot = dedicated_multiclick_slot
        if dedicated_multiclick_slot:
            num_outputs += 1
        self.num_outputs = num_outputs

        # Transformer
        self.transformer = transformer

        # Token embeddings (optim uses nn.Embedding detection to set WD to 0)
        self.output_embed = nn.Embedding(number_of_additional_tokens, transformer_dim)
        # Using a ModuleList instead of an (n, transformer_dim) sized embedding
        # allows for easy backwards compatible loading of old checkpoints.
        self.num_point_embeddings = num_point_embeddings
        point_embeddings = [nn.Embedding(1, transformer_dim) for i in range(num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, transformer_dim)

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
        self.output_hypernetworks = nn.ModuleList(
            [DynamicLinear(transformer_dim // 8, 1) for i in range(num_outputs)]
        )

        self.add_mask_pred = add_mask_pred
        if add_mask_pred:
            self.pred_downscaling = nn.Sequential(
                nn.Conv2d(1, mask_pred_dim // 4, kernel_size=2, stride=2),
                LayerNorm(mask_pred_dim // 4),
                self.activation(),
                nn.Conv2d(mask_pred_dim // 4, mask_pred_dim, kernel_size=2, stride=2),
                LayerNorm(mask_pred_dim),
                self.activation(),
                nn.Conv2d(mask_pred_dim, transformer_dim, kernel_size=1),
            )
            self.no_pred_embed = nn.Embedding(1, transformer_dim)
        self.transformer_dim = transformer_dim

        self.iou_prediction_head = iou_prediction_head

    def forward(
        self,
        low_res_embeddings: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: Optional[torch.Tensor],
        is_first_iter: bool = False,
    ) -> Dict[str, torch.Tensor]:

        # Generate positional embeddings
        pos = self.pe_layer(low_res_embeddings)
        # Ignore positional embedding for non-points
        point_coords_pe = self.pe_layer._pe_encoding(point_coords)
        ignore_mask = point_labels == -1
        point_coords_pe[ignore_mask] = 0.0

        # Prepare transformer queries
        queries = point_coords_pe  # B x N x C
        for i in range(self.num_point_embeddings):
            queries[point_labels == i] += self.point_embeddings[i].weight
        queries[point_labels == -1] += self.not_a_point_embed.weight

        # Concatenate annotation token, output is B x (N + 1) x C
        # Later code assumes the output tokens are concatenated at the front
        output_token = self.output_embed.weight.unsqueeze(0).expand(queries.size(0), -1, -1)
        queries = torch.cat((output_token, queries), dim=1)

        # when point_coords is empty, queries should just be output_token
        if point_labels.shape[1] == 0:
            assert torch.equal(queries, output_token)

        queries = queries.to(torch.float)  # Preserved from old version, maybe unncessary

        # Get prediction from last iteration
        has_prev_pred = (mask_input is not None)
        if self.add_mask_pred and has_prev_pred:
            low_res_pred_masks = self.pred_downscaling(mask_input)
            no_pred_indicator = 0
        else:
            low_res_pred_masks = None
            no_pred_indicator = 1

        # Expand per-image data in batch direction to be per-mask
        # FIXME: handle num_masks_per_image better
        num_masks_per_image = torch.LongTensor([point_coords.shape[0]]).to(pos.device)
        src = torch.repeat_interleave(low_res_embeddings, num_masks_per_image, dim=0)
        pos_src = torch.repeat_interleave(pos, num_masks_per_image, dim=0)
        b, c, h, w = src.shape

        if self.add_mask_pred:
            # pretend to add no_pred_embed to make ddp happy
            src += no_pred_indicator * self.no_pred_embed.weight.unsqueeze(-1).unsqueeze(-1)
            if low_res_pred_masks is not None:
                src += low_res_pred_masks

        # Apply transformer, output is (deep supervision x B x N x C) (TODO: Check this)
        hs, src = self.transformer(src, pos_src, queries)

        src = src.transpose(1, 2).view(b, c, h, w)
        outputs_seg_masks = self.output_upscaling(src)
        out_masks = []
        for i in range(self.num_outputs):
            hyper_in = hs[:, i + self.n_iou_token, :]
            if self.output_hypernetworks_mlps is not None:
                hyper_in = self.output_hypernetworks_mlps[i](hyper_in)
            out_masks.append(
                self.output_hypernetworks[i](
                    outputs_seg_masks,
                    hyper_in,
                    force_no_chunk=False,
                )
            )
        outputs_seg_masks = torch.cat(out_masks, dim=1)

        # Select the dedicated slot or other slots if
        # dedicated_multiclick_slot = True
        if self.dedicated_multiclick_slot and not is_first_iter:
            channel_slice = slice(0, 1)
        elif self.dedicated_multiclick_slot and is_first_iter:
            channel_slice = slice(1, None)
        else:
            channel_slice = slice(None, None)
        outputs_seg_masks = outputs_seg_masks[:, channel_slice, :, :]

        iou_pred = self.iou_prediction_head(hs[:, 0, :])

        # Prepare output
        return outputs_seg_masks, iou_pred[:, channel_slice]
