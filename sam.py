from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from image_encoder import ImageEncoderViT
from prompt_encoder import PromptEncoder
from mask_decoder import MaskDecoder, MLP
from transformer import Transformer, TwoWayDecoderLayer

class Sam(nn.Module):

    mask_threshold = 0.0
    image_format = "RGB"

    def __init__(
        self,
        image_encoder: torch.nn.Module,
        prompt_encoder: torch.nn.Module,
        mask_decoder: torch.nn.Module,
        pixel_mean = [123.675, 116.28, 103.53],
        pixel_std = [58.395, 57.12, 57.375],
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)


    def forward(
        self, 
        batched_input: List[Dict[str, torch.Tensor]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:
        input_images = torch.stack(
            [self.preprocess(x['image']) for x in batched_input], dim=0
        )
        image_embeddings = self.image_encoder(input_images)

        outputs = []
        for image_record, image_embeddings in zip(batched_input, image_embeddings):
            if 'point_coords' in image_record:
                points = (image_record['point_coords'], image_record['point_labels'])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=image_record.get('boxes', None),
                masks=image_record.get('masks', None),
            )
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_embeddings=sparse_embeddings,
                dense_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks(
                low_res_masks,
                input_size=image_record['image'].shape[-2],
                original_size=image_record['original_size'],
            )
            masks = masks > self.mask_threshold
            outputs.append({
                "masks" : masks,
                "iou_predictions": iou_predictions,
                "low_res_predictions": low_res_masks,
            })
        return outputs

    def postprocess_masks(
        self, masks: torch.Tensor, input_size: Tuple[int], original_size: Tuple[int],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks, (self.image_encoder.img_size, self.image_encoder.img_size), mode="bilinear", align_corners=False
        )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    def preprocess(self, x):
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x


def build_sam():
    return Sam(
        image_encoder=ImageEncoderViT(
            depth=32,
            embed_dim=1280,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_heads=16,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            window_block_indexes=[0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30],
            window_size=14,
            out_chans=256,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16
        ),
        mask_decoder=MaskDecoder(
            final_layer_hypernetwork_mlp=True,
            iou_prediction_head=MLP(
                hidden_dim=256,
                input_dim=256,
                output_dim=4,
                num_layers=3,
            ),
            num_multimask_outputs=3,
            dedicated_multiclick_slot=True,
            transformer=Transformer(
                add_pe_to_first_layer=True,
                decoder_layer=TwoWayDecoderLayer,
                depth=2,
                embedding_dim=256,
                final_attention_by_clicks=True,
                mlp_dim=2048,
                num_heads=8,
                p_dropout=0.1,
                pre_norm=False,
            ),
            transformer_dim=256
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

if __name__=="__main__":
    sam = build_sam()
    print(sam.image_encoder)
    print(sam.mask_decoder)

