from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from image_encoder import ImageEncoderViT
from interactive_module import InteractiveModule
from transformer import Transformer, TwoWayDecoderLayer
from layers import MLP

from PIL import Image


class SAM(nn.Module):

    mask_threshold = 0.0

    def __init__(
        self,
        image_encoder: torch.nn.Module,
        mask_decoder: torch.nn.Module,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.reset_image()

    # FIXME: currently a staticmethod so people can offload to dataloader workers
    # Do we want this structure?
    @staticmethod
    def preprocess(
        image: np.ndarray,
        target_size: int, 
        image_format: str = "RGB"
    ) -> torch.Tensor:
        assert image_format in ["RGB", "BGR"], \
            f"image_format must be 'RGB' or 'BGR', is {image_format}."
        if image_format == "BGR":
            image = image[:,:,::-1]


        # Rescale 
        h, w = image.shape[:2]
        new_h, new_w = SAM.get_preprocess_shape(h, w, target_size, target_size)
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((new_w, new_h), Image.BILINEAR)
        image = torch.as_tensor(np.asarray(pil_image))
        image = image.permute(2, 0, 1).unsqueeze(0).contiguous()
        return image

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, short_edge_length: int, max_size: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target short edge length.
        """
        h, w = oldh, oldw
        size = short_edge_length * 1.0
        scale = size / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


    def reset_image(self):
        self.is_image_set = False
        self.features = None
        self.orig_h, self.orig_w = None, None
        self.input_h, self.input_w = None, None


    @torch.no_grad()
    def set_image(
        self,
        image: torch.Tensor,
        image_format: str = "RGB",
    ) -> None:
        self.reset_image()
        
        self.orig_h, self.orig_w = image.shape[:2]
        input_image = self.preprocess(
            image=image, 
            target_size=self.image_encoder.img_size, 
            image_format=image_format,
        )
        self.input_h, self.input_w = input_image.shape[2:]
        self.features = self.image_encoder(input_image)
        self.is_image_set = True

    @staticmethod
    def preprocess_points(
        point_coords: torch.Tensor, 
        new_h: int,
        new_w: int,
        old_h: int, 
        old_w: int,
        target_size: int,
        floor: bool = False,
    ):
        # FIXME: It is fussy to need to have this +0.5 in the middle of rescaling,
        # but it matches the current behavior.
        scale = torch.Tensor([[[new_w/old_w, new_h/old_h]]]).to(point_coords.device)
        # FIXME: There is a bug in InteractivePredictor where concatenating the
        # padding point implicitly casts the points from int to float, causing
        # the transform to not floor the result. This means there is a very subtle
        # difference in results for box inputs (no padding point) and point inputs
        # (padding point). To keep exact reproducability and aid development, I
        # am keeping both options here, but at the end we will just want to pick
        # one. It never affects results in a way that actually matters.
        point_coords = point_coords * scale
        if floor:
            point_coords = torch.floor(point_coords)
        point_coords = point_coords + 0.5
        return point_coords / target_size

    def postprocess_masks(self, masks: torch.Tensor) -> torch.Tensor:
        masks = F.interpolate(
            masks, (self.image_encoder.img_size, self.image_encoder.img_size), mode="bilinear", align_corners=False
        )
        masks = self.image_encoder.postprocess_masks(masks, self.input_h, self.input_w)
        masks = F.interpolate(
            masks, (self.orig_h, self.orig_w), mode="bilinear", align_corners=False
        )
        return masks

    @torch.no_grad()
    def predict(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: Optional[torch.Tensor] = None,
        is_first_iter: Optional[bool] = None,
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Auto-detect if the input is the first iteration
        if is_first_iter is None:
            if len(point_coords) == 1:
                is_first_iter = True
            elif len(point_coords) == 2 and torch.any(
                torch.logical_or(point_labels == 2, point_labels == 3)
            ):
                is_first_iter = True
            elif len(point_coords) == 2 and torch.any(point_labels == -1):
                is_first_iter = True
            else:
                is_first_iter = False

        # Hacky workaround to fix the point2mask inference of a box2mask
        # model, which will expect a single padded point.
        if (
            self.mask_decoder.num_point_embeddings == 4
            and not torch.any(torch.logical_or(point_labels == 2, point_labels == 3))
        ):
            point_coords = torch.cat([point_coords, torch.zeros((1, 1, 2))], axis=1)
            point_labels = torch.cat([point_labels, torch.tensor([[-1]])], axis=1)

        # Transform
        point_coords = self.preprocess_points(
            point_coords, 
            self.input_h,
            self.input_w,
            self.orig_h, 
            self.orig_w,
            self.image_encoder.img_size,
            # FIXME: See function def for explanation of this nonsense
            floor=torch.any(torch.logical_or(point_labels == 2, point_labels == 3))
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            self.features,
            point_coords,
            point_labels,
            mask_input,
            is_first_iter,
        )

        masks = self.postprocess_masks(low_res_masks)

        masks = masks > self.mask_threshold
        return masks, iou_predictions, low_res_masks

    def forward(self):
        raise NotImplementedError(
            "SAM has no end-to-end execution, run set_image(...) and "
            "then predict(...)."
        )


def build_sam():
    return SAM(
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
            pixel_mean=[123.675, 116.28, 103.53],
            pixel_std=[58.395, 57.12, 57.375],
            out_chans=256,
        ),
        mask_decoder=InteractiveModule(
            add_mask_pred=True,
            dedicated_multiclick_slot=True,
            final_layer_hypernetwork_mlp=True,
            iou_prediction_head=MLP(
                hidden_dim=256,
                input_dim=256,
                output_dim=4,
                num_layers=3,
            ),
            mask_dim=None, # To remove
            mask_pred_dim=16,
            mlp_hidden_dim=256,
            num_outputs=3,
            num_point_embeddings=4,
            number_of_additional_tokens=5,
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
    )

if __name__=="__main__":
    sam = build_sam()
    print(sam.image_encoder)
    print(sam.mask_decoder)

