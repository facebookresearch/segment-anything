from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transforms import ResizeLongestSide

class SamPredictor:

    def __init__(
        self,
        sam_model: nn.Module,
    ):
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def reset_image(self):
        self.is_image_set = False
        self.features = None
        self.orig_h, self.orig_w = None, None
        self.input_h, self.input_w = None, None

    def set_image(
        self,
        image: np.ndarray,
        image_format = "RGB",
    ) -> None:
        self.reset_image()
        assert image_format in ["RGB", "BGR"], \
            f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[...,::-1]
        
        input_image = self.transform.apply_image(image)
        input_image = torch.as_tensor(input_image).permute(2,0,1).contiguous()
        input_image = input_image[None,:,:,:]
        self.set_torch_image(input_image, image.shape[:2])


    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor,
        original_image_size: Tuple[int, int],
    ) -> None:
        self.original_size = original_image_size
        self.input_size = transformed_image.shape[-2:]
        input_image = self.model.preprocess(transformed_image)
        self.features = self.model.image_encoder(input_image)
        self.is_image_set = True


    def predict(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        mask_input: Optional[np.ndarray],
        multimask: Optional[bool] = None,
        mask_input_transform: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # FIXME: Will become cleaner when moving to prompt-based model.
        if not np.any(np.logical_or(point_labels == 2, point_labels == 3)):
            point_coords = np.concatenate([point_coords, np.zeros((1, 1, 2))], axis=1)
            point_labels = np.concatenate([point_labels, np.array([[-1]])], axis=1)

        # Auto-detect if the input is the first iteration
        if multimask is None:
            multimask = (point_coords.shape[1] == 2)

        point_coords = self.transform.apply_coords(
            point_coords, self.original_size
        )
        point_coords = torch.as_tensor(point_coords).to(torch.float)
        point_labels = torch.as_tensor(point_labels).to(torch.int)
        if mask_input_transform:
            mask_input = self.transform.apply_mask(mask_input)
        mask_input = torch.as_tensor(mask_input) if mask_input is not None else None

        masks, iou_predictions, low_res_masks = self.predict_torch(
            point_coords,
            point_labels,
            mask_input,
            multimask,
        )

        return masks.numpy(), iou_predictions.numpy(), low_res_masks.numpy()

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        mask_input: Optional[torch.Tensor],
        multimask: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        point_coords = point_coords / self.model.image_encoder.img_size
        low_res_masks, iou_predictions = self.model.mask_decoder(
            self.features,
            point_coords,
            point_labels,
            mask_input,
            multimask,
        )

        masks = self.model.postprocess_masks(
            low_res_masks, self.input_size, self.original_size
        )
        masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks
