# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from segment_anything.build_sam import (
    build_sam,
    build_sam_vit_h,
    build_sam_vit_l,
    build_sam_vit_b,
    sam_model_registry,
)
from segment_anything.predictor import SamPredictor
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

__version__ = "1.0.0"
