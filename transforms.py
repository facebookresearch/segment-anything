from typing import Tuple
from torchvision.transforms.functional import resize, to_pil_image
from copy import deepcopy
import numpy as np

class ResizeLongestSide:
    def __init__(self, target_length: int):
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length
        )
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(
        self, coords: np.ndarray, original_size: Tuple[int]
    ) -> np.ndarray:
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords)
        coords[...,0] = coords[...,0] * (new_w / old_w)
        coords[...,1] = coords[...,1] * (new_h / old_h)
        coords = coords + 0.5 #FIXME: This +0.5 is somewhat orphaned
        return coords

    def apply_boxes(
        self, boxes: np.ndarray, original_size: Tuple[int]
    ) -> np.ndarray:
        boxes = self.apply_coords(
            boxes.reshape(-1, 2, 2), original_size
        )
        return boxes.reshape(-1, 4)

    def apply_mask(self, mask: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

