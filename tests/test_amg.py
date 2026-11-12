# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from segment_anything.utils.amg import box_xyxy_to_xywh


def test_box_xyxy_to_xywh_single_box():
    box = torch.tensor([1.0, 2.0, 5.0, 8.0])
    out = box_xyxy_to_xywh(box)
    assert torch.equal(out, torch.tensor([1.0, 2.0, 4.0, 6.0]))
    # input is not modified
    assert torch.equal(box, torch.tensor([1.0, 2.0, 5.0, 8.0]))


def test_box_xyxy_to_xywh_batched():
    boxes = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 15.0, 20.0],
        ]
    )
    out = box_xyxy_to_xywh(boxes)
    expected = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [5.0, 5.0, 10.0, 15.0],
        ]
    )
    assert torch.equal(out, expected)
    assert torch.equal(
        boxes,
        torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [5.0, 5.0, 15.0, 20.0],
            ]
        ),
    )
