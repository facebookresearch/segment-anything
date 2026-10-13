# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from segment_anything import SamAutomaticMaskGenerator, SamPredictor
from segment_anything.modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)
from segment_anything.utils.amg import calculate_stability_score
from segment_anything.utils.transforms import ResizeLongestSide

import pytest
from types import SimpleNamespace


@pytest.fixture
def default_dtype(request):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    try:
        yield request.param
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("width", [1024, 32767, 32768, 40000, 65536])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("threshold,offset", [(0.0, 1.0), (3.0, 0.5)])
def test_stability_score_counts_wide_masks(width, dtype, threshold, offset):
    counts = [width, width - 1, 7 * width // 8]
    masks = torch.full((3, 2, width), threshold, dtype=dtype)
    for index, count in enumerate(counts):
        masks[index, :, :count] = threshold + 2 * offset
    expected = torch.tensor([count / width for count in counts])
    actual = calculate_stability_score(masks, threshold, offset)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.get_default_dtype()
    assert actual.device == masks.device
    assert torch.isfinite(actual).all()
    assert ((actual >= 0) & (actual <= 1)).all()


def test_stability_score_preserves_batch_and_strict_thresholds():
    values = np.tile(np.array([-1.0, 0.0, 1.0, 2.0]), 10000)
    masks = torch.from_numpy(np.broadcast_to(values, (2, 3, 2, 40000)).copy())
    expected = torch.full((2, 3), 1.0 / 3.0)
    for layout in [masks, masks.transpose(-1, -2)]:
        torch.testing.assert_close(calculate_stability_score(layout, 0.0, 1.0), expected)


def test_stability_score_empty_batch_and_existing_empty_mask_behavior():
    empty = torch.empty(0, 2, 40000)
    assert calculate_stability_score(empty, 0.0, 1.0).shape == (0,)
    assert torch.isnan(calculate_stability_score(torch.full((1, 2, 8), -2.0), 0.0, 1.0)).all()


@pytest.mark.parametrize("transpose", [False, True])
def test_stability_score_torchscript(transpose):
    masks = torch.zeros(1, 2, 40000)
    masks[..., :35000] = 2
    if transpose:
        masks = masks.transpose(-1, -2)
    scripted = torch.jit.script(calculate_stability_score)
    torch.testing.assert_close(scripted(masks, 0.0, 1.0), torch.tensor([0.875]))


def _assert_only_stable_masks_are_kept(width):
    masks = torch.zeros(1, 3, 2, width, dtype=torch.float32)
    for index, count in enumerate([width, width - 1, 7 * width // 8]):
        masks[:, index, :, :count] = 2
    generator = SamAutomaticMaskGenerator.__new__(SamAutomaticMaskGenerator)
    generator.predictor = SimpleNamespace(
        transform=ResizeLongestSide(1024),
        device=torch.device("cpu"),
        model=SimpleNamespace(mask_threshold=0.0),
        predict_torch=lambda *args, **kwargs: (masks, torch.ones(1, 3, dtype=torch.float32), masks),
    )
    generator.pred_iou_thresh = 0.0
    generator.stability_score_thresh = 0.95
    generator.stability_score_offset = 1.0
    data = generator._process_batch(
        np.array([[width / 2, 1.0]]), (2, width), [0, 0, width, 2], (2, width)
    )
    assert len(data["rles"]) == 2
    torch.testing.assert_close(data["stability_score"], torch.tensor([1.0, (width - 1) / width]))


@pytest.mark.parametrize("width", [32768, 40000, 65536])
def test_automatic_mask_generator_keeps_only_stable_masks(width):
    _assert_only_stable_masks_are_kept(width)


@pytest.mark.parametrize(
    "default_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64], indirect=True
)
@pytest.mark.parametrize("scripted", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
def test_stability_score_preserves_default_dtype(default_dtype, scripted, transpose):
    score = calculate_stability_score
    if scripted:
        torch.set_default_dtype(torch.float32)
        try:
            score = torch.jit.script(calculate_stability_score)
        finally:
            torch.set_default_dtype(default_dtype)
    for width in [1024, 32767, 32768, 40000, 65536]:
        counts = [width, width - 1, 7 * width // 8]
        masks = torch.zeros(3, 2, width, dtype=torch.float32)
        for index, count in enumerate(counts):
            masks[index, :, :count] = 2
        if transpose:
            masks = masks.transpose(-1, -2)
        actual = score(masks, 0.0, 1.0)
        torch.testing.assert_close(actual, torch.tensor([count / width for count in counts]))
        assert actual.dtype == default_dtype
        assert actual.device == masks.device
        assert torch.isfinite(actual).all()
        assert ((actual >= 0) & (actual <= 1)).all()
    fractional = torch.tensor([[[2.0, 0.0, 0.0]]], dtype=torch.float32)
    torch.testing.assert_close(
        score(fractional, 0.0, 1.0), torch.tensor([1.0 / 3.0]), rtol=0, atol=0
    )
    assert score(torch.empty(0, 2, 40000, dtype=torch.float32), 0.0, 1.0).shape == (0,)
    assert torch.isnan(score(torch.full((1, 2, 8), -2.0, dtype=torch.float32), 0.0, 1.0)).all()


@pytest.mark.parametrize(
    "default_dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64], indirect=True
)
def test_automatic_mask_generator_filter_with_default_dtype(default_dtype):
    for width in [32768, 40000, 65536]:
        _assert_only_stable_masks_are_kept(width)


def test_predictor_scores_original_resolution_with_controlled_decoder(monkeypatch):
    model = Sam(
        ImageEncoderViT(
            img_size=1024, patch_size=16, embed_dim=8, depth=1, num_heads=1, out_chans=8
        ),
        PromptEncoder(8, (64, 64), (1024, 1024), 4),
        MaskDecoder(
            transformer_dim=8,
            transformer=TwoWayTransformer(1, 8, 1, 16),
            iou_head_hidden_dim=8,
        ),
    )
    predictor = SamPredictor(model)
    predictor.input_size = ResizeLongestSide.get_preprocess_shape(64, 40000, 1024)
    predictor.original_size = (64, 40000)
    predictor.is_image_set = True
    predictor.features = torch.zeros(1, 8, 64, 64)
    low_res = torch.full((1, 1, 8, 8), 2.0)
    low_res[..., -1] = 0
    monkeypatch.setattr(model.mask_decoder, "forward", lambda **kwargs: (low_res, torch.ones(1, 1)))
    masks, _, _ = predictor.predict_torch(None, None, return_logits=True)
    assert masks.shape == (1, 1, 64, 40000)
    logits = masks.numpy()
    expected = np.count_nonzero(logits > 1.0, axis=(-2, -1)) / np.count_nonzero(
        logits > -1.0, axis=(-2, -1)
    )
    np.testing.assert_allclose(calculate_stability_score(masks, 0.0, 1.0).numpy(), expected)
