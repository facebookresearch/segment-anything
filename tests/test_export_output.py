# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from segment_anything.modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)

import importlib.util
import pytest
from pathlib import Path


@pytest.mark.parametrize("single_mask", [False, True])
@pytest.mark.parametrize("use_stability", [False, True])
def test_run_export_passes_output_path_to_serializer(
    tmp_path, monkeypatch, single_mask, use_stability
):
    script = Path(__file__).parents[1] / "scripts" / "export_onnx_model.py"
    spec = importlib.util.spec_from_file_location("sam_export_output_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sam = Sam(
        ImageEncoderViT(img_size=16, patch_size=8, embed_dim=8, depth=1, num_heads=1, out_chans=8),
        PromptEncoder(8, (2, 2), (16, 16), 4),
        MaskDecoder(
            transformer_dim=8,
            transformer=TwoWayTransformer(1, 8, 1, 16),
            iou_head_hidden_dim=8,
        ),
    )
    monkeypatch.setitem(module.sam_model_registry, "test", lambda checkpoint: sam)
    monkeypatch.setattr(module, "onnxruntime_exists", False)
    output = tmp_path / "mask decoder.onnx"
    calls = []

    def serialize(model, inputs, target, **kwargs):
        assert target == str(output)
        assert kwargs["opset_version"] == 17
        assert kwargs["input_names"] == [
            "image_embeddings",
            "point_coords",
            "point_labels",
            "mask_input",
            "has_mask_input",
            "orig_im_size",
        ]
        assert kwargs["output_names"] == ["masks", "iou_predictions", "low_res_masks"]
        assert kwargs["dynamic_axes"] == {
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
        }
        with torch.no_grad():
            masks, scores, logits = model(*inputs)
        assert masks.shape[-2:] == (1500, 2250)
        assert masks.shape[1] == (1 if single_mask else 4)
        assert scores.shape == masks.shape[:2]
        assert logits.shape[:2] == masks.shape[:2]
        assert torch.isfinite(masks).all()
        calls.append(target)
        Path(target).write_bytes(b"serializer contract control")

    monkeypatch.setattr(torch.onnx, "export", serialize)
    with torch.no_grad():
        module.run_export(
            "test",
            "unused local fixture",
            str(output),
            17,
            single_mask,
            use_stability_score=use_stability,
        )
    assert calls == [str(output)]
    assert output.read_bytes() == b"serializer contract control"
