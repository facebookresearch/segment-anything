# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from segment_anything import build_sam, build_sam_vit_b, build_sam_vit_l
from segment_anything.modeling.sam import Sam
from segment_anything import sam_model_registry
from segment_anything.utils.onnx import SamOnnxModel
import onnx
from onnx.external_data_helper import convert_model_to_external_data

import argparse
import warnings

try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

parser = argparse.ArgumentParser(
    description="Export the SAM prompt encoder and mask decoder to an ONNX model."
)

parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="The path to the SAM model checkpoint.",
)

parser.add_argument(
    "--encoder-output",
    type=str,
    required=True,
    help="The filename to save the encoder ONNX model to.",
)

parser.add_argument(
    "--encoder-data-file",
    type=str,
    help="The filename to save the external data for encoder ONNX model to. Use this if the encoder model is too large to be saved in a single file.",
)

parser.add_argument(
    "--decoder-output",
    type=str,
    required=True,
    help="The filename to save the decoder ONNX model to.",
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="In ['default', 'vit_h', 'vit_l', 'vit_b']. Which type of SAM model to export.",
)

parser.add_argument(
    "--return-single-mask",
    action="store_true",
    help=(
        "If true, the exported ONNX model will only return the best mask, "
        "instead of returning multiple masks. For high resolution images "
        "this can improve runtime when upscaling masks is expensive."
    ),
)

parser.add_argument(
    "--opset",
    type=int,
    default=17,
    help="The ONNX opset version to use. Must be >=11",
)

parser.add_argument(
    "--quantize-encoder-out",
    type=str,
    default=None,
    help=(
        "If set, will quantize the encoder model and save it with this name. "
        "Quantization is performed with quantize_dynamic from onnxruntime.quantization.quantize."
    ),
)

parser.add_argument(
    "--quantize-decoder-out",
    type=str,
    default=None,
    help=(
        "If set, will quantize the decoder model and save it with this name. "
        "Quantization is performed with quantize_dynamic from onnxruntime.quantization.quantize."
    ),
)

parser.add_argument(
    "--gelu-approximate",
    action="store_true",
    help=(
        "Replace GELU operations with approximations using tanh. Useful "
        "for some runtimes that have slow or unimplemented erf ops, used in GELU."
    ),
)

parser.add_argument(
    "--use-stability-score",
    action="store_true",
    help=(
        "Replaces the model's predicted mask quality score with the stability "
        "score calculated on the low resolution masks using an offset of 1.0. "
    ),
)

parser.add_argument(
    "--return-extra-metrics",
    action="store_true",
    help=(
        "The model will return five results: (masks, scores, stability_scores, "
        "areas, low_res_logits) instead of the usual three. This can be "
        "significantly slower for high resolution outputs."
    ),
)


def run_export(
    model_type: str,
    checkpoint: str,
    encoder_output: str,
    encoder_data_file: str,
    decoder_output: str,
    opset: int,
    return_single_mask: bool,
    gelu_approximate: bool = False,
    use_stability_score: bool = False,
    return_extra_metrics=False,
):
    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)

    export_encoder(sam, encoder_output, opset, encoder_data_file)

    export_decoder(
        sam,
        decoder_output,
        opset,
        return_single_mask,
        gelu_approximate,
        use_stability_score,
        return_extra_metrics,
    )


def export_encoder(sam: Sam, output: str, opset: int, encoder_data_file: str):
    dynamic_axes = {
        "x": {0: "batch"},
    }
    dummy_inputs = {
        "x": torch.randn(1, 3, 1024, 1024, dtype=torch.float),
    }
    _ = sam.image_encoder(**dummy_inputs)

    output_names = ["image_embeddings"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        print(f"Exporing onnx model to {output}...")
        torch.onnx.export(
            sam.image_encoder,
            tuple(dummy_inputs.values()),
            output,
            export_params=True,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    if encoder_data_file:
        onnx_model = onnx.load(output)
        convert_model_to_external_data(
            onnx_model,
            all_tensors_to_one_file=True,
            location=encoder_data_file,
            size_threshold=1024,
            convert_attribute=False,
        )
        onnx.save_model(onnx_model, output)

    if onnxruntime_exists:
        ort_inputs = {k: to_numpy(v) for k, v in dummy_inputs.items()}
        ort_session = onnxruntime.InferenceSession(output)
        _ = ort_session.run(None, ort_inputs)
        print("Encoder has successfully been run with ONNXRuntime.")


def export_decoder(
    sam: Sam,
    output: str,
    opset: int,
    return_single_mask: bool,
    gelu_approximate: bool,
    use_stability_score: bool,
    return_extra_metrics: bool,
):
    onnx_model = SamOnnxModel(
        model=sam,
        return_single_mask=return_single_mask,
        use_stability_score=use_stability_score,
        return_extra_metrics=return_extra_metrics,
    )

    if gelu_approximate:
        for n, m in onnx_model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(
            low=0, high=1024, size=(1, 5, 2), dtype=torch.float
        ),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }

    _ = onnx_model(**dummy_inputs)

    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(output, "wb") as f:
            print(f"Exporting onnx model to {output}...")
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

    if onnxruntime_exists:
        ort_inputs = {k: to_numpy(v) for k, v in dummy_inputs.items()}
        ort_session = onnxruntime.InferenceSession(output)
        _ = ort_session.run(None, ort_inputs)
        print("Decoder has successfully been run with ONNXRuntime.")


def to_numpy(tensor):
    return tensor.cpu().numpy()


if __name__ == "__main__":
    args = parser.parse_args()
    run_export(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        encoder_output=args.encoder_output,
        encoder_data_file=args.encoder_data_file,
        decoder_output=args.decoder_output,
        opset=args.opset,
        return_single_mask=args.return_single_mask,
        gelu_approximate=args.gelu_approximate,
        use_stability_score=args.use_stability_score,
        return_extra_metrics=args.return_extra_metrics,
    )

    if args.quantize_encoder_out is not None:
        assert onnxruntime_exists, "onnxruntime is required to quantize the model."
        from onnxruntime.quantization import QuantType  # type: ignore
        from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore

        print(f"Quantizing encoder model and writing to {args.quantize_encoder_out}...")
        quantize_dynamic(
            model_input=args.encoder_output,
            model_output=args.quantize_encoder_out,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        print("Done!")

    if args.quantize_decoder_out is not None:
        assert onnxruntime_exists, "onnxruntime is required to quantize the model."
        from onnxruntime.quantization import QuantType  # type: ignore
        from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore

        print(f"Quantizing decoder model and writing to {args.quantize_decoder_out}...")
        quantize_dynamic(
            model_input=args.decoder_output,
            model_output=args.quantize_decoder_out,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        print("Done!")
