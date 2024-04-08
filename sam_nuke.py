import os
import torch
import segment_anything
from segment_anything import sam_model_registry
import logging
from torch import nn
import segment_anything.modeling

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Define the paths and model type
BASE_PATH = "./nuke/Cattery/SegmentAnything"
SAM_MODELS = {
    "vit_b": "../models/sam_vit_b_01ec64.pth",
    "vit_l": "../models/sam_vit_l_0b3195.pth",
    "vit_h": "../models/sam_vit_h_4b8939.pth",
}


class SamEncoderNuke(nn.Module):
    """
    A wrapper around the SAM model that allows it to be used as a TorchScript model.
    """

    def __init__(self, sam_model: segment_anything.modeling.Sam) -> torch.Tensor:
        super().__init__()
        self.sam_model = sam_model

    def forward(self, x: torch.Tensor):
        device = torch.device("cuda") if x.is_cuda else torch.device("cpu")
        dtype = x.dtype

        if dtype == torch.float32:
            x = x.half()

        image = x.to(device)
        image = image * 255.0

        output = self.sam_model.encode(image)
        output = output.reshape(1, 1, 1024, 1024)
        return output.contiguous()


class SamDecoderNuke(nn.Module):
    """
    A wrapper around the SAM model that allows it to be used as a TorchScript model.

    The model is designed to be used in Nuke, where the user can provide up to 24 points.

    The reason for using floats for the points is that Nuke 2D and 3D knobs
    lose their links on the 'Inference' node when reopening a saved Nuke script.
    """

    def __init__(
        self,
        sam_model: segment_anything.modeling.Sam,
        point_01_x=0.0, point_01_y=0.0, label_01=0,
        point_02_x=0.0, point_02_y=0.0, label_02=0,
        point_03_x=0.0, point_03_y=0.0, label_03=0,
        point_04_x=0.0, point_04_y=0.0, label_04=0,
        point_05_x=0.0, point_05_y=0.0, label_05=0,
        point_06_x=0.0, point_06_y=0.0, label_06=0,
        point_07_x=0.0, point_07_y=0.0, label_07=0,
        point_08_x=0.0, point_08_y=0.0, label_08=0,
        point_09_x=0.0, point_09_y=0.0, label_09=0,
        point_10_x=0.0, point_10_y=0.0, label_10=0,
        point_11_x=0.0, point_11_y=0.0, label_11=0,
        point_12_x=0.0, point_12_y=0.0, label_12=0,
        point_13_x=0.0, point_13_y=0.0, label_13=0,
        point_14_x=0.0, point_14_y=0.0, label_14=0,
        point_15_x=0.0, point_15_y=0.0, label_15=0,
        point_16_x=0.0, point_16_y=0.0, label_16=0,
    ) -> torch.Tensor:
        super().__init__()
        self.point_01_x, self.point_01_y, self.label_01 = point_01_x, point_01_y, label_01
        self.point_02_x, self.point_02_y, self.label_02 = point_02_x, point_02_y, label_02,
        self.point_03_x, self.point_03_y, self.label_03 = point_03_x, point_03_y, label_03,
        self.point_04_x, self.point_04_y, self.label_04 = point_04_x, point_04_y, label_04,
        self.point_05_x, self.point_05_y, self.label_05 = point_05_x, point_05_y, label_05,
        self.point_06_x, self.point_06_y, self.label_06 = point_06_x, point_06_y, label_06,
        self.point_07_x, self.point_07_y, self.label_07 = point_07_x, point_07_y, label_07,
        self.point_08_x, self.point_08_y, self.label_08 = point_08_x, point_08_y, label_08,
        self.point_09_x, self.point_09_y, self.label_09 = point_09_x, point_09_y, label_09,
        self.point_10_x, self.point_10_y, self.label_10 = point_10_x, point_10_y, label_10,
        self.point_11_x, self.point_11_y, self.label_11 = point_11_x, point_11_y, label_11,
        self.point_12_x, self.point_12_y, self.label_12 = point_12_x, point_12_y, label_12,
        self.point_13_x, self.point_13_y, self.label_13 = point_13_x, point_13_y, label_13,
        self.point_14_x, self.point_14_y, self.label_14 = point_14_x, point_14_y, label_14,
        self.point_15_x, self.point_15_y, self.label_15 = point_15_x, point_15_y, label_15,
        self.point_16_x, self.point_16_y, self.label_16 = point_16_x, point_16_y, label_16,

        # Segment Anything Model
        self.sam_model = sam_model

    def forward(self, x: torch.Tensor):
        """
        Predicts as mask end-to-end from provided image and the center of the image.

        Args:
            image_embeddings: (torch.Tensor) The image embeddings from the original image, in shape 1x1xHxW.

        Returns:
            mask: (torch.Tensor) The image mask, in shape 1x1xHxW.
        """
        device = torch.device("cuda") if x.is_cuda else torch.device("cpu")

        points = torch.tensor([
            [self.point_01_x, self.point_01_y],
            [self.point_02_x, self.point_02_y],
            [self.point_03_x, self.point_03_y],
            [self.point_04_x, self.point_04_y],
            [self.point_05_x, self.point_05_y],
            [self.point_06_x, self.point_06_y],
            [self.point_07_x, self.point_07_y],
            [self.point_08_x, self.point_08_y],
            [self.point_09_x, self.point_09_y],
            [self.point_10_x, self.point_10_y],
            [self.point_11_x, self.point_11_y],
            [self.point_12_x, self.point_12_y],
            [self.point_13_x, self.point_13_y],
            [self.point_14_x, self.point_14_y],
            [self.point_15_x, self.point_15_y],
            [self.point_16_x, self.point_16_y],
        ]).to(device)

        labels = torch.tensor([
            self.label_01,
            self.label_02,
            self.label_03,
            self.label_04,
            self.label_05,
            self.label_06,
            self.label_07,
            self.label_08,
            self.label_09,
            self.label_10,
            self.label_11,
            self.label_12,
            self.label_13,
            self.label_14,
            self.label_15,
            self.label_16,
        ]).to(device)

        labels = labels < 1  # 0 mode is additive, 1 mode is subtractive in Nuke

        # Remove Trackers in Nuke out of the image bounds (below 1)
        mask = torch.all(points[:, :] >= 1, dim=1)
        active_points = points[mask]
        labels = labels[mask]

        # If no active points, return a blank mask
        if active_points.size(0) == 0:
            return torch.zeros(1, 1, 1024, 1024)

        # Nuke coordinates start from bottom left corner
        active_points[:, 1] = 1024 - active_points[:, 1]

        image_embeddings = x.to(device)
        image_embeddings = image_embeddings.reshape(1, 256, 64, 64)

        # Add batch dimension
        point_coords = active_points[None, :, :]
        point_labels = labels[None, :]

        mask = self.sam_model(image_embeddings, point_coords, point_labels, True)
        return mask


def main():
    """
    Convert SAM to TorchScript and save it.

    See: http://docs.djl.ai/docs/pytorch/how_to_convert_your_model_to_torchscript.html
    """

    # Trace the models
    for model_type, checkpoint in SAM_MODELS.items():
        print("=" * 80)
        print(f"Tracing {model_type} model...")

        # Trace the encoder and decoder
        trace_encoder(model_type, checkpoint)
        trace_decoder(model_type, checkpoint)

        print(f"Finished tracing {model_type} model.")


def trace_encoder(model_type, checkpoint):
    sam_model = sam_model_registry[model_type](checkpoint)

    sam_encoder_nuke = SamEncoderNuke(sam_model)
    sam_encoder_nuke.eval()
    sam_encoder_nuke.half()
    sam_encoder_nuke.cuda()

    # Test the model
    sam_encoder_nuke(torch.randn([1, 3, 1024, 1024], device="cuda"))  # RGB image, 1024x1024

    # Trace the model
    with torch.jit.optimized_execution(True):
        scripted_model = torch.jit.script(sam_encoder_nuke)

    # Save the TorchScript model
    DESTINATION = f"{BASE_PATH}/sam_{model_type}_encoder.pt"
    scripted_model.save(DESTINATION)
    print(f"Saved TorchScript model to {DESTINATION} - {file_size(DESTINATION)} MB")


def trace_decoder(model_type, checkpoint):
    sam_model = sam_model_registry[model_type](checkpoint)

    sam_decoder_nuke = SamDecoderNuke(sam_model)
    sam_decoder_nuke.eval()
    sam_decoder_nuke.cuda()

    # Test the model
    sam_decoder_nuke(torch.randn([1, 1, 1024, 1024], device="cuda"))  # 1024x1024 mask

    # Remove the image encoder for the decoding only pass - saving disk space.
    # We need to make sure we don't use the image encoder in the forward pass.
    sam_decoder_nuke.sam_model.image_encoder = None

    with torch.jit.optimized_execution(True):
        # torch.jit.enable_onednn_fusion(True)  # Not supported in PyTorch 1.6
        scripted_model = torch.jit.script(sam_decoder_nuke)
        # scripted_model = torch.jit.freeze(scripted_model.eval())  # Not supported in PyTorch 1.6

    # Save the TorchScript model
    DESTINATION = f"{BASE_PATH}/sam_{model_type}_decoder.pt"
    scripted_model.save(DESTINATION)
    print(f"Saved TorchScript model to {DESTINATION} - {file_size(DESTINATION)} MB")


def file_size(file_path):
    size_in_bytes = os.path.getsize(file_path)
    return int(size_in_bytes / (1024 * 1024))


if __name__ == "__main__":
    main()
