import numpy as np
import tritonclient.http as httpclient
from typing import Optional, Tuple
from PIL import Image
from copy import deepcopy

from .utils.transforms import ResizeLongestSide

class MockModel():

    mask_threshold = 0


class TritonSamPredictor():

    def __init__(
        self,
        host='localhost',
        encoder_model_name='sam_encoder',
        decoder_model_name='sam_decoder',
        proxy_host=None,
        proxy_port=None,
        img_size=1024
    ):
        self._host = host
        self._encoder_model_name = encoder_model_name
        self._decoder_model_name = decoder_model_name
        self._client = httpclient.InferenceServerClient(url=host, proxy_host=proxy_host, proxy_port=proxy_port)
        self.img_size = img_size
        self.transform = ResizeLongestSide(img_size)
        self.model = MockModel()
        self.reset_image()

    def get_image_embedding(self) -> np.ndarray:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None

    def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB"
    ):
        # Resize image preserving aspect ratio using 1024 as a long side
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        orig_width, orig_height = image.size
        resized_width, resized_height = image.size

        if orig_width > orig_height:
            resized_width = self.img_size
            resized_height = int(self.img_size / orig_width * orig_height)
        else:
            resized_height = self.img_size
            resized_width = int(self.img_size / orig_height * orig_width)

        res_image = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)

        # Prepare input tensor from image
        input_tensor = np.array(res_image)

        # Normalize input tensor numbers
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([[58.395, 57.12, 57.375]])
        input_tensor = (input_tensor - mean) / std

        # Transpose input tensor to shape (Batch,Channels,Height,Width
        input_tensor = input_tensor.transpose(2,0,1)[None,:,:,:].astype(np.float32)

        # Make image square 1024x1024 by padding short side by zeros
        if resized_height < resized_width:
            input_tensor = np.pad(input_tensor,((0,0),(0,0),(0,self.img_size-resized_height),(0,0)))
        else:
            input_tensor = np.pad(input_tensor,((0,0),(0,0),(0,0),(0,self.img_size-resized_width)))
        
        self.is_image_set = True
        self.orig_w, self.orig_h = image.size
        self.input_w, self.input_h = resized_width, resized_height
        self.input_tensor = input_tensor

    def _run_encoder(self):
        # Set Inputs
        input_tensors = [
            httpclient.InferInput("images", self.input_tensor.shape, datatype="FP32")
        ]
        input_tensors[0].set_data_from_numpy(self.input_tensor)
        # Set outputs
        outputs = [
            httpclient.InferRequestedOutput("embeddings")
        ]

        # Query
        query_response = self._client.infer(model_name=self._encoder_model_name,
                                        inputs=input_tensors,
                                        outputs=outputs)
        self.features = query_response.as_numpy("embeddings")
        return self.features

    def _run_decoder(self, params):
        input_tensors = []
        for k, v in params.items():
            it = httpclient.InferInput(k, v.shape, datatype="FP32")
            it.set_data_from_numpy(v)
            input_tensors.append(it)
        output_keys = ["masks", "iou_predictions", "low_res_masks"]
        outputs = [httpclient.InferRequestedOutput(k) for k in output_keys]
        query_response = self._client.infer(model_name='sam_decoder',
                                        inputs=input_tensors,
                                        outputs=outputs)
        return {k: query_response.as_numpy(k) for k in output_keys}

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ):
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        embeddings = self._run_encoder()

        # onnx_coord = np.concatenate([point_coords, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        # onnx_label = np.concatenate([point_labels, np.array([-1])])[None, :].astype(np.float32)

        batch_size = 1  # len(point_coords)

        onnx_coord = point_coords
        onnx_label = point_labels.astype(np.float32)

        coords = deepcopy(onnx_coord).astype(float)
        coords[..., 0] = coords[..., 0] * (self.input_w / self.orig_w)
        coords[..., 1] = coords[..., 1] * (self.input_h / self.orig_h)

        onnx_coord = coords.astype("float32")

        # RUN DECODER TO GET MASK
        onnx_mask_input = np.zeros((batch_size, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(batch_size, dtype=np.float32)

        decoder_inputs = {
            "image_embeddings": np.repeat(embeddings, batch_size, axis=0),
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array([self.orig_h, self.orig_w], dtype=np.float32)
        }

        retvals = self._run_decoder(decoder_inputs)
        return retvals['masks'], retvals["iou_predictions"], retvals['low_res_masks']

    def predict_np(
        self,
        point_coords: Optional[np.ndarray],
        point_labels: Optional[np.ndarray],
        boxes: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        retvals = self.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=boxes,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits
        )
        return retvals

    def get_image_embedding(self):
        pass

    @property
    def device(self):
        return 'cpu'
