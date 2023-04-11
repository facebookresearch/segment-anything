## Segment Anything Simple Web demo

This **front-end only** demo shows how to load a fixed image and `.npy` file of the SAM image embedding, and run the SAM ONNX model in the browser using Web Assembly with mulithreading enabled by `SharedArrayBuffer`, Web Worker, and SIMD128.

<img src="https://github.com/facebookresearch/segment-anything/raw/main/assets/minidemo.gif" width="500"/>

## Run the app

```
yarn && yarn start
```

Navigate to [`http://localhost:8081/`](http://localhost:8081/)

Move your cursor around to see the mask prediction update in real time.

## Change the image, embedding and ONNX model

In the [ONNX Model Example notebook](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb) upload the image of your choice and generate and save corresponding embedding.

Initialize the predictor

```python
checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cuda')
predictor = SamPredictor(sam)
```

Set the new image and export the embedding

```
image = cv2.imread('src/assets/dogs.jpg')
predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()
np.save("dogs_embedding.npy", image_embedding)
```

Save the new image and embedding in `/assets/data`and update the following paths to the files at the top of`App.tsx`:

```py
const IMAGE_PATH = "/assets/data/dogs.jpg";
const IMAGE_EMBEDDING = "/assets/data/dogs_embedding.npy";
const MODEL_DIR = "/model/sam_onnx_quantized_example.onnx";
```

Optionally you can also export the ONNX model. Currently the example ONNX model from the notebook is saved at `/model/sam_onnx_quantized_example.onnx`.

**NOTE: if you change the ONNX model by using a new checkpoint you need to also re-export the embedding.**

## ONNX multithreading with SharedArrayBuffer

To use multithreading, the appropriate headers need to be set to create a cross origin isolation state which will enable use of `SharedArrayBuffer` (see this [blog post](https://cloudblogs.microsoft.com/opensource/2021/09/02/onnx-runtime-web-running-your-machine-learning-model-in-browser/) for more details)

The headers below are set in `configs/webpack/dev.js`:

```js
headers: {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "credentialless",
}
```

## Structure of the app

**`App.tsx`**

- Initializes ONNX model
- Loads image embedding and image
- Runs the ONNX model based on input prompts

**`Stage.tsx`**

- Handles mouse move interaction to update the ONNX model prompt

**`Tool.tsx`**

- Renders the image and the mask prediction

**`helpers/maskUtils.tsx`**

- Conversion of ONNX model output from array to an HTMLImageElement

**`helpers/onnxModelAPI.tsx`**

- Formats the inputs for the ONNX model

**`helpers/scaleHelper.tsx`**

- Handles image scaling logic for SAM (longest size 1024)

**`hooks/`**

- Handle shared state for the app
