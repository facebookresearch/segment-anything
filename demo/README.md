## Segment Anything Simple Web demo

This **front-end only** React based web demo shows how to load a fixed image and corresponding `.npy` file of the SAM image embedding, and run the SAM ONNX model in the browser using Web Assembly with mulithreading enabled by `SharedArrayBuffer`, Web Worker, and SIMD128.

<img src="https://github.com/facebookresearch/segment-anything/raw/main/assets/minidemo.gif" width="500"/>

## Run the app

Install Yarn

```
npm install --g yarn
```

Build and run:

```
yarn && yarn start
```

Navigate to [`http://localhost:8081/`](http://localhost:8081/)

Move your cursor around to see the mask prediction update in real time.

## Export the image embedding

In the [ONNX Model Example notebook](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb) upload the image of your choice and generate and save corresponding embedding.

Initialize the predictor:

```python
checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cuda')
predictor = SamPredictor(sam)
```

Set the new image and export the embedding:

```
image = cv2.imread('src/assets/dogs.jpg')
predictor.set_image(image)
image_embedding = predictor.get_image_embedding().cpu().numpy()
np.save("dogs_embedding.npy", image_embedding)
```

Save the new image and embedding in `src/assets/data`.

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
