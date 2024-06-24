<script setup lang="ts">

const IMAGE_PATH = "src/assets/dogs.jpg";
const IMAGE_EMBEDDING = "src/assets/dogs_embedding.npy";
const MODEL_DIR = "src/assets/sam_onnx_quantized_example.onnx";

import npyjs from "npyjs";
import {InferenceSession, Tensor} from "onnxruntime-web";
import {onBeforeMount, ref} from "vue";
import * as ort from "onnxruntime-web";
import * as _ from "underscore";
import {onnxMaskToImage} from "./maskUtils";



const loadNpyTensor = async (tensorFile: string, dType: string) => {
  let npLoader = new npyjs();
  const npArray = await npLoader.load(tensorFile);
  const tensor = new ort.Tensor(dType, npArray.data, npArray.shape);
  return tensor;
};

const model= ref<InferenceSession | null>(null);
const initModel = async () => {
  try {
    if (MODEL_DIR === undefined) return;
    const URL: string = MODEL_DIR;
    // or download the model from a CDN ,and put it in the src folder
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/"
    model.value = await InferenceSession.create(URL);
  } catch (e) {
    console.log(e);
  }
};
const clicks=ref()

const handleImageScale = (image: HTMLImageElement) => {
  // Input images to SAM must be resized so the longest side is 1024
  const LONG_SIDE_LENGTH = 1024;
  let w = image.naturalWidth;
  let h = image.naturalHeight;
  const samScale = LONG_SIDE_LENGTH / Math.max(h, w);
  return { height: h, width: w, samScale };
};

const imageSrc=ref()
const modelScale=ref()
const shouldFitToWidth=ref()
const loadImage = async (url: URL) => {
  try {
    const img = new Image();
    img.src = url;
    img.onload = () => {
      const { height, width, samScale } = handleImageScale(img);
      modelScale.value={
        height: height,  // original image height
        width: width,  // original image width
        samScale: samScale, // scaling factor for image which has been resized to longest side 1024
      }
      img.width = width;
      img.height = height;
         const imageAspectRatio = width / height;
    const screenAspectRatio = window.innerWidth / window.innerHeight;
    shouldFitToWidth.value=imageAspectRatio > screenAspectRatio
      imageSrc.value=img
    };
  } catch (error) {
    console.log(error);
  }
};
const modelData = ({ clicks, tensor, modelScale }: any) => {
  const imageEmbedding = tensor;
  let pointCoords;
  let pointLabels;
  let pointCoordsTensor;
  let pointLabelsTensor;

  // Check there are input click prompts
  if (clicks) {
    let n = clicks.length;

    // If there is no box input, a single padding point with
    // label -1 and coordinates (0.0, 0.0) should be concatenated
    // so initialize the array to support (n + 1) points.
    pointCoords = new Float32Array(2 * (n + 1));
    pointLabels = new Float32Array(n + 1);

    // Add clicks and scale to what SAM expects
    for (let i = 0; i < n; i++) {
      pointCoords[2 * i] = clicks[i].x * modelScale.samScale;
      pointCoords[2 * i + 1] = clicks[i].y * modelScale.samScale;
      pointLabels[i] = clicks[i].clickType;
    }

    // Add in the extra point/label when only clicks and no box
    // The extra point is at (0, 0) with label -1
    pointCoords[2 * n] = 0.0;
    pointCoords[2 * n + 1] = 0.0;
    pointLabels[n] = -1.0;

    // Create the tensor
    pointCoordsTensor = new Tensor("float32", pointCoords, [1, n + 1, 2]);
    pointLabelsTensor = new Tensor("float32", pointLabels, [1, n + 1]);
  }
  const imageSizeTensor = new Tensor("float32", [
    modelScale.height,
    modelScale.width,
  ]);

  if (pointCoordsTensor === undefined || pointLabelsTensor === undefined)
    return;

  // There is no previous mask, so default to an empty tensor
  const maskInput = new Tensor(
    "float32",
    new Float32Array(256 * 256),
    [1, 1, 256, 256]
  );
  // There is no previous mask, so default to 0
  const hasMaskInput = new Tensor("float32", [0]);

  return {
    image_embeddings: imageEmbedding,
    point_coords: pointCoordsTensor,
    point_labels: pointLabelsTensor,
    orig_im_size: imageSizeTensor,
    mask_input: maskInput,
    has_mask_input: hasMaskInput,
  };
};

const handleMouseMove = _.throttle(async (e: any) => {
  let el = e.target;
  const rect = el.getBoundingClientRect();
  let x = e.clientX - rect.left;
  let y = e.clientY - rect.top;
  const imageScale = imageSrc.value ? imageSrc.value.width / el.offsetWidth : 1;
  x *= imageScale;
  y *= imageScale;
  clicks.value=[{x:x,y:y,clickType:1}]
  await runONNX()
}, 15);

const handleMouseout= async () => {
  clicks.value=null
  predictImg.value=null
};
const runONNX = async () => {
    try {
      if (
        model.value === null ||
        clicks.value === null ||
        tensor.value === null ||
        modelScale.value === null
      )
        return;
      else {
        // Preapre the model input in the correct format for SAM.
        // The modelData function is from onnxModelAPI.tsx.
        const feeds = modelData({
          clicks:clicks.value,
          tensor:tensor.value,
          modelScale:modelScale.value,
        });
        if (feeds === undefined) return;
        // Run the SAM ONNX model with the feeds returned from modelData()
        const results = await model.value.run(feeds);
        const output = results[model.value.outputNames[0]];
        // The predicted mask returned from the ONNX model is an array which is
        // rendered as an HTML image using onnxMaskToImage() from maskUtils.tsx.
        predictImg.value = onnxMaskToImage(output.data, output.dims[2], output.dims[3]).src
      }
    } catch (e) {
      console.error("Error running ONNX model")
      console.log(e);
    }
  };

const predictImg=ref()

const tensor = ref<ort.Tensor | null>(null);
onBeforeMount(async () => {
   await loadImage(IMAGE_PATH)
  await initModel();
  tensor.value = await loadNpyTensor(IMAGE_EMBEDDING, "float32")
});

</script>

<template>
  <div class="flex items-center justify-center w-full h-full">
      <div class="flex items-center justify-center relative w-[90%] h-[90%]">
    <img :src="IMAGE_PATH" :class="shouldFitToWidth?'w-full': 'h-full'"
         @mousemove="handleMouseMove"
         @mouseout="handleMouseout"
        />
    <img :src='predictImg' class="absolute opacity-40 pointer-events-none"/>
  </div>
  </div>
</template>

<style scoped>
.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}
.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}
.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}
</style>
