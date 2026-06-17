/* ============================================================
   SAM inference worker
   Runs the Segment Anything encoder + decoder fully on-device
   using transformers.js (WebGPU, with a WASM fallback).

   Messages IN:
     { type: 'load', backend? }
     { type: 'encode', id, width, height, data(RGBA buffer) }   // transfer data.buffer
     { type: 'decode', x, y }                                    // x,y normalized 0..1
   Messages OUT:
     { type: 'progress', pct, text }
     { type: 'ready', device }
     { type: 'encoded', id }
     { type: 'masks', dims:[H,W], order:[idx...], masks:[Uint8Array], areas:[], scores:[] }
     { type: 'error', message, fatal }
   ============================================================ */

import {
  SamModel,
  AutoProcessor,
  RawImage,
  Tensor,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

// Pull weights from the Hugging Face hub (not from this origin).
env.allowLocalModels = false;

const MODEL_ID = "Xenova/slimsam-77-uniform";

let model = null;
let processor = null;
let device = "wasm";

// Per-image state, reused across every hover so we never re-run the encoder.
let imageInputs = null;
let imageEmbeddings = null;

/* ---- Model loading ---------------------------------------- */
async function load(preferred) {
  processor = await AutoProcessor.from_pretrained(MODEL_ID);

  const tryOrder =
    preferred === "wasm" ? ["wasm"] : ["webgpu", "wasm"];

  let lastErr = null;
  for (const dev of tryOrder) {
    try {
      const opts =
        dev === "webgpu"
          ? { dtype: "fp16", device: "webgpu" }
          : { dtype: "q8", device: "wasm" };

      model = await SamModel.from_pretrained(MODEL_ID, {
        ...opts,
        progress_callback: (p) => {
          if (p.status === "progress" && p.total) {
            const pct = Math.round((p.loaded / p.total) * 100);
            self.postMessage({
              type: "progress",
              pct,
              text: `Downloading model · ${pct}%`,
            });
          } else if (p.status === "ready") {
            self.postMessage({ type: "progress", pct: 100, text: "Warming up…" });
          }
        },
      });
      device = dev;
      return;
    } catch (e) {
      lastErr = e;
      model = null;
      // try next backend
    }
  }
  throw lastErr || new Error("Could not initialise the model.");
}

/* ---- Encode one image ------------------------------------- */
async function encode(msg) {
  // Rebuild the image from the transferred RGBA buffer.
  const rgba = new Uint8ClampedArray(msg.data);
  const image = new RawImage(rgba, msg.width, msg.height, 4).rgb();

  imageInputs = await processor(image);
  imageEmbeddings = await model.get_image_embeddings(imageInputs);

  self.postMessage({ type: "encoded", id: msg.id });
}

/* ---- Decode a single point -------------------------------- */
async function decode(msg) {
  if (!imageEmbeddings) return;

  const reshaped = imageInputs.reshaped_input_sizes[0]; // [h, w]
  // One positive point, scaled into the model's reshaped input space.
  const points = [msg.x * reshaped[1], msg.y * reshaped[0]];
  const input_points = new Tensor("float32", points, [1, 1, 1, 2]);
  const input_labels = new Tensor("int64", [1n], [1, 1, 1]);

  const outputs = await model({
    ...imageEmbeddings,
    input_points,
    input_labels,
  });

  // Upscale masks back to the original (= encoded) image size.
  const processed = await processor.post_process_masks(
    outputs.pred_masks,
    imageInputs.original_sizes,
    imageInputs.reshaped_input_sizes
  );

  const maskTensor = processed[0]; // dims [num, H, W]
  const [num, H, W] = maskTensor.dims;
  const flat = maskTensor.data; // 0/1 per pixel
  const scores = Array.from(outputs.iou_scores.data); // length num

  const masks = [];
  const areas = [];
  const plane = H * W;
  for (let i = 0; i < num; i++) {
    const m = new Uint8Array(plane);
    let area = 0;
    const base = i * plane;
    for (let p = 0; p < plane; p++) {
      const v = flat[base + p] > 0 ? 1 : 0;
      m[p] = v;
      area += v;
    }
    masks.push(m);
    areas.push(area);
  }

  // Rank biggest → smallest (the "scrub" order the UI exposes).
  const order = masks.map((_, i) => i).sort((a, b) => areas[b] - areas[a]);

  self.postMessage(
    {
      type: "masks",
      dims: [H, W],
      order,
      masks,
      areas,
      scores,
    },
    masks.map((m) => m.buffer) // zero-copy transfer
  );
}

/* ---- Dispatch --------------------------------------------- */
self.addEventListener("message", async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "load") {
      await load(msg.backend);
      self.postMessage({ type: "ready", device });
    } else if (msg.type === "encode") {
      await encode(msg);
    } else if (msg.type === "decode") {
      await decode(msg);
    }
  } catch (err) {
    self.postMessage({
      type: "error",
      message: (err && err.message) || String(err),
      fatal: msg.type === "load",
    });
  }
});
