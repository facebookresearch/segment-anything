/* ============================================================
   SAM Section — main thread UI / interaction controller
   Talks to sam-worker.js (the on-device SAM engine) and drives
   the canvas, the hover-to-pick loop, freeze and removal.
   ============================================================ */

const MAX_SIDE = 1024; // SAM operates on a 1024px long side.

const SAMPLES = [
  { name: "Dog", src: "assets/img/samples/dog.jpg" },
  { name: "Truck", src: "assets/img/samples/truck.jpg" },
  { name: "Groceries", src: "assets/img/samples/groceries.jpg" },
];

/* ---- DOM --------------------------------------------------- */
const $ = (id) => document.getElementById(id);
const statusPill = $("statusPill"), statusText = $("statusText");
const devicePill = $("devicePill"), deviceText = $("deviceText");
const dropzone = $("dropzone"), fileInput = $("fileInput");
const browseBtn = $("browseBtn"), pasteHint = $("pasteHint"), sampleRow = $("sampleRow");
const newImageBtn = $("newImageBtn"), helpBtn = $("helpBtn"), help = $("help");
const stage = $("stage"), canvasWrap = $("canvasWrap");
const imageCanvas = $("imageCanvas"), overlayCanvas = $("overlayCanvas");
const hud = $("hud"), hudPick = $("hudPick"), hudScale = $("hudScale"), hudScore = $("hudScore"), hudMeter = $("hudMeter");
const loader = $("loader"), loaderTitle = $("loaderTitle"), loaderMsg = $("loaderMsg"), loaderBar = $("loaderBar");
const dock = $("dock"), scaleLabel = $("scaleLabel");
const sizeUp = $("sizeUp"), sizeDown = $("sizeDown");
const freezeBtn = $("freezeBtn"), removeBtn = $("removeBtn");
const undoBtn = $("undoBtn"), resetBtn = $("resetBtn"), downloadBtn = $("downloadBtn");
const toastEl = $("toast");

const imageCtx = imageCanvas.getContext("2d", { willReadFrequently: true });
const overlayCtx = overlayCanvas.getContext("2d");

/* ---- Colors for the mask overlay --------------------------- */
const PICK = { fill: [124, 92, 255, 84], edge: [186, 160, 255, 255] };
const FROZEN = { fill: [255, 92, 200, 86], edge: [255, 178, 224, 255] };

/* ---- State ------------------------------------------------- */
const state = {
  ready: false,
  device: "—",
  hasImage: false,
  encoded: false,
  pendingEncode: false,
  encodeId: 0,
  W: 0, H: 0,
  originalImageData: null,
  overlayImage: null,         // reusable ImageData for the overlay
  candidates: null,           // { masks, order, areas, scores }
  index: 0,                   // index into candidates.order (0 = biggest)
  frozen: false,
  frozenMask: null,
  pendingPoint: null,
  inFlight: false,
  undo: [],
};
const MAX_UNDO = 15;

/* ---- Worker ------------------------------------------------ */
const params = new URLSearchParams(location.search);
const worker = new Worker("assets/js/sam-worker.js", { type: "module" });
worker.addEventListener("message", onWorkerMessage);
worker.addEventListener("error", (e) =>
  setStatus("error", "Worker failed to start")
);
worker.postMessage({ type: "load", backend: params.get("backend") || undefined });

/* ---- Small helpers ----------------------------------------- */
let toastTimer;
function toast(msg, isErr = false) {
  toastEl.textContent = msg;
  toastEl.classList.toggle("err", isErr);
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), 2600);
}
function setStatus(kind, text) {
  statusPill.classList.remove("is-loading", "is-ready", "is-error");
  statusPill.classList.add(`is-${kind}`);
  statusText.textContent = text;
}
function showLoader(title, msg, pct) {
  loaderTitle.textContent = title;
  loaderMsg.textContent = msg || "";
  loader.classList.add("show");
  loaderBar.style.width = (pct ?? 0) + "%";
  loaderBar.parentElement.style.visibility = pct == null ? "hidden" : "visible";
}
function hideLoader() { loader.classList.remove("show"); }

/* ---- Worker message handling ------------------------------- */
function onWorkerMessage(e) {
  const m = e.data;
  switch (m.type) {
    case "progress":
      showLoader("Preparing SAM", m.text, m.pct);
      setStatus("loading", "Loading model…");
      break;
    case "ready":
      state.ready = true;
      state.device = m.device;
      deviceText.textContent = m.device === "webgpu" ? "WebGPU" : "WASM · CPU";
      devicePill.title = m.device === "webgpu"
        ? "Running on your GPU via WebGPU"
        : "WebGPU unavailable — running on CPU (slower)";
      setStatus("ready", "Ready");
      if (state.pendingEncode) doEncode();
      else hideLoader();
      break;
    case "encoded":
      if (m.id !== state.encodeId) return; // stale
      state.encoded = true;
      hideLoader();
      setStatus("ready", "Pick anything");
      hud.classList.add("show");
      dock.classList.add("show");
      canvasWrap.classList.remove("cursor-busy");
      canvasWrap.classList.add("cursor-pick");
      break;
    case "masks":
      onMasks(m);
      break;
    case "error":
      console.error("[SAM]", m.message);
      if (m.fatal) {
        setStatus("error", "Model failed to load");
        showLoader("Couldn't start SAM", m.message + " — try reloading.", null);
      } else {
        toast("Inference error: " + m.message, true);
        state.inFlight = false;
        canvasWrap.classList.remove("cursor-busy");
      }
      break;
  }
}

/* ---- Image intake ------------------------------------------ */
browseBtn.addEventListener("click", () => fileInput.click());
newImageBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
  const f = e.target.files && e.target.files[0];
  if (f) loadFromFile(f);
  fileInput.value = "";
});
pasteHint.addEventListener("click", () =>
  toast("Press ⌘/Ctrl + V to paste an image")
);

// Drag & drop
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); })
);
dropzone.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) loadFromFile(f);
});
// Paste
window.addEventListener("paste", (e) => {
  const items = e.clipboardData && e.clipboardData.items;
  if (!items) return;
  for (const it of items) {
    if (it.type.startsWith("image/")) { loadFromFile(it.getAsFile()); break; }
  }
});

// Samples
SAMPLES.forEach((s) => {
  const b = document.createElement("button");
  b.className = "sample";
  b.title = s.name;
  b.innerHTML = `<img src="${s.src}" alt="${s.name}" />`;
  b.addEventListener("click", () => loadFromURL(s.src));
  sampleRow.appendChild(b);
});

function loadFromFile(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => { onImageReady(img); URL.revokeObjectURL(url); };
  img.onerror = () => { toast("Could not read that image.", true); URL.revokeObjectURL(url); };
  img.src = url;
}
function loadFromURL(src) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => onImageReady(img);
  img.onerror = () => toast("Could not load image (blocked or offline).", true);
  img.src = src;
}

/* ---- Prepare an image for segmentation --------------------- */
function onImageReady(img) {
  // Downscale so the long side is <= MAX_SIDE (SAM's working resolution).
  const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
  const W = Math.max(1, Math.round(img.naturalWidth * scale));
  const H = Math.max(1, Math.round(img.naturalHeight * scale));

  state.W = W; state.H = H;
  imageCanvas.width = W; imageCanvas.height = H;
  overlayCanvas.width = W; overlayCanvas.height = H;
  imageCtx.clearRect(0, 0, W, H);
  imageCtx.drawImage(img, 0, 0, W, H);

  state.originalImageData = imageCtx.getImageData(0, 0, W, H);
  state.overlayImage = overlayCtx.createImageData(W, H);

  // Reset interaction state
  state.hasImage = true;
  state.encoded = false;
  state.candidates = null;
  state.index = 0;
  state.frozen = false;
  state.frozenMask = null;
  state.pendingPoint = null;
  state.inFlight = false;
  state.undo = [];
  clearOverlay();
  refreshButtons();

  dropzone.style.display = "none";
  canvasWrap.classList.remove("hidden");
  newImageBtn.hidden = false;
  fitImage();

  if (state.ready) doEncode();
  else { state.pendingEncode = true; showLoader("Preparing SAM", "Waiting for the model to finish loading…", null); }
}

function doEncode() {
  state.pendingEncode = false;
  state.encodeId++;
  const id = state.encodeId;
  showLoader("Reading the image", "Encoding on " + (state.device === "webgpu" ? "your GPU" : "CPU") + "…", null);
  setStatus("loading", "Encoding…");
  canvasWrap.classList.add("cursor-busy");

  const snapshot = imageCtx.getImageData(0, 0, state.W, state.H);
  worker.postMessage(
    { type: "encode", id, width: state.W, height: state.H, data: snapshot.data.buffer },
    [snapshot.data.buffer]
  );
}

/* ---- Display sizing ---------------------------------------- */
function fitImage() {
  if (!state.hasImage) return;
  const pad = 48;
  const availW = stage.clientWidth - pad;
  const availH = stage.clientHeight - 150; // leave room for dock + HUD
  const ar = state.W / state.H;
  let w = Math.min(availW, 1100);
  let h = w / ar;
  if (h > availH) { h = availH; w = h * ar; }
  canvasWrap.style.width = Math.round(w) + "px";
  canvasWrap.style.height = Math.round(h) + "px";
}
let resizeRAF;
window.addEventListener("resize", () => {
  cancelAnimationFrame(resizeRAF);
  resizeRAF = requestAnimationFrame(fitImage);
});

/* ---- Hover → decode loop ----------------------------------- */
canvasWrap.addEventListener("mousemove", (e) => {
  if (!state.encoded || state.frozen) return;
  const r = imageCanvas.getBoundingClientRect();
  const x = (e.clientX - r.left) / r.width;
  const y = (e.clientY - r.top) / r.height;
  if (x < 0 || x > 1 || y < 0 || y > 1) return;
  state.pendingPoint = { x: clamp01(x), y: clamp01(y) };
  pump();
});
canvasWrap.addEventListener("mouseleave", () => {
  state.pendingPoint = null;
  if (!state.frozen) { state.candidates = null; clearOverlay(); refreshButtons(); updateHUD(); }
});

function pump() {
  if (state.inFlight || state.frozen || !state.pendingPoint) return;
  const p = state.pendingPoint;
  state.pendingPoint = null;
  state.inFlight = true;
  worker.postMessage({ type: "decode", x: p.x, y: p.y });
}

function onMasks(m) {
  state.inFlight = false;
  if (!state.encoded) return;
  state.candidates = { masks: m.masks, order: m.order, areas: m.areas, scores: m.scores, dims: m.dims };
  // Keep the user's chosen granularity across hovers, clamped to range.
  state.index = Math.min(state.index, m.order.length - 1);
  if (!state.frozen) { renderCurrent(); refreshButtons(); updateHUD(); }
  // Coalesce: if the cursor moved while we were busy, go again.
  if (state.pendingPoint && !state.frozen) requestAnimationFrame(pump);
}

/* ---- Current mask helpers ---------------------------------- */
function currentMask() {
  const c = state.candidates;
  if (!c) return null;
  const idx = c.order[state.index];
  return c.masks[idx];
}
function currentScore() {
  const c = state.candidates;
  if (!c) return 0;
  return c.scores[c.order[state.index]] ?? 0;
}

/* ---- Overlay rendering ------------------------------------- */
function clearOverlay() {
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}
function renderCurrent() {
  const mask = state.frozen ? state.frozenMask : currentMask();
  if (!mask) { clearOverlay(); return; }
  renderMask(mask, state.frozen ? FROZEN : PICK);
}
function renderMask(mask, color) {
  const W = state.W, H = state.H;
  const img = state.overlayImage;
  const data = img.data;
  data.fill(0);
  const [fr, fg, fb, fa] = color.fill;
  const [er, eg, eb, ea] = color.edge;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const p = y * W + x;
      if (!mask[p]) continue;
      const edge =
        x === 0 || y === 0 || x === W - 1 || y === H - 1 ||
        !mask[p - 1] || !mask[p + 1] || !mask[p - W] || !mask[p + W];
      const o = p << 2;
      if (edge) { data[o] = er; data[o + 1] = eg; data[o + 2] = eb; data[o + 3] = ea; }
      else { data[o] = fr; data[o + 1] = fg; data[o + 2] = fb; data[o + 3] = fa; }
    }
  }
  overlayCtx.putImageData(img, 0, 0);
}

/* ---- HUD --------------------------------------------------- */
function updateHUD() {
  const c = state.candidates;
  const n = c ? c.order.length : 0;
  const k = n ? state.index + 1 : 0;
  let tag = "";
  if (n) tag = state.index === 0 ? " · largest" : state.index === n - 1 ? " · smallest" : "";
  scaleLabel.textContent = n ? `Size ${k}/${n}` : "Size —";
  hudScale.textContent = n ? `${k}/${n}${tag}` : "—";

  const score = currentScore();
  hudScore.textContent = n ? score.toFixed(2) : "—";
  hudMeter.style.width = n ? Math.round(score * 100) + "%" : "0%";

  const sw = hud.querySelector(".swatch");
  if (sw) sw.style.background = state.frozen ? "var(--magenta)" : "var(--accent)";
  hudPick.textContent = state.frozen
    ? "Frozen — press Delete to remove"
    : n ? "Click or press F to freeze" : "Move the cursor to pick";
}

/* ---- Freeze / unfreeze ------------------------------------- */
canvasWrap.addEventListener("click", () => {
  if (!state.encoded) return;
  if (state.frozen) unfreeze();
  else if (state.candidates) freeze();
});
function freeze() {
  const mask = currentMask();
  if (!mask) return;
  state.frozen = true;
  state.frozenMask = mask.slice(); // own a stable copy
  renderCurrent();
  refreshButtons();
  updateHUD();
}
function unfreeze() {
  state.frozen = false;
  state.frozenMask = null;
  clearOverlay();
  refreshButtons();
  updateHUD();
}

/* ---- Scale scrubbing --------------------------------------- */
function nudgeSize(dir) {
  // dir = -1 → bigger (toward index 0); +1 → smaller
  if (!state.candidates || state.frozen) return;
  const n = state.candidates.order.length;
  const next = Math.min(n - 1, Math.max(0, state.index + dir));
  if (next === state.index) return;
  state.index = next;
  renderCurrent();
  updateHUD();
}
sizeUp.addEventListener("click", () => nudgeSize(-1));
sizeDown.addEventListener("click", () => nudgeSize(1));
canvasWrap.addEventListener("wheel", (e) => {
  if (!state.candidates || state.frozen) return;
  e.preventDefault();
  nudgeSize(e.deltaY > 0 ? 1 : -1);
}, { passive: false });

/* ---- Removal / history ------------------------------------- */
freezeBtn.addEventListener("click", () => {
  if (state.frozen) unfreeze();
  else freeze();
});
removeBtn.addEventListener("click", removeSelection);
function removeSelection() {
  // Use the frozen mask if present, otherwise the live one.
  const mask = state.frozen ? state.frozenMask : currentMask();
  if (!mask) { toast("Hover an object first."); return; }

  pushUndo();
  const W = state.W, H = state.H;
  const frame = imageCtx.getImageData(0, 0, W, H);
  const d = frame.data;
  for (let p = 0; p < mask.length; p++) {
    if (mask[p]) d[(p << 2) + 3] = 0; // erase → transparent
  }
  imageCtx.putImageData(frame, 0, 0);

  unfreeze();
  state.candidates = null;
  state.index = 0;
  refreshButtons();
  updateHUD();
  toast("Object removed");
}
function pushUndo() {
  state.undo.push(imageCtx.getImageData(0, 0, state.W, state.H));
  if (state.undo.length > MAX_UNDO) state.undo.shift();
  refreshButtons();
}
undoBtn.addEventListener("click", undo);
function undo() {
  const prev = state.undo.pop();
  if (!prev) return;
  imageCtx.putImageData(prev, 0, 0);
  unfreeze();
  state.candidates = null;
  refreshButtons();
  updateHUD();
}
resetBtn.addEventListener("click", () => {
  if (!state.originalImageData) return;
  imageCtx.putImageData(state.originalImageData, 0, 0);
  state.undo = [];
  unfreeze();
  state.candidates = null;
  refreshButtons();
  updateHUD();
  toast("Restored original");
});
downloadBtn.addEventListener("click", () => {
  const a = document.createElement("a");
  a.download = "sam-cutout.png";
  a.href = imageCanvas.toDataURL("image/png");
  a.click();
});

/* ---- Button enable/disable --------------------------------- */
function refreshButtons() {
  const hasSel = !!(state.candidates || state.frozenMask);
  const canScrub = !!state.candidates && !state.frozen;
  freezeBtn.disabled = !hasSel;
  removeBtn.disabled = !hasSel;
  sizeUp.disabled = !canScrub || state.index === 0;
  sizeDown.disabled = !canScrub || (state.candidates && state.index === state.candidates.order.length - 1);
  undoBtn.disabled = state.undo.length === 0;
  resetBtn.disabled = state.undo.length === 0;
  downloadBtn.disabled = !state.hasImage;
  freezeBtn.classList.toggle("is-active", state.frozen);
}

/* ---- Keyboard ---------------------------------------------- */
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  const k = e.key;
  if ((e.metaKey || e.ctrlKey) && (k === "z" || k === "Z")) { e.preventDefault(); undo(); return; }
  if (!state.hasImage) return;
  switch (k) {
    case "f": case "F":
      if (state.encoded) { e.preventDefault(); state.frozen ? unfreeze() : freeze(); }
      break;
    case "Delete": case "Backspace":
      e.preventDefault(); removeSelection(); break;
    case "Escape":
      if (state.frozen) unfreeze(); break;
    case "ArrowUp":
      e.preventDefault(); nudgeSize(-1); break;
    case "ArrowDown":
      e.preventDefault(); nudgeSize(1); break;
    case "?":
      help.classList.toggle("show"); break;
  }
});

/* ---- Help toggle ------------------------------------------- */
helpBtn.addEventListener("click", () => help.classList.toggle("show"));
document.addEventListener("click", (e) => {
  if (help.classList.contains("show") && !help.contains(e.target) && e.target !== helpBtn && !helpBtn.contains(e.target))
    help.classList.remove("show");
});

/* ---- utils ------------------------------------------------- */
function clamp01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
