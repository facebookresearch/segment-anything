import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor

# Load image
image_path = "grubber_ferrule.jpg"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# Set the image
predictor.set_image(image)

# Your YOLO-format bounding box [x_center, y_center, width, height] (all relative)
#yolo_box = [0.556217, 0.528522, 0.263889, 0.199901]  # Replace with your values
yolo_box = [0.305886, 0.578497, 0.102513, 0.0751488]

# Convert to absolute pixel coordinates
img_h, img_w, _ = image.shape
x_center, y_center, w_rel, h_rel = yolo_box
w = w_rel * img_w
h = h_rel * img_h
x0 = int((x_center * img_w) - w / 2)
y0 = int((y_center * img_h) - h / 2)
x1 = int(x0 + w)
y1 = int(y0 + h)
box = np.array([x0, y0, x1, y1])

# Get segmentation mask
masks, scores, _ = predictor.predict(box=box, multimask_output=False)
best_mask = masks[np.argmax(scores)]

# Convert mask to uint8 for OpenCV
mask_uint8 = (best_mask * 255).astype(np.uint8)

# ---- Step 1: Dilate mask to ensure polygon estimation wraps around shape ----
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilated_mask = cv2.dilate(mask_uint8, kernel, iterations=1)

# ---- Step 2: Find contour from dilated mask and approximate polygon ----
contours, _ = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)

# Approximate polygon from dilated contour
epsilon = 0.005 * cv2.arcLength(contour, True)  # Smaller epsilon = tighter fit
approx_polygon = cv2.approxPolyDP(contour, epsilon, True)

# Flatten points
pts = approx_polygon.reshape(-1, 2)

# ---- Step 3: Extract 4 longest lines from polygon ----
edges = []
for i in range(len(pts)):
    pt1 = pts[i]
    pt2 = pts[(i + 1) % len(pts)]
    length = np.linalg.norm(pt2 - pt1)
    edges.append((length, pt1, pt2))

# Sort and take top 4
edges.sort(key=lambda x: x[0], reverse=True)
top_edges = edges[:4]

# ---- Step 4: Draw everything ----
canvas = np.ones_like(image) * 255

# Draw original (non-dilated) contour in red
orig_contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(canvas, orig_contours, -1, color=(255, 0, 0), thickness=2)

# Draw the new polygon in blue
cv2.polylines(canvas, [approx_polygon], isClosed=True, color=(0, 0, 255), thickness=2)

# Draw top 4 longest lines (extended)
extension = 60
colors = [(255, 0, 255), (0, 255, 255), (0, 128, 255), (0, 255, 0)]

for idx, (_, pt1, pt2) in enumerate(top_edges):
    pt1 = pt1.astype(np.float32)
    pt2 = pt2.astype(np.float32)

    direction = pt2 - pt1
    norm = np.linalg.norm(direction)
    if norm == 0:
        continue
    direction = direction / norm

    extended_pt1 = pt1 - direction * extension
    extended_pt2 = pt2 + direction * extension

    cv2.line(canvas,
             tuple(np.intp(extended_pt1)),
             tuple(np.intp(extended_pt2)),
             color=colors[idx % 4],
             thickness=2)

# ---- Step 5: Display ----
#plt.figure(figsize=(10, 10))
plt.imshow(canvas)
plt.title("Red: Original Contour | Blue: Wrapped Polygon | Longest Lines Extended")
plt.axis('off')
plt.show()
