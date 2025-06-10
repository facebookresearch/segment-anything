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

# Find contours (edges of the mask)
contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a white canvas
canvas = np.ones_like(image) * 255  # RGB white background

# Draw original contours in red
cv2.drawContours(canvas, contours, -1, color=(255, 0, 0), thickness=2)  # Red (BGR)

# Combine all contour points
all_points = np.concatenate(contours)

# Approximate polygon (simplified shape)
epsilon = 0.01 * cv2.arcLength(all_points, True)
approx_polygon = cv2.approxPolyDP(all_points, epsilon, True)

# Draw the simplified polygon in blue
#cv2.polylines(canvas, [approx_polygon], isClosed=True, color=(0, 0, 255), thickness=2)

# --- Line classification logic starts here ---
# Flatten and convert to list of point pairs (edges)
pts = approx_polygon.reshape(-1, 2)
edges = []

for i in range(len(pts)):
    pt1 = pts[i]
    pt2 = pts[(i + 1) % len(pts)]
    length = np.linalg.norm(pt1 - pt2)
    edges.append((length, pt1, pt2))

# Sort edges by length (descending)
edges.sort(key=lambda x: x[0], reverse=True)

# Take top 4 longest edges
top_edges = edges[:4]

# Colors for the 4 lines
line_colors = [(255, 0, 255),   # Purple
               (0, 255, 255),   # Yellow
               (0, 128, 255),   # Orange
               (0, 255, 0)]     # Green

# Line extension length
extension = 50  # pixels

# Draw extended lines
for idx, (_, pt1, pt2) in enumerate(top_edges):
    pt1 = pt1.astype(np.float32)
    pt2 = pt2.astype(np.float32)

    direction = pt2 - pt1
    norm = np.linalg.norm(direction)
    if norm == 0:
        continue  # Skip zero-length edges
    direction = direction / norm

    extended_pt1 = pt1 - direction * extension
    extended_pt2 = pt2 + direction * extension

    # Convert to int for drawing
    p1 = tuple(np.intp(extended_pt1))
    p2 = tuple(np.intp(extended_pt2))

    cv2.line(canvas, p1, p2, color=line_colors[idx % 4], thickness=2)

# Show the result
#plt.figure(figsize=(10, 10))
plt.imshow(canvas)
plt.title("Contour + Simplified Polygon + Classified Edges (Top, Bottom, Left, Right)")
plt.axis('off')
plt.show()

