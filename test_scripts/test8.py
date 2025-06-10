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
contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Create a white canvas
canvas = np.ones_like(image) * 255  # RGB white background

# Draw original contours in red
cv2.drawContours(canvas, contours, -1, color=(255, 0, 0), thickness=1)

# Use the largest contour
contour = max(contours, key=cv2.contourArea)
contour_points = contour.reshape(-1, 2)

# Break contour into N chunks to fit lines
N = 30  # more segments gives better coverage
step = len(contour_points) // N
line_segments = []

for i in range(0, len(contour_points), step):
    segment = contour_points[i:i + step]

    if len(segment) < 2:
        continue

    # Fit a line to the segment
    [vx, vy, x, y] = cv2.fitLine(segment, cv2.DIST_L2, 0, 0.01, 0.01)
    direction = np.array([vx, vy]).reshape(-1)
    origin = np.array([x, y]).reshape(-1)

    # Project endpoints of segment onto line to get line endpoints
    proj = np.dot(segment - origin, direction)
    min_proj = np.min(proj)
    max_proj = np.max(proj)

    pt1 = origin + direction * min_proj
    pt2 = origin + direction * max_proj
    length = np.linalg.norm(pt2 - pt1)

    line_segments.append((length, pt1, pt2))

# Take 4 longest line segments
line_segments.sort(key=lambda x: x[0], reverse=True)
top_lines = line_segments[:4]

# Colors for lines
colors = [(255, 0, 255), (0, 255, 255), (0, 128, 255), (0, 255, 0)]
extension = 60  # pixels

for idx, (_, pt1, pt2) in enumerate(top_lines):
    # Extend the line in both directions
    dir_vector = pt2 - pt1
    dir_unit = dir_vector / np.linalg.norm(dir_vector)
    pt1_ext = pt1 - dir_unit * extension
    pt2_ext = pt2 + dir_unit * extension

    cv2.line(canvas,
             tuple(np.intp(pt1_ext)),
             tuple(np.intp(pt2_ext)),
             color=colors[idx % 4],
             thickness=2)

# Show result
#plt.figure(figsize=(10, 10))
plt.imshow(canvas)
plt.title("Red: Original Contour | Colored: 4 Longest Lines (Extended)")
plt.axis('off')
plt.show()