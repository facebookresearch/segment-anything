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
cv2.polylines(canvas, [approx_polygon], isClosed=True, color=(0, 0, 255), thickness=2)


### Trying cornerSubPix

# Convert original image or mask to grayscale for corner refinement
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Create a mask to limit the search area to the object region
mask_for_corners = (mask_uint8 > 0).astype(np.uint8)

# Define criteria for termination of the cornerSubPix algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)

# Convert approximate polygon points to float32 for cornerSubPix
initial_corners = approx_polygon.reshape(-1, 1, 2).astype(np.float32)

# Apply cornerSubPix
refined_corners = cv2.cornerSubPix(
    gray, initial_corners, winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria
)

# Draw refined polygon in green for comparison
refined_corners_int = np.intp(refined_corners)
cv2.polylines(canvas, [refined_corners_int], isClosed=True, color=(0, 255, 0), thickness=2)

### End cornerSubPix

# Show the result
#plt.figure(figsize=(10, 10))
plt.imshow(canvas)
plt.title("Red: Original Contours | Blue: Simplified Polygon (cone-like)")
plt.axis('off')
plt.show()
