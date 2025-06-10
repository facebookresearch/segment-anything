import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# Load image
image_path = "grubber_ferrule.jpg"  # Path to your input image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load SAM model

# vit h model
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

# vit b model
#sam_checkpoint = "sam_vit_b_01ec64.pth"
#model_type = "vit_b"

#sam_checkpoint = "sam_vit_l_0b3195.pth"
#model_type = "vit_l"


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

# Get segmentation
masks, scores, _ = predictor.predict(box=box, multimask_output=True)
best_mask = masks[np.argmax(scores)]

# Isolate object
masked_image = np.zeros_like(image)
for c in range(3):
    masked_image[:, :, c] = image[:, :, c] * best_mask

# Get bounding box of mask
#ys, xs = np.where(best_mask)
#min_x, max_x = np.min(xs), np.max(xs)
#min_y, max_y = np.min(ys), np.max(ys)

# Convert mask to uint8 for contour detection
mask_uint8 = (best_mask * 255).astype(np.uint8)

# Find contours
contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get minimum area rectangle (rotated bounding box)
rot_rect = cv2.minAreaRect(contours[0])  # ((cx, cy), (w, h), angle)
box_points = cv2.boxPoints(rot_rect)  # Returns 4 corner points
box_points = np.intp(box_points)  # Convert to integer coords


# Define corner points
#top_left = (min_x, min_y)
#top_right = (max_x, min_y)
#bottom_right = (max_x, max_y)
#bottom_left = (min_x, max_y)

# Draw red dots at the corners
dot_radius = 6
dot_color = (255, 0, 0)  # Red in RGB
dot_thickness = -1  # Filled circle

#corner_points = [top_left, top_right, bottom_right, bottom_left]
corner_points = [tuple(pt) for pt in box_points]

for pt in corner_points:
    cv2.circle(masked_image, pt, radius=dot_radius, color=dot_color, thickness=dot_thickness)

# Draw red rotated box outline
cv2.polylines(masked_image, [box_points], isClosed=True, color=(255, 0, 0), thickness=2)

# Show result
plt.figure(figsize=(10, 10))
plt.imshow(masked_image)
plt.title("Masked Object with Red Dots on Corners")
plt.axis('off')
plt.show()