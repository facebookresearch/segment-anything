# Polygon representation of the mask

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
#yolo_box = [0.291832, 0.549231, 0.118056, 0.129712] # rich ferrule
#yolo_box = [0.556217, 0.528522, 0.263889, 0.199901]  # grubber ball
yolo_box = [0.305886, 0.578497, 0.102513, 0.0751488] # grubber ferrule

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

# Draw the mask contours as red polygons
cv2.drawContours(canvas, contours, -1, color=(255, 0, 0), thickness=2)  # Red color (BGR)

# Show the result
#plt.figure(figsize=(10, 10))

plt.imshow(canvas)
plt.title("Polygon Representation of Mask")
plt.axis('off')
plt.show()