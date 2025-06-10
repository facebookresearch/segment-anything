# just find the ferrule, and outline it

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
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# Set the image
predictor.set_image(image)

# Your YOLO-format bounding box [x_center, y_center, width, height] (all relative)
#yolo_box = [0.556217, 0.528522, 0.263889, 0.199901]
yolo_box = [0.305886, 0.578497, 0.102513, 0.0751488] # Ferrule
#yolo_box = [0.612269, 0.582713, 0.270833, 0.221974] # Ball

# Convert YOLO box to absolute pixel coordinates (x0, y0, x1, y1)
img_h, img_w, _ = image.shape
x_center, y_center, w_rel, h_rel = yolo_box

w = w_rel * img_w
h = h_rel * img_h
x0 = int((x_center * img_w) - w / 2)
y0 = int((y_center * img_h) - h / 2)
x1 = int(x0 + w)
y1 = int(y0 + h)

box = np.array([x0, y0, x1, y1])

# Predict segmentation mask
masks, scores, logits = predictor.predict(box=box, multimask_output=True)

# Select the best mask (highest score)
best_mask = masks[np.argmax(scores)]

# Show result
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.imshow(best_mask, alpha=0.6, cmap='jet')
plt.title("Segmented Primary Object")
plt.axis('off')
plt.show()