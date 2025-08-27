import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# Load image
image_path = "../test_data/grubber_ferrule.jpg"  # Path to your input image
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
# yolo_box = [0.556217, 0.528522, 0.263889, 0.199901]
yolo_box = [0.305886, 0.578497, 0.102513, 0.0751488]  # Ferrule
# yolo_box = [0.612269, 0.582713, 0.270833, 0.221974]  # Ball

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

# Convert mask to uint8 for OpenCV contour finding
mask_uint8 = (best_mask * 255).astype(np.uint8)

# Find contours
contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Assuming largest contour corresponds to object
contour = max(contours, key=cv2.contourArea).squeeze()  # shape (N, 2)

# Find extreme points based on sums/differences
sum_pts = contour[:, 0] + contour[:, 1]
diff_pts = contour[:, 0] - contour[:, 1]

left_top = contour[np.argmin(sum_pts)]
right_bottom = contour[np.argmax(sum_pts)]
left_bottom = contour[np.argmin(diff_pts)]
right_top = contour[np.argmax(diff_pts)]

# Draw red dots on these points on a copy of the original image
image_with_dots = image.copy()
red_color = (255, 0, 0)  # RGB red
radius = 8
thickness = -1  # filled circle

for point in [left_top, right_top, left_bottom, right_bottom]:
    cv2.circle(image_with_dots, tuple(point), radius, red_color, thickness)

# Show result
#plt.figure(figsize=(10, 10))
plt.imshow(image_with_dots)
plt.imshow(best_mask, alpha=0.6, cmap='jet')
plt.title("Segmented Object with Extreme Points")
plt.axis('off')
plt.show()
