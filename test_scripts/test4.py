# Extended box around the mask

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor

def extend_line(p1, p2, length=100):
    if p1 is None or p2 is None:
        raise ValueError("extend_line received None as input point.")

    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)

    direction = p2 - p1
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Cannot extend a line with zero length (p1 == p2).")

    direction = direction / norm  # unit vector
    new_p1 = (p1 - direction * length).astype(int)
    new_p2 = (p2 + direction * length).astype(int)
    return tuple(new_p1), tuple(new_p2)

def draw_extended_box(mask, image, extension=100, color=(255, 0, 0), thickness=2):
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Get contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    # Get the largest contour
    largest = max(contours, key=cv2.contourArea)

    # Get minimum area rotated rectangle
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)  # 4 points in float32

    # Extend each of the 4 edges
    extended_image = image.copy()
    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        ext_p1, ext_p2 = extend_line(p1, p2, length=extension)
        cv2.line(extended_image, ext_p1, ext_p2, color=color, thickness=thickness)

    return extended_image, box

def draw_extended_box2(mask, masked_image, extension=100, color=(255, 0, 0), thickness=2):
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Get contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    # Get the largest contour
    largest = max(contours, key=cv2.contourArea)

    # Get minimum area rotated rectangle
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)  # 4 points in float32

    # Extend each of the 4 edges
    for i in range(4):
        p1 = box[i]
        p2 = box[(i + 1) % 4]
        ext_p1, ext_p2 = extend_line(p1, p2, length=extension)
        cv2.line(masked_image, ext_p1, ext_p2, color=color, thickness=thickness)

    return extended_image, box

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
masks, scores, _ = predictor.predict(box=box, multimask_output=False)
best_mask = masks[np.argmax(scores)]

# Isolate object
masked_image = np.zeros_like(image)
for c in range(3):
    masked_image[:, :, c] = image[:, :, c] * best_mask

# Convert mask to uint8
mask_uint8 = (best_mask * 255).astype(np.uint8)

# Extract (x, y) coordinates of mask pixels
ys, xs = np.where(best_mask)
points = np.column_stack((xs, ys))  # shape: (N, 2)

# Perform PCA to get orientation
mean = np.mean(points, axis=0)
centered_points = points - mean

# Compute covariance matrix and eigenvectors
cov = np.cov(centered_points, rowvar=False)
eigvals, eigvecs = np.linalg.eigh(cov)

# Get the principal component (largest eigenvector)
principal_axis = eigvecs[:, np.argmax(eigvals)]

# Create a long line segment along the principal axis
line_length = max(image.shape[:2])  # long enough to cross mask
pt1 = (mean + principal_axis * line_length).astype(int)
pt2 = (mean - principal_axis * line_length).astype(int)

#quad = mask_to_quad_polygon(best_mask)

# Draw on image (optional)
#for i in range(4):
#    pt1 = tuple(quad[i])
#    pt2 = tuple(quad[(i + 1) % 4])
#    cv2.line(masked_image, pt1, pt2, color=(255, 0, 0), thickness=2)


# Show result
#plt.figure(figsize=(10, 10))
#plt.imshow(masked_image)
#plt.title("Masked Object with Red Dots on Corners")
#plt.axis('off')
#plt.show()

# Assume `image` is the original RGB image and `best_mask` is from SAM
#extended_image, quad = draw_extended_box(best_mask, image, extension=100)
extended_image, quad = draw_extended_box2(best_mask, masked_image, extension=100)

# Show it
import matplotlib.pyplot as plt
plt.imshow(extended_image)
plt.title("Extended Box Around Mask")
plt.axis('off')
plt.show()