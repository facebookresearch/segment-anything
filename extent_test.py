import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import argparse

def run_sam(predictor, img, yolo_box):
    predictor.set_image(img)
    h, w, _ = img.shape
    xc, yc, w_rel, h_rel = yolo_box
    w_abs = w_rel * w
    h_abs = h_rel * h
    x0 = int(xc * w - w_abs / 2)
    y0 = int(yc * h - h_abs / 2)
    x1 = int(x0 + w_abs)
    y1 = int(y0 + h_abs)
    box = np.array([x0, y0, x1, y1])
    masks, scores, _ = predictor.predict(box=box, multimask_output=True)
    return masks[np.argmax(scores)]

def get_extreme_points(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea).squeeze()
    if contour.ndim != 2:  # edge case: only one point
        return []
    sum_pts = contour[:, 0] + contour[:, 1]
    diff_pts = contour[:, 0] - contour[:, 1]
    left_top = contour[np.argmin(sum_pts)]
    right_bottom = contour[np.argmax(sum_pts)]
    left_bottom = contour[np.argmin(diff_pts)]
    right_top = contour[np.argmax(diff_pts)]
    return [tuple(left_top), tuple(right_top), tuple(left_bottom), tuple(right_bottom)]

def draw_points(image, points, color=(255, 0, 0), radius=8):
    img_copy = image.copy()
    for pt in points:
        cv2.circle(img_copy, pt, radius, color, -1)
    return img_copy

def load_yolo_annotations(file_path):
    annotations = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip malformed lines
            class_id, x_center, y_center, width, height = map(float, parts)
            annotations.append({
                "class_id": int(class_id),
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height
            })
    return annotations


def main(args):

    # load annotations
    annotations = load_yolo_annotations(args.yolo)
    an = [ann for ann in annotations if ann["class_id"] == args.yolo_id]
    ab = an[0]
    print(ab['x_center'])

    # Bounding Box (YOLO format: [xc, yc, w, h])
    yolo_box = [ab['x_center'], ab['y_center'], ab['width'], ab['height']]

    # Load SAM Model
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)

    # Load Image
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess Image: Grayscale + Contrast Increase
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    contrast_enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)  # 50% contrast boost
    preprocessed_image = cv2.cvtColor(contrast_enhanced, cv2.COLOR_GRAY2RGB)

    # Run SAM on Original
    #predictor_original = SamPredictor(sam)
    mask_original = run_sam(predictor, image, yolo_box)
    points_original = get_extreme_points(mask_original)
    image_original_with_dots = draw_points(image, points_original)

    # Run SAM on Preprocessed
    #predictor_pre = SamPredictor(sam)
    mask_preprocessed = run_sam(predictor, preprocessed_image, yolo_box)
    points_pre = get_extreme_points(mask_preprocessed)
    image_preprocessed_with_dots = draw_points(preprocessed_image, points_pre)

    # Plot Results Side-by-Side
    fig, axs = plt.subplots(1, 2, figsize=(16, 10))

    axs[0].imshow(image_original_with_dots)
    axs[0].imshow(mask_original, alpha=0.5, cmap='jet')
    axs[0].set_title("Original Image with Mask + Dots")
    axs[0].axis('off')

    axs[1].imshow(image_preprocessed_with_dots)
    axs[1].imshow(mask_preprocessed, alpha=0.5, cmap='jet')
    axs[1].set_title("Preprocessed Image with Mask + Dots")
    axs[1].axis('off')

    plt.tight_layout()
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Image file")
parser.add_argument("--yolo", type=str, required=True, help="Yolo prediction file")
parser.add_argument("--yolo-id", type=int, required=True, help="Yolo prediction identifier")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
