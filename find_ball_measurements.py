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


def load_yolo_annotations(file_path):
    annotations = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, width, height = map(float, parts)
            annotations.append({
                "class_id": int(class_id),
                "x_center": x_center,
                "y_center": y_center,
                "width": width,
                "height": height
            })
    return annotations

def draw_enclosing_circle(image, mask, real_world_diameter_in=1.680):
    """
    Draws a minimum enclosing circle around the largest contour in the mask and calculates real-world radius.

    Args:
        image (np.ndarray): Original RGB image.
        mask (np.ndarray): Binary mask (bool or 0/1 array).
        real_world_diameter_in (float): Actual diameter of the ball in inches.

    Returns:
        image_with_circle (np.ndarray): Image with the enclosing circle drawn.
        circle_info (dict): Dictionary with pixel and real-world center/radius.
    """
    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found in mask.")
        return image.copy(), None

    largest_contour = max(contours, key=cv2.contourArea)
    (x, y), radius_px = cv2.minEnclosingCircle(largest_contour)
    center = (int(x), int(y))
    radius_px = float(radius_px)
    diameter_px = 2 * radius_px
    inch_per_pixel = real_world_diameter_in / diameter_px
    radius_in = radius_px * inch_per_pixel

    image_with_circle = image.copy()
    cv2.circle(image_with_circle, center, int(radius_px), (255, 0, 0), 3)  # Blue circle

    return image_with_circle, {
        "center_px": center,
        "radius_px": radius_px,
        "radius_in": radius_in
    }

def convert_pixels_to_inches(circle_info, pixel_length):
    """
    Converts a pixel measurement to inches using calibration from a reference circle.

    Args:
        circle_info (dict): Output from `draw_enclosing_circle`, must contain "radius_px" and "radius_in".
        pixel_length (float): Length in pixels to convert to inches.

    Returns:
        float: Equivalent length in inches.
    """
    if not circle_info or "radius_px" not in circle_info or "radius_in" not in circle_info:
        raise ValueError("Invalid circle_info provided.")

    # Calculate inch-per-pixel based on the known object
    inch_per_pixel = circle_info["radius_in"] / circle_info["radius_px"]
    return pixel_length * inch_per_pixel

def main(args):
    annotations = load_yolo_annotations(args.yolo)
    an = [ann for ann in annotations if ann["class_id"] == args.yolo_id]
    ab = an[0]
    yolo_box = [ab['x_center'], ab['y_center'], ab['width'], ab['height']]

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_original = run_sam(predictor, image, yolo_box)

    # Draw enclosing circle
    image_with_circle, circle_info = draw_enclosing_circle(image, mask_original)

    length_in_pixels = 233
    length_in_inches = convert_pixels_to_inches(circle_info, length_in_pixels)
    print(f"Length: {length_in_pixels}px ≈ {length_in_inches:.3f}\"")

    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    axs.imshow(image_with_circle)
    axs.imshow(mask_original, alpha=0.5, cmap='jet')
    if circle_info:
        _, _, radius = circle_info
        axs.set_title(f"Ball measurements: radius = {radius}px")
    else:
        axs.set_title("Ball measurements: no circle found")
    axs.axis('off')
    plt.tight_layout()
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Image file")
parser.add_argument("--yolo", type=str, required=True, help="Yolo prediction file")
parser.add_argument("--yolo-id", type=int, required=True, help="Yolo prediction identifier")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)