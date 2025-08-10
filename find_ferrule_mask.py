import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from segment_anything import sam_model_registry, SamPredictor

def load_yolo_annotations(fp):
    anns = []
    with open(fp) as f:
        for line in f:
            cid, xc, yc, w, h = map(float, line.split())
            anns.append(dict(
                class_id=int(cid),
                x_center=xc,
                y_center=yc,
                width=w,
                height=h
            ))
    return anns

def main(args):
    # Hardcoded class_id (change here if needed)
    target_class_id = 1  # 0 = ball, 1 = ferrule

    # Load image
    image_path = args.image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load annotations
    print("Loading annotations")
    annotations = load_yolo_annotations(args.yolo)

    # Pick annotation for the hardcoded class
    target_ann = next((ann for ann in annotations if ann["class_id"] == target_class_id), None)
    if target_ann is None:
        raise ValueError(f"No annotation found for class_id {target_class_id} in {args.yolo}")

    # Load SAM model
    sam_checkpoint = "sam_vit_h_4b8939.pth"  # adjust path if needed
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)

    # Set the image
    predictor.set_image(image)

    # Convert YOLO to absolute pixel coords
    img_h, img_w, _ = image.shape
    x_center, y_center = target_ann["x_center"], target_ann["y_center"]
    w_rel, h_rel = target_ann["width"], target_ann["height"]
    w = w_rel * img_w
    h = h_rel * img_h
    x0 = int((x_center * img_w) - w / 2)
    y0 = int((y_center * img_h) - h / 2)
    x1 = int(x0 + w)
    y1 = int(y0 + h)

    box_abs = np.array([x0, y0, x1, y1])

    # Predict segmentation mask
    masks, scores, _ = predictor.predict(box=box_abs, multimask_output=True)

    # Take the best mask
    best_mask = masks[np.argmax(scores)]
    mask_uint8 = (best_mask * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # White background
    mask_img = np.ones_like(mask_uint8) * 255

    # Fill mask with black
    cv2.drawContours(mask_img, contours, -1, color=0, thickness=-1)

    # Save or show
    if args.write:
        out_path = "mask.png"
        cv2.imwrite(out_path, mask_img)
        print(f"Saved mask to {out_path}")
    else:
        plt.figure(figsize=(8, 8))
        plt.imshow(mask_img, cmap='gray')
        plt.title(f"Class {target_class_id} Mask")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--yolo", required=True, help="Path to YOLO annotation file")
    p.add_argument("--write", action="store_true", help="Save mask instead of displaying it")
    main(p.parse_args())
