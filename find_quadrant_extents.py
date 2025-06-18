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
        return [], None, None

    contour = max(contours, key=cv2.contourArea).squeeze()
    if contour.ndim != 2 or len(contour) < 4:
        return [], None, None

    pts = contour.astype(np.float32)
    center = np.mean(pts, axis=0)
    rel_pts = pts - center

    # Use cv2.fitEllipse to get stable orientation
    if len(contour) < 5:
        return [], None, None  # fitEllipse requires at least 5 points
    ellipse = cv2.fitEllipse(contour)
    angle = np.deg2rad(ellipse[2])  # convert to radians

    # Rotate points so major axis aligns with x-axis
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    rotated_pts = rel_pts @ R.T

    angles = np.arctan2(rotated_pts[:, 1], rotated_pts[:, 0])
    angles = (angles + 2 * np.pi) % (2 * np.pi)

    quadrant_points = [[] for _ in range(4)]
    for i, ang in enumerate(angles):
        sector = int(ang // (np.pi / 2)) % 4
        quadrant_points[sector].append(i)

    result_pts = []
    for sector in quadrant_points:
        if sector:
            dists = np.linalg.norm(rotated_pts[sector], axis=1)
            idx = sector[np.argmax(dists)]
            result_pts.append(tuple(pts[idx]))

    if len(result_pts) < 4:
        used = set(tuple(p) for p in result_pts)
        dists = np.linalg.norm(rotated_pts, axis=1)
        remaining = sorted([(i, d) for i, d in enumerate(dists) if tuple(pts[i]) not in used], key=lambda x: -x[1])
        for i, _ in remaining:
            pt = tuple(pts[i])
            if pt not in result_pts:
                result_pts.append(pt)
            if len(result_pts) == 4:
                break

    return result_pts, tuple(center.astype(int)), angle

def sort_points_clockwise(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    top_left = pts[np.argmin(s)]
    bottom_right = pts[np.argmax(s)]
    top_right = pts[np.argmin(diff)]
    bottom_left = pts[np.argmax(diff)]
    ordered = [top_left, top_right, bottom_right, bottom_left]
    return [tuple(map(int, pt)) for pt in ordered]

def draw_directional_lines(image, points, length=50, color=(0, 255, 0), thickness=2):
    img_copy = image.copy()
    for i, pt in enumerate(points):
        adj_indices = [j for j in range(4) if j != i]
        adj_vecs = [np.array(points[j]) - np.array(pt) for j in adj_indices]
        avg_vec = -np.mean(adj_vecs, axis=0)
        away_vec = avg_vec / (np.linalg.norm(avg_vec) + 1e-6) * length
        pt_start = tuple(map(int, pt))
        pt_end = tuple(map(int, (np.array(pt) + away_vec).astype(int)))
        cv2.line(img_copy, pt_start, pt_end, color, thickness)
    return img_copy

def draw_points(image, points, color=(255, 0, 0), radius=8):
    img_copy = image.copy()
    for pt in points:
        cv2.circle(img_copy, pt, radius, color, -1)
    return img_copy

def draw_rotated_quadrants(image, center, angle_rad, length=500, color=(0, 255, 255), thickness=2):
    img_copy = image.copy()
    cx, cy = map(int, center)
    base_angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    for a in base_angles:
        total_angle = a + angle_rad
        dx = int(np.cos(total_angle) * length)
        dy = int(np.sin(total_angle) * length)
        end_point = (cx + dx, cy + dy)
        cv2.line(img_copy, (cx, cy), end_point, color, thickness)
    return img_copy

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
    extreme_points, center, angle = get_extreme_points(mask_original)
    if not extreme_points:
        print("No extreme points found.")
        return

    points_original = sort_points_clockwise(extreme_points)
    image_with_quadrants = draw_rotated_quadrants(image, center, angle)
    image_original_with_dots = draw_points(image_with_quadrants, points_original)
    image_original_with_dots = draw_directional_lines(image_original_with_dots, points_original)

    for i, pt in enumerate(points_original):
        cv2.putText(image_original_with_dots, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    axs.imshow(image_original_with_dots)
    axs.imshow(mask_original, alpha=0.5, cmap='jet')
    axs.set_title("Oriented Quadrants + Extreme Points")
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