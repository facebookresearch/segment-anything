import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
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

'''
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

def get_extreme_points3(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)  # returns 4 points as float32
    box = np.int8(box)
    return [tuple(pt) for pt in box]
'''

def get_extreme_points(mask):
    # Preprocess mask
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea).squeeze()
    if contour.ndim != 2 or len(contour) < 4:
        return []

    pts = contour.astype(np.float32)
    center = np.mean(pts, axis=0)

    # PCA alignment
    cov = np.cov((pts - center).T)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(-eigvals)
    principal_axes = eigvecs[:, order]

    # Transform into PCA space
    aligned = (pts - center) @ principal_axes

    # Find extremes in aligned space
    left_idx   = np.argmin(aligned[:, 0])
    right_idx  = np.argmax(aligned[:, 0])
    top_idx    = np.argmin(aligned[:, 1])
    bottom_idx = np.argmax(aligned[:, 1])

    indices = [left_idx, right_idx, top_idx, bottom_idx]
    unique_indices = list(dict.fromkeys(indices))  # Remove duplicates if any

    result_pts = [tuple(pts[i]) for i in unique_indices]

    # If fewer than 4 points (e.g. perfect symmetry), pad with farthest unused points
    if len(result_pts) < 4:
        used = set(unique_indices)
        dists = np.linalg.norm(aligned, axis=1)
        remaining = sorted([(i, d) for i, d in enumerate(dists) if i not in used], key=lambda x: -x[1])
        for i, _ in remaining:
            result_pts.append(tuple(pts[i]))
            if len(result_pts) == 4:
                break

    result_pts = [tuple(map(int, pts[i])) for i in unique_indices]

'''
def get_extreme_points(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea).squeeze()
    if contour.ndim != 2 or len(contour) < 4:
        return []

    # Get convex hull
    hull = cv2.convexHull(contour)

    # Find extreme points directly in image coordinates
    leftmost  = tuple(hull[hull[:, :, 0].argmin()][0])
    rightmost = tuple(hull[hull[:, :, 0].argmax()][0])
    topmost   = tuple(hull[hull[:, :, 1].argmin()][0])
    bottommost= tuple(hull[hull[:, :, 1].argmax()][0])

    return [leftmost, topmost, rightmost, bottommost]
'''

def sort_points_clockwise(pts):
    """
    Sort points in clockwise order starting from the top-left.
    Assumes pts is a list of 4 (x, y) tuples.
    """
    pts = np.array(pts)
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
    sort_idx = np.argsort(angles)
    return [tuple(pts[i]) for i in sort_idx]

def draw_directional_lines(image, points, length=50, color=(0, 255, 0), thickness=2):
    """
    Draw lines from each point away from the direction of its adjacent points.
    Assumes 4 points ordered as: [top-left, top-right, bottom-left, bottom-right]
    """
    img_copy = image.copy()
    for i, pt in enumerate(points):
        # Get adjacent indices in the cyclic order of 4
        adj_indices = [j for j in range(4) if j != i]
        adj_vecs = []

        for j in adj_indices:
            adj_vec = np.array(points[j]) - np.array(pt)
            adj_vecs.append(adj_vec)

        # Average direction towards neighbors
        avg_vec = np.mean(adj_vecs, axis=0)

        # Reverse direction to point away
        away_vec = -avg_vec
        away_vec = away_vec / (np.linalg.norm(away_vec) + 1e-6) * length  # Normalize and scale

        pt_start = tuple(map(int, pt))
        pt_end = tuple(map(int, (np.array(pt) + away_vec).astype(int)))

        cv2.line(img_copy, pt_start, pt_end, color, thickness)

    return img_copy

def draw_points(image, points, color=(255, 0, 0), radius=8):
    img_copy = image.copy()
    for pt in points:
        cv2.circle(img_copy, pt, radius, color, -1)
    return img_copy


def compute_away_vectors(points):
    """
    Compute outward unit vectors away from adjacent points for each point.
    """
    vectors = []
    for i, pt in enumerate(points):
        adj_indices = [j for j in range(4) if j != i]
        adj_vecs = []

        for j in adj_indices:
            adj_vec = np.array(points[j]) - np.array(pt)
            adj_vecs.append(adj_vec)

        # Average vector toward neighbors, then negate and normalize
        avg_vec = -np.mean(adj_vecs, axis=0)
        norm = np.linalg.norm(avg_vec)
        unit_vec = avg_vec / norm if norm != 0 else avg_vec
        vectors.append(unit_vec)
    return vectors


def move_points_outside_mask(points, directions, mask, max_iters=100, step_size=2):
    points = np.array(points, dtype=np.float32)
    h, w = mask.shape
    mask_edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)

    def polygon_intersects_mask(polygon, mask_edges):
        polygon = np.array(polygon, dtype=np.int32)
        temp_mask = np.zeros_like(mask_edges)
        cv2.polylines(temp_mask, [polygon], isClosed=True, color=255, thickness=1)
        intersection = cv2.bitwise_and(temp_mask, mask_edges)
        return np.any(intersection)

    for _ in range(max_iters):
        polygon = [tuple(pt) for pt in points]
        if not polygon_intersects_mask(polygon, mask_edges):
            break  # Stop moving if no intersection
        for i in range(len(points)):
            points[i] += directions[i] * step_size

    return [tuple(pt.astype(int)) for pt in points]

def draw_polygon(image, points, color=(0, 0, 255), thickness=2):
    img_copy = image.copy()
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_copy, [pts], isClosed=True, color=color, thickness=thickness)
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
    points_original = sort_points_clockwise(get_extreme_points(mask_original))

    #image_original_with_dots = draw_points(image, points_original)
    image_original_with_dots = draw_points(image, points_original)
    # Compute directions
    directions = compute_away_vectors(points_original)

    # Move points outward until no intersection
    points_moved = move_points_outside_mask(points_original, directions, mask_original)

    # Draw polygon
    image_original_with_dots = draw_polygon(image_original_with_dots, points_moved)
    image_original_with_dots = draw_directional_lines(image_original_with_dots, points_original)

    for i, pt in enumerate(points_original):
        cv2.putText(image_original_with_dots, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    # Run SAM on Preprocessed
    #predictor_pre = SamPredictor(sam)
    mask_preprocessed = run_sam(predictor, preprocessed_image, yolo_box)
    points_pre = get_extreme_points(mask_preprocessed)
    #image_preprocessed_with_dots = draw_points(preprocessed_image, points_pre)
    image_preprocessed_with_dots = draw_points(preprocessed_image, points_pre)
    image_preprocessed_with_dots = draw_directional_lines(image_preprocessed_with_dots, points_pre)

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
