import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import argparse

def visualize_step(image, points, movement_vectors, edge_flags, reversal_flags, step_num):
    img = image.copy()
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw mask in background
    ax.imshow(img, alpha=0.4)

    # Draw polygon
    poly_pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [poly_pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # Draw points and vectors
    for i, pt in enumerate(points):
        color = 'green'
        if edge_flags[i] or edge_flags[(i - 1) % len(points)]:
            color = 'red'
        if reversal_flags[i] == -1:
            color = 'yellow'

        ax.plot(pt[0], pt[1], 'o', color=color, markersize=8)

        # Draw movement arrow
        mv = movement_vectors[i]
        if np.linalg.norm(mv) > 0:
            ax.arrow(
                pt[0], pt[1], mv[0]*5, mv[1]*5,
                head_width=3, head_length=5, fc='blue', ec='blue'
            )

    ax.set_title(f"Iteration {step_num}")
    ax.axis('off')
    plt.tight_layout()
    plt.pause(0.5)  # Pause to allow rendering without blocking
    plt.close()


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

def get_extreme_points(mask):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    contour = max(contours, key=cv2.contourArea).squeeze()
    if contour.ndim != 2 or len(contour) < 4:
        return []

    pts = contour.astype(np.float32)
    mean = np.mean(pts, axis=0)

    # PCA to get orientation
    cov = np.cov((pts - mean).T)
    eigvals, eigvecs = np.linalg.eig(cov)
    order = np.argsort(-eigvals)
    principal_axes = eigvecs[:, order]

    # Project to PCA frame
    centered = pts - mean
    projected = centered @ principal_axes

    # Divide into quadrants
    q1, q2, q3, q4 = [], [], [], []
    for i, pt in enumerate(projected):
        x, y = pt
        if x <= 0 and y <= 0:
            q1.append(i)
        elif x > 0 and y <= 0:
            q2.append(i)
        elif x > 0 and y > 0:
            q3.append(i)
        elif x <= 0 and y > 0:
            q4.append(i)

    def get_farthest(indices):
        if not indices:
            return None
        dists = np.linalg.norm(projected[indices], axis=1)
        far_idx = indices[np.argmax(dists)]
        return tuple(contour[far_idx])

    pts_out = []
    for q in [q1, q2, q3, q4]:
        pt = get_farthest(q)
        if pt is not None:
            pts_out.append(pt)

    # Fallback if < 4 unique quadrants found
    while len(pts_out) < 4:
        # Add remaining farthest points (unique)
        remaining = set(map(tuple, contour)) - set(pts_out)
        if not remaining:
            break
        far = max(remaining, key=lambda p: np.linalg.norm(p - mean))
        pts_out.append(tuple(far))

    return pts_out

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

def compute_direction_vectors(points):
    """
    For each point, compute:
    - 'prev': away from the previous point
    - 'next': away from the next point
    - 'avg': away from both neighbors (average)

    Returns: list of dicts of vectors for each point
    """
    n = len(points)
    directions = []

    for i in range(n):
        curr = np.array(points[i], dtype=np.float32)
        prev = np.array(points[(i - 1) % n], dtype=np.float32)
        next = np.array(points[(i + 1) % n], dtype=np.float32)

        to_prev = prev - curr
        to_next = next - curr

        # Vectors away
        away_prev = -to_prev
        away_next = -to_next
        away_avg = -(to_prev + to_next)

        def normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v

        directions.append({
            "prev": normalize(away_prev),
            "next": normalize(away_next),
            "avg": normalize(away_avg),
        })

    return directions

def compute_outward_normals(points):
    """
    Compute outward unit normal vector at each polygon vertex.
    Assumes points are ordered clockwise or counterclockwise.
    """
    n = len(points)
    points = np.array(points, dtype=np.float32)

    # Determine polygon winding (clockwise or counterclockwise)
    def polygon_area(pts):
        # Shoelace formula
        return 0.5 * np.sum(
            pts[:-1,0]*pts[1:,1] - pts[1:,0]*pts[:-1,1]
        )
    pts_closed = np.vstack([points, points[0]])
    area = polygon_area(pts_closed)
    clockwise = area < 0  # Negative area means clockwise

    normals = []
    for i in range(n):
        p_prev = points[(i - 1) % n]
        p_curr = points[i]
        p_next = points[(i + 1) % n]

        # Edge vectors
        edge1 = p_curr - p_prev
        edge2 = p_next - p_curr

        # Normalize edges
        edge1 /= (np.linalg.norm(edge1) + 1e-8)
        edge2 /= (np.linalg.norm(edge2) + 1e-8)

        # Normals to edges (rotate edge vector by 90 degrees)
        # For 2D vector (x, y), normal can be (y, -x) or (-y, x)
        if clockwise:
            normal1 = np.array([edge1[1], -edge1[0]])
            normal2 = np.array([edge2[1], -edge2[0]])
        else:
            normal1 = np.array([-edge1[1], edge1[0]])
            normal2 = np.array([-edge2[1], edge2[0]])

        # Average normals at vertex and normalize
        normal = (normal1 + normal2) / 2
        normal /= (np.linalg.norm(normal) + 1e-8)

        normals.append(normal)

    return normals


def compute_internal_angles(points):
    """
    Compute the internal angle (in radians) at each point in a polygon.
    Returns a list of angles [angle_0, angle_1, ..., angle_n-1]
    """
    n = len(points)
    angles = []

    for i in range(n):
        prev_pt = np.array(points[(i - 1) % n])
        curr_pt = np.array(points[i])
        next_pt = np.array(points[(i + 1) % n])

        v1 = prev_pt - curr_pt
        v2 = next_pt - curr_pt

        # Normalize
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)

        # Angle between vectors
        dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(dot)
        angles.append(angle)

    return angles  # In radians

'''
def move_points_adaptively(points, directions, mask, max_iters=500, step_size=2):
    points = np.array(points, dtype=np.float32)
    h, w = mask.shape
    mask_edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)

    def edge_intersects_mask(p1, p2):
        temp_mask = np.zeros_like(mask_edges)
        cv2.line(temp_mask, tuple(p1.astype(int)), tuple(p2.astype(int)), 255, thickness=1)
        intersection = cv2.bitwise_and(temp_mask, mask_edges)
        return np.any(intersection)

    n = len(points)

    for iteration in range(max_iters):
        edge_flags = [False] * n

        angles = compute_internal_angles(points)
        angle_weights = [(np.pi - a) / np.pi for a in angles]  # Normalize to [0, 1]; sharp = higher weight

        # Step 1: Identify which edges intersect the mask
        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            if edge_intersects_mask(p1, p2):
                edge_flags[i] = True

        if not any(edge_flags):
            break  # All edges are clear

        movement_vectors = [np.zeros(2, dtype=np.float32) for _ in range(n)]

        # Step 2: Calculate movement vectors for intersecting edges
        for i in range(n):
            if not edge_flags[i]:
                continue

            i_next = (i + 1) % n

            # Weight movement by how "corner-like" the point is
            movement_vectors[i] += directions[i]["next"] * angle_weights[i]
            movement_vectors[i_next] += directions[i_next]["prev"] * angle_weights[i_next]

        # Step 3: Normalize and apply movement
        for i in range(n):
            mv = movement_vectors[i]
            if np.linalg.norm(mv) > 0:
                # Decay step size over time to reduce jitter
                scaled_step = step_size * (0.9 ** iteration)

                # Cap movement to avoid flying off
                mv = mv / np.linalg.norm(mv) * min(scaled_step, step_size)
                mv *= angle_weights[i]

                points[i] += mv

    return [tuple(pt.astype(int)) for pt in points]
'''

def move_points_adaptively(points, mask, image=None, max_iters=200, step_size=2):
    points = np.array(points, dtype=np.float32)
    h, w = mask.shape
    mask_edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)

    def edge_intersects_mask(p1, p2, mask):
        temp_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.line(temp_mask, tuple(p1.astype(int)), tuple(p2.astype(int)), 255, thickness=1)
        intersection = cv2.bitwise_and(temp_mask, (mask * 255).astype(np.uint8))
        return np.any(intersection)

    n = len(points)
    intersect_counts = [0] * n
    reversal_flags = [1] * n  # 1 for normal, -1 for reversed

    for iteration in range(max_iters):
        print("iteration: {0}".format(iteration))
        edge_flags = [False] * n
        angles = compute_internal_angles(points)
        angle_weights = [(np.pi - a) / np.pi for a in angles]
        #directions = compute_direction_vectors(points)
        directions = compute_outward_normals(points)

        # Step 1: Identify which edges intersect
        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            if edge_intersects_mask(p1, p2, mask):
                edge_flags[i] = True
                intersect_counts[i] += 1
                intersect_counts[(i + 1) % n] += 1
            else:
                intersect_counts[i] = max(0, intersect_counts[i] - 1)
                intersect_counts[(i + 1) % n] = max(0, intersect_counts[(i + 1) % n] - 1)

        if not any(edge_flags):
            print("converged")
            break

        # Reverse directions if a point has been stuck too long
        for i in range(n):
            if intersect_counts[i] >= 4:
                reversal_flags[i] = -1
            elif intersect_counts[i] == 0:
                reversal_flags[i] = 1

        movement_vectors = [np.zeros(2, dtype=np.float32) for _ in range(n)]

        for i in range(n):
            if not edge_flags[i]:
                continue
            i_next = (i + 1) % n
            #movement_vectors[i] += directions[i]["avg"] * angle_weights[i]
            #movement_vectors[i_next] += directions[i_next]["avg"] * angle_weights[i_next]
            movement_vectors[i] += directions[i] * angle_weights[i]
            movement_vectors[i_next] += directions[i_next] * angle_weights[i_next]

        # Compute centroid for outward check
        centroid = np.mean(points, axis=0)

        for i in range(n):
            mv = movement_vectors[i]
            if np.linalg.norm(mv) > 0:
                mv = mv / np.linalg.norm(mv) * step_size
                mv *= angle_weights[i] * reversal_flags[i]
                points[i] += mv

        if image is not None and iteration % 10 == 0:
            visualize_step(image, points, movement_vectors, edge_flags, reversal_flags, iteration)

    return [tuple(pt.astype(int)) for pt in points]


def draw_polygon(image, points, color=(0, 0, 255), thickness=2):
    img_copy = image.copy()
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img_copy, [pts], isClosed=True, color=color, thickness=thickness)
    return img_copy

def find_parallel_edges_as_top_bottom(points):
    """
    Finds the two most parallel edges in a 4-point polygon (assumed cone-like shape).
    Returns the indices of those edges (each index is the start point of an edge).
    """
    def normalize(v):
        norm = np.linalg.norm(v)
        return v / norm if norm != 0 else v

    n = len(points)
    edges = []
    directions = []

    # Compute direction vectors for each edge
    for i in range(n):
        p1 = np.array(points[i])
        p2 = np.array(points[(i + 1) % n])
        vec = normalize(p2 - p1)
        edges.append((i, (p1, p2)))
        directions.append(vec)

    # Compare all pairs of edges (non-adjacent)
    parallel_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n == 1 or (i - j) % n == 1:
                continue  # skip adjacent edges

            dot = np.abs(np.dot(directions[i], directions[j]))  # closeness to being parallel
            parallel_pairs.append(((i, j), dot))

    # Choose the pair with the most parallel vectors
    (edge1_idx, edge2_idx), _ = max(parallel_pairs, key=lambda x: x[1])
    return edge1_idx, edge2_idx


def draw_edge_lines(image, points, edge_indices, color=(255, 255, 0), thickness=2):
    """
    Draws lines over specified edges.
    edge_indices: list or tuple of edge start indices.
    """
    img_copy = image.copy()
    for i in edge_indices:
        pt1 = tuple(np.array(points[i]).astype(int))
        pt2 = tuple(np.array(points[(i + 1) % len(points)]).astype(int))
        cv2.line(img_copy, pt1, pt2, color, thickness)
    return img_copy

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
    mask_original = run_sam(predictor, image, yolo_box)
    points_original = sort_points_clockwise(get_extreme_points(mask_original))
    points_moved = move_points_adaptively(points_original, mask_original, image=image)

    image_original_with_dots = draw_points(image, points_moved)
    image_original_with_dots = draw_polygon(image_original_with_dots, points_moved)

    top_idx, bottom_idx = find_parallel_edges_as_top_bottom(points_moved)
    image_original_with_dots = draw_edge_lines(image_original_with_dots, points_moved, [top_idx, bottom_idx])

    for i, pt in enumerate(points_original):
        cv2.putText(image_original_with_dots, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    # Run SAM on Processed Image
    mask_preprocessed = run_sam(predictor, preprocessed_image, yolo_box)
    points_pre = sort_points_clockwise(get_extreme_points(mask_preprocessed))
    points_moved = move_points_adaptively(points_pre, mask_preprocessed)
    image_preprocessed_with_dots = draw_points(preprocessed_image, points_moved)
    image_preprocessed_with_dots = draw_polygon(image_preprocessed_with_dots, points_moved)

    for i, pt in enumerate(points_original):
        cv2.putText(image_preprocessed_with_dots, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

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