import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from shapely.geometry import Polygon
from segment_anything import sam_model_registry, SamPredictor

# ────────────────────────────────────────────────────────────────
# DRAWING HELPERS  ░ (all pure side-effects – no geometry/math) ░
# ────────────────────────────────────────────────────────────────

def visualize_final_polygon(image, mask, polygon, title="Final Polygon"):
    """
    Visualizes the image with the mask and final 4-point polygon.

    Args:
        image (np.ndarray): Original RGB image.
        mask (np.ndarray): Binary mask (bool or 0/1 array).
        polygon (list of tuples): List of 4 (x, y) points.
        title (str): Optional plot title.
    """
    img_copy = image.copy()

    # Draw the mask as an overlay
    colored_mask = np.zeros_like(img_copy)
    colored_mask[:, :, 1] = (mask * 255).astype(np.uint8)  # green channel
    overlay = cv2.addWeighted(img_copy, 0.8, colored_mask, 0.5, 0)

    # Draw the polygon
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(overlay, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    # Draw polygon points
    for (x, y) in polygon:
        cv2.circle(overlay, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Show the result
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def visualize_step(image, points, movement_vectors=None, edge_flags=None, reversal_flags=None, step_num=0, label="", rectangles=None):
    img = image.copy()
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw mask in background
    ax.imshow(img, alpha=0.4)

    # Convert polygon points
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # Draw rectangles (as quadrilaterals)
    if rectangles:
        for rect_pts in rectangles:
            rect = np.array(rect_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [rect], isClosed=True, color=(255, 255, 0), thickness=1)

    # Bounding box to zoom into
    margin = 40
    pts_flat = np.array(points, dtype=np.int32)
    xmin = max(pts_flat[:, 0].min() - margin, 0)
    xmax = min(pts_flat[:, 0].max() + margin, img.shape[1])
    ymin = max(pts_flat[:, 1].min() - margin, 0)
    ymax = min(pts_flat[:, 1].max() + margin, img.shape[0])

    # Draw each point
    for i, pt in enumerate(points):
        color = 'green'
        if edge_flags and (edge_flags[i] or edge_flags[(i - 1) % len(points)]):
            color = 'red'
        if reversal_flags and reversal_flags[i] == -1:
            color = 'yellow'
        ax.plot(pt[0], pt[1], 'o', color=color, markersize=8)

        if movement_vectors is not None:
            mv = movement_vectors[i]
            if np.linalg.norm(mv) > 0:
                ax.arrow(
                    pt[0], pt[1], mv[0]*5, mv[1]*5,
                    head_width=3, head_length=5, fc='blue', ec='blue'
                )

    ax.set_title(f"{label} – Iteration {step_num}")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)  # Y axis flipped for image coordinates
    ax.axis('off')
    plt.tight_layout()
    plt.pause(0.5)
    plt.close()

def draw_points(img, pts, color=(255, 0, 0), r=8):
    out = img.copy()
    for p in pts:
        cv2.circle(out, tuple(map(int, p)), r, color, -1)
    return out

def draw_rotated_quadrants(img, center, angle_rad,
                           length=500, color=(0, 255, 255), thickness=2):
    """purely visual – needs only already-known center & ellipse angle"""
    out = img.copy()
    cx, cy = center
    for a in (0, np.pi/2, np.pi, 3*np.pi/2):
        t = a + angle_rad
        dx, dy = int(np.cos(t)*length), int(np.sin(t)*length)
        cv2.line(out, (cx, cy), (cx+dx, cy+dy), color, thickness)
    return out

def draw_polygon(img, pts, color=(0, 0, 255), thickness=2):
    out = img.copy()
    poly = np.array(pts, np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [poly], True, color, thickness)
    return out

def draw_edge_lines(img, pts, edge_ids, color=(255, 255, 0), thickness=2):
    out = img.copy()
    n = len(pts)
    for idx in edge_ids:
        p1, p2 = tuple(map(int, pts[idx])), tuple(map(int, pts[(idx+1) % n]))
        cv2.line(out, p1, p2, color, thickness)
    return out

def draw_height_measurement(img, line_triplet, avg_px,
                            thickness=2, font_scale=0.9):
    """
    line_triplet: [(pTop, pBot), (pTop, pBot), (pTop, pBot)]
                  returned by FerruleDimensions.get_height_lines()
    """
    out = img.copy()
    colors = [(0,255,0), (0,0,255), (0,255,0)]  # center red, outer green
    mid_line = line_triplet[1]

    for (p1, p2), col in zip(line_triplet, colors):
        cv2.line(out, p1, p2, col, thickness)

    # label at the midpoint of centre line
    mx = (mid_line[0][0] + mid_line[1][0]) // 2
    my = (mid_line[0][1] + mid_line[1][1]) // 2
    cv2.putText(out, f"{int(avg_px)}px", (mx, my),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,255), 2,
                cv2.LINE_AA)
    return out

def draw_enclosing_circle(image, center_px, radius_px):
    out = image.copy()
    if center_px and radius_px:
        cv2.circle(out, center_px, int(radius_px), (255, 0, 0), 3)
    return out

# ────────────────────────────────────────────────────────────────
# CALCULATION CORE  ░  all heavy lifting lives here  ░
# ────────────────────────────────────────────────────────────────
class FerruleDimensions:
    def __init__(self, image, yolo_box_rel, predictor):
        self.image = image
        self.yolo_box_rel = yolo_box_rel
        self.predictor = predictor

        self.mask = None
        self.refined_pts = None
        self.top_idx = None
        self.bottom_idx = None
        self.height_lines = None
        self.height_px = None

    def process(self):
        self.mask = self._run_sam()
        self.refined_pts = self._find_min_area_enclosing_polygon()
        #self.top_idx, self.bottom_idx = self._find_parallel_edges()
        #self.height_lines, self.height_px = self._compute_height_triplet()

    def _run_sam(self):
        h, w, _ = self.image.shape
        xc, yc, wr, hr = self.yolo_box_rel
        box = np.array([
            int(xc * w - wr * w / 2), int(yc * h - hr * h / 2),
            int(xc * w + wr * w / 2), int(yc * h + hr * h / 2)
        ])
        self.predictor.set_image(self.image)
        masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
        return masks[np.argmax(scores)]

    def debug_grid_and_mask(self, image, mask, grid):
        img_copy = image.copy()
        for (x, y) in grid:
            cv2.circle(img_copy, (x, y), 2, (0, 0, 255), -1)
        overlay = cv2.addWeighted(img_copy, 0.6, (np.dstack([mask*255]*3)).astype(np.uint8), 0.4, 0)
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.title("Grid Points vs Mask")
        plt.axis('off')
        plt.show()

    def _find_min_area_enclosing_polygon(self, num_samples=20000, grid_density=12, pad=30, visualize=True):
        """
        Finds a 4-point convex polygon that fully contains the mask and
        minimizes the area between the polygon and mask boundary.

        - Ensures polygon does not intersect or cut through mask.
        - All polygon corners are outside the mask.
        - The entire mask lies inside the polygon.
        """
        msk = (self.mask * 255).astype(np.uint8)
        h, w = msk.shape
        contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(cnt)

        # Expanded grid around mask
        xs = np.linspace(x - pad, x + w_box + pad, grid_density, dtype=int)
        ys = np.linspace(y - pad, y + h_box + pad, grid_density, dtype=int)
        grid = [(xi, yi) for xi in xs for yi in ys]

        best_poly = None
        best_area = float("inf")

        for attempt in range(num_samples):
            combo = random.sample(grid, 4)
            poly = np.array(combo, dtype=np.int32)

            if cv2.contourArea(poly) < 1:
                continue
            if not cv2.isContourConvex(poly):
                continue

            # Ensure all polygon corners are outside the mask
            if any(cv2.pointPolygonTest(cnt, tuple(map(float, p)), False) >= 0 for p in poly):
                continue

            # 2. Check all contour points are inside the polygon
            if not all(cv2.pointPolygonTest(poly, tuple(map(float, pt[0])), False) >= 0 for pt in cnt):
                continue

            poly_area = cv2.contourArea(poly)
            if poly_area < best_area:
                best_area = poly_area
                best_poly = combo

        # Optional: visualize result
        if visualize and best_poly is not None:
            vis = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
            cv2.polylines(vis, [np.array(best_poly, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.drawContours(vis, [cnt], -1, (0, 0, 255), 1)
            plt.figure(figsize=(8, 8))
            plt.imshow(vis)
            plt.title("Best Polygon vs. Mask")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        return best_poly

    @staticmethod
    def _sort_clockwise(pts):
        pts = np.array(pts, dtype=np.float32)
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        return [tuple(pt) for pt in pts[np.argsort(angles)]]

    def _find_parallel_edges(self):
        pts = np.array(self.refined_pts, float)
        n = len(pts)
        vecs = [pts[(i + 1) % n] - pts[i] for i in range(n)]
        vecs = [v / (np.linalg.norm(v) + 1e-8) for v in vecs]
        best = (-1, -1, -1)
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n == 1 or (i - j) % n == 1:
                    continue
                dot = abs(np.dot(vecs[i], vecs[j]))
                if dot > best[2]:
                    best = (i, j, dot)
        return best[0], best[1]

    def _compute_height_triplet(self, t_samples=(0.1, 0.5, 0.9)):
        p = np.array(self.refined_pts, float)
        top, bottom = self.top_idx, self.bottom_idx
        ptT1, ptT2 = p[top], p[(top + 1) % 4]
        ptB1, ptB2 = p[bottom], p[(bottom + 1) % 4]

        dirT = ptT2 - ptT1
        dirT /= np.linalg.norm(dirT) + 1e-8
        normal = np.array([-dirT[1], dirT[0]])

        lines = []
        dists = []
        for t in t_samples:
            a = (1 - t) * ptT1 + t * ptT2
            b = (1 - t) * ptB1 + t * ptB2
            proj = np.dot(b - a, normal)
            b_proj = a + proj * normal
            lines.append((tuple(map(int, a)), tuple(map(int, b_proj))))
            dists.append(abs(proj))
        return lines, float(np.mean(dists))


# ───────────────────────────────────────────────────────────────
# CALCULATION CLASS – BallDimensions
# ───────────────────────────────────────────────────────────────
class BallDimensions:
    def __init__(self, image_rgb, yolo_box_rel, predictor, real_world_diameter_in=1.680):
        """
        Args:
            image_rgb (np.ndarray): The RGB image.
            yolo_box_rel (tuple): (x_center, y_center, width, height) in YOLO format.
            predictor (SamPredictor): Initialized SAM predictor.
            real_world_diameter_in (float): Real-world diameter of the ball in inches.
        """
        self.image = image_rgb
        self.yolo_box_rel = yolo_box_rel
        self.predictor = predictor
        self.real_world_diameter_in = real_world_diameter_in

        # Will be populated after running .process()
        self.mask = None
        self.center_px = None
        self.radius_px = None
        self.radius_in = None

    def process(self):
        self.mask = self._run_sam()
        self._calculate_enclosing_circle()

    def convert_pixels_to_inches(self, pixel_length):
        if not self.radius_px or not self.radius_in:
            raise ValueError("Measurement not initialized. Run process() first.")
        inch_per_pixel = self.radius_in / self.radius_px
        return pixel_length * inch_per_pixel

    # ───── INTERNAL HELPERS ─────

    def _run_sam(self):
        h, w, _ = self.image.shape
        xc, yc, w_rel, h_rel = self.yolo_box_rel
        w_abs = w_rel * w
        h_abs = h_rel * h
        x0 = int(xc * w - w_abs / 2)
        y0 = int(yc * h - h_abs / 2)
        x1 = int(x0 + w_abs)
        y1 = int(y0 + h_abs)
        box = np.array([x0, y0, x1, y1])
        self.predictor.set_image(self.image)
        masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
        return masks[np.argmax(scores)]

    def _calculate_enclosing_circle(self):
        mask_uint8 = (self.mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            print("No contours found in mask.")
            return

        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius_px = cv2.minEnclosingCircle(largest_contour)

        self.center_px = (int(x), int(y))
        self.radius_px = float(radius_px)
        diameter_px = 2 * self.radius_px
        inch_per_pixel = self.real_world_diameter_in / diameter_px
        self.radius_in = self.radius_px * inch_per_pixel


# ────────────────────────────────────────────────────────────────
# IO UTILS  (kept outside the class because they’re pure file I/O)
# ────────────────────────────────────────────────────────────────
def load_yolo_annotations(fp):
    anns=[]
    with open(fp) as f:
        for line in f:
            cid,xc,yc,w,h = map(float, line.split())
            anns.append(dict(class_id=int(cid), x_center=xc,
                             y_center=yc, width=w, height=h))
    return anns

# ────────────────────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────────────────────

def main(args):
    # Load image
    print("Loading image")
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")
    print("Converting to rgb")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("Loading annotations")
    # Load annotations and find objects by hardcoded IDs
    annotations = load_yolo_annotations(args.yolo)
    ball_ann = next((ann for ann in annotations if ann["class_id"] == 0), None)
    ferrule_ann = next((ann for ann in annotations if ann["class_id"] == 1), None)

    if not ball_ann or not ferrule_ann:
        raise ValueError("Could not find both class_id=0 (ball) and class_id=1 (ferrule) in the YOLO file.")

    ball_box = (ball_ann['x_center'], ball_ann['y_center'], ball_ann['width'], ball_ann['height'])
    ferrule_box = (ferrule_ann['x_center'], ferrule_ann['y_center'], ferrule_ann['width'], ferrule_ann['height'])

    # Load SAM model
    sam_ckpt = "sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
    print("Loading SAM")
    predictor = SamPredictor(sam)

    # Analyze ferrule
    ferrule = FerruleDimensions(image_rgb, ferrule_box, predictor)
    print ("Analyzing ferrule")
    ferrule.process()
    visualize_final_polygon(ferrule.image, ferrule.mask, ferrule.refined_pts)

'''
    print ("Drawing ferrule on canvas")
    canvas = image_rgb.copy()
    canvas = draw_rotated_quadrants(canvas, ferrule.center, ferrule.quadrants_angle)
    canvas = draw_points(canvas, ferrule.refined_pts)
    canvas = draw_polygon(canvas, ferrule.refined_pts)
    canvas = draw_edge_lines(canvas, ferrule.refined_pts, [ferrule.top_idx, ferrule.bottom_idx])
    canvas = draw_height_measurement(canvas, ferrule.height_lines, ferrule.height_px)

    # Analyze ball
    ball = BallDimensions(image_rgb, ball_box, predictor)
    print ("Analyzing ball")
    ball.process()
    print ("Drawing ball on canvas")
    canvas = draw_enclosing_circle(canvas, ball.center_px, ball.radius_px)

    # Convert ferrule length from pixels to inches using ball scale
    try:
        length_in_inches = ball.convert_pixels_to_inches(ferrule.height_px)
        print(f"Ferrule height: {ferrule.height_px:.1f}px ≈ {length_in_inches:.3f}\"")
    except ValueError:
        print("Could not convert ferrule length to inches – ball not found.")

    print ("Displaying final result")
    # Display final result
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(canvas)
    if ferrule.mask is not None:
        ax.imshow(ferrule.mask, alpha=0.3, cmap='jet')
    if ball.mask is not None:
        ax.imshow(ball.mask, alpha=0.3, cmap='jet')
    ax.axis('off')
    ax.set_title("Ball + Ferrule Measurement")
    plt.tight_layout()
    plt.show()
'''

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--yolo",  required=True)
    #p.add_argument("--yolo-id", type=int, required=True)
    main(p.parse_args())