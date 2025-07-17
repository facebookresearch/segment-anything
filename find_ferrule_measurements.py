import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# ────────────────────────────────────────────────────────────────
# DRAWING HELPERS  ░ (all pure side-effects – no geometry/math) ░
# ────────────────────────────────────────────────────────────────
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
                            thickness=2, font_scale=0.9, label_override=None):
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
    label = label_override if label_override else f"{int(avg_px)}px"
    cv2.putText(out, label, (mx, my),
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
        self.extreme_pts = None
        self.center = None
        self.ellipse_ang = None
        self.refined_pts = None
        self.top_idx = None
        self.bottom_idx = None
        self.height_lines = None
        self.height_px = None

    def process(self, visualize=False):
        self.mask = self._run_sam()
        raw_pts, self.center, self.ellipse_ang = self._get_extreme_points()
        if len(raw_pts) != 4:
            raise RuntimeError("Unable to locate 4 extreme points")

        self.refined_pts = self._move_points(raw_pts, visualize=visualize)
        self._optimize_edges_area(visualize=visualize)

        self.top_idx, self.bottom_idx = self._find_parallel_edges()
        self.height_lines, self.height_px = self._compute_height_triplet()

    @property
    def quadrants_angle(self):
        return self.ellipse_ang

    @staticmethod
    def _sort_clockwise(pts):
        pts = np.array(pts, np.float32)
        s = pts.sum(1)
        diff = np.diff(pts, axis=1).ravel()
        return [tuple(int(x) for x in pt)
                for pt in (pts[np.argmin(s)],
                           pts[np.argmin(diff)],
                           pts[np.argmax(s)],
                           pts[np.argmax(diff)])]

    @staticmethod
    def _polygon_outward_normals(pts):
        pts = np.array(pts, np.float32)
        n = len(pts)
        centroid = np.mean(pts, axis=0)
        normals = []

        for i in range(n):
            prev = pts[(i - 1) % n]
            curr = pts[i]
            next = pts[(i + 1) % n]

            edge1 = curr - prev
            edge2 = next - curr
            edge = (edge1 + edge2) / 2
            edge /= (np.linalg.norm(edge) + 1e-8)

            normal = np.array([edge[1], -edge[0]])
            to_point = curr - centroid
            if np.dot(normal, to_point) < 0:
                normal = -normal
            normals.append(normal)

        return normals

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

    def compute_edge_lengths(self):
        pts = np.array(self.refined_pts, float)
        top = np.linalg.norm(pts[(self.top_idx + 1) % 4] - pts[self.top_idx])
        bottom = np.linalg.norm(pts[(self.bottom_idx + 1) % 4] - pts[self.bottom_idx])
        return top, bottom

    def _get_extreme_points(self):
        m = (self.mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [], None, None
        cnt = max(contours, key=cv2.contourArea).squeeze()

        center = np.mean(cnt, 0)
        if len(cnt) < 5:
            return [], None, None
        ellipse = cv2.fitEllipse(cnt)
        ang = np.deg2rad(ellipse[2])

        R = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
        rel = (cnt - center) @ R.T
        thetas = (np.arctan2(rel[:, 1], rel[:, 0]) + 2 * np.pi) % (2 * np.pi)

        buckets = [[] for _ in range(4)]
        for i, t in enumerate(thetas):
            buckets[int(t // (np.pi / 2)) % 4].append(i)

        pick = []
        for b in buckets:
            if b:
                d = np.linalg.norm(rel[b], axis=1)
                pick.append(tuple(cnt[b[np.argmax(d)]]))

        if len(pick) < 4:
            taken = set(pick)
            left = sorted(range(len(cnt)), key=lambda i: -np.linalg.norm(rel[i]))
            for i in left:
                p = tuple(cnt[i])
                if p not in taken:
                    pick.append(p)
                    taken.add(p)
                if len(pick) == 4:
                    break
        ordered = self._sort_clockwise(pick)
        return ordered, tuple(map(int, center)), float(ang)

    def _move_points(self, init_pts, max_iters=200, step=2, visualize=False):
        pts = np.array(init_pts, np.float32)
        h, w = self.mask.shape

        def seg_intersects(p1, p2):
            line_mask = np.zeros_like(self.mask, dtype=np.uint8)
            cv2.line(line_mask, tuple(p1.astype(int)), tuple(p2.astype(int)), 255, thickness=1)
            intersection = cv2.bitwise_and(line_mask, (self.mask * 255).astype(np.uint8))
            return np.any(intersection)

        n = len(pts)
        frozen = [False] * n
        rev = [1] * n

        for it in range(max_iters):
            if visualize:
                visualize_step(
                    self.image,
                    points=pts,
                    step_num=it,
                    label="MovePoints",
                    rectangles=None
                )
            edges = [seg_intersects(pts[i], pts[(i + 1) % n]) for i in range(n)]
            if not any(edges):
                print(f"✅ Converged at iteration {it}")
                break

            for i in range(n):
                e1 = edges[i]
                e2 = edges[(i - 1) % n]
                frozen[i] = not (e1 or e2)

            angles = self._internal_angles(pts)
            wts = [max(0.2, (np.pi - a) / np.pi) for a in angles]
            dirs = self._polygon_outward_normals(pts)
            mv = [np.zeros(2, dtype=np.float32) for _ in range(n)]

            for i, intersects in enumerate(edges):
                if not intersects:
                    continue
                j = (i + 1) % n
                mv[i] += dirs[i] * wts[i]
                mv[j] += dirs[j] * wts[j]

            for i in range(n):
                if frozen[i]:
                    continue
                if np.linalg.norm(mv[i]) > 0:
                    mv_norm = mv[i] / (np.linalg.norm(mv[i]) + 1e-8)
                    delta = mv_norm * step * wts[i]
                    pts[i] += delta
                else:
                    print(f"⚠️ Point {i} has no movement at iteration {it}")

        return [tuple(map(int, pt)) for pt in pts]

    def _optimize_edges_area(self, max_iters=100, step=1.0, visualize=False):
        """
        Refines polygon edges by minimizing the area between each edge's rectangle
        and the object mask. Keeps adjusting points until no better configuration exists.
        """
        mask_bin = (self.mask * 255).astype(np.uint8)
        pts = np.array(self.refined_pts, dtype=np.float32)
        n = len(pts)

        def edge_area(p1, p2):
            edge_vec = p2 - p1
            length = np.linalg.norm(edge_vec)
            if length < 1e-5:
                return np.inf

            direction = edge_vec / length
            normal = np.array([-direction[1], direction[0]])
            thickness = 6  # pixels

            # Rectangle around the edge
            corners = [
                p1 + normal * thickness,
                p2 + normal * thickness,
                p2 - normal * thickness,
                p1 - normal * thickness
            ]
            poly = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))

            mask_edge = np.zeros_like(mask_bin)
            cv2.fillPoly(mask_edge, [poly], 255)

            inter = cv2.bitwise_and(mask_edge, mask_bin)
            return np.count_nonzero(mask_edge) - np.count_nonzero(inter)

        for iter_num in range(max_iters):
            best_pts = None
            best_area = None
            best_rects = []

            for i in range(n):
                j = (i + 1) % n
                p1, p2 = pts[i], pts[j]
                area_orig = edge_area(p1, p2)

                for sign in [+1, -1]:
                    for idx in [i, j]:
                        test_pts = pts.copy()
                        move_vec = self._polygon_outward_normals(pts)[idx]
                        candidate_pt = test_pts[idx] + sign * step * move_vec

                        # Prevent moving into the mask
                        y, x = int(candidate_pt[1]), int(candidate_pt[0])
                        if (0 <= y < mask_bin.shape[0]) and (0 <= x < mask_bin.shape[1]):
                            if mask_bin[y, x] > 0:
                                # This point would go inside the mask — skip it
                                continue

                        test_pts[idx] = candidate_pt
                        test_area = edge_area(test_pts[i], test_pts[j])

                        if test_area < area_orig:
                            if best_area is None or test_area < best_area:
                                best_area = test_area
                                best_pts = test_pts.copy()

                                # For visualization, store the improved edge rectangle
                                edge_vec = best_pts[j] - best_pts[i]
                                dir = edge_vec / (np.linalg.norm(edge_vec) + 1e-8)
                                norm = np.array([-dir[1], dir[0]])
                                thickness = 6
                                best_rects = [[
                                    best_pts[i] + norm * thickness,
                                    best_pts[j] + norm * thickness,
                                    best_pts[j] - norm * thickness,
                                    best_pts[i] - norm * thickness,
                                ]]

            if best_pts is not None:
                pts = best_pts
                if visualize:
                    visualize_step(
                        self.image,
                        points=pts,
                        step_num=iter_num,
                        label="AreaOptimize",
                        rectangles=best_rects
                    )
            else:
                if visualize:
                    print(f"Converged at iteration {iter_num}")
                break

        #return [tuple(map(int, pt)) for pt in pts]

    @staticmethod
    def _internal_angles(pts):
        pts = np.array(pts, float)
        n = len(pts)
        angs = []
        for i in range(n):
            a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
            v1, v2 = a - b, c - b
            v1 /= np.linalg.norm(v1) + 1e-8
            v2 /= np.linalg.norm(v2) + 1e-8
            angs.append(np.arccos(np.clip(np.dot(v1, v2), -1, 1)))
        return angs

    def _find_parallel_edges(self):
        pts = np.array(self.refined_pts, float)
        n = len(pts)
        vec = [pts[(i + 1) % n] - pts[i] for i in range(n)]
        vec = [v / np.linalg.norm(v) + 1e-8 for v in vec]
        best = (-1, -1, -1)
        for i in range(n):
            for j in range(i + 1, n):
                if (j - i) % n == 1 or (i - j) % n == 1:
                    continue
                d = abs(np.dot(vec[i], vec[j]))
                if d > best[2]:
                    best = (i, j, d)
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
    import json
    import os

    print("Loading image")
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")
    print("Converting to rgb")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    print("Loading annotations")
    annotations = load_yolo_annotations(args.yolo)
    ball_ann = next((ann for ann in annotations if ann["class_id"] == 0), None)
    ferrule_ann = next((ann for ann in annotations if ann["class_id"] == 1), None)

    if not ball_ann or not ferrule_ann:
        raise ValueError("Could not find both class_id=0 (ball) and class_id=1 (ferrule) in the YOLO file.")

    ball_box = (ball_ann['x_center'], ball_ann['y_center'], ball_ann['width'], ball_ann['height'])
    ferrule_box = (ferrule_ann['x_center'], ferrule_ann['y_center'], ferrule_ann['width'], ferrule_ann['height'])

    sam_ckpt = "sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt)
    print("Loading SAM")
    predictor = SamPredictor(sam)

    ferrule = FerruleDimensions(image_rgb, ferrule_box, predictor)
    print("Analyzing ferrule")
    ferrule.process(visualize=True)

    print("Drawing ferrule on canvas")
    canvas = image_rgb.copy()
    canvas = draw_rotated_quadrants(canvas, ferrule.center, ferrule.quadrants_angle)
    canvas = draw_points(canvas, ferrule.refined_pts)
    canvas = draw_polygon(canvas, ferrule.refined_pts)
    canvas = draw_edge_lines(canvas, ferrule.refined_pts, [ferrule.top_idx, ferrule.bottom_idx])

    ball = BallDimensions(image_rgb, ball_box, predictor)
    print("Analyzing ball")
    ball.process()
    print("Drawing ball on canvas")
    canvas = draw_enclosing_circle(canvas, ball.center_px, ball.radius_px)

    top_px, bot_px = ferrule.compute_edge_lengths()

    measurements = {}
    try:
        top_in = ball.convert_pixels_to_inches(top_px)
        bot_in = ball.convert_pixels_to_inches(bot_px)
        height_in = ball.convert_pixels_to_inches(ferrule.height_px)

        print(f"Top edge: {top_px:.1f}px ≈ {top_in:.3f}\"")
        print(f"Bottom edge: {bot_px:.1f}px ≈ {bot_in:.3f}\"")
        print(f"Ferrule height: {ferrule.height_px:.1f}px ≈ {height_in:.3f}\"")

        measurements = {
            "top_edge": {"pixels": round(top_px, 1), "inches": round(top_in, 3)},
            "bottom_edge": {"pixels": round(bot_px, 1), "inches": round(bot_in, 3)},
            "ferrule_height": {"pixels": round(ferrule.height_px, 1), "inches": round(height_in, 3)}
        }

        canvas = draw_height_measurement(canvas, ferrule.height_lines, ferrule.height_px, label_override=f"{height_in:.3f}\"")

        top_mid = (
            (ferrule.refined_pts[ferrule.top_idx][0] +
             ferrule.refined_pts[(ferrule.top_idx + 1) % 4][0]) // 2,
            (ferrule.refined_pts[ferrule.top_idx][1] +
             ferrule.refined_pts[(ferrule.top_idx + 1) % 4][1]) // 2,
        )
        cv2.putText(canvas, f"{top_in:.3f}\"", top_mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        bot_mid = (
            (ferrule.refined_pts[ferrule.bottom_idx][0] +
             ferrule.refined_pts[(ferrule.bottom_idx + 1) % 4][0]) // 2,
            (ferrule.refined_pts[ferrule.bottom_idx][1] +
             ferrule.refined_pts[(ferrule.bottom_idx + 1) % 4][1]) // 2,
        )
        cv2.putText(canvas, f"{bot_in:.3f}\"", bot_mid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    except ValueError:
        print("Could not convert top/bottom edge lengths – ball not found.")

    if args.write:
        print("📝 Saving output files...")

        # Save processed image (no mask)
        processed_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        cv2.imwrite("processed.jpg", processed_bgr)
        print("✔️ Saved 'processed.jpg'")

        # Save masked image using matplotlib to overlay masks
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(canvas)
        if ferrule.mask is not None:
            ax.imshow(ferrule.mask, alpha=0.3, cmap='jet')
        if ball.mask is not None:
            ax.imshow(ball.mask, alpha=0.3, cmap='jet')
        ax.axis('off')
        ax.set_title("Ball + Ferrule Measurement (with Mask)")
        plt.tight_layout()
        fig.savefig("masked.jpg", format="jpg", dpi=300)
        plt.close()
        print("✔️ Saved 'masked.jpg'")

        # Save measurements
        with open("measurements.json", "w") as f:
            json.dump(measurements, f, indent=2)
        print("✔️ Saved 'measurements.json'")
    else:
        print("📺 Displaying final result")
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


# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--yolo",  required=True)
    p.add_argument("--write", action="store_true", help="Save final image to disk instead of displaying it.")
    main(p.parse_args())
