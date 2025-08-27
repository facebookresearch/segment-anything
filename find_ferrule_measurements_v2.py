import cv2
from scipy.optimize import minimize
import argparse
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# ---- Loss configuration (reverted to count) ----

PENALTY_WEIGHT = 5.0

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


class FerruleDimensions:
    def __init__(self, image_rgb, yolo_box_rel, predictor):
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

        self.mask = None
        self.points = None
        self.result = None

    def process(self):
        print("→ Running SAM segmentation")
        self.mask = self._run_sam()

        # Convert mask to binary and fill
        mask_uint8 = (self.mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.ones_like(mask_uint8) * 255
        cv2.drawContours(filled_mask, contours, -1, color=0, thickness=-1)

        print("→ Extracting black points")
        self.points = self.extract_black_points(filled_mask)

        print("→ Fitting frustum")
        self.result = self.fit_frustum(self.points, self.image.shape)

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

    def get_filled_mask_image(self):
        """
        Returns a white-background image with the SAM mask filled in black.
        Also useful for saving to disk.
        """
        if self.mask is None:
            raise ValueError("SAM mask not computed yet. Run process() or _run_sam() first.")

        mask_uint8 = (self.mask.astype(np.uint8) * 255)

        # Find external contours
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # White background
        filled_mask = np.ones_like(mask_uint8) * 255

        # Fill the mask with black
        cv2.drawContours(filled_mask, contours, -1, color=0, thickness=-1)

        return filled_mask

    def calculate_measurements(self, params):
        corners = self.trapezoid_corners(params)

        # Sort corners for consistent labeling
        sorted_by_y = corners[np.argsort(corners[:, 1])]
        top_two = sorted_by_y[:2]
        bottom_two = sorted_by_y[2:]
        tl, tr = sorted(top_two, key=lambda pt: pt[0])
        bl, br = sorted(bottom_two, key=lambda pt: pt[0])

        # Top and bottom width
        top_width_px = np.linalg.norm(tr - tl)
        bottom_width_px = np.linalg.norm(br - bl)

        # Length (vertical distance along center axis)
        top_center = (tl + tr) / 2
        bottom_center = (bl + br) / 2
        length_px = np.linalg.norm(bottom_center - top_center)

        # Determine large/small diameter
        large = max(top_width_px, bottom_width_px)
        small = min(top_width_px, bottom_width_px)
        taper_ratio = (large - small) / length_px if length_px > 0 else 0.0

        return {
            "top_width_px": top_width_px,
            "bottom_width_px": bottom_width_px,
            "length_px": length_px,
            "large_diameter_px": large,
            "small_diameter_px": small,
            "taper_ratio": taper_ratio,
            "is_inverted": bottom_width_px < top_width_px,
        }, np.array([tl, tr, br, bl])  # ensure corners follow this order

    def extract_black_points2(self, image_path, visualize=False):
        # Read image and convert to grayscale
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Invert threshold to extract black areas
        _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

        # Extract coordinates of black pixels
        ys, xs = np.where(mask > 0)
        points = np.column_stack((xs, ys))

        # Optional visualization
        if visualize:
            # Create a white canvas
            canvas = np.full_like(gray, 255)
            for x, y in points:
                canvas[y, x] = 0  # draw black point

            # Optionally save or display the canvas
            #cv2.imwrite("black_points_visualized.png", canvas)
            cv2.imshow("Points", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(f"img.shape {img.shape}")
        return img, points

    def extract_black_points(self, binary_mask, visualize=False):
        if len(binary_mask.shape) == 3:
            gray = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
        else:
            gray = binary_mask
        _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        ys, xs = np.where(mask > 0)
        points = np.column_stack((xs, ys))

        if visualize:
            canvas = np.full_like(gray, 255)
            for x, y in points:
                canvas[y, x] = 0
            cv2.imshow("Points", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return points

    def trapezoid_corners(self, params):
        # center_x, center_y, w_top, h, w_bottom, theta
        # This parameterization is an isosceles trapezoid (perfect frustum):
        # - top/bottom centers are aligned along u (no skew)
        # - left/right offsets are ± along v (symmetric)
        center_x, center_y, w_top, h, w_bottom, theta = params
        C = np.array([center_x, center_y], dtype=float)
        c, s = np.cos(theta), np.sin(theta)
        u = np.array([-s,  c])  # vertical axis after rotation
        v = np.array([ c,  s])  # horizontal axis after rotation
        top_center = C - (h / 2.0) * u
        bot_center = C + (h / 2.0) * u
        tl = top_center - (w_top / 2.0) * v
        tr = top_center + (w_top / 2.0) * v
        bl = bot_center - (w_bottom / 2.0) * v
        br = bot_center + (w_bottom / 2.0) * v
        return np.array([tl, tr, br, bl], dtype=float)

    def point_in_trapezoid(self, points, corners):
        # Revert: call pointPolygonTest for every point (edge-inclusive)
        contour = corners.astype(np.float32).reshape((-1,1,2))
        return np.array([
            cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0
            for pt in points
        ])

    def frustum_loss(self, params, points):
        # Revert: step penalty with shoelace area
        corners = self.trapezoid_corners(params)
        inside = self.point_in_trapezoid(points, corners)
        x = corners[:, 0]
        y = corners[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        loss = area + PENALTY_WEIGHT * np.sum(~inside)
        return loss


    def fit_frustum(self, points, img_shape, restarts=4, jitter=0.10):
        # Revert to simple percentile-based initializer and single Powell call
        h, w = img_shape[:2]

        # Compute central region of point cloud
        x_min, x_max = np.percentile(points[:, 0], [5, 95])
        y_min, y_max = np.percentile(points[:, 1], [5, 95])
        x_center = np.mean([x_min, x_max])
        y_center = np.mean([y_min, y_max])

        # Compute reasonable widths and height
        w_top0 = (x_max - x_min) * 0.5
        w_bottom0 = (x_max - x_min) * 0.9  # more generous base
        h0 = y_max - y_min
        theta0 = 0.0

        # Initial guess
        init = [x_center, y_center, w_top0, h0, w_bottom0, theta0]

        # Allow 20% breathing room in bounds
        x_pad = (x_max - x_min) * 0.2
        y_pad = (y_max - y_min) * 0.2
        w = img_shape[1]
        h = img_shape[0]

        bounds = [
            (max(0, x_min - x_pad), min(w - 1, x_max + x_pad)),  # center_x
            (max(0, y_min - y_pad), min(h - 1, y_max + y_pad)),  # center_y
            (1, w),   # w_top
            (1, h),   # height
            (1, w),   # w_bottom
            (-np.pi, np.pi)  # theta
        ]

        # one run with per-iteration loss and params capture
        def run_one(x0, run_num=0):
            loss_history = []
            params_history = []

            def _cb(xk):
                current_loss = self.frustum_loss(xk, points)
                loss_history.append(current_loss)
                params_history.append(np.array(xk, dtype=float))

            res = minimize(
                self.frustum_loss, x0, args=(points,), method='Powell', bounds=bounds, callback=_cb,
                # Revert higher tolerance: relax tolerances and reduce caps
                options={'maxiter': 50, 'maxfev': 10000, 'xtol': 1e-3, 'ftol': 1e-3}
            )
            return res, loss_history, params_history

        # baseline run
        print("Starting baseline optimization run...")
        best_res, best_hist, best_params_hist = run_one(init, 0)
        best_loss = self.frustum_loss(best_res.x, points)
        print(f"Baseline run completed. Loss: {best_loss:.2f}")

        # multi-start with small jitter to escape plateaus
        rng = np.random.default_rng(0)
        for restart_num in range(restarts):
            print(f"Starting restart {restart_num + 1}/{restarts}...")
            xj = np.array(init, dtype=float)
            # jitter center a few pixels
            xj[0] = np.clip(xj[0] + rng.normal(0, 5.0), 0, w-1)
            xj[1] = np.clip(xj[1] + rng.normal(0, 5.0), 0, h-1)
            # jitter widths/heights multiplicatively
            xj[2] = np.clip(xj[2] * np.clip(1 + rng.normal(0, jitter), 0.5, 2.0), 1, w)
            xj[3] = np.clip(xj[3] * np.clip(1 + rng.normal(0, jitter), 0.5, 2.0), 1, h)
            xj[4] = np.clip(xj[4] * np.clip(1 + rng.normal(0, jitter), 0.5, 2.0), 1, w)
            # jitter theta modestly
            xj[5] = np.clip(xj[5] + rng.normal(0, 0.1), -np.pi, np.pi)

            res, hist, phist = run_one(xj, restart_num + 1)
            loss = self.frustum_loss(res.x, points)
            if loss < best_loss:
                print(f"Found better solution! Loss: {loss:.2f} (was {best_loss:.2f})")
                best_res, best_hist, best_params_hist, best_loss = res, hist, phist, loss
            else:
                print(f"Restart {restart_num + 1} completed. Loss: {loss:.2f} (best: {best_loss:.2f})")

        # Calculate physical measurements from the fitted frustum
        corners = self.trapezoid_corners(best_res.x)
        tl, tr, br, bl = corners

        # Calculate widths at top and bottom
        top_width = np.linalg.norm(tr - tl)
        bottom_width = np.linalg.norm(br - bl)

        # Assign large/small diameter based on actual size (not position)
        large_diameter = max(top_width, bottom_width)
        small_diameter = min(top_width, bottom_width)

        # Length (height between top and bottom centers)
        top_center = (tl + tr) / 2
        bottom_center = (bl + br) / 2
        length = np.linalg.norm(bottom_center - top_center)

        measurements, sorted_corners = self.calculate_measurements(best_res.x)

        result = {
            'params': best_res.x,
            'loss_history': best_hist,
            'params_history': best_params_hist,
            'corners': sorted_corners,
            'measurements': measurements
        }

        return result


    def plot_frustum(self, params, points):
        """
        Draws the frustum and points on a copy of the original image using OpenCV.
        Returns the image as a NumPy BGR array.
        """
        img_out = self.image.copy()
        overlay = img_out.copy()

        # === Draw black points as semi-transparent green dots ===
        alpha = 0.2
        dot_radius = 1
        green = (0, 255, 0)

        for x, y in points.astype(int):
            if (
                    0 <= y < overlay.shape[0] and
                    0 <= x < overlay.shape[1]
            ):
                cv2.circle(overlay, (x, y), dot_radius, color=green, thickness=-1)

        # Blend once
        img_out = cv2.addWeighted(overlay, alpha, img_out, 1 - alpha, 0)

        # === Draw frustum trapezoid in red ===
        corners = self.trapezoid_corners(params).astype(int)
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])
            cv2.line(img_out, pt1, pt2, color=(0, 0, 255), thickness=1)  # red

        # Add label/title
        cv2.putText(
            img_out,
            "Best Fit Frustum",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )

        return img_out


    def plot_frustum2(self, params, points):
        """
        Draws the frustum, points, and mask on a copy of the original image using OpenCV.
        Returns the image as a NumPy BGR array.
        """
        img_out = self.image.copy()

        # === Overlay green mask (if available) ===
        if hasattr(self, "mask") and self.mask is not None:
            if self.mask.ndim == 3:
                mask_binary = self.mask[:, :, 0].astype(bool)
            else:
                mask_binary = self.mask.astype(bool)

            green_overlay = np.zeros_like(img_out, dtype=np.uint8)
            green_overlay[:, :] = (0, 255, 0)  # BGR green

            alpha = 0.2
            img_out[mask_binary] = (
                    (1 - alpha) * img_out[mask_binary] + alpha * green_overlay[mask_binary]
            ).astype(np.uint8)

        # === Draw black points as semi-transparent green dots ===
        alpha = 0.2
        dot_radius = 1
        green = (0, 255, 0)  # BGR tuple

        for x, y in points.astype(int):
            if (
                    0 <= y - dot_radius < img_out.shape[0]
                    and 0 <= x - dot_radius < img_out.shape[1]
                    and y + dot_radius < img_out.shape[0]
                    and x + dot_radius < img_out.shape[1]
            ):
                # Draw a tiny filled circle on a separate mask
                overlay = img_out.copy()
                cv2.circle(overlay, (x, y), dot_radius, color=tuple(green), thickness=-1)
                # Blend just that region
                img_out = cv2.addWeighted(overlay, alpha, img_out, 1 - alpha, 0)

        # === Draw frustum trapezoid in red ===
        corners = self.trapezoid_corners(params).astype(int)
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])
            cv2.line(img_out, pt1, pt2, color=(0, 0, 255), thickness=1)  # red

        # === Add label ===
        cv2.putText(
            img_out,
            "Best Fit Frustum",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )

        return img_out

    def plot_frustum_with_loss(self, img, params, points, loss_history, params_history=None):
        corners = self.trapezoid_corners(params)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Left: image + frustum
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0].scatter(points[:,0], points[:,1], c='k', s=2, label='Black Dots')

        # Optional: overlay trapezoid at each iteration to visualize progression
        if params_history and len(params_history) > 0:
            cmap = plt.cm.viridis
            n = len(params_history)
            for i, p in enumerate(params_history):
                cs = self.trapezoid_corners(p)
                col = cmap(i / max(n - 1, 1))
                axes[0].plot(*np.append(cs, [cs[0]], axis=0).T, color=col, alpha=0.25, linewidth=1)

        # Final result highlighted
        axes[0].plot(*np.append(corners, [corners[0]], axis=0).T, 'r-', linewidth=2.0, label='Final Frustum')
        axes[0].legend()
        axes[0].set_title('Best Fit 2D Frustum (progression)')
        axes[0].set_xlim(0, img.shape[1])
        axes[0].set_ylim(img.shape[0], 0)
        axes[0].set_aspect('equal', adjustable='box')

        # Right: loss curve (per Powell iteration)
        if loss_history:
            axes[1].plot(np.arange(1, len(loss_history) + 1), loss_history, 'b-')
        else:
            axes[1].plot([], [])
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss per Iteration (Powell)')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_frustum_with_measurements(self, img, result, points):
        """Plot the final frustum fit with measurement annotations"""
        params = result['params']
        corners = result['corners']
        measurements = result['measurements']

        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.scatter(points[:,0], points[:,1], c='k', s=2, alpha=0.5, label='Black Points')

        # Draw the fitted trapezoid
        trap_x = np.append(corners[:, 0], corners[0, 0])
        trap_y = np.append(corners[:, 1], corners[0, 1])
        plt.plot(trap_x, trap_y, 'r-', linewidth=3, label='Fitted Frustum')

        # Extract corner coordinates for annotations
        tl, tr, br, bl = corners

        # Draw measurement lines and annotations
        top_center = (tl + tr) / 2
        bottom_center = (bl + br) / 2

        # Top width line
        plt.plot([tl[0], tr[0]], [tl[1], tr[1]], 'b-', linewidth=2, alpha=0.7)
        plt.text(top_center[0], top_center[1] - 15, f'Top: {measurements["top_width_px"]:.1f}px',
                 ha='center', va='bottom', color='blue', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Bottom width line
        plt.plot([bl[0], br[0]], [bl[1], br[1]], 'g-', linewidth=2, alpha=0.7)
        plt.text(bottom_center[0], bottom_center[1] + 15, f'Bottom: {measurements["bottom_width_px"]:.1f}px',
                 ha='center', va='top', color='green', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Length line (center axis)
        plt.plot([top_center[0], bottom_center[0]], [top_center[1], top_center[1]],
                 'm--', linewidth=2, alpha=0.7)
        length_mid = (top_center + bottom_center) / 2
        plt.text(length_mid[0] + 20, length_mid[1], f'Length: {measurements["length_px"]:.1f}px',
                 ha='left', va='center', color='magenta', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Summary text box
        summary_text = f"""Measurements Summary:
            Large Diameter: {measurements['large_diameter_px']:.1f}px
            Small Diameter: {measurements['small_diameter_px']:.1f}px
            Length: {measurements['length_px']:.1f}px
            Taper Ratio: {measurements['taper_ratio']:.3f}"""

        if measurements['is_inverted']:
            summary_text += "\n⚠️ Inverted cone detected"

        plt.text(0.02, 0.98, summary_text, transform=plt.gca().transAxes,
                 verticalalignment='top', fontsize=9, fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

        plt.legend(loc='upper right')
        plt.title('Frustum Fit with Pixel Measurements', fontsize=14, fontweight='bold')
        plt.xlim(0, img.shape[1])
        plt.ylim(img.shape[0], 0)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

def create_measured_image(ball: BallDimensions, ferrule: FerruleDimensions):
    """
    Renders a single image showing the ball mask, frustum outline, and
    annotated measurements (in pixels) using OpenCV.
    Returns the rendered image as a NumPy array.
    """
    img_out = ball.image.copy()
    m = ferrule.result["measurements"]
    corners = ferrule.trapezoid_corners(ferrule.result['params']).astype(int)

    # === Sort corners reliably ===
    # Sort by Y (top to bottom)
    sorted_by_y = corners[np.argsort(corners[:, 1])]
    top_two = sorted_by_y[:2]
    bottom_two = sorted_by_y[2:]

    # Sort left/right within each row
    top_left, top_right = sorted(top_two, key=lambda pt: pt[0])
    bottom_left, bottom_right = sorted(bottom_two, key=lambda pt: pt[0])

    # Rebuild ordered corners
    corners_ordered = np.array([top_left, top_right, bottom_right, bottom_left])
    tl, tr, br, bl = corners_ordered

    # === Draw frustum outline ===
    for i in range(4):
        pt1 = tuple(corners_ordered[i])
        pt2 = tuple(corners_ordered[(i + 1) % 4])
        cv2.line(img_out, pt1, pt2, color=(0, 0, 255), thickness=2)

    # === Draw ball circle ===
    if ball.center_px and ball.radius_px:
        center = tuple(map(int, ball.center_px))
        radius = int(ball.radius_px)
        cv2.circle(img_out, center, radius, color=(255, 165, 0), thickness=2)
        cv2.putText(img_out, f"Ball: {2*radius:.1f}px", (center[0], center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2, lineType=cv2.LINE_AA)

    # === Measurement lines ===
    top_center = ((tl + tr) / 2).astype(int)
    bottom_center = ((bl + br) / 2).astype(int)
    length_mid = ((top_center + bottom_center) / 2).astype(int)

    # Top width
    cv2.arrowedLine(img_out, tuple(tl), tuple(tr), color=(255, 0, 0), thickness=1, tipLength=0.02)
    cv2.arrowedLine(img_out, tuple(tr), tuple(tl), color=(255, 0, 0), thickness=1, tipLength=0.02)
    cv2.putText(img_out, f"Top: {m['top_width_px']:.1f}px",
                tuple(top_center - np.array([0, 10])), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 0), 2, lineType=cv2.LINE_AA)

    # Bottom width
    cv2.arrowedLine(img_out, tuple(bl), tuple(br), color=(0, 128, 0), thickness=1, tipLength=0.02)
    cv2.arrowedLine(img_out, tuple(br), tuple(bl), color=(0, 128, 0), thickness=1, tipLength=0.02)
    cv2.putText(img_out, f"Bottom: {m['bottom_width_px']:.1f}px",
                tuple(bottom_center + np.array([0, 25])), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 128, 0), 2, lineType=cv2.LINE_AA)

    # Length
    cv2.arrowedLine(img_out, tuple(top_center), tuple(bottom_center),
                    color=(255, 0, 255), thickness=1, tipLength=0.02)
    cv2.arrowedLine(img_out, tuple(bottom_center), tuple(top_center),
                    color=(255, 0, 255), thickness=1, tipLength=0.02)
    cv2.putText(img_out, f"Length: {m['length_px']:.1f}px",
                (length_mid[0] + 15, length_mid[1]), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 255), 2, lineType=cv2.LINE_AA)

    # === Summary box ===
    summary = [
        f"Ball Diameter: {2*ball.radius_px:.1f}px",
        f"Large Diameter: {m['large_diameter_px']:.1f}px",
        f"Small Diameter: {m['small_diameter_px']:.1f}px",
        f"Taper Ratio: {m['taper_ratio']:.3f}",
    ]
    if m["is_inverted"]:
        summary.append("⚠️ Inverted cone")

    box_x, box_y = 20, 40
    for i, line in enumerate(summary):
        y = box_y + i * 25
        cv2.putText(img_out, line, (box_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, lineType=cv2.LINE_AA)

    return img_out


def main(args):
    # Load image
    image_bgr = cv2.imread(args.image)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Load YOLO annotations
    print("Loading annotations")
    annotations = load_yolo_annotations(args.yolo)

    ball_ann = next((ann for ann in annotations if ann["class_id"] == 0), None)
    ferrule_ann = next((ann for ann in annotations if ann["class_id"] == 1), None)

    if not ball_ann or not ferrule_ann:
        raise ValueError("Could not find both class_id=0 (ball) and class_id=1 (ferrule) in the YOLO file.")

    ball_box = (ball_ann['x_center'], ball_ann['y_center'], ball_ann['width'], ball_ann['height'])
    ferrule_box = (ferrule_ann['x_center'], ferrule_ann['y_center'], ferrule_ann['width'], ferrule_ann['height'])

    print("Loading SAM model")
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)

    print("Processing ball...")
    ball = BallDimensions(image_rgb, ball_box, predictor)
    ball.process()
    print(f"Ball: center px={ball.center_px}, radius px={ball.radius_px}, radius in={ball.radius_in:.3f}")

    print("Processing ferrule...")
    ferrule = FerruleDimensions(image_rgb, ferrule_box, predictor)
    ferrule.process()

    # Save the white-background + filled black mask
    filled_mask = ferrule.get_filled_mask_image()
    cv2.imwrite("mask.png", filled_mask)
    print("🖼️  Saved filled mask to mask.png")

    print("Rendering frustum image")
    frustum_img = ferrule.plot_frustum(ferrule.result["params"], ferrule.points)
    cv2.imwrite("frustum.jpg", frustum_img)
    print("🖼️  Saved frustum overlay to frustum.png")

    print("Generating final measured image...")
    measured_img = create_measured_image(ball, ferrule)
    cv2.imwrite("processed.jpg", measured_img)
    print("🖼️  Saved annotated measurement image to processed.jpg")

    measurements = ferrule.result['measurements']
    print(f"Ferrule taper ratio: {measurements['taper_ratio']:.3f}")
    print(f"Measurements {measurements}")

    #print("Plotting...")
    #ferrule.plot_frustum_with_measurements(ferrule.image, ferrule.result, ferrule.points)

    print("✅ Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--yolo", required=True, help="Path to YOLO annotation file")
    main(p.parse_args())
