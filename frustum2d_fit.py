import cv2
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ---- Loss configuration (reverted to count) ----

PENALTY_WEIGHT = 5.0

def extract_black_points(image_path):
    # Revert: simple threshold and use all black pixels
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    ys, xs = np.where(mask > 0)
    points = np.column_stack((xs, ys))  # revert: no float32 cast
    return img, points



def trapezoid_corners(params):
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

def point_in_trapezoid(points, corners):
    # Revert: call pointPolygonTest for every point (edge-inclusive)
    contour = corners.astype(np.float32).reshape((-1,1,2))
    return np.array([
        cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0
        for pt in points
    ])

def frustum_loss(params, points):
    # Revert: step penalty with shoelace area
    corners = trapezoid_corners(params)
    inside = point_in_trapezoid(points, corners)
    x = corners[:, 0]
    y = corners[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    loss = area + PENALTY_WEIGHT * np.sum(~inside)
    return loss

def fit_frustum(points, img_shape, restarts=4, jitter=0.10, live_plot=True):
    # Revert to simple percentile-based initializer and single Powell call
    h, w = img_shape[:2]
    center_x0 = np.mean(points[:,0])
    center_y0 = np.mean(points[:,1])
    w_top0 = (np.percentile(points[:,0], 90) - np.percentile(points[:,0], 10)) * 0.5
    h0 = points[:,1].max() - points[:,1].min()
    w_bottom0 = (np.percentile(points[:,0], 95) - np.percentile(points[:,0], 5))
    theta0 = 0.0
    init = [center_x0, center_y0, w_top0, h0, w_bottom0, theta0]
    bounds = [
        (0, w-1), (0, h-1), (1, w), (1, h), (1, w), (-np.pi, np.pi)
    ]

    # Setup live plotting
    if live_plot:
        plt.ion()  # Turn on interactive mode
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left plot: Image with current trapezoid
        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax1.scatter(points[:,0], points[:,1], c='k', s=1, alpha=0.5, label='Points')
        ax1.set_xlim(0, w)
        ax1.set_ylim(h, 0)
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_title('Current Best Fit')
        ax1.legend()

        # Right plot: Loss curve
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Progress')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # one run with per-iteration loss and params capture
    def run_one(x0, run_num=0):
        loss_history = []
        params_history = []

        def _cb(xk):
            current_loss = frustum_loss(xk, points)
            loss_history.append(current_loss)
            params_history.append(np.array(xk, dtype=float))

            # Update live plot every few iterations
            if live_plot and len(loss_history) % 2 == 0:  # Update every 2 iterations
                # Clear and update left plot
                ax1.clear()
                ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax1.scatter(points[:,0], points[:,1], c='k', s=1, alpha=0.5, label='Points')

                # Draw current trapezoid
                corners = trapezoid_corners(xk)
                trap_x = np.append(corners[:, 0], corners[0, 0])
                trap_y = np.append(corners[:, 1], corners[0, 1])
                ax1.plot(trap_x, trap_y, 'r-', linewidth=2, label=f'Current Fit (Loss: {current_loss:.0f})')

                ax1.set_xlim(0, w)
                ax1.set_ylim(h, 0)
                ax1.set_aspect('equal', adjustable='box')
                ax1.set_title(f'Run {run_num+1}, Iteration {len(loss_history)}')
                ax1.legend()

                # Update right plot
                ax2.clear()
                ax2.plot(range(1, len(loss_history) + 1), loss_history, 'b-', linewidth=2)
                ax2.set_xlabel('Iteration')
                ax2.set_ylabel('Loss')
                ax2.set_title(f'Loss Progress (Current: {current_loss:.0f})')
                ax2.grid(True, alpha=0.3)

                plt.draw()
                plt.pause(0.01)  # Small pause to update display

        res = minimize(
            frustum_loss, x0, args=(points,), method='Powell', bounds=bounds, callback=_cb,
            # Revert higher tolerance: relax tolerances and reduce caps
            options={'maxiter': 50, 'maxfev': 10000, 'xtol': 1e-3, 'ftol': 1e-3}
        )
        return res, loss_history, params_history

    # baseline run
    print("Starting baseline optimization run...")
    best_res, best_hist, best_params_hist = run_one(init, 0)
    best_loss = frustum_loss(best_res.x, points)
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
        loss = frustum_loss(res.x, points)
        if loss < best_loss:
            print(f"Found better solution! Loss: {loss:.2f} (was {best_loss:.2f})")
            best_res, best_hist, best_params_hist, best_loss = res, hist, phist, loss
        else:
            print(f"Restart {restart_num + 1} completed. Loss: {loss:.2f} (best: {best_loss:.2f})")

    if live_plot:
        plt.ioff()  # Turn off interactive mode

    # Calculate physical measurements from the fitted frustum
    corners = trapezoid_corners(best_res.x)
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

    # Create result dictionary with measurements
    result = {
        'params': best_res.x,
        'loss_history': best_hist,
        'params_history': best_params_hist,
        'corners': corners,
        'measurements': {
            'large_diameter_px': large_diameter,
            'small_diameter_px': small_diameter,
            'top_width_px': top_width,
            'bottom_width_px': bottom_width,
            'length_px': length,
            'taper_ratio': small_diameter / large_diameter if large_diameter > 0 else 0,
            'is_inverted': top_width > bottom_width  # True if cone is upside down
        }
    }

    return result

def plot_frustum(img, params, points):
    corners = trapezoid_corners(params)
    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.scatter(points[:,0], points[:,1], c='k', s=2, label='Black Dots')
    plt.plot(*np.append(corners, [corners[0]], axis=0).T, 'r-', label='Fitted Frustum')
    plt.legend()
    plt.title('Best Fit 2D Frustum (Trapezoid)')
    plt.xlim(0, img.shape[1])
    plt.ylim(img.shape[0], 0)
    plt.gca().set_aspect('equal', adjustable='box')  # keep geometry visually correct
    plt.show()

def plot_frustum_with_loss(img, params, points, loss_history, params_history=None):
    corners = trapezoid_corners(params)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: image + frustum
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].scatter(points[:,0], points[:,1], c='k', s=2, label='Black Dots')

    # Optional: overlay trapezoid at each iteration to visualize progression
    if params_history and len(params_history) > 0:
        cmap = plt.cm.viridis
        n = len(params_history)
        for i, p in enumerate(params_history):
            cs = trapezoid_corners(p)
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

def plot_frustum_with_measurements(img, result, points):
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



if __name__ == "__main__":
    img, points = extract_black_points("test_data/Nickey_02.png")
    print(f"Loaded {len(points)} points from image")

    # Run with live plotting enabled
    result = fit_frustum(points, img.shape, live_plot=True)
    params = result['params']
    loss_history = result['loss_history']
    params_history = result['params_history']
    measurements = result['measurements']

    print(f"Optimized parameters: {params}")
    print(f"Corners: {trapezoid_corners(params)}")
    print(f"Total loss: {frustum_loss(params, points)}")
    print(f"\nMeasurements (pixels):")
    print(f"  Large diameter: {measurements['large_diameter_px']:.1f}")
    print(f"  Small diameter: {measurements['small_diameter_px']:.1f}")
    print(f"  Top width: {measurements['top_width_px']:.1f}")
    print(f"  Bottom width: {measurements['bottom_width_px']:.1f}")
    print(f"  Length: {measurements['length_px']:.1f}")
    print(f"  Taper ratio: {measurements['taper_ratio']:.3f}")
    if measurements['is_inverted']:
        print(f"  Note: Cone appears inverted (wider at top)")

    # Show optimization result
    plot_frustum_with_loss(img, params, points, loss_history, params_history)

    # Show final result with measurements
    plot_frustum_with_measurements(img, result, points)