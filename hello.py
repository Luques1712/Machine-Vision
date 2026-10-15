"""
Brain MRI tumour segmentation (GMM) with ground-truth comparison.

Folder layout (beside this script):

    images/   <- MRI scans            (e.g. brisc2025_test_00010_gl_ax_t1.jpg)
    masks/    <- matching masks        (e.g. brisc2025_test_00010_gl_ax_t1.png)
    loading.gif                        (optional, shown on the loading screen)

A mask is matched to its image by file name (same name, any image extension;
"<name>_mask" is also accepted).

How to use:
  1. Press "Choose image" (top left) and pick a scan from the images folder.
  2. Click once inside the suspected tumour (the pixel is highlighted).
     Pick a "Method" (all "Adaptive" methods first grow a region from your
     seed until it would leak into normal brain, then refine its border):
       - Adaptive + Watershed (default): snaps the border to the strongest
         edge around the grown region.
       - Adaptive + Random walker: like the watershed but a graph-based
         border (needs scikit-image: pip install scikit-image).
       - Adaptive + GMM: refines the border with a Gaussian Mixture Model
         fitted around the tumour. "Clusters" sets its number of components.
       - Whole-image GMM: the original method (cluster the full image and
         take the cluster that contains the seed). Needs about 3-6 clusters.
     The Adaptive methods work best for tumours brighter than the tissue
     around them.
  3. Press ">>". A loading screen is shown while the image is segmented.
  4. The output appears on the right. Use the "Output" selector to switch
     between Prediction, Ground truth (your mask) and Error map.
  5. The accuracy of the generated mask versus your mask is shown below.

Saved to segmentation_output/:
  <name>_mask.png, <name>_overlay.png, <name>_comparison.png and a running
  results.csv with one row of scores per segmented image.

Educational prototype only.
NOT a diagnostic medical device.
"""

import base64
import csv
import queue
import threading
import tkinter as tk
import warnings
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

try:  # optional: only needed for the random walker method
    from skimage.segmentation import random_walker
except ImportError:
    random_walker = None


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

IMAGES_DIR_NAME = "images"
MASKS_DIR_NAME = "masks"
OUTPUT_DIR_NAME = "segmentation_output"

PANEL_SIZE = 400  # width and height of each image panel (pixels)

# GMM settings
METHOD_WATERSHED = "Adaptive + Watershed"
METHOD_RANDOM_WALKER = "Adaptive + Random walker"
METHOD_GMM = "Adaptive + GMM"
METHOD_CLASSIC = "Whole-image GMM"
METHODS = (METHOD_WATERSHED, METHOD_RANDOM_WALKER, METHOD_GMM, METHOD_CLASSIC)
DEFAULT_METHOD = METHOD_WATERSHED
GMM_METHODS = (METHOD_GMM, METHOD_CLASSIC)   # methods that use "Clusters"
DEFAULT_COMPONENTS = {METHOD_GMM: 2, METHOD_CLASSIC: 3}
MIN_COMPONENTS, MAX_COMPONENTS = 2, 8
FIT_SAMPLES = None       # None = fit the GMM on every pixel (most accurate).
                         # A number such as 20000 is faster but can miss
                         # small tumours.

# GIF shown on the loading screen. Put it in the same folder as this script.
LOADING_GIF_NAME = "loading.gif"
GIF_MAX_SIZE = 250     # larger GIFs are shrunk to fit (pixels)
GIF_FRAME_DELAY = 60   # milliseconds between GIF frames

VIEW_NAMES = ("Prediction", "Ground truth", "Error map")
VIEW_LEGENDS = {
    "Prediction": "Red = region segmented by the GMM",
    "Ground truth": "Green = mask from your masks folder",
    "Error map": "Green = correct   |   Red = false positive   |   "
                 "Blue = missed tumour",
}


# --------------------------------------------------------------------------
# Image processing
# --------------------------------------------------------------------------
def load_grayscale_image(image_path: Path) -> np.ndarray:
    """Load an image and normalize its intensity range."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    low, high = np.percentile(image, [1, 99])

    if high > low:
        image = np.clip(
            (image.astype(np.float32) - low) / (high - low), 0, 1
        )
        image = (image * 255).astype(np.uint8)

    return image


def find_mask_file(masks_dir: Path, stem: str):
    """Find the mask that belongs to an image (matched by file name)."""
    if masks_dir is None or not masks_dir.is_dir():
        return None

    for name in (stem, f"{stem}_mask"):
        for ext in SUPPORTED_EXTENSIONS:
            candidate = masks_dir / f"{name}{ext}"
            if candidate.is_file():
                return candidate

    # Case-insensitive fallback (e.g. .PNG on case-sensitive systems)
    wanted = {stem.lower(), f"{stem}_mask".lower()}
    for f in masks_dir.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS \
                and f.stem.lower() in wanted:
            return f
    return None


def load_mask(mask_path: Path, shape) -> np.ndarray:
    """Load a ground-truth mask as a 0/255 uint8 array matching `shape`."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {mask_path}")

    if mask.shape != shape:
        mask = cv2.resize(
            mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return (mask > 127).astype(np.uint8) * 255


def build_features(image: np.ndarray) -> np.ndarray:
    """Per-pixel features: intensity, x and y (spatial ones down-weighted)."""
    height, width = image.shape

    intensity = image.astype(np.float32) / 255.0
    yy, xx = np.mgrid[0:height, 0:width]

    x_norm = xx.astype(np.float32) / max(width - 1, 1)
    y_norm = yy.astype(np.float32) / max(height - 1, 1)

    return np.column_stack([
        intensity.ravel(),
        (x_norm * 0.35).ravel(),
        (y_norm * 0.35).ravel(),
    ])


def segment_image(image: np.ndarray, seed_xy,
                  n_components=DEFAULT_COMPONENTS[METHOD_CLASSIC],
                  fit_samples=FIT_SAMPLES) -> np.ndarray:
    """
    Segment using the GMM cluster that contains the seed pixel.
    By default the GMM is fitted on every pixel. If fit_samples is a number,
    only a random subsample is used for fitting (faster, less accurate).
    """
    height, width = image.shape

    if height * width < max(30, n_components * 10):
        raise ValueError("The image is too small for segmentation.")

    sx, sy = seed_xy
    features = build_features(image)

    if fit_samples is not None and features.shape[0] > fit_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(features.shape[0], fit_samples, replace=False)
        fit_data = features[idx]
    else:
        fit_data = features

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        reg_covar=1e-5,
        random_state=42,
        n_init=2,
    ).fit(fit_data)

    labels = gmm.predict(features).reshape(height, width)

    mask = (labels == labels[sy, sx]).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    seed_component = component_labels[sy, sx]

    if seed_component != 0:
        mask = np.where(component_labels == seed_component, 255, 0)
    elif count > 1:
        # Seed pixel was removed by morphology: keep the largest component
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = int(np.argmax(areas)) + 1
        mask = np.where(component_labels == largest_label, 255, 0)

    return mask.astype(np.uint8)


# --------------------------------------------------------------------------
# Adaptive + GMM method
# --------------------------------------------------------------------------
def fill_holes(mask: np.ndarray) -> np.ndarray:
    flood = mask.copy()
    fill_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, fill_mask, (0, 0), 255)
    return mask | cv2.bitwise_not(flood)


def seed_component(mask: np.ndarray, seed_xy) -> np.ndarray:
    """Keep the connected component with the seed (else the largest one)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    if count <= 1:
        return mask
    component = labels[seed_xy[1], seed_xy[0]]
    if component == 0:
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == component).astype(np.uint8) * 255


def grow_region(image: np.ndarray, seed_xy, step=3, jump_ratio=1.35,
                jump_frac=0.02, cap=0.30, floor=0.4):
    """
    Stage 1 (all Adaptive methods): seeded region growing by threshold sweep.

    A brightness threshold is lowered step by step. At each step the region
    connected to the area around the seed is kept (holes are filled, so a dark
    tumour core does not matter). While the region is inside the tumour it
    grows slowly; when it leaks into normal tissue its area suddenly jumps.
    Sweeping stops at the first jump, and the most stable region before it
    (the one whose area changes least between steps) is returned.

    Returns a coarse mask, or None if no sensible region was found.
    """
    height, width = image.shape
    sx, sy = seed_xy

    smooth = cv2.GaussianBlur(image, (0, 0), 2.0)

    radius = max(4, round(0.03 * min(height, width)))
    disc = np.zeros((height, width), np.uint8)
    cv2.circle(disc, (sx, sy), radius, 1, -1)

    # Tumour brightness = upper part of the values around the seed
    start = float(np.percentile(smooth[disc > 0], 75))

    open_r = max(2, round(0.01 * min(height, width)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * open_r + 1, 2 * open_r + 1)
    )

    candidates = []  # (threshold, area, mask)
    threshold = int(start)
    lowest = int(0.45 * start)

    while threshold >= lowest:
        mask = ((smooth >= threshold) * 255).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        count, labels, _, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        if count > 1:
            overlap = np.bincount(labels[disc > 0], minlength=count)
            overlap[0] = 0
            component = int(np.argmax(overlap))

            if overlap[component] > 0:
                region = fill_holes(
                    (labels == component).astype(np.uint8) * 255
                )
                area = int((region > 0).sum())

                if candidates:
                    previous = candidates[-1][1]
                    if (area >= jump_ratio * previous
                            and area - previous >= jump_frac * height * width):
                        break  # leaked into the surroundings
                if area > cap * height * width:
                    break

                candidates.append((threshold, area, region))

        threshold -= step

    if len(candidates) < 3:
        return None

    areas = np.array([c[1] for c in candidates], dtype=float)
    growth = np.r_[np.nan, np.diff(areas) / areas[:-1]]

    # Stability = relative area growth averaged over three neighbouring steps
    stability = np.full(len(candidates), np.inf)
    for i in range(1, len(candidates)):
        stability[i] = np.nanmean(growth[max(1, i - 1):i + 2])
    stability[areas < floor * areas[-1]] = np.inf  # ignore tiny inner specks

    return candidates[int(np.argmin(stability))][2]


def refine_with_gmm(image: np.ndarray, seed_xy, coarse: np.ndarray,
                    n_components=2) -> np.ndarray:
    """
    Stage 2: fit a GMM only in a window around the coarse region (where the
    tumour is a large share of the pixels) and use it to refine the border.
    """
    height, width = image.shape
    ys, xs = np.nonzero(coarse)
    box_w, box_h = xs.max() - xs.min(), ys.max() - ys.min()
    margin_x, margin_y = int(box_w * 0.35) + 5, int(box_h * 0.35) + 5

    x0, x1 = max(0, xs.min() - margin_x), min(width, xs.max() + margin_x + 1)
    y0, y1 = max(0, ys.min() - margin_y), min(height, ys.max() + margin_y + 1)

    smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.5)
    window = smooth[y0:y1, x0:x1]
    data = window.reshape(-1, 1) / 255.0

    if data.shape[0] < 50 * n_components:
        return coarse

    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full",
        reg_covar=1e-4, random_state=42, n_init=3,
    ).fit(data)
    labels = gmm.predict(data).reshape(window.shape)

    # Tumour cluster = the cluster most common inside the coarse region
    core = cv2.erode(coarse, np.ones((5, 5), np.uint8))[y0:y1, x0:x1] > 0
    if not core.any():
        core = coarse[y0:y1, x0:x1] > 0
    tumour_label = np.bincount(labels[core]).argmax()

    mask = np.zeros_like(image)
    mask[y0:y1, x0:x1] = ((labels == tumour_label) * 255).astype(np.uint8)

    # The refined border may only move slightly beyond the coarse region
    grow = 2 * int(0.04 * max(box_w, box_h)) + 3
    limit = cv2.dilate(
        coarse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow, grow))
    )
    mask = cv2.bitwise_and(mask, limit)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    if not mask[seed_xy[1], seed_xy[0]] and not mask.any():
        return coarse

    return fill_holes(seed_component(mask, seed_xy))


def band_markers(coarse: np.ndarray, fraction=0.12):
    """Sure-foreground (shrunk) and sure-background (grown) markers."""
    area = int((coarse > 0).sum())
    d = max(3, int(fraction * 2 * np.sqrt(area / np.pi)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    sure_fg = cv2.erode(coarse, kernel)
    outside = cv2.dilate(coarse, kernel)
    if not sure_fg.any():
        sure_fg = coarse
    return sure_fg, outside


def finish_mask(mask: np.ndarray, seed_xy, fallback: np.ndarray):
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    if not mask.any():
        return fallback
    return fill_holes(seed_component(mask, seed_xy))


def refine_with_watershed(image: np.ndarray, seed_xy, coarse: np.ndarray):
    """
    Stage 2: marker-controlled watershed. The border is placed on the
    strongest intensity edge in the uncertain band around the coarse region.
    """
    smooth = cv2.GaussianBlur(image, (0, 0), 1.5)
    sure_fg, outside = band_markers(coarse)

    markers = np.zeros(image.shape, np.int32)
    markers[outside == 0] = 1   # background
    markers[sure_fg > 0] = 2    # tumour
    cv2.watershed(cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR), markers)

    mask = (markers == 2).astype(np.uint8) * 255
    return finish_mask(mask, seed_xy, coarse)


def refine_with_random_walker(image: np.ndarray, seed_xy, coarse: np.ndarray):
    """
    Stage 2: random walker (graph partitioning). Each uncertain pixel joins
    the tumour or the background depending on which marker a random walk
    starting from it is more likely to reach first.
    """
    smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.5) / 255.0
    sure_fg, outside = band_markers(coarse)

    labels = np.zeros(image.shape, np.int32)
    labels[outside == 0] = 1
    labels[sure_fg > 0] = 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = random_walker(smooth, labels, beta=130, mode="cg_j")

    mask = (result == 2).astype(np.uint8) * 255
    return finish_mask(mask, seed_xy, coarse)


NO_REGION_NOTE = (
    "No clear region was found around the click. The tumour may be too faint, "
    "or the click may be outside it. Try another spot inside the tumour, or "
    "use the Whole-image GMM method."
)


def run_segmentation(image, seed_xy, method, n_components):
    """Run the chosen method. Returns (mask, note)."""
    if method == METHOD_CLASSIC:
        return segment_image(image, seed_xy, n_components), ""

    coarse = grow_region(image, seed_xy)
    if coarse is None:
        return np.zeros_like(image), NO_REGION_NOTE

    if method == METHOD_GMM:
        return refine_with_gmm(image, seed_xy, coarse, n_components), ""

    if method == METHOD_RANDOM_WALKER:
        if random_walker is None:
            return (refine_with_watershed(image, seed_xy, coarse),
                    "scikit-image is not installed (pip install "
                    "scikit-image), so the watershed was used instead.")
        return refine_with_random_walker(image, seed_xy, coarse), ""

    return refine_with_watershed(image, seed_xy, coarse), ""


# --------------------------------------------------------------------------
# Comparison with the ground-truth mask
# --------------------------------------------------------------------------
def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, g = pred > 0, gt > 0
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    tn = int(np.logical_and(~p, ~g).sum())

    def ratio(num, den):
        return num / den if den > 0 else float("nan")

    return {
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "pixel_accuracy": ratio(tp + tn, tp + tn + fp + fn),
        "gt_area": int(g.sum()),
        "pred_area": int(p.sum()),
    }


def tint(base_bgr, mask, colour, alpha=0.4):
    """Tint a mask region on the image and outline it."""
    out = base_bgr.copy()
    layer = out.copy()
    layer[mask > 0] = colour
    out = cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)

    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(out, contours, -1, colour, 2)
    return out


def make_error_map(image, pred, gt):
    """Green = true positive, red = false positive, blue = missed tumour."""
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    p, g = pred > 0, gt > 0

    error = (base * 0.5).astype(np.uint8)
    error[np.logical_and(p, g)] = (0, 200, 0)
    error[np.logical_and(p, ~g)] = (0, 0, 255)
    error[np.logical_and(~p, g)] = (255, 100, 0)
    return error


def with_title(panel, text):
    bar = np.full((28, panel.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def save_results(image, pred, views, metrics, seed_xy, output_dir: Path,
                 stem: str, n_components: int, method: str):
    """Save mask, overlay, comparison image and append to results.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_dir / f"{stem}_mask.png"), pred)
    cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), views["Prediction"])

    base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if "Ground truth" in views:
        panels = [
            with_title(base, "Input MRI"),
            with_title(views["Ground truth"], "Ground truth"),
            with_title(views["Prediction"], "Prediction"),
            with_title(views["Error map"], "TP green | FP red | FN blue"),
        ]
    else:
        panels = [with_title(base, "Input MRI"),
                  with_title(views["Prediction"], "Prediction")]
    cv2.imwrite(str(output_dir / f"{stem}_comparison.png"), np.hstack(panels))

    if metrics is not None:
        csv_path = output_dir / "results.csv"
        columns = ["time", "image", "seed_x", "seed_y", "dice", "iou",
                   "precision", "recall", "pixel_accuracy",
                   "gt_area", "pred_area", "clusters", "method"]

        # An older results.csv with different columns is kept under a new name
        if csv_path.is_file():
            with open(csv_path, encoding="utf-8") as f:
                header = f.readline().strip().split(",")
            if header != columns:
                backup = output_dir / (
                    f"results_old_{datetime.now():%Y%m%d_%H%M%S}.csv"
                )
                csv_path.rename(backup)

        new_file = not csv_path.is_file()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(columns)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), stem,
                seed_xy[0], seed_xy[1],
                *(f"{metrics[c]:.4f}" for c in columns[4:9]),
                metrics["gt_area"], metrics["pred_area"], n_components,
                method,
            ])


# --------------------------------------------------------------------------
# User interface
# --------------------------------------------------------------------------
def to_photo(bgr_or_gray: np.ndarray) -> tk.PhotoImage:
    """Convert a numpy image to a Tk PhotoImage (no Pillow needed)."""
    ok, buffer = cv2.imencode(".png", bgr_or_gray)
    if not ok:
        raise RuntimeError("Could not encode image for display.")
    return tk.PhotoImage(data=base64.b64encode(buffer.tobytes()))


def percent(value) -> str:
    return "n/a" if value is None or np.isnan(value) else f"{value * 100:.1f}%"


class SegmentationApp:
    def __init__(self, root: tk.Tk, project_dir: Path):
        self.root = root
        self.images_dir = project_dir / IMAGES_DIR_NAME
        self.masks_dir = project_dir / MASKS_DIR_NAME
        self.output_dir = project_dir / OUTPUT_DIR_NAME
        self.gif_path = project_dir / LOADING_GIF_NAME

        # Current image state
        self.image = None
        self.image_path = None
        self.mask_path = None
        self.seed = None
        self.scale = 1.0
        self.disp_w = self.disp_h = PANEL_SIZE
        self.off_x = self.off_y = 0
        self.input_photo = None
        self.view_photos = {}

        self.result_queue = queue.Queue()

        root.title("Brain MRI Tumour Segmentation (GMM)")
        root.resizable(False, False)

        container = tk.Frame(root)
        container.pack(fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.main_frame = tk.Frame(container, padx=20, pady=15)
        self.loading_frame = tk.Frame(container, padx=20, pady=15)
        for frame in (self.main_frame, self.loading_frame):
            frame.grid(row=0, column=0, sticky="nsew")

        self._build_main_ui()
        self._build_loading_ui()
        self.show_main()

    # ---------------- UI #1 ----------------
    def _build_main_ui(self):
        f = self.main_frame

        # Top bar: "Choose image" button at the top left
        top = tk.Frame(f)
        top.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        tk.Button(
            top, text="Choose image", font=("Arial", 11, "bold"),
            command=self.choose_image,
        ).pack(side="left")

        tk.Button(
            top, text="Masks folder...", command=self.choose_masks_folder,
        ).pack(side="left", padx=(8, 0))

        tk.Label(top, text="Method:").pack(side="left", padx=(15, 2))
        self.method_var = tk.StringVar(value=DEFAULT_METHOD)
        tk.OptionMenu(
            top, self.method_var, *METHODS, command=self.on_method_change
        ).pack(side="left")

        tk.Label(top, text="Clusters:").pack(side="left", padx=(10, 2))
        self.components_var = tk.IntVar(value=DEFAULT_COMPONENTS[METHOD_GMM])
        self.components_box = tk.Spinbox(
            top, from_=MIN_COMPONENTS, to=MAX_COMPONENTS, width=3,
            textvariable=self.components_var,
            state="readonly" if DEFAULT_METHOD in GMM_METHODS else "disabled",
        )
        self.components_box.pack(side="left")

        self.file_label = tk.Label(
            top, text="No image selected", anchor="w", font=("Arial", 10)
        )
        self.file_label.pack(side="left", padx=(15, 0))

        # Titles
        tk.Label(f, text="Input image", font=("Arial", 12, "bold")).grid(
            row=1, column=0
        )

        selector = tk.Frame(f)
        selector.grid(row=1, column=2)
        tk.Label(selector, text="Output:", font=("Arial", 12, "bold")).pack(
            side="left", padx=(0, 6)
        )
        self.view_var = tk.StringVar(value=VIEW_NAMES[0])
        for name in VIEW_NAMES:
            tk.Radiobutton(
                selector, text=name, value=name, variable=self.view_var,
                command=self.refresh_output_view,
            ).pack(side="left")

        # Input canvas
        self.input_canvas = tk.Canvas(
            f, width=PANEL_SIZE, height=PANEL_SIZE,
            bg="#222222", highlightthickness=0, borderwidth=0,
            cursor="crosshair",
        )
        self.input_canvas.grid(row=2, column=0, padx=10, pady=8)
        self.input_canvas.bind("<Button-1>", self.on_click)
        self._show_placeholder(
            self.input_canvas, "Press 'Choose image' to begin"
        )

        # ">>" button
        self.next_button = tk.Button(
            f, text=">>", font=("Arial", 16, "bold"),
            width=5, command=self.on_next,
        )
        self.next_button.grid(row=2, column=1, padx=10)

        # Output canvas
        self.output_canvas = tk.Canvas(
            f, width=PANEL_SIZE, height=PANEL_SIZE,
            bg="#222222", highlightthickness=1,
        )
        self.output_canvas.grid(row=2, column=2, padx=10, pady=8)
        self._show_placeholder(self.output_canvas, "Output will appear here")

        # Status and legend
        self.status = tk.Label(
            f, text="Choose an image, click inside the tumour, then press  >>",
            font=("Arial", 10),
        )
        self.status.grid(row=3, column=0, sticky="w", padx=10)

        self.legend = tk.Label(f, text="", font=("Arial", 10), fg="#444444")
        self.legend.grid(row=3, column=2, sticky="w", padx=10)

        # Accuracy panel
        box = tk.LabelFrame(
            f, text=" Accuracy vs. ground-truth mask ", padx=12, pady=6,
            font=("Arial", 10, "bold"),
        )
        box.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 0))

        self.dice_label = tk.Label(
            box, text="Dice: -", font=("Arial", 20, "bold")
        )
        self.dice_label.pack(anchor="w")

        self.metrics_label = tk.Label(
            box, text="Run a segmentation to see the scores.",
            font=("Arial", 11), justify="left",
        )
        self.metrics_label.pack(anchor="w")

        tk.Label(
            box,
            text="Dice is the main score (overlap with your mask). Pixel "
                 "accuracy looks high even for poor masks, because most "
                 "pixels are background.",
            font=("Arial", 9), fg="#666666", justify="left",
        ).pack(anchor="w", pady=(4, 0))

    @staticmethod
    def _show_placeholder(canvas, text):
        canvas.delete("all")
        canvas.create_text(
            PANEL_SIZE // 2, PANEL_SIZE // 2, text=text,
            fill="#aaaaaa", font=("Arial", 12),
        )

    def on_method_change(self, method):
        """Clusters only applies to the GMM methods."""
        if method in GMM_METHODS:
            self.components_var.set(DEFAULT_COMPONENTS[method])
            self.components_box.config(state="readonly")
        else:
            self.components_box.config(state="disabled")

    # ---------------- choosing files ----------------
    def choose_image(self):
        start = self.images_dir if self.images_dir.is_dir() \
            else self.images_dir.parent

        path = filedialog.askopenfilename(
            title="Choose an MRI image",
            initialdir=str(start),
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.load_image(Path(path))

    def choose_masks_folder(self):
        start = self.masks_dir if self.masks_dir.is_dir() \
            else self.masks_dir.parent

        folder = filedialog.askdirectory(
            title="Choose the folder containing the masks",
            initialdir=str(start),
        )
        if folder:
            self.masks_dir = Path(folder)
            if self.image_path is not None:
                self.mask_path = find_mask_file(
                    self.masks_dir, self.image_path.stem
                )
                self.update_file_label()

    def load_image(self, path: Path):
        try:
            image = load_grayscale_image(path)
        except ValueError as exc:
            messagebox.showerror("Cannot open image", str(exc))
            return

        self.image = image
        self.image_path = path
        self.mask_path = find_mask_file(self.masks_dir, path.stem)
        self.seed = None
        self.update_file_label()

        height, width = image.shape
        self.scale = min(PANEL_SIZE / width, PANEL_SIZE / height)
        self.disp_w = max(1, int(round(width * self.scale)))
        self.disp_h = max(1, int(round(height * self.scale)))
        self.off_x = (PANEL_SIZE - self.disp_w) // 2
        self.off_y = (PANEL_SIZE - self.disp_h) // 2

        self.input_photo = to_photo(
            cv2.resize(image, (self.disp_w, self.disp_h),
                       interpolation=cv2.INTER_NEAREST)
        )
        self.input_canvas.delete("all")
        self.input_canvas.create_image(
            self.off_x, self.off_y, anchor="nw", image=self.input_photo
        )

        # Clear any previous result
        self.view_photos = {}
        self.view_var.set(VIEW_NAMES[0])
        self._show_placeholder(self.output_canvas, "Output will appear here")
        self.legend.config(text="")
        self.dice_label.config(text="Dice: -")
        self.metrics_label.config(text="Run a segmentation to see the scores.")
        self.status.config(text="Click inside the suspected tumour, "
                                "then press  >>")

    def update_file_label(self):
        if self.image_path is None:
            return
        mask_text = (f"mask: {self.mask_path.name}" if self.mask_path
                     else "mask: NOT FOUND in masks folder")
        self.file_label.config(
            text=f"{self.image_path.name}    ({mask_text})",
            fg="black" if self.mask_path else "#b00000",
        )

    # ---------------- seed selection ----------------
    def on_click(self, event):
        if self.image is None:
            return

        ix = int((event.x - self.off_x) / self.scale)
        iy = int((event.y - self.off_y) / self.scale)

        height, width = self.image.shape
        if not (0 <= ix < width and 0 <= iy < height):
            return

        self.seed = (ix, iy)
        self.draw_seed()
        self.status.config(
            text=f"Seed pixel: x={ix}, y={iy}  -  click again to move it, "
                 "or press  >>"
        )

    def draw_seed(self):
        """Highlight the seeded pixel (filled box plus a ring marker)."""
        self.input_canvas.delete("seed")
        sx, sy = self.seed

        x0 = self.off_x + sx * self.scale
        y0 = self.off_y + sy * self.scale
        size = max(self.scale, 3)
        cx, cy = x0 + self.scale / 2, y0 + self.scale / 2

        self.input_canvas.create_rectangle(
            cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2,
            fill="yellow", outline="red", width=1, tags="seed",
        )
        self.input_canvas.create_oval(
            cx - 10, cy - 10, cx + 10, cy + 10,
            outline="red", width=2, tags="seed",
        )

    # ---------------- run ----------------
    def on_next(self):
        if self.image is None:
            messagebox.showwarning(
                "No image", "Press 'Choose image' and pick an MRI scan first."
            )
            return
        if self.seed is None:
            messagebox.showwarning(
                "No seed", "Please click a seed point on the input image first."
            )
            return

        gt = None
        if self.mask_path is not None:
            try:
                gt = load_mask(self.mask_path, self.image.shape)
            except ValueError as exc:
                messagebox.showwarning("Mask problem", str(exc))

        method = self.method_var.get()
        components = ""  # only used by the GMM methods
        if method in GMM_METHODS:
            try:
                components = int(self.components_var.get())
            except (tk.TclError, ValueError):
                components = DEFAULT_COMPONENTS[method]

        self.show_loading()
        threading.Thread(
            target=self.worker,
            args=(self.image, self.seed, gt, self.image_path.stem,
                  method, components),
            daemon=True,
        ).start()
        self.root.after(100, self.poll_worker)

    # ---------------- UI #2 ----------------
    def _load_gif_frames(self):
        """Load every frame of the loading GIF (empty list if unavailable)."""
        frames = []
        if not self.gif_path.is_file():
            print(f"Loading GIF not found: {self.gif_path} "
                  "(showing text only)")
            return frames

        index = 0
        while True:
            try:
                frame = tk.PhotoImage(
                    file=str(self.gif_path), format=f"gif -index {index}"
                )
            except tk.TclError:
                break

            factor = max(1, -(-max(frame.width(), frame.height())
                              // GIF_MAX_SIZE))
            if factor > 1:
                frame = frame.subsample(factor, factor)

            frames.append(frame)
            index += 1

        return frames

    def _build_loading_ui(self):
        f = self.loading_frame

        box = tk.Frame(f, highlightbackground="black",
                       highlightthickness=2, padx=60, pady=40)
        box.pack(expand=True, padx=120, pady=100)

        self.gif_frames = self._load_gif_frames()
        self.gif_label = tk.Label(box)
        if self.gif_frames:
            self.gif_label.config(image=self.gif_frames[0])
            self.gif_label.pack(pady=(0, 15))

        self.loading_label = tk.Label(
            box, text="Loading", font=("Arial", 18, "bold")
        )
        self.loading_label.pack()

        self._loading_active = False
        self._gif_index = 0
        self._dots = 0
        self._gif_job = None
        self._dots_job = None

    def _animate_gif(self):
        if not self._loading_active or not self.gif_frames:
            return
        self._gif_index = (self._gif_index + 1) % len(self.gif_frames)
        self.gif_label.config(image=self.gif_frames[self._gif_index])
        self._gif_job = self.root.after(GIF_FRAME_DELAY, self._animate_gif)

    def _animate_dots(self):
        if not self._loading_active:
            return
        self._dots = (self._dots + 1) % 4
        self.loading_label.config(text="Loading" + "." * self._dots)
        self._dots_job = self.root.after(400, self._animate_dots)

    # ---------------- screen switching ----------------
    def show_main(self):
        self._loading_active = False
        for job in (self._gif_job, self._dots_job):
            if job is not None:
                self.root.after_cancel(job)
        self._gif_job = self._dots_job = None
        self.main_frame.tkraise()

    def show_loading(self):
        self.loading_frame.tkraise()
        self._loading_active = True
        self._gif_index = 0
        self._animate_gif()
        self._animate_dots()

    # ---------------- background work ----------------
    def worker(self, image, seed, gt, stem, method, components):
        try:
            pred, method_note = run_segmentation(
                image, seed, method, components
            )

            base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            views = {"Prediction": tint(base, pred, (0, 0, 255))}
            metrics = None

            if gt is not None:
                views["Ground truth"] = tint(base, gt, (0, 200, 0))
                views["Error map"] = make_error_map(image, pred, gt)
                metrics = compute_metrics(pred, gt)

            save_note = ""
            try:
                save_results(image, pred, views, metrics, seed,
                             self.output_dir, stem, components, method)
            except OSError as exc:  # e.g. results.csv open in Excel
                save_note = f"(could not save some files: {exc})"

            self.result_queue.put(
                ("ok", views, metrics, " ".join([method_note, save_note]))
            )
        except Exception as exc:
            self.result_queue.put(("error", exc, None, ""))

    def poll_worker(self):
        try:
            kind, payload, metrics, note = self.result_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self.poll_worker)
            return

        self.show_main()

        if kind == "error":
            messagebox.showerror("Segmentation failed", str(payload))
            return

        # Create display images (must happen in the main thread)
        self.view_photos = {
            name: to_photo(cv2.resize(
                img, (self.disp_w, self.disp_h),
                interpolation=cv2.INTER_NEAREST))
            for name, img in payload.items()
        }
        self.view_var.set(VIEW_NAMES[0])
        self.refresh_output_view()
        self.show_metrics(metrics)

        if self.mask_path is None:
            self.status.config(text="Done (no mask found, so no accuracy). "
                                    f"Saved in: {self.output_dir.name}/ "
                                    f"{note}")
        else:
            self.status.config(
                text=f"Done. Saved in: {self.output_dir.name}/ {note}"
            )

    def refresh_output_view(self):
        name = self.view_var.get()
        self.legend.config(text=VIEW_LEGENDS.get(name, ""))

        if not self.view_photos:
            return

        self.output_canvas.delete("all")
        if name in self.view_photos:
            self.output_canvas.create_image(
                self.off_x, self.off_y, anchor="nw",
                image=self.view_photos[name],
            )
        else:
            self._show_placeholder(
                self.output_canvas, "No mask found for this image"
            )

    def show_metrics(self, metrics):
        if metrics is None:
            self.dice_label.config(text="Dice: n/a")
            self.metrics_label.config(
                text="No matching mask was found in the masks folder.\n"
                     "Use 'Masks folder...' or check the file names."
            )
            return

        self.dice_label.config(text=f"Dice: {percent(metrics['dice'])}")
        self.metrics_label.config(
            text=(
                f"IoU: {percent(metrics['iou'])}      "
                f"Precision: {percent(metrics['precision'])}      "
                f"Recall: {percent(metrics['recall'])}      "
                f"Pixel accuracy: {percent(metrics['pixel_accuracy'])}\n"
                f"Mask area:  generated {metrics['pred_area']} px   |   "
                f"ground truth {metrics['gt_area']} px"
            )
        )


def main():
    project_dir = Path(__file__).resolve().parent

    root = tk.Tk()
    SegmentationApp(root, project_dir)
    root.mainloop()


if __name__ == "__main__":
    main()