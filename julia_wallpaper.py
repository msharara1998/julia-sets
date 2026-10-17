import os
import sys

import matplotlib
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap
from PIL import Image

# ============================================================
# Configuration
# ============================================================

WIDTH: int = 1920
HEIGHT: int = 1080

# Render at 2x resolution, then downsample for cleaner antialiasing
SCALE: int = 2
RENDER_W: int = WIDTH * SCALE
RENDER_H: int = HEIGHT * SCALE

# Colors from the Lorenz script
BG: str = "#0b1020"
GRID: str = "#192238"  # the axis/tick color used in the Lorenz script
CYAN: str = "#00c7d8"
GOLD: str = "#e5ad00"

# Julia set parameter z -> z^2 + c (override with JULIA_C, e.g. "-0.4+0.6j")
DEFAULT_C: complex = complex(-0.79, 0.15)

# View: half-width of the visible region on the real axis
VIEW_HALF_WIDTH: float = 1.6
CENTER: complex = complex(0.0, 0.0)

BAILOUT: float = 256.0  # large bailout gives smooth, band-free values

VERSION: str = "1.0"

# Per-theme settings
#   cmap_name / palette : how values map to color
#   max_iter            : iteration cap
#   smooth              : True = continuous coloring, False = discrete bands
#                         (the reference image shows discrete bands)
#   gamma               : <1 brightens the glow near the set, 1 = linear
THEMES: dict[str, dict] = {
    "image": {
        "max_iter": 100,
        "smooth": False,
        "gamma": 1.0,
    },
    "lorenz": {
        "max_iter": 200,
        "smooth": True,
        "gamma": 0.55,
    },
}

# ============================================================
# Theme selection (environment variable)
# ============================================================

THEME: str = os.environ.get("JULIA_THEME", "image").strip().lower()

if THEME not in THEMES:
    sys.exit(f"Unknown JULIA_THEME '{THEME}'. Choose from: {', '.join(THEMES)}")

C: complex = complex(os.environ.get("JULIA_C", str(DEFAULT_C)).replace(" ", ""))
SETTINGS: dict = THEMES[THEME]


def build_colormap(theme: str) -> Colormap:
    """Builds the colormap for the chosen theme.

    Args:
        theme (str): Either "image" (viridis, as in the reference image)
            or "lorenz" (dark navy -> cyan -> gold, as in the Lorenz script).

    Returns:
        Colormap: A matplotlib colormap mapping [0, 1] to colors.
    """
    if theme == "image":
        return matplotlib.colormaps["viridis"]

    return LinearSegmentedColormap.from_list(
        "lorenz_theme",
        [
            (0.00, BG),
            (0.30, GRID),
            (0.70, CYAN),
            (1.00, GOLD),
        ],
        N=1024,
    )


# ============================================================
# Julia set computation
# ============================================================


def julia_values(
    width: int,
    height: int,
    c: complex,
    max_iter: int,
    smooth: bool,
) -> np.ndarray:
    """Computes normalized escape-time values for a Julia set.

    Only points that have not yet escaped are iterated, which keeps
    the 4K supersampled render fast.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        c (complex): The Julia constant.
        max_iter (int): Maximum number of iterations.
        smooth (bool): Use continuous (smooth) escape values if True,
            integer iteration counts if False.

    Returns:
        np.ndarray: Array of shape (height, width), float in [0, 1].
            Points that never escape (inside the set) are 1.0.
    """
    half_h: float = VIEW_HALF_WIDTH * height / width

    re: np.ndarray = np.linspace(
        CENTER.real - VIEW_HALF_WIDTH, CENTER.real + VIEW_HALF_WIDTH, width
    )
    im: np.ndarray = np.linspace(CENTER.imag + half_h, CENTER.imag - half_h, height)

    z: np.ndarray = (re[None, :] + 1j * im[:, None]).ravel()
    values: np.ndarray = np.full(z.shape, float(max_iter), dtype=np.float64)

    active: np.ndarray = np.arange(z.size)
    z_active: np.ndarray = z.copy()

    log_bailout_term: float = np.log(2.0)

    for n in range(max_iter):
        z_active = z_active * z_active + c
        mag: np.ndarray = np.abs(z_active)
        escaped: np.ndarray = mag > BAILOUT

        if escaped.any():
            idx: np.ndarray = active[escaped]
            if smooth:
                values[idx] = n + 1 - np.log(np.log(mag[escaped])) / log_bailout_term
            else:
                values[idx] = n + 1

            keep: np.ndarray = ~escaped
            active = active[keep]
            z_active = z_active[keep]

        if active.size == 0:
            break

    values = np.clip(values / max_iter, 0.0, 1.0)
    return values.reshape(height, width)


# ============================================================
# Render
# ============================================================

values: np.ndarray = julia_values(
    RENDER_W, RENDER_H, C, SETTINGS["max_iter"], SETTINGS["smooth"]
)

values = values ** SETTINGS["gamma"]

cmap: Colormap = build_colormap(THEME)
rgb: np.ndarray = (cmap(values)[..., :3] * 255).astype(np.uint8)

img: Image.Image = Image.fromarray(rgb, mode="RGB")

# ============================================================
# Downsample to exact 1920x1080
# ============================================================

img = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)

output_file: str = f"julia_fullhd_wallpaper_{THEME}_{VERSION}.png"

img.save(output_file, format="PNG", optimize=True)

print(f"Saved: {output_file}")
print(f"Resolution: {WIDTH}x{HEIGHT}")
print(f"Theme: {THEME}")
print(f"c = {C}")
