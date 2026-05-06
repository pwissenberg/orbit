"""Generate a header banner image for the ORBIT README.

Visualizes the core idea: a species-A embedding cloud is gradually rotated
along an elliptical "orbit" until it aligns with the species-B cloud.
Three ghost clouds at intermediate angles trace the orbital path, with a
curved green arrow labelled R* indicating the optimal Procrustes rotation.

Outputs both SVG (vector, primary) and PNG (300 DPI raster fallback) to
`assets/header.{svg,png}`.

Layout: 3:1 banner. Left third: ORBIT title + acronym expansion. Right
two-thirds: orbital trajectory with 5 cloud snapshots (start, three ghosts,
end) and the rotation indicator.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

# ------------------------------------------------------------------
# Manuscript palette
# ------------------------------------------------------------------
BLUE = "#7aaed4"
ORANGE = "#e7672a"
GREEN = "#2e7d31"
GRAY = "#808080"
LIGHT_GRAY = "#cccccc"

OUT_DIR = Path.home() / "orbit" / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"

# ------------------------------------------------------------------
# Synthetic shape — same underlying cloud rotated through an orbit
# ------------------------------------------------------------------
rng = np.random.default_rng(7)
N = 22
cov = np.array([[1.0, 0.55], [0.55, 0.45]])
shape = rng.multivariate_normal(mean=[0, 0], cov=cov, size=N)
shape *= 0.55  # keep clouds compact along the orbit


def rotate(pts: np.ndarray, deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    R = np.array([[np.cos(th), -np.sin(th)],
                  [np.sin(th),  np.cos(th)]])
    return pts @ R.T


def blend(t: float):
    a = np.array(mc.to_rgb(BLUE))
    b = np.array(mc.to_rgb(ORANGE))
    return tuple((1.0 - t) * a + t * b)


# Orbital path: gentle arc, 5 stops from species A to species B
n_steps = 5
ts = np.linspace(0.0, 1.0, n_steps)
angles_deg = np.linspace(0.0, 90.0, n_steps)

# x positions evenly spaced; y follows an inverted parabola for the arc
xs = np.linspace(-3.0, 3.0, n_steps)
arc_height = 0.65
ys = arc_height * (1.0 - (xs / 3.0) ** 2) + 0.15

# ------------------------------------------------------------------
# Figure: 12 in x 4 in -> 1200 x 400 px (3:1 banner)
# ------------------------------------------------------------------
fig = plt.figure(figsize=(12, 4), dpi=100)
fig.patch.set_facecolor("white")

gs = GridSpec(
    nrows=1, ncols=2,
    width_ratios=[1.0, 2.0],
    left=0.02, right=0.98, top=0.98, bottom=0.02,
    wspace=0.02,
    figure=fig,
)

# ---- Text panel (left) ----
ax_text = fig.add_subplot(gs[0, 0])
ax_text.set_facecolor("white")
ax_text.set_xlim(0, 1)
ax_text.set_ylim(0, 1)
ax_text.set_xticks([])
ax_text.set_yticks([])
for spine in ax_text.spines.values():
    spine.set_visible(False)

ax_text.text(
    0.06, 0.62, "ORBIT",
    fontsize=56,
    fontweight="bold",
    color=GREEN,
    ha="left", va="center",
    family="DejaVu Sans",
)
ax_text.text(
    0.06, 0.30,
    "Orthogonal Rotation for Biological\nInter-species Transfer",
    fontsize=12,
    fontstyle="italic",
    color=GRAY,
    ha="left", va="center",
    linespacing=1.3,
)

# ---- Orbit panel (right) ----
ax = fig.add_subplot(gs[0, 1])
ax.set_facecolor("white")

# Dashed light-gray orbital trace — connects the cloud centers smoothly
trace_x = np.linspace(-3.0, 3.0, 200)
trace_y = arc_height * (1.0 - (trace_x / 3.0) ** 2) + 0.15
ax.plot(trace_x, trace_y, linestyle=(0, (4, 3)),
        linewidth=1.2, color=LIGHT_GRAY, zorder=1)

# Plot the 5 cloud snapshots along the orbit
for i, (cx, cy, ang, t) in enumerate(zip(xs, ys, angles_deg, ts)):
    pts = rotate(shape, ang)
    pts = pts + np.array([cx, cy])

    # Endpoints: full opacity, full color. Ghosts: faded.
    if i == 0:
        col, alpha, edge_w = BLUE, 0.95, 0.8
    elif i == n_steps - 1:
        col, alpha, edge_w = ORANGE, 0.95, 0.8
    else:
        col, alpha, edge_w = blend(t), 0.45, 0.0

    ax.scatter(
        pts[:, 0], pts[:, 1],
        s=42, c=[col], edgecolors="white", linewidths=edge_w,
        alpha=alpha, zorder=3 + (1 if i in (0, n_steps - 1) else 0),
    )

# R* curved arrow: rides above the orbit between endpoints
arrow = FancyArrowPatch(
    posA=(-2.4, 1.55), posB=(2.4, 1.55),
    connectionstyle="arc3,rad=-0.30",
    arrowstyle="-|>",
    mutation_scale=20,
    linewidth=2.4,
    color=GREEN,
    zorder=5,
)
ax.add_patch(arrow)

ax.text(
    0.0, 2.25, r"$\mathbf{R^{*}}$",
    fontsize=22,
    fontweight="bold",
    color=GREEN,
    ha="center", va="center",
    zorder=6,
)

# Cosmetics
ax.set_xlim(-3.8, 3.8)
ax.set_ylim(-1.2, 2.6)
ax.set_aspect("equal", adjustable="box")
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

svg_path = OUT_DIR / "header.svg"
png_path = OUT_DIR / "header.png"

# Do NOT use bbox_inches="tight" — it crops to content and breaks the 3:1 ratio.
fig.savefig(svg_path, format="svg", facecolor="white")
fig.savefig(png_path, format="png", dpi=300, facecolor="white")
plt.close(fig)

print(f"Wrote: {svg_path} ({svg_path.stat().st_size} bytes)")
print(f"Wrote: {png_path} ({png_path.stat().st_size} bytes)")
