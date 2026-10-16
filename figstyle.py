"""Shared figure style, implementing the project style guide.

The style guide refers to a `geodesic_style` package which is not available on
the cluster, so its specification is implemented here directly, mirroring
HMM_MEG/analysis/viz/style.py: the Wong colourblind-safe palette, serif
typography, the type scale, L-shaped axes, the standard figure sizes, a house
reference line, rounded-top bars, and `save_figure` writing PNG and PDF.

Colours are assigned by meaning and keep that meaning in every figure:
  K (number of groups)  K_COLOURS, one fixed Wong colour per K
  preterm / term        vermillion / blue
  ordered quantities    the sequential Blues map (age, q)
  signed quantities     the diverging RdBu_r map (loadings, correlations):
                        red = higher, blue = lower, white = zero

Reference: Wong, B. (2011). Points of view: Color blindness. Nature Methods,
8(6), 441.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

# --- Wong palette -----------------------------------------------------------

COLORS = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Black is reserved for axes and text; yellow has low contrast on white, so
# both are kept out of the automatic cycle but remain available by key.
COLOR_CYCLE = [
    COLORS["orange"], COLORS["blue"], COLORS["bluish_green"], COLORS["vermillion"],
    COLORS["reddish_purple"], COLORS["sky_blue"], COLORS["yellow"],
]

NEUTRAL_LIGHT_BLUE = "#B8D4E8"
NEUTRAL_GREY = "#AAAAAA"

TEXT_PRIMARY = COLORS["black"]
TEXT_SECONDARY = "#333333"
TEXT_MUTED = "#555555"

# --- semantic assignments -----------------------------------------------------

K_COLOURS = {k: COLOR_CYCLE[k - 1] for k in range(1, 6)}
Q_GRID = (0, 1, 2, 5, 10, 15, 25)                 # q is ordered: sequential shades,
Q_COLOURS = None                                  # filled in below, same in every figure
K_MARKERS = {1: "o", 2: "s", 3: "D", 4: "^", 5: "v"}
PRETERM_COLOUR = COLORS["vermillion"]
TERM_COLOUR = NEUTRAL_GREY                       # the reference group
POSITIVE_COLOUR = COLORS["vermillion"]          # bars for positive loadings
NEGATIVE_COLOUR = COLORS["blue"]
MEASURE_COLOURS = {"strength": COLORS["orange"], "log_clustering": COLORS["bluish_green"],
                   "communicability": COLORS["reddish_purple"]}

SEQUENTIAL_CMAP = "Blues"
DIVERGING_CMAP = "RdBu_r"
SEQUENTIAL = plt.get_cmap(SEQUENTIAL_CMAP)
DIVERGING = plt.get_cmap(DIVERGING_CMAP)


def sequential_shades(n: int, lo: float = 0.35, hi: float = 1.0) -> list:
    """n ordered shades of the sequential map, light to dark. The lightest
    steps of Blues vanish on white, so the range starts at `lo`."""
    return [SEQUENTIAL(v) for v in np.linspace(lo, hi, n)]


# Continuous version of the same truncation, for colouring points by an ordered
# quantity such as age: the palest Blues would make the youngest infants vanish.
SEQUENTIAL_VISIBLE = LinearSegmentedColormap.from_list("blues_visible",
                                                       sequential_shades(256, 0.3, 1.0))
Q_COLOURS = dict(zip(Q_GRID, sequential_shades(len(Q_GRID))))


# --- sizes and type -------------------------------------------------------------

FIGURE_SIZES = {
    "single_column": (5.5, 4.0),
    "double_column": (10.0, 4.5),
    "full_page": (10.0, 10.0),
    "multi_panel_tall": (14.0, 15.0),
}

# Times New Roman is not installed on the cluster; DejaVu Serif is the style
# guide's alternative, with Liberation Serif and Nimbus Roman as fallbacks.
SERIF_STACK = ["Times New Roman", "DejaVu Serif", "Liberation Serif", "Nimbus Roman"]

NOTE_SIZE = 11          # annotations and notes: the tick/legend size
VALUE_SIZE = 9          # value labels on bars, where 11 pt would collide


def apply_style() -> None:
    """Apply the project figure style. Call once before building figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": SERIF_STACK,
        "mathtext.fontset": "dejavuserif",
        "axes.prop_cycle": plt.cycler(color=COLOR_CYCLE),

        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "font.size": 11,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_PRIMARY,
        "axes.grid": False,
        "text.color": TEXT_PRIMARY,
        "xtick.color": TEXT_PRIMARY,
        "ytick.color": TEXT_PRIMARY,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,

        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 150,
    })


# --- helpers --------------------------------------------------------------------

def panel_title(ax, letter: str, title: str) -> None:
    """Panel letter and title as one bold 16 pt axes title."""
    ax.set_title(f"{letter}   {title}", pad=12)


def reference_line(ax, value: float, axis: str = "y") -> None:
    """Grey, dashed, thin, low-opacity reference line (e.g. zero, chance)."""
    drawer = ax.axhline if axis == "y" else ax.axvline
    drawer(value, color=NEUTRAL_GREY, linestyle="--", linewidth=0.7, alpha=0.5, zorder=0)


def rounded_bars(ax, pos, values, width=0.72, color=COLORS["blue"], horizontal=False,
                 radius=0.02, value_fmt=None, zorder=2):
    """Bars with subtly rounded ends away from the baseline and a thin black edge.

    `radius` is in axes-fraction units, so set the axis limits before calling.
    `color` may be one colour or one per bar. `value_fmt` (e.g. "{:.1f}") adds a
    bold value label beyond each bar's end.
    """
    pos, values = np.asarray(pos, float), np.asarray(values, float)
    # A list means one colour per bar; a string or tuple is a single colour (an
    # RGBA tuple must never be read as four bar colours).
    colors = list(color) if isinstance(color, list) else [color] * len(pos)
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    r_along = radius * abs((x1 - x0) if horizontal else (y1 - y0))
    r_across = radius * abs((y1 - y0) if horizontal else (x1 - x0))
    for p, v, c in zip(pos, values, colors):
        s = 1.0 if v >= 0 else -1.0
        ra = min(r_along, abs(v) / 2)
        rc = min(r_across, width / 2)
        a, b = p - width / 2, p + width / 2
        # outline in (along, across) coordinates, rounded at the far end
        pts = [(0, a), (v - s * ra, a), (v, a), (v, a + rc), (v, b - rc), (v, b),
               (v - s * ra, b), (0, b), (0, a)]
        codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
                 MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
                 MplPath.CLOSEPOLY]
        verts = pts if horizontal else [(q, l) for l, q in pts]
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=c, edgecolor=TEXT_PRIMARY,
                               linewidth=0.6, zorder=zorder))
        if value_fmt:
            off = 4 * s
            if horizontal:
                ax.annotate(value_fmt.format(v), (v, p), xytext=(off, 0),
                            textcoords="offset points", ha="left" if s > 0 else "right",
                            va="center", fontsize=VALUE_SIZE, fontweight="bold")
            else:
                ax.annotate(value_fmt.format(v), (p, v), xytext=(0, off),
                            textcoords="offset points", ha="center",
                            va="bottom" if s > 0 else "top", fontsize=VALUE_SIZE,
                            fontweight="bold")


def horizontal_colorbar(fig, mappable, rect, label):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(labelsize=11)
    cb.set_label(label, fontsize=11)
    return cb


def save_figure(fig, path_without_extension: str | Path) -> list[Path]:
    """Save a figure as both PNG (150 DPI) and PDF, per the style guide."""
    stem = Path(path_without_extension)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix, kwargs in ((".png", {"dpi": 150}), (".pdf", {})):
        target = stem.with_suffix(suffix)
        fig.savefig(target, bbox_inches="tight", **kwargs)
        written.append(target)
    plt.close(fig)
    return written
