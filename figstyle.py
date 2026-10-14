"""Figure style matching the research proposal's figure.

Warm sequential ramp for postmenstrual age, light panels with thin borders,
bold panel letters, horizontal colourbar beneath.

Every palette here was checked with the data-viz validator rather than chosen by
eye (node scripts/validate_palette.js):

  AGE_RAMP      ordinal, light mode: PASS — monotone light->dark, adjacent
                delta-L >= 0.06, light end >= 2:1 on the surface, hue spread < 40 deg.
                The proposal's own yellow->red ramp fails both the light-end
                contrast (1.26:1) and the single-hue gate (70 deg spread), so this
                keeps its warmth while staying legible.
  SERIES        categorical slots 1-5, light, adjacent pairs: PASS.
                Three slots sit below 3:1 contrast, so the relief rule applies —
                every series carries a visible direct label, never colour alone.
  CLUSTER       first three categorical slots; passes the all-pairs gate used
                for scatter.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8982"
PANEL_EDGE = "#d9d8d2"
GRID = "#ececea"

AGE_STEPS = ["#e59542", "#d97a37", "#ca5c2e", "#b44027", "#99291f", "#7a1518"]
AGE_RAMP = LinearSegmentedColormap.from_list("pma_warm", AGE_STEPS)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
CLUSTER = SERIES[:3]
NEUTRAL = "#9a9a94"


def apply():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": PANEL_EDGE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 9,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "axes.titlecolor": INK_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "font.size": 9,
        "font.family": "sans-serif",
        "lines.linewidth": 2.0,
        "lines.markersize": 4.0,
        "figure.dpi": 160,
    })


def panel_letter(ax, letter, label=None):
    """Bold letter top-left, optional small caption beside it — as in the proposal."""
    ax.text(-0.02, 1.10, letter, transform=ax.transAxes, fontsize=10.5,
            fontweight="bold", va="top", ha="right", color=INK)
    if label:
        ax.text(0.0, 1.10, label, transform=ax.transAxes, fontsize=8.5,
                va="top", ha="left", color=INK_2)


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def age_colorbar(fig, mappable, rect=(0.34, 0.045, 0.32, 0.016)):
    """Horizontal age colourbar beneath the panels, as in the proposal figure."""
    cax = fig.add_axes(rect)
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, colors=INK_MUTED, length=0, pad=3)
    cb.set_label("postmenstrual age (weeks)", fontsize=8, color=INK_2, labelpad=5)
    return cb
