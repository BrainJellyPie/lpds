"""Shared figure style: white background, Inter typography, a validated
categorical palette (checked for colour-vision deficiency separation on a
white surface), hairline grids, ringed markers, direct labels and wide
layouts sized for an IEEE two-column figure (7.16 in)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects as pe
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

W = 7.16                      # IEEE text width in inches

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#ebeae6"
SURF = "#ffffff"
NEUTRAL = "#9a9993"           # reference series (not a categorical slot)
PALE = "#c9c8c3"

# categorical slots in validated order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
    "#6250d6", "#e34948")
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
ORANGE_RAMP = ["#fde3d6", "#fbc8b0", "#f7a582", "#f18559", "#eb6834", "#cf5222",
               "#ad4019", "#8a3214"]
LIGHT_BLUE = "#86b6ef"

COLORS = {
    "LPM": BLUE, "IsoForest": ORANGE, "kNN": AQUA, "AA-Range": YELLOW,
    "Mahalanobis": MAGENTA, "IBUG": GREEN, "ErrModel": VIOLET,
    "VirtualEns": RED, "LOF": "#7a7974",
}
LABELS = {
    "LPM": "LPM", "IsoForest": "Isolation Forest", "kNN": "kNN distance",
    "AA-Range": "AA-Range", "Mahalanobis": "Mahalanobis", "IBUG": "IBUG",
    "ErrModel": "Error model", "VirtualEns": "Virtual ensemble", "LOF": "LOF",
}
MARKERS = {"LPM": "o", "IsoForest": "s", "kNN": "D", "AA-Range": "^",
           "Mahalanobis": "v", "IBUG": "P", "ErrModel": "X",
           "VirtualEns": "h", "LOF": "<"}


def setup():
    mpl.rcParams.update({
        "font.family": "Inter",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.4,
        "xtick.labelsize": 7.7,
        "ytick.labelsize": 7.7,
        "legend.fontsize": 7.6,
        "axes.edgecolor": PALE,
        "axes.linewidth": 0.7,
        "axes.labelcolor": INK2,
        "axes.titlecolor": INK,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 7,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "xtick.major.pad": 3.5,
        "ytick.major.pad": 3.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": SURF,
        "axes.facecolor": SURF,
        "savefig.facecolor": SURF,
        "legend.frameon": False,
        "legend.handletextpad": 0.35,
        "legend.columnspacing": 1.1,
        "legend.borderaxespad": 0.3,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Inter",
        "mathtext.it": "Inter:italic",
        "mathtext.bf": "Inter:bold",
        "mathtext.sf": "Inter",
        "mathtext.cal": "Inter",
        "mathtext.tt": "Inter",
        "pdf.fonttype": 42,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.03,
        "figure.constrained_layout.w_pad": 0.03,
        "figure.constrained_layout.hspace": 0.02,
        "figure.constrained_layout.wspace": 0.03,
    })


def panel(ax, letter, title):
    """Bold panel letter and a short semibold title, left-aligned above the axes."""
    ax.set_title(rf"$\mathbf{{{letter}}}$" + "   " + title, loc="left",
                 fontsize=9.0, fontweight="semibold", color=INK)


def glow_line(ax, x, y, color, lw=1.9, glow=True, zorder=3, **kw):
    if glow:
        ax.plot(x, y, color=color, lw=lw * 3.6, alpha=0.12, zorder=zorder - 0.2,
                solid_capstyle="round")
    return ax.plot(x, y, color=color, lw=lw, zorder=zorder, **kw)


def dot(ax, x, y, color, marker="o", s=30, z=4, ring=1.0, **kw):
    return ax.scatter(x, y, s=s, color=color, marker=marker, edgecolor=SURF,
                      linewidth=ring, zorder=z, **kw)


def _aspect(ax):
    """Data-y per data-x at the current axes size (for round corners)."""
    bb = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return (abs(y1 - y0) / max(bb.height, 1)) / (abs(x1 - x0) / max(bb.width, 1))


def rbar(ax, x, h, w, color, bottom=0.0, frac=0.3, alpha=1.0, z=3, horizontal=False):
    """Bar with a rounded data end and a square baseline end. Set the axis
    limits before drawing bars so that the corners come out circular."""
    asp = _aspect(ax)
    if horizontal:
        rad = min(frac * w / asp, abs(h) / 2)          # in x units
        lo = bottom if h >= 0 else bottom + h
        p = FancyBboxPatch((lo, x - w / 2), abs(h), w, mutation_aspect=asp,
                           boxstyle=f"round,pad=0,rounding_size={rad}",
                           linewidth=0, facecolor=color, alpha=alpha, zorder=z)
        ax.add_patch(p)
        if abs(h) > 2 * rad:
            base = bottom if h >= 0 else bottom - rad
            ax.add_patch(plt.Rectangle((base, x - w / 2), rad, w, facecolor=color,
                                       linewidth=0, alpha=alpha, zorder=z))
    else:
        rad = min(frac * w, abs(h) / 2 / asp)           # in x units
        lo = bottom if h >= 0 else bottom + h
        p = FancyBboxPatch((x - w / 2, lo), w, abs(h), mutation_aspect=asp,
                           boxstyle=f"round,pad=0,rounding_size={rad}",
                           linewidth=0, facecolor=color, alpha=alpha, zorder=z)
        ax.add_patch(p)
        ry = rad * asp
        if abs(h) > 2 * ry:
            base = bottom if h >= 0 else bottom - ry
            ax.add_patch(plt.Rectangle((x - w / 2, base), w, ry, facecolor=color,
                                       linewidth=0, alpha=alpha, zorder=z))
    return p


def halo(lw=2.4):
    return [pe.withStroke(linewidth=lw, foreground=SURF)]


def label(ax, x, y, text, color=INK, dx=4, dy=0, size=7.6, weight="medium",
          ha="left", va="center", coords="data"):
    return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                       xycoords=coords, ha=ha, va=va, fontsize=size, color=color,
                       fontweight=weight, path_effects=halo(), annotation_clip=False)


def spread_labels(ax, x_end, ys, texts, colors, x_text=None, gap_pt=8.5, size=7.4,
                  weights=None, connector=True):
    """Direct labels at the right end of lines, pushed apart vertically so they
    never collide. Works on linear and logarithmic axes."""
    fig = ax.figure
    fig.canvas.draw()
    to_ax = ax.transData + ax.transAxes.inverted()
    pts = np.array([to_ax.transform((x_end, y)) for y in ys])
    fy = pts[:, 1].copy()
    h_pt = ax.get_window_extent().height * 72.0 / fig.dpi
    g = gap_pt / h_pt
    order = np.argsort(fy)
    pos = fy[order].copy()
    for _ in range(200):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < g:
                mid = (pos[i] + pos[i - 1]) / 2
                pos[i - 1], pos[i] = mid - g / 2, mid + g / 2
                moved = True
        if not moved:
            break
    new = np.empty_like(fy)
    new[order] = pos
    xt = pts[0, 0] + 0.03 if x_text is None else x_text
    for k, (t, c) in enumerate(zip(texts, colors)):
        w = "medium" if weights is None else weights[k]
        ax.annotate(t, xy=(pts[k, 0] + 0.008, fy[k]), xytext=(xt, new[k]),
                    xycoords="axes fraction", textcoords="axes fraction",
                    ha="left", va="center", fontsize=size, color=INK if w == "bold" else INK2,
                    fontweight=w, annotation_clip=False, path_effects=halo(),
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.8, alpha=0.8,
                                    shrinkA=0, shrinkB=1.5) if connector and abs(new[k] - fy[k]) > 0.012 else None)


def save(fig, path, caption):
    fig.savefig(path, dpi=600)
    with open(path.rsplit(".", 1)[0] + ".txt", "w") as f:
        f.write(caption.strip() + "\n")
    plt.close(fig)


def soft_band(ax, x, lo, hi, color, alpha=0.12):
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0, zorder=1)


def hline(ax, y, text=None, color=INK2, ls=(0, (3, 3)), lw=0.9, xpos=0.99, va="bottom"):
    ax.axhline(y, color=color, lw=lw, ls=ls, zorder=2)
    if text:
        ax.text(xpos, y, text, transform=ax.get_yaxis_transform(), ha="right",
                va=va, fontsize=7.2, color=color, path_effects=halo())


def log_ticks(axis, ticks, labels=None):
    axis.set_major_locator(ticker.FixedLocator(ticks))
    axis.set_minor_locator(ticker.NullLocator())
    axis.set_major_formatter(ticker.FixedFormatter(labels or [f"{t:g}" for t in ticks]))
    axis.set_minor_formatter(ticker.NullFormatter())


def handle(color, marker="o", lw=0, ms=6.2, ls="-"):
    return Line2D([], [], color=color, marker=marker, lw=lw, ms=ms, ls=ls,
                  markeredgecolor=SURF, markeredgewidth=0.8)


def fig_legend(fig, handles, labels, ncol, size=7.4, spacing=1.2):
    return fig.legend(handles, labels, loc="outside upper center", ncol=ncol,
                      fontsize=size, handletextpad=0.3, columnspacing=spacing,
                      frameon=False)


def jitter_offsets(n, width=0.62):
    if n == 1:
        return np.zeros(1)
    return np.linspace(-width / 2, width / 2, n)
