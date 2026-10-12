"""Rugged-landscape figure for §8 of the Peixoto (2025) walkthrough.

The toy model is the one in peixoto-viz/js/replicas.js: five modules of twelve ±1 spins, all +1 inside a
module, sparse ±0.6 couplings between modules drawn with the same seeded RNG (mulberry32, seed 7). With every
module fully aligned, a macrostate is a sign pattern σ ∈ {±1}^5 and its weight is

    P(α) ∝ exp(β Σ_{k<l} J_kl σ_k σ_l),    J_kl = sum of the couplings between modules k and l.

Each of the 32 patterns becomes a peak; small random bumps make the landscape rugged. Six macrostates are drawn
as configurations on a plane above it, with dotted lines down to their peaks.

    python week-5/code/landscape_figure.py   ->   week-5/code/peixoto-viz/img/landscape.{png,pdf}
"""

from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

NM, MS, BETA = 5, 12, 0.2
OUT = Path(__file__).parent / "peixoto-viz" / "img" / "landscape.png"

# Page palette (week-3/code/viz-common/styles.css)
BG, INK, MUTED, PLUS, MINUS, MODEL = "#1a1a19", "#c3c2b7", "#898781", "#d95926", "#383835", "#9085e9"


def mulberry32(seed):
    s = seed & 0xFFFFFFFF

    def uniform():
        nonlocal s
        s = (s + 0x6D2B79F5) & 0xFFFFFFFF
        t = s
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return uniform


def module_couplings():
    """J_kl: summed inter-module couplings, drawn exactly as in replicas.js."""
    rng, N, J = mulberry32(7), NM * MS, np.zeros((NM, NM))
    for i in range(N):
        for j in range(i + 1, N):
            if i // MS == j // MS:
                continue
            if rng() < 0.08:
                w = -0.6 if rng() < 0.5 else 0.6
                J[i // MS, j // MS] += w
    return J + J.T


def main():
    J = module_couplings()
    pats = np.array(list(product([1, -1], repeat=NM)))
    logw = np.array([BETA * 0.5 * s @ J @ s for s in pats])
    P = np.exp(logw - logw.max())
    P /= P.sum()

    # 2-D layout: a 4 × 8 Gray-code grid (modules 1–2 pick the row, 3–5 the column),
    # so neighbouring peaks differ by flipping a single module
    ang = -np.pi / 2 + 2 * np.pi * np.arange(NM) / NM
    gray = lambda bits: (lambda b: b ^ (b >> 1))(int("".join("1" if v < 0 else "0" for v in bits), 2))
    inv = lambda code, n: next(r for r in range(2**n) if r ^ (r >> 1) == code)
    xy = np.array([[-1.54 + 0.44 * inv(gray(s[2:]), 3), -1.05 + 0.7 * inv(gray(s[:2]), 2)] for s in pats])

    g = np.linspace(-1.85, 1.85, 300)
    X, Y = np.meshgrid(g, g)
    Z = np.zeros_like(X)
    height = 0.15 + 0.85 * P / P.max()
    for (px, py), h in zip(xy, height):
        Z += h * np.exp(-((X - px) ** 2 + (Y - py) ** 2) / (2 * 0.085**2))
    rng = np.random.default_rng(3)
    for _ in range(140):  # ruggedness: small bumps in the valleys
        cx, cy = rng.uniform(-1.7, 1.7, 2)
        Z += rng.uniform(0.01, 0.06) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * rng.uniform(0.04, 0.09) ** 2))

    fig = plt.figure(figsize=(12, 9), facecolor=BG)
    ax = fig.add_subplot(projection="3d", computed_zorder=False, facecolor=BG)
    cmap = LinearSegmentedColormap.from_list("model", ["#1a1a19", "#2c2850", "#474090", "#6b62c6", "#9085e9", "#d2cdf8"])
    ax.plot_surface(X, Y, Z, cmap=cmap, rstride=2, cstride=2, linewidth=0, antialiased=True, zorder=1, rasterized=True)

    # the plane of configurations
    zt = 2.0
    ax.plot_surface(*np.meshgrid([-1.85, 1.85], [-1.85, 1.85]), np.full((2, 2), zt), color="#2a2a28", alpha=0.35, zorder=2, shade=False)
    for x0, x1, y0, y1 in [(-1.85, 1.85, -1.85, -1.85), (-1.85, 1.85, 1.85, 1.85), (-1.85, -1.85, -1.85, 1.85), (1.85, 1.85, -1.85, 1.85)]:
        ax.plot([x0, x1], [y0, y1], [zt, zt], color=MUTED, lw=0.8, zorder=3)

    # six macrostates, from most to least probable, kept far enough apart that their configurations don't overlap
    picks = []
    for k in np.argsort(-P)[[0, 1, 2, 3, 6, 8, 10, 12, 14, 18, 22, 26, 29, 31]]:
        if len(picks) < 6 and all(np.hypot(*(xy[k] - xy[q])) > 0.95 for q in picks):
            picks.append(k)

    for k in picks:
        s, (px, py) = pats[k], xy[k]
        gx, gy = px, py
        pz = Z[np.abs(g - py).argmin(), np.abs(g - px).argmin()]
        # glyph: five modules on a pentagon, each a ring of twelve spins
        for m in range(NM):
            mx, my = gx + 0.22 * np.cos(ang[m]), gy + 0.22 * np.sin(ang[m])
            t = 2 * np.pi * np.arange(MS) / MS
            ax.scatter(mx + 0.07 * np.cos(t), my + 0.07 * np.sin(t), zt, s=8, color=PLUS if s[m] > 0 else MINUS,
                       edgecolors=INK if s[m] < 0 else "none", linewidths=0.3, depthshade=False, zorder=6)
        ax.plot([gx, px], [gy, py], [zt - 0.02, pz + 0.02], ls=(0, (1.5, 2.5)), color=INK, lw=1.1, zorder=5)
        ax.scatter(px, py, pz, s=22, color="#ffffff", depthshade=False, zorder=5)
        ax.text(gx + 0.45, gy - 0.06, zt, f"P(α) = {P[k]:.3f}", color=INK, fontsize=9.5, ha="left", zorder=7)

    ax.text2D(0.02, 0.95, "metastable configurations x ∈ α, one per macrostate (orange +1, grey −1; one ring per module)",
              transform=ax.transAxes, color=INK, fontsize=11)
    ax.text2D(0.02, 0.06, "P(x | W) over configuration space: one peak per macrostate α, valleys of low probability between them",
              transform=ax.transAxes, color=INK, fontsize=11)
    ax.text2D(0.02, 0.025, f"toy model of §8: 5 modules × 12 spins, β = {BETA}; peak heights ∝ exp(β Σ J_kl σ_k σ_l), the 32 sign patterns σ",
              transform=ax.transAxes, color=MUTED, fontsize=9.5)

    ax.set_xlim(-1.85, 1.85); ax.set_ylim(-1.85, 1.85); ax.set_zlim(0, zt)
    ax.view_init(elev=24, azim=-30)
    ax.set_box_aspect((1, 1, 0.78))
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=BG)
    fig.savefig(OUT.with_suffix(".pdf"), dpi=200, facecolor=BG)  # surface rasterized, text and lines vector
    print(OUT, "picked P(α):", np.round(P[picks], 3))


if __name__ == "__main__":
    main()
