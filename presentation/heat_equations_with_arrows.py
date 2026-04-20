import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Setup ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

x = np.linspace(0, np.pi, 500)

# Wobbly initial condition (similar to the reference image)
u = 50 + 25*np.sin(x) - 18*np.sin(2*x) + 10*np.cos(3*x) - 6*np.sin(4*x)

# Numerical second derivative (curvature ∝ ∂²u/∂x²)
dx = x[1] - x[0]
d2u = np.gradient(np.gradient(u, dx), dx)

# ── Colormap along the curve ───────────────────────────────────
from matplotlib.collections import LineCollection
points = np.array([x, u]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
norm = plt.Normalize(x.min(), x.max())
lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2.5, zorder=3)
lc.set_array(x)
ax.add_collection(lc)

# ── Arrows showing curvature ───────────────────────────────────
arrow_xs = np.linspace(0.15, np.pi - 0.15, 14)
cmap = plt.cm.viridis

for xi in arrow_xs:
    idx = np.argmin(np.abs(x - xi))
    yi = u[idx]
    curv = d2u[idx]          # ∂²u/∂x²
    color = cmap(norm(xi))

    # Arrow length proportional to curvature magnitude, capped
    scale = 0.18
    dy = np.clip(curv * scale, -18, 18)

    if abs(dy) < 0.8:        # skip near-zero curvature
        continue

    ax.annotate(
        '',
        xy=(xi, yi + dy),           # arrow tip
        xytext=(xi, yi),            # arrow base (on curve)
        arrowprops=dict(
            arrowstyle='->',
            color=color,
            lw=1.8 + abs(dy) / 12,
        ),
        zorder=4,
    )

# ── Colorbar strip at the bottom (like the reference) ─────────
cb_ax = fig.add_axes([0.09, 0.08, 0.83, 0.03])
gradient = np.linspace(0, 1, 256).reshape(1, -1)
cb_ax.imshow(gradient, aspect='auto', cmap='plasma')
cb_ax.set_axis_off()

# ── Equation text ─────────────────────────────────────────────
ax.text(0.38, 0.92,
        r'$\frac{\partial u}{\partial t}(x,t) = D \cdot \frac{\partial^2 u}{\partial x^2}(x,t)$',
        transform=ax.transAxes, fontsize=16, color='black',
        ha='center', va='top')

# ── Axes styling ──────────────────────────────────────────────
ax.set_xlim(-0.05, np.pi + 0.05)
ax.set_ylim(10, 95)
ax.spines[['top', 'right', 'bottom']].set_visible(False)
ax.spines['left'].set_color('black')
ax.tick_params(colors='black', labelsize=10)
ax.yaxis.set_tick_params(length=4)
ax.set_xticks([])
ax.set_ylabel('T', color='black', fontsize=12, rotation=0, labelpad=10)

# Black arrow for x-axis
ax.annotate('', xy=(np.pi + 0.05, 10), xytext=(-0.05, 10),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.4))
# Black arrow for y-axis
ax.annotate('', xy=(-0.05, 95), xytext=(-0.05, 10),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.4))

plt.tight_layout(rect=[0, 0.14, 1, 1])
plt.show()