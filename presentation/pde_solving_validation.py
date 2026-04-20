import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Color palette ─────────────────────────────────────────────
palette = {
    'Physical system':       '#2c7bb6',
    'Derive PDE':            '#5aaa6f',
    'Analytical':            '#e8a838',
    'Numerical':             '#d9534f',
    'Parameter Inference':   '#8e6bbf',
    'Validation':            '#3aabaa',
}

# ── Node definitions ──────────────────────────────────────────
# (label, x_center, y_center, sublabel)
nodes = [
    ("Physical\nSystem",       1.2,  3.0,  "observations\n& assumptions"),
    ("Derive\nPDE",            3.4,  3.0,  "governing\nequation"),
    ("Analytical\nSolution",   6.1,  4.6,  "exact / series\nsolution"),
    ("Numerical\nSolver",      6.1,  1.4,  "FD / FE / spectral"),
    ("Parameter\nInference",   9.5,  3.0,  "fit α, BCs …"),
    ("Validation",            12.2,  3.0,  "error metrics\n& ground truth"),
]

BOX_W, BOX_H = 1.9, 1.05

def draw_node(ax, label, xc, yc, sublabel, color):
    x0, y0 = xc - BOX_W/2, yc - BOX_H/2
    fancy = FancyBboxPatch((x0, y0), BOX_W, BOX_H,
                           boxstyle="round,pad=0.08",
                           facecolor=color, edgecolor='white',
                           linewidth=2, zorder=3)
    ax.add_patch(fancy)
    ax.text(xc, yc + 0.10, label, ha='center', va='center',
            fontsize=10, fontweight='bold', color='white',
            zorder=4, linespacing=1.3)
    ax.text(xc, yc - 0.38, sublabel, ha='center', va='center',
            fontsize=7.2, color='white', alpha=0.85,
            zorder=4, linespacing=1.3)

for label, xc, yc, sub in nodes:
    key = label.replace('\n', ' ').split()[0] + \
          (' ' + label.replace('\n',' ').split()[1] if len(label.replace('\n',' ').split()) > 1 else '')
    # match palette by first word
    color = next((v for k, v in palette.items() if label.replace('\n','').replace(' ','') in k.replace(' ','')), '#555')
    draw_node(ax, label, xc, yc, sub, color)

# ── Arrows ────────────────────────────────────────────────────
arrow_kw = dict(arrowstyle='->', color='#333', lw=1.6,
                connectionstyle='arc3,rad=0.0',
                mutation_scale=18, zorder=2)

def arrow(ax, x1, y1, x2, y2, rad=0.0, label='', color='#333'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.6,
                                connectionstyle=f'arc3,rad={rad}',
                                mutation_scale=18), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.18, label, ha='center', fontsize=7.5,
                color='#555', style='italic')

# Physical → Derive PDE
arrow(ax, 1.2 + BOX_W/2, 3.0,  3.4 - BOX_W/2, 3.0)

# Derive PDE → Analytical (up-right)
arrow(ax, 3.4 + BOX_W/2, 3.35, 6.1 - BOX_W/2, 4.6, rad=-0.25)

# Derive PDE → Numerical (down-right)
arrow(ax, 3.4 + BOX_W/2, 2.65, 6.1 - BOX_W/2, 1.4, rad=0.25)

# Analytical → Parameter Inference
arrow(ax, 6.1 + BOX_W/2, 4.6,  9.5 - BOX_W/2, 3.35, rad=0.25)

# Numerical → Parameter Inference
arrow(ax, 6.1 + BOX_W/2, 1.4,  9.5 - BOX_W/2, 2.65, rad=-0.25)

# Analytical ↔ Numerical (double-headed comparison arrow)
ax.annotate('', xy=(6.1, 1.4 + BOX_H/2 + 0.05), xytext=(6.1, 4.6 - BOX_H/2 - 0.05),
            arrowprops=dict(arrowstyle='<->', color='#aaa', lw=1.3,
                            connectionstyle='arc3,rad=0.0', mutation_scale=14), zorder=2)
ax.text(6.1 + 0.18, 3.0, 'compare', ha='left', fontsize=7, color='#999', style='italic', rotation=90, va='center')

# Parameter Inference → Validation
arrow(ax, 9.5 + BOX_W/2, 3.0, 12.2 - BOX_W/2, 3.0)

# Validation → back to Physical (feedback loop, curved under)
ax.annotate('', xy=(1.2, 3.0 - BOX_H/2 - 0.05), xytext=(12.2, 3.0 - BOX_H/2 - 0.05),
            arrowprops=dict(arrowstyle='->', color='#b04040', lw=1.3,
                            connectionstyle='arc3,rad=0.3', mutation_scale=14), zorder=2)
ax.text(6.7, 0.55, 'refine model / update assumptions', ha='center',
        fontsize=7.5, color='#b04040', style='italic')

# ── Title ─────────────────────────────────────────────────────
ax.text(7.0, 5.75, 'PDE Solving and Validation — Workflow',
        ha='center', va='top', fontsize=13, fontweight='bold', color='#222')

plt.tight_layout()
plt.savefig('pde_workflow.png', dpi=150, bbox_inches='tight')
plt.show()