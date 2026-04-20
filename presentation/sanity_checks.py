import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

fig = plt.figure(figsize=(15, 9))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38)

x   = np.linspace(0, np.pi, 400)
t_vals = [0.0, 0.5, 1.5, 4.0, 10.0]
alphas = [0.02, 0.08, 0.20, 0.50]

modes = [(1, 1.0), (3, 0.4), (5, 0.12), (7, 0.04)]

def u(x, t, alpha):
    return sum(a * np.sin(n*x) * np.exp(-alpha * n**2 * t) for n, a in modes)

def add_panel_label(ax, text, color):
    ax.text(0.02, 0.97, text, transform=ax.transAxes,
            fontsize=8, color='white', fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='none'))

# ── shared style ──────────────────────────────────────────────
def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=8.5, color='#333')
    ax.set_ylabel(ylabel, fontsize=8.5, color='#333')
    ax.set_title(title, fontsize=9.5, fontweight='bold', color='#222', pad=7)
    ax.tick_params(labelsize=7.5, color='#888')
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#ccc')
    ax.set_facecolor('#fafafa')

cmap_time  = plt.cm.plasma
cmap_alpha = plt.cm.viridis

# ════════════════════════════════════════════════════════════════
# PANEL 1 — Sanity check: smoothing over time
# ════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0])
for i, t in enumerate(t_vals):
    c = cmap_time(i / (len(t_vals)-1) * 0.85 + 0.05)
    lw = 2.2 if t == 0 else 1.5
    ls = '-' if t == 0 else '--'
    ax1.plot(x, u(x, t, 0.08), color=c, lw=lw, ls=ls,
             label=f't = {t}', alpha=0.92)

ax1.axhline(0, color='#ccc', lw=0.8)
ax1.set_xticks([0, np.pi/2, np.pi])
ax1.set_xticklabels(['0', 'π/2', 'π'])
style(ax1, 'x', 'u(x, t)', '① Smoothing over time')
ax1.legend(fontsize=6.8, framealpha=0.5, loc='upper right')
add_panel_label(ax1, 'SANITY CHECK', '#2c7bb6')
ax1.text(0.5, -0.22,
    "u → 0 as t → ∞  (BCs = 0, no source)\nHigher modes decay faster: e⁻ᵅⁿ²ᵗ",
    transform=ax1.transAxes, ha='center', fontsize=7.2,
    color='#555', style='italic', linespacing=1.5)

# ════════════════════════════════════════════════════════════════
# PANEL 2 — Sanity check: mode decay rates
# ════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1])
t_fine = np.linspace(0, 12, 300)
alpha  = 0.08
mode_colors = ['#2c7bb6', '#e8a838', '#d9534f', '#8e6bbf']
for (n, a), c in zip(modes, mode_colors):
    decay = a * np.exp(-alpha * n**2 * t_fine)
    ax2.plot(t_fine, decay, color=c, lw=2.0, label=f'n={n}, a={a}')

ax2.axhline(0, color='#ccc', lw=0.8)
style(ax2, 't', 'aₙ · e⁻ᵅⁿ²ᵗ', '② Mode decay rates  (α = 0.08)')
ax2.legend(fontsize=7, framealpha=0.5)
add_panel_label(ax2, 'SANITY CHECK', '#2c7bb6')
ax2.text(0.5, -0.22,
    "Mode n=1 persists longest — dominant shape at large t\n"
    "Mode n=7 vanishes almost instantly",
    transform=ax2.transAxes, ha='center', fontsize=7.2,
    color='#555', style='italic', linespacing=1.5)

# ════════════════════════════════════════════════════════════════
# PANEL 3 — Sanity check: energy decay
# ════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[0, 2])
t_fine = np.linspace(0, 12, 300)
# L2 norm squared = sum a_n^2 * exp(-2*alpha*n^2*t) * pi/2
E = lambda t, a: sum(
    an**2 * np.exp(-2*alpha*n**2*t) for n, an in modes
) * (np.pi/2)

E_vals = [E(t, alpha) for t in t_fine]
ax3.plot(t_fine, E_vals, color='#3aabaa', lw=2.2)
ax3.fill_between(t_fine, E_vals, alpha=0.12, color='#3aabaa')
ax3.axhline(0, color='#ccc', lw=0.8)
style(ax3, 't', '‖u(·,t)‖²', '③ Energy dissipation')
add_panel_label(ax3, 'SANITY CHECK', '#2c7bb6')
ax3.text(0.5, -0.22,
    "‖u‖² strictly decreasing (no source term)\n"
    "Consistent with 2nd law — entropy increases",
    transform=ax3.transAxes, ha='center', fontsize=7.2,
    color='#555', style='italic', linespacing=1.5)

# ════════════════════════════════════════════════════════════════
# PANEL 4 — Identifiability: sensitivity of u to α
# ════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 0])
t_obs = 1.5
for i, a in enumerate(alphas):
    c = cmap_alpha(i / (len(alphas)-1))
    ax4.plot(x, u(x, t_obs, a), color=c, lw=2.0, label=f'α = {a}')

ax4.set_xticks([0, np.pi/2, np.pi])
ax4.set_xticklabels(['0', 'π/2', 'π'])
ax4.axhline(0, color='#ccc', lw=0.8)
style(ax4, 'x', f'u(x, t={t_obs})', f'④ Sensitivity to α  (t = {t_obs})')
ax4.legend(fontsize=7, framealpha=0.5)
add_panel_label(ax4, 'IDENTIFIABILITY', '#8e6bbf')
ax4.text(0.5, -0.22,
    "Different α → clearly distinct profiles at t = 1.5\n"
    "α is identifiable from a single spatial snapshot",
    transform=ax4.transAxes, ha='center', fontsize=7.2,
    color='#555', style='italic', linespacing=1.5)

# ════════════════════════════════════════════════════════════════
# PANEL 5 — Identifiability: log-linear decay → read off α
# ════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[1, 1])
t_fine = np.linspace(0.3, 12, 300)
# At large t, mode n=1 dominates: log u_max ≈ log(a1) - alpha*t
# We show u at x=pi/2 (sin(pi/2)=1, sin(3*pi/2)=-1, etc.)
x_probe = np.pi / 2
for i, a in enumerate(alphas):
    u_probe = np.array([u(x_probe, t, a) for t in t_fine])
    # only plot where positive
    mask = u_probe > 1e-4
    log_u = np.log(np.abs(u_probe[mask]))
    ax5.plot(t_fine[mask], log_u, color=cmap_alpha(i/(len(alphas)-1)),
             lw=2.0, label=f'α = {a}')
    # overlay slope annotation for last alpha
    if i == len(alphas)-1:
        t0, t1 = 1.0, 3.0
        y0 = np.log(abs(u(x_probe, t0, a)))
        y1 = np.log(abs(u(x_probe, t1, a)))
        ax5.annotate('', xy=(t1, y1), xytext=(t0, y0),
                     arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))
        slope = (y1-y0)/(t1-t0)
        ax5.text((t0+t1)/2 + 0.3, (y0+y1)/2,
                 f'slope = −α·n²\n≈ {slope:.2f}',
                 fontsize=6.8, color='#c0392b')

style(ax5, 't', 'log |u(π/2, t)|', '⑤ Log-linear decay → infer α')
ax5.legend(fontsize=7, framealpha=0.5)
add_panel_label(ax5, 'IDENTIFIABILITY', '#8e6bbf')
ax5.text(0.5, -0.22,
    "At large t, log u ≈ log a₁ − α t  (n=1 dominates)\n"
    "Slope of log-plot directly gives α — no solver needed",
    transform=ax5.transAxes, ha='center', fontsize=7.2,
    color='#555', style='italic', linespacing=1.5)

# ════════════════════════════════════════════════════════════════
# PANEL 6 — Identifiability: loss landscape L(α)
# ════════════════════════════════════════════════════════════════
ax6 = fig.add_subplot(gs[1, 2])
alpha_true = 0.08
t_obs_pts  = [0.5, 1.0, 2.0]
x_obs      = np.linspace(0.3, np.pi-0.3, 8)
alpha_grid = np.linspace(0.01, 0.5, 300)
ls_styles  = ['-', '--', ':']
obs_colors = ['#2c7bb6', '#e8a838', '#d9534f']

for t_o, ls, c in zip(t_obs_pts, ls_styles, obs_colors):
    u_true = u(x_obs, t_o, alpha_true)
    loss   = [np.mean((u(x_obs, t_o, a) - u_true)**2) for a in alpha_grid]
    loss   = np.array(loss) / max(loss)   # normalize for readability
    ax6.plot(alpha_grid, loss, lw=2.0, ls=ls, color=c, label=f't = {t_o}')

ax6.axvline(alpha_true, color='black', lw=1.3, ls='--', label=f'α_true = {alpha_true}')
ax6.set_xlim(0.01, 0.5)
style(ax6, 'α', 'Normalised loss  L(α)', '⑥ Loss landscape L(α)')
ax6.legend(fontsize=7, framealpha=0.5)
add_panel_label(ax6, 'IDENTIFIABILITY', '#8e6bbf')
ax6.text(0.5, -0.22,
    "Sharp, unique minimum → α is well-identifiable\n"
    "Earlier observations yield a wider (flatter) bowl",
    transform=ax6.transAxes, ha='center', fontsize=7.2,
    color='#555', style='italic', linespacing=1.5)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

fig = plt.figure(figsize=(15, 9))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38)

x   = np.linspace(0, np.pi, 400)
t_vals = [0.0, 0.5, 1.5, 4.0, 10.0]
alphas = [0.02, 0.08, 0.20, 0.50]

modes = [(1, 1.0), (3, 0.4), (5, 0.12), (7, 0.04)]

def u(x, t, alpha):
    return sum(a * np.sin(n*x) * np.exp(-alpha * n**2 * t) for n, a in modes)

def add_panel_label(ax, text, color):
    ax.text(0.02, 0.97, text, transform=ax.transAxes,
            fontsize=8, color='white', fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor='none'))

# ── shared style ──────────────────────────────────────────────
def style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=8.5, color='#333')
    ax.set_ylabel(ylabel, fontsize=8.5, color='#333')
    ax.set_title(title, fontsize=9.5, fontweight='bold', color='#222', pad=7)
    ax.tick_params(labelsize=7.5, color='#888')
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#ccc')
    ax.set_facecolor('#fafafa')

cmap_time  = plt.cm.plasma
cmap_alpha = plt.cm.viridis

# ════════════════════════════════════════════════════════════════
# PANEL 1 — Sanity check: smoothing over time
# ════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0])
for i, t in enumerate(t_vals):
    c = cmap_time(i / (len(t_vals)-1) * 0.85 + 0.05)
    lw = 2.2 if t == 0 else 1.5
    ls = '-' if t == 0 else '--'
    ax1.plot(x, u(x, t, 0.08), color=c, lw=lw, ls=ls,
             label=f't = {t}', alpha=0.92)

ax1.axhline(0, color='#ccc', lw=0.8)
ax1.set_xticks([0, np.pi/2, np.pi])
ax1.set_xticklabels(['0', 'π/2', 'π'])
style(ax1, 'x', 'u(x, t)', '① Smoothing over time')
ax1.legend(fontsize=6.8, framealpha=0.5, loc='upper right')
add_panel_label(ax1, 'SANITY CHECK', '#2c7bb6')
ax1.text(0.5, -0.22,
    "u → 0 as t → ∞  (BCs = 0, no source)\nHigher modes decay faster: e⁻ᵅⁿ²ᵗ",
    transform=ax1.transAxes, ha='center', fontsize=7.2,
    color='#555', style='italic', linespacing=1.5)

# ════════════════════════════════════════════════════════════════
# PANEL 2 — Sanity check: mode decay rates
# ════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1])
t_fine = np.linspace(0, 12, 300)
alpha  = 0.08
mode_colors = ['#2c7bb6', '#e8a838', '#d9534f', '#8e6bbf']
for (n, a), c in zip(modes, mode_colors):
    decay = a * np.exp(-alpha * n**2 * t_fine)
    ax2.plot(t_fine, decay, color=c, lw=2.0, label=f'n={n}, a={a}')

ax2.axhline(0, color='#ccc', lw=0.8)
style(ax2, 't', 'aₙ · e⁻ᵅⁿ²ᵗ', '② Mode decay rates  (α = 0.08)')
ax2.legend(fontsize=7, framealpha=0.5)
add_panel_label(ax2, 'SANITY CHECK', '#2c7bb6')
ax2.text(0.5, -0.22,
    "Mode n=1 persists longest — dominant shape at large t\n"
    "Mode n=7 vanishes almost instantly",
    transform=ax2.transAxes, ha='center', fontsize=7.2,
    color='#555', style='italic', linespacing=1.5)

# ════════════════════════════════════════════════════════════════
# PANEL 3 — Sanity check: energy decay
# ════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[0, 2])
t_fine = np.linspace(0, 12, 300)
# L2 norm squared = sum a_n^2 * exp(-2*alpha*n^2*t) * pi/2
E = lambda t, a: sum(
    an**2 * np.exp(-2*alpha*n**2*t) for n, an in modes
) * (np.pi/2)

E_vals = [E(t, alpha) for t in t_fine]
ax3.plot(t_fine, E_vals, color='#3aabaa', lw=2.2)
ax3.fill_between(t_fine, E_vals, alpha=0.12, color='#3aabaa')
ax3.axhline(0, color='#ccc', lw=0.8)
style(ax3, 't', '‖u(·,t)‖²', '③ Energy dissipation')
add_panel_label(ax3, 'SANITY CHECK', '#2c7bb6')
ax3.text(0.5, -0.22,
    "‖u‖² strictly decreasing (no source term)\n"
    "Consistent with 2nd law — entropy increases",
    transform=ax3.transAxes, ha='center', fontsize=7.2,
    color='#555', style='italic', linespacing=1.5)

# ════════════════════════════════════════════════════════════════
# PANEL 4 — Identifiability: sensitivity of u to α
# ════════════════════════════════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 0])
t_obs = 1.5
for i, a in enumerate(alphas):
    c = cmap_alpha(i / (len(alphas)-1))
    ax4.plot(x, u(x, t_obs, a), color=c, lw=2.0, label=f'α = {a}')

ax4.set_xticks([0, np.pi/2, np.pi])
ax4.set_xticklabels(['0', 'π/2', 'π'])
ax4.axhline(0, color='#ccc', lw=0.8)
style(ax4, 'x', f'u(x, t={t_obs})', f'④ Sensitivity to α  (t = {t_obs})')
ax4.legend(fontsize=7, framealpha=0.5)
add_panel_label(ax4, 'IDENTIFIABILITY', '#8e6bbf')
ax4.text(0.5, -0.22,
    "Different α → clearly distinct profiles at t = 1.5\n"
    "α is identifiable from a single spatial snapshot",
    transform=ax4.transAxes, ha='center', fontsize=7.2,
    color='#555', style='italic', linespacing=1.5)

# ════════════════════════════════════════════════════════════════
# PANEL 5 — Identifiability: log-linear decay → read off α
# ════════════════════════════════════════════════════════════════
ax5 = fig.add_subplot(gs[1, 1])
t_fine = np.linspace(0.3, 12, 300)
# At large t, mode n=1 dominates: log u_max ≈ log(a1) - alpha*t
# We show u at x=pi/2 (sin(pi/2)=1, sin(3*pi/2)=-1, etc.)
x_probe = np.pi / 2
for i, a in enumerate(alphas):
    u_probe = np.array([u(x_probe, t, a) for t in t_fine])
    # only plot where positive
    mask = u_probe > 1e-4
    log_u = np.log(np.abs(u_probe[mask]))
    ax5.plot(t_fine[mask], log_u, color=cmap_alpha(i/(len(alphas)-1)),
             lw=2.0, label=f'α = {a}')
    # overlay slope annotation for last alpha
    if i == len(alphas)-1:
        t0, t1 = 1.0, 3.0
        y0 = np.log(abs(u(x_probe, t0, a)))
        y1 = np.log(abs(u(x_probe, t1, a)))
        ax5.annotate('', xy=(t1, y1), xytext=(t0, y0),
                     arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))
        slope = (y1-y0)/(t1-t0)
        ax5.text((t0+t1)/2 + 0.3, (y0+y1)/2,
                 f'slope = −α·n²\n≈ {slope:.2f}',
                 fontsize=6.8, color='#c0392b')

style(ax5, 't', 'log |u(π/2, t)|', '⑤ Log-linear decay → infer α')
ax5.legend(fontsize=7, framealpha=0.5)
add_panel_label(ax5, 'IDENTIFIABILITY', '#8e6bbf')
ax5.text(0.5, -0.22,
    "At large t, log u ≈ log a₁ − α t  (n=1 dominates)\n"
    "Slope of log-plot directly gives α — no solver needed",
    transform=ax5.transAxes, ha='center', fontsize=7.2,
    color='#555', style='italic', linespacing=1.5)

# ════════════════════════════════════════════════════════════════
# PANEL 6 — Identifiability: loss landscape L(α)
# ════════════════════════════════════════════════════════════════
ax6 = fig.add_subplot(gs[1, 2])
alpha_true = 0.08
t_obs_pts  = [0.5, 1.0, 2.0]
x_obs      = np.linspace(0.3, np.pi-0.3, 8)
alpha_grid = np.linspace(0.01, 0.5, 300)
ls_styles  = ['-', '--', ':']
obs_colors = ['#2c7bb6', '#e8a838', '#d9534f']

for t_o, ls, c in zip(t_obs_pts, ls_styles, obs_colors):
    u_true = u(x_obs, t_o, alpha_true)
    loss   = [np.mean((u(x_obs, t_o, a) - u_true)**2) for a in alpha_grid]
    loss   = np.array(loss) / max(loss)   # normalize for readability
    ax6.plot(alpha_grid, loss, lw=2.0, ls=ls, color=c, label=f't = {t_o}')

ax6.axvline(alpha_true, color='black', lw=1.3, ls='--', label=f'α_true = {alpha_true}')
ax6.set_xlim(0.01, 0.5)
style(ax6, 'α', 'Normalised loss  L(α)', '⑥ Loss landscape L(α)')
ax6.legend(fontsize=7, framealpha=0.5)
add_panel_label(ax6, 'IDENTIFIABILITY', '#8e6bbf')
ax6.text(0.5, -0.22,
    "Sharp, unique minimum → α is well-identifiable\n"
    "Earlier observations yield a wider (flatter) bowl",
    transform=ax6.transAxes, ha='center', fontsize=7.2,
    color='#555', style='italic', linespacing=1.5)

# ── Main title ────────────────────────────────────────────────
fig.text(0.5, 0.97,
         'Analytical solution — sanity checks & identifiability insights\n'
         'Heat equation:  ∂u/∂t = α ∂²u/∂x²',
         ha='center', va='top', fontsize=12.5, fontweight='bold',
         color='#222', linespacing=1.5)

plt.savefig('sanity_identifiability.png', dpi=150, bbox_inches='tight')
plt.show()
# ── Main title ────────────────────────────────────────────────
fig.text(0.5, 0.97,
         'Analytical solution — sanity checks & identifiability insights\n'
         'Heat equation:  ∂u/∂t = α ∂²u/∂x²',
         ha='center', va='top', fontsize=12.5, fontweight='bold',
         color='#222', linespacing=1.5)

plt.savefig('sanity_identifiability.png', dpi=150, bbox_inches='tight')
plt.show()