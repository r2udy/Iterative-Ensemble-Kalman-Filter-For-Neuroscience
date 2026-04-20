#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IEnKF Sensitivity Analysis  —  v2
===================================
Combines:
  A) One-at-a-time (OAT) sweeps over 6 axes
  B) 2D cross-sweeps over 4 scientifically motivated pairs

Sweep axes
----------
  cmro2_true  : [1.0, 1.5, 2.0, 2.5, 3.0]  µmol/cm³/min
  R0_true     : [80, 90, 100, 110, 120, 130] µm
  obs_var     : [1², 5², 10², 50², 100²]      mmHg²
  cmro2_var   : [5e-8, 0.005, 0.5, 5.0]      (physical prior variance)
  R0_var      : [0.1², 2², 10², 20², 50²]    µm²
  pvessel_var : [0.5², 1², 5², 10², 20²]     mmHg²

2D cross-sweeps (final iteration only, heatmap + scatter output)
----------------------------------------------------------------
  1. cmro2_true  × obs_var      — signal vs noise trade-off
  2. cmro2_true  × cmro2_var    — prior width vs ground truth
  3. R0_true     × obs_var      — geometry uncertainty vs noise
  4. obs_var     × cmro2_var    — observation vs prior variance

Priority metrics (front-and-centre in every figure)
----------------------------------------------------
  • CMRO2 estimation bias     |true − mean|
  • Posterior std             (uncertainty width)
  • pO2 reconstruction error  (absolute and relative)

Settings: n_ensembles=100, iterations_max=20, seed=42

Output layout
-------------
  <SAVE_PATH>/
    oat/<sweep_name>/          5 figures + raw .npy arrays
    cross/<pair_name>/         2 figures + raw .npy arrays
    SUMMARY_bias_all_OAT.png
    SUMMARY_priority_metrics.png
    SUMMARY_cross_sweeps.png
    sensitivity_summary.csv
"""

import sys, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
# 0.  PATHS  —  edit to match your environment
# ══════════════════════════════════════════════════════
ROOT       = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/"
CLASSES    = os.path.join(ROOT, "classes")
IMPL       = os.path.join(ROOT, "Implementation")
CLUSTERING = os.path.join(IMPL, "Clustering")
SAVE_PATH  = os.path.join(ROOT, "Data/EnKF_plots/sensitivity_analysis_v4/")

for p in [ROOT, CLASSES, IMPL, CLUSTERING]:
    sys.path.append(p)

from EnKF_FEM_3 import EnKF
from FEM_code.generateMesh_Solver_multiple_holes import SolverParameters, HoleGeometry
from MapGenerator import MapGenerator

for sub in ["oat", "cross"]:
    os.makedirs(os.path.join(SAVE_PATH, sub), exist_ok=True)

# ══════════════════════════════════════════════════════
# 1.  CONSTANTS & DEFAULTS
# ══════════════════════════════════════════════════════
D      = 4.0e3
ALPHA  = 1.39e-15
C2M    = 60 * D * ALPHA * 1e12    # CMRO2 → M unit conversion

GRID   = 20
X_AX, Y_AX = np.meshgrid(
    np.linspace(-190, 190, GRID),
    np.linspace(-190, 190, GRID),
)
OBS_DIM  = GRID * GRID
N_ENS    = 100
ITER_MAX = 8
STATE_DIM= 3
SIG_NOISE= 2.0
SEED = np.random.seed(1)

DEF = dict(
    cmro2_true    = 2.0,
    R0_true       = 100.0,
    Pves_true     = 80.0,
    Rves          = 17.5,
    obs_var       = 2.0 ** 2,
    cmro2_var     = 0.5 ** 2,      # physical units
    R0_var        = 1.0 ** 2,
    pvessel_var   = 2.0 ** 2,
    cmro2_mean0   = 2.0,
    R0_mean0      = 100.0,
    pvessel_mean0 = 80.0,
)

# ── OAT grids ──────────────────────────────────────────
SWEEPS = {
    "cmro2_true"  : np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
    "R0_true"     : np.array([80., 90., 100., 110., 120.]),
    "obs_var"     : np.array([1.**2, 5.**2, 10.**2]),
    "cmro2_var"   : np.array([5e-8, 0.005, 0.5, 5.0]),
    "R0_var"      : np.array([0.1**2, 2.**2, 10.**2]),
    "pvessel_var" : np.array([.5**2, 1.**2, 5.**2, 10.**2]),
}

# (axis label, tick formatter, is_log_axis)
SWEEP_META = {
    "cmro2_true"  : ("CMRO2 true [µmol/cm³/min]",   lambda v: f"{v:.2f}",          False),
    "R0_true"     : ("R0 true [µm]",                 lambda v: f"{v:.0f}",           False),
    "obs_var"     : ("Obs noise σ [mmHg]",            lambda v: f"{np.sqrt(v):.0f}",  True),
    "cmro2_var"   : ("CMRO2 prior σ [µmol/cm³/min]", lambda v: f"{np.sqrt(v):.3g}",  True),
    "R0_var"      : ("R0 prior σ [µm]",              lambda v: f"{np.sqrt(v):.1f}",  True),
    "pvessel_var" : ("pvessel prior σ [mmHg]",        lambda v: f"{np.sqrt(v):.1f}",  False),
}

# ── 2D cross-sweep pairs ────────────────────────────────
CROSS_SWEEPS = [
    ("cmro2_true", np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
     "obs_var",    np.array([1.**2, 5.**2, 10.**2, 50.**2, 100.**2]),
     "CMRO2_true x obs_var  —  signal vs noise"),

    ("cmro2_true", np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
     "cmro2_var",  np.array([5e-8, 0.005, 0.5, 5.0]),
     "CMRO2_true x cmro2_var  —  prior width vs ground truth"),

    ("R0_true",    np.array([80., 100., 120., 130.]),
     "obs_var",    np.array([1.**2, 10.**2, 50.**2, 100.**2]),
     "R0_true x obs_var  —  geometry vs noise"),

    ("obs_var",    np.array([1.**2, 5.**2, 10.**2]),
     "cmro2_var",  np.array([0.005, 0.5, 5.0]),
     "obs_var x cmro2_var  —  observation vs prior variance"),
]

# ══════════════════════════════════════════════════════
# 2.  CORE ENGINE
# ══════════════════════════════════════════════════════

def _make_enkf(obs_var, cmro2_var_phys, R0_var, pvessel_var,
               cmro2_mean0, R0_mean0, pvessel_mean0):
    cmro2_var_M = cmro2_var_phys / C2M ** 2
    def dyn(x): return x
    enkf = EnKF(STATE_DIM, OBS_DIM, N_ENS, dyn, seed=SEED)
    a = np.array([cmro2_mean0 / C2M, R0_mean0, pvessel_mean0] * 3)
    b = np.array([cmro2_var_M, R0_var, pvessel_var] * 3)
    enkf.initialize_ensemble(a, b)
    B = np.diag([cmro2_var_M, R0_var, pvessel_var])
    R = obs_var * np.eye(OBS_DIM)
    enkf.set_process_noise(B)
    enkf.set_observation_noise(R)
    return enkf


def _true_map(cmro2, Pves, Rves, R0):
    params = SolverParameters(filename="square_holes")
    holes  = [HoleGeometry(center=(0., 0., 0.), cmro2=cmro2,
                           Pves=Pves, radius_ves=Rves, radius_0=R0,
                           marker=params.marker)]
    return MapGenerator(holes=holes, params=params, X=X_AX, Y=Y_AX).pO2_array


def _est_map(cmro2, pves, Rves, R0):
    params = SolverParameters(filename="square_holes")
    holes  = [HoleGeometry(center=(0., 0., 0.), cmro2=cmro2,
                           Pves=pves, radius_ves=Rves, radius_0=R0,
                           marker=params.marker)]
    return MapGenerator(holes=holes, params=params, X=X_AX, Y=Y_AX).pO2_array


def run_single(p):
    """Run one full IEnKF experiment; return dict of per-iteration arrays."""
    enkf = _make_enkf(
        obs_var        = p["obs_var"],
        cmro2_var_phys = p["cmro2_var"],
        R0_var         = p["R0_var"],
        pvessel_var    = p["pvessel_var"],
        cmro2_mean0    = p["cmro2_mean0"],
        R0_mean0       = p["R0_mean0"],
        pvessel_mean0  = p["pvessel_mean0"],
    )
    true_pO2 = _true_map(p["cmro2_true"], p["Pves_true"], p["Rves"], p["R0_true"])

    out = {k: [] for k in [
        "cmro2_mean", "cmro2_std",
        "R0_mean",    "R0_std",
        "pves_mean",  "pves_std",
        "bias",       "rel_bias",
        "pO2_abs",    "pO2_rel",
        "spread",     "gain_norm",
    ]}

    for _ in range(ITER_MAX):
        obs = true_pO2.flatten() + np.random.normal(0, SIG_NOISE, OBS_DIM)
        enkf.predict()
        enkf.update(obs, X_AX, Y_AX)
        mu, cov = enkf.get_state_estimate()

        cm  = mu[0] * C2M
        cs  = np.sqrt(max(cov[0, 0], 0)) * C2M
        R0m = mu[1];  R0s = np.sqrt(max(cov[1, 1], 0))
        pvm = mu[2];  pvs = np.sqrt(max(cov[2, 2], 0))

        ep    = _est_map(cm, pvm, p["Rves"], R0m)
        tf    = true_pO2.flatten()
        ef    = ep.flatten()
        denom = np.abs(obs) + 1e-9

        out["cmro2_mean"].append(cm)
        out["cmro2_std"].append(cs)
        out["R0_mean"].append(R0m)
        out["R0_std"].append(R0s)
        out["pves_mean"].append(pvm)
        out["pves_std"].append(pvs)
        out["bias"].append(abs(p["cmro2_true"] - cm))
        out["rel_bias"].append(abs(p["cmro2_true"] - cm) / p["cmro2_true"] * 100)
        out["pO2_abs"].append(float(np.mean(np.abs(tf - ef))))
        out["pO2_rel"].append(float(np.mean(np.abs(tf - ef) / denom) * 100))
        out["spread"].append(float(np.std(enkf.ensemble[0, :]) * C2M))
        try:
            out["gain_norm"].append(float(np.linalg.norm(enkf.K)))
        except Exception:
            out["gain_norm"].append(np.nan)

    return {k: np.array(v) for k, v in out.items()}


# ══════════════════════════════════════════════════════
# 3.  OAT RUNNER
# ══════════════════════════════════════════════════════

def run_oat_sweep(sweep_name, sweep_values):
    n    = len(sweep_values)
    keys = ["cmro2_mean","cmro2_std","R0_mean","R0_std","pves_mean","pves_std",
            "bias","rel_bias","pO2_abs","pO2_rel","spread","gain_norm"]
    res  = {k: np.zeros((n, ITER_MAX)) for k in keys}

    for i, val in enumerate(sweep_values):
        p = {k: v for k, v in DEF.items()}
        p[sweep_name] = val
        if sweep_name == "R0_true":
            p["R0_mean0"] = val

        single = run_single(p)
        for k in keys:
            res[k][i] = single[k]

        print(f"    [{sweep_name}={val:.4g}]  "
              f"CMRO2={res['cmro2_mean'][i,-1]:.3f}±{res['cmro2_std'][i,-1]:.3f}  "
              f"bias={res['bias'][i,-1]:.3f}  "
              f"pO2_abs={res['pO2_abs'][i,-1]:.2f} mmHg")
    return res


# ══════════════════════════════════════════════════════
# 4.  2D CROSS-SWEEP RUNNER
# ══════════════════════════════════════════════════════

def run_cross_sweep(pA, valsA, pB, valsB):
    nA, nB    = len(valsA), len(valsB)
    bias_grid = np.zeros((nA, nB))
    std_grid  = np.zeros((nA, nB))
    pO2_grid  = np.zeros((nA, nB))

    for i, va in enumerate(valsA):
        for j, vb in enumerate(valsB):
            p = {k: v for k, v in DEF.items()}
            p[pA] = va
            p[pB] = vb
            if pA == "R0_true": p["R0_mean0"] = va
            if pB == "R0_true": p["R0_mean0"] = vb

            r = run_single(p)
            bias_grid[i, j] = r["bias"][-1]
            std_grid[i, j]  = r["cmro2_std"][-1]
            pO2_grid[i, j]  = r["pO2_abs"][-1]

        print(f"    [row {i+1}/{nA}: {pA}={va:.4g}] done")

    return bias_grid, std_grid, pO2_grid


# ══════════════════════════════════════════════════════
# 5.  PLOTTING — OAT
# ══════════════════════════════════════════════════════

def _colors(n, cmap="plasma"):
    return plt.cm.get_cmap(cmap)(np.linspace(0.12, 0.92, n))


def fig_priority_panel(sweep_name, sweep_values, res):
    """Hero figure: 3 priority metrics vs iteration for every sweep value."""
    meta   = SWEEP_META[sweep_name]
    colors = _colors(len(sweep_values))
    iters  = np.arange(1, ITER_MAX + 1)

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    configs = [
        ("bias",      "|CMRO2 bias|  [µmol/cm³/min]",     True,  None),
        ("cmro2_std", "Posterior σ (CMRO2)  [µmol/cm³/min]", False, None),
        ("pO2_abs",   "Mean pO2 abs error  [mmHg]",          False, None),
    ]
    for ax, (metric, ylabel, log_y, _) in zip(axes, configs):
        for idx, val in enumerate(sweep_values):
            ax.plot(iters, res[metric][idx], color=colors[idx],
                    lw=2, marker="o", ms=3, label=meta[1](val))
        if log_y:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        ax.grid(True, alpha=0.25)

    # Add shaded final-iteration column
    for ax in axes:
        ax.axvspan(ITER_MAX - 0.5, ITER_MAX + 0.5, color="gold", alpha=0.15,
                   label="final iter" if ax is axes[0] else "")

    axes[-1].set_xlabel("Iteration", fontsize=10)
    fig.suptitle(
        f"Priority Metrics vs Iteration\nSweep: {sweep_name}  ({meta[0]})",
        fontweight="bold", fontsize=11)
    plt.tight_layout()
    return fig


def fig_final_trio(sweep_name, sweep_values, res):
    """Bar chart of the 3 priority metrics at final iteration."""
    meta   = SWEEP_META[sweep_name]
    labels = [meta[1](v) for v in sweep_values]
    x      = np.arange(len(sweep_values))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric, ylabel, clr in zip(
        axes,
        ["bias",    "cmro2_std",                        "pO2_abs"],
        ["|CMRO2 bias|\n[µmol/cm³/min]",
         "Posterior σ (CMRO2)\n[µmol/cm³/min]",
         "Mean pO2 abs error\n[mmHg]"],
        ["#2E75B6", "#ED7D31", "#70AD47"],
    ):
        vals = res[metric][:, -1]
        bars = ax.bar(x, vals, color=clr, edgecolor="k", linewidth=0.6, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel(meta[0], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Final-Iteration Priority Metrics  —  sweep: {sweep_name}",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    return fig


def fig_heatmap_pair(sweep_name, sweep_values, res):
    """Dual heatmap: |bias| and pO2 error across sweep_value × iteration."""
    meta  = SWEEP_META[sweep_name]
    ylbls = [meta[1](v) for v in sweep_values]
    ext   = [1, ITER_MAX, len(sweep_values) - 0.5, -0.5]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(sweep_values) * 0.75 + 2)))
    for ax, metric, title, use_log in zip(
        axes,
        ["bias",        "pO2_abs"],
        ["|CMRO2 bias|","pO2 abs error [mmHg]"],
        [True,           False],
    ):
        data = res[metric]
        norm = LogNorm(vmin=data.min() + 1e-6, vmax=data.max()) if use_log else None
        im   = ax.imshow(data, aspect="auto", cmap="RdYlGn_r",
                         norm=norm, extent=ext)
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel(meta[0], fontsize=9)
        ax.set_yticks(range(len(sweep_values)))
        ax.set_yticklabels(ylbls, fontsize=8)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax)

    fig.suptitle(f"Priority-Metric Heatmaps  —  sweep: {sweep_name}",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    return fig


def fig_uncertainty_collapse(sweep_name, sweep_values, res):
    """Posterior std vs ensemble spread — consistency check."""
    meta   = SWEEP_META[sweep_name]
    colors = _colors(len(sweep_values), "coolwarm")
    iters  = np.arange(1, ITER_MAX + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for idx, val in enumerate(sweep_values):
        lbl = meta[1](val)
        axes[0].plot(iters, res["cmro2_std"][idx], color=colors[idx], lw=2, label=lbl)
        axes[1].plot(iters, res["spread"][idx],    color=colors[idx], lw=2, label=lbl)
        lo = np.minimum(res["cmro2_std"][idx], res["spread"][idx])
        hi = np.maximum(res["cmro2_std"][idx], res["spread"][idx])
        axes[0].fill_between(iters, lo, hi, alpha=0.08, color=colors[idx])

    for ax, title, yl in zip(
        axes,
        ["Posterior σ  (from covariance)", "Ensemble spread  (std of particles)"],
        ["σ  [µmol/cm³/min]"] * 2,
    ):
        ax.set_xlabel("Iteration"); ax.set_ylabel(yl, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Uncertainty Collapse — sweep: {sweep_name}\n"
        "Shaded = gap between posterior σ and ensemble spread "
        "(large gap → filter inconsistency / over-collapse)",
        fontweight="bold", fontsize=10)
    plt.tight_layout()
    return fig


def fig_pO2_both_errors(sweep_name, sweep_values, res):
    """Absolute and relative pO2 reconstruction error vs iteration."""
    meta   = SWEEP_META[sweep_name]
    colors = _colors(len(sweep_values))
    iters  = np.arange(1, ITER_MAX + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, val in enumerate(sweep_values):
        lbl = meta[1](val)
        axes[0].plot(iters, res["pO2_abs"][idx], color=colors[idx],
                     lw=2, marker="o", ms=3, label=lbl)
        axes[1].plot(iters, res["pO2_rel"][idx], color=colors[idx],
                     lw=2, marker="s", ms=3, label=lbl)

    axes[0].set_ylabel("Mean |true − est|  [mmHg]", fontsize=9)
    axes[1].set_ylabel("Mean relative error  [%]", fontsize=9)
    for ax, title in zip(axes, ["pO2 Absolute Error", "pO2 Relative Error"]):
        ax.set_xlabel("Iteration"); ax.set_title(title)
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.25)

    fig.suptitle(f"pO2 Reconstruction Error — sweep: {sweep_name}",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════
# 6.  PLOTTING — 2D CROSS-SWEEPS
# ══════════════════════════════════════════════════════

def fig_cross_heatmaps(pA, valsA, pB, valsB, bias_g, std_g, pO2_g, title):
    """3-panel heatmap (bias / posterior std / pO2) for one 2D cross-sweep."""
    metaA = SWEEP_META[pA]
    metaB = SWEEP_META[pB]
    xlbls = [metaB[1](v) for v in valsB]
    ylbls = [metaA[1](v) for v in valsA]

    fig, axes = plt.subplots(1, 3, figsize=(16, max(4, len(valsA) * 0.9 + 2)))
    panels = [
        (bias_g, "|CMRO2 bias|\n[µmol/cm³/min]", "RdYlGn_r", True),
        (std_g,  "Posterior σ (CMRO2)\n[µmol/cm³/min]", "YlOrRd",   False),
        (pO2_g,  "pO2 abs error\n[mmHg]",               "RdYlGn_r", False),
    ]
    for ax, (data, cblabel, cmap, use_log) in zip(axes, panels):
        norm = LogNorm(vmin=data.min() + 1e-9, vmax=data.max()) if use_log else None
        im   = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto", origin="upper")
        ax.set_xticks(range(len(valsB)))
        ax.set_yticks(range(len(valsA)))
        ax.set_xticklabels(xlbls, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(ylbls, fontsize=8)
        ax.set_xlabel(metaB[0], fontsize=9)
        ax.set_ylabel(metaA[0], fontsize=9)
        # Annotate cells with values
        for r in range(len(valsA)):
            for c in range(len(valsB)):
                txt_clr = "white" if data[r, c] > (data.max() * 0.6) else "black"
                ax.text(c, r, f"{data[r,c]:.2f}", ha="center", va="center",
                        fontsize=7, color=txt_clr, fontweight="bold")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        plt.colorbar(im, cax=cax, label=cblabel)
        ax.set_title(cblabel.replace("\n", " "), fontsize=9)

    fig.suptitle(f"2D Cross-sweep  —  {title}\n(all values at final iteration)",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    return fig


def fig_cross_scatter(pA, valsA, pB, valsB, bias_g, std_g, pO2_g, title):
    """
    Bubble scatter: x=|CMRO2 bias|, y=pO2 abs error, bubble size ∝ posterior σ.
    Each point is one (pA, pB) combination — reveals trade-off structure.
    """
    metaA  = SWEEP_META[pA]
    metaB  = SWEEP_META[pB]
    colors = _colors(len(valsA), "tab10")

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, va in enumerate(valsA):
        sc = ax.scatter(
            bias_g[i, :],
            pO2_g[i, :],
            s=np.clip(std_g[i, :] * 800, 20, 600),
            c=[colors[i]] * len(valsB),
            label=f"{pA}={metaA[1](va)}",
            alpha=0.78, edgecolors="k", linewidths=0.5,
        )
    # Label pB values on first row only
    for j, vb in enumerate(valsB):
        ax.annotate(f"{pB}={metaB[1](vb)}", (bias_g[0, j], pO2_g[0, j]),
                    fontsize=7, textcoords="offset points", xytext=(5, 4))

    ax.set_xlabel("|CMRO2 bias|  [µmol/cm³/min]", fontsize=10)
    ax.set_ylabel("Mean pO2 abs error  [mmHg]", fontsize=10)
    ax.set_title(
        f"Bias vs pO2 error  —  {title}\n"
        "Bubble size ∝ posterior σ  (larger = more uncertain)",
        fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════
# 7.  SUMMARY FIGURES
# ══════════════════════════════════════════════════════

def fig_summary_bias(all_oat):
    """2×3 grid: final-iteration |bias| ± posterior σ for all 6 OAT sweeps."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for idx, (sname, svals) in enumerate(SWEEPS.items()):
        ax   = axes[idx]
        res  = all_oat[sname]
        meta = SWEEP_META[sname]
        x    = np.arange(len(svals))
        ax.bar(x, res["bias"][:, -1], yerr=res["cmro2_std"][:, -1],
               capsize=4, color="#2E75B6", edgecolor="k", alpha=0.82)
        ax.set_xticks(x)
        ax.set_xticklabels([meta[1](v) for v in svals],
                           rotation=28, ha="right", fontsize=7)
        ax.set_xlabel(meta[0], fontsize=8)
        ax.set_ylabel("|bias|  [µmol/cm³/min]", fontsize=8)
        ax.set_title(sname, fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
    for idx in range(len(SWEEPS), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle("CMRO2 Estimation Bias at Final Iteration  —  All OAT Sweeps\n"
                 "Error bars = posterior σ  (uncertainty width)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def fig_summary_priority(all_oat):
    """3-row × 6-col grid: all priority metrics at final iteration, all sweeps."""
    snames  = list(SWEEPS.keys())
    metrics = [
        ("bias",      "|CMRO2 bias| [µmol/cm³/min]"),
        ("cmro2_std", "Posterior σ [µmol/cm³/min]"),
        ("pO2_abs",   "pO2 abs error [mmHg]"),
    ]
    clrs = ["#2E75B6", "#ED7D31", "#70AD47"]

    fig, axes = plt.subplots(3, len(snames), figsize=(3.2 * len(snames), 10),
                             sharey="row")
    for row, ((metric, row_label), clr) in enumerate(zip(metrics, clrs)):
        for col, sname in enumerate(snames):
            ax    = axes[row, col]
            res   = all_oat[sname]
            svals = SWEEPS[sname]
            meta  = SWEEP_META[sname]
            vals  = res[metric][:, -1]
            x     = np.arange(len(svals))
            ax.bar(x, vals, color=clr, edgecolor="k", linewidth=0.5, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([meta[1](v) for v in svals],
                               rotation=35, ha="right", fontsize=6)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=8)
            if row == 0:
                ax.set_title(sname, fontsize=8, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Priority Metrics at Final Iteration — All OAT Sweeps",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def fig_summary_cross(cross_results):
    """2×2 thumbnail of bias heatmaps for all four cross-sweeps."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, (pA, valsA, pB, valsB, title, bias_g, std_g, pO2_g) in \
            enumerate(cross_results):
        ax    = axes[idx]
        metaA = SWEEP_META[pA]
        metaB = SWEEP_META[pB]
        im    = ax.imshow(bias_g, cmap="RdYlGn_r", aspect="auto", origin="upper")
        ax.set_xticks(range(len(valsB)))
        ax.set_yticks(range(len(valsA)))
        ax.set_xticklabels([metaB[1](v) for v in valsB],
                           rotation=30, ha="right", fontsize=7)
        ax.set_yticklabels([metaA[1](v) for v in valsA], fontsize=7)
        ax.set_xlabel(metaB[0], fontsize=8)
        ax.set_ylabel(metaA[0], fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, label="|bias| [µmol/cm³/min]")
    fig.suptitle("2D Cross-Sweep Summary  —  CMRO2 Bias Heatmaps (final iteration)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════
# 8.  CSV
# ══════════════════════════════════════════════════════

def save_csv(all_oat, cross_results, path):
    rows = []
    for sname, svals in SWEEPS.items():
        res  = all_oat[sname]
        meta = SWEEP_META[sname]
        for i, v in enumerate(svals):
            rows.append(dict(
                section="OAT", sweep_param=sname,
                sweep_value=v, sweep_label=meta[1](v),
                param_B="", value_B="",
                cmro2_bias=res["bias"][i, -1],
                cmro2_rel_bias=res["rel_bias"][i, -1],
                cmro2_mean=res["cmro2_mean"][i, -1],
                cmro2_std=res["cmro2_std"][i, -1],
                R0_mean=res["R0_mean"][i, -1],
                pves_mean=res["pves_mean"][i, -1],
                pO2_abs_err=res["pO2_abs"][i, -1],
                pO2_rel_err=res["pO2_rel"][i, -1],
                spread=res["spread"][i, -1],
            ))
    for (pA, valsA, pB, valsB, title, bias_g, std_g, pO2_g) in cross_results:
        metaA = SWEEP_META[pA]; metaB = SWEEP_META[pB]
        for i, va in enumerate(valsA):
            for j, vb in enumerate(valsB):
                rows.append(dict(
                    section="CROSS", sweep_param=pA,
                    sweep_value=va, sweep_label=metaA[1](va),
                    param_B=pB, value_B=vb,
                    cmro2_bias=bias_g[i, j],
                    cmro2_rel_bias=np.nan,
                    cmro2_mean=np.nan,
                    cmro2_std=std_g[i, j],
                    R0_mean=np.nan, pves_mean=np.nan,
                    pO2_abs_err=pO2_g[i, j],
                    pO2_rel_err=np.nan, spread=np.nan,
                ))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"\n✅  CSV saved → {path}")
    return df


# ══════════════════════════════════════════════════════
# 9.  MAIN
# ══════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    t0 = time.time()

    # ────────────────────────────────────
    # A. OAT sweeps
    # ────────────────────────────────────
    all_oat = {}
    for sweep_name, sweep_values in SWEEPS.items():
        print(f"\n{'='*62}")
        print(f"  OAT  ·  {sweep_name}  ({len(sweep_values)} values)")
        print(f"{'='*62}")

        res = run_oat_sweep(sweep_name, sweep_values)
        all_oat[sweep_name] = res

        sub = os.path.join(SAVE_PATH, "oat", sweep_name)
        os.makedirs(sub, exist_ok=True)

        for k, arr in res.items():
            np.save(os.path.join(sub, f"{k}.npy"), arr)
        np.save(os.path.join(sub, "sweep_values.npy"), sweep_values)

        # 1. Priority panel (hero figure — 3 metrics × ITER_MAX)
        fig = fig_priority_panel(sweep_name, sweep_values, res)
        fig.savefig(os.path.join(sub, "priority_metrics.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

        # 2. Final-iteration bar chart (3 metrics side by side)
        fig = fig_final_trio(sweep_name, sweep_values, res)
        fig.savefig(os.path.join(sub, "final_trio_bars.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

        # 3. Heatmap (sweep × iteration for bias and pO2 error)
        fig = fig_heatmap_pair(sweep_name, sweep_values, res)
        fig.savefig(os.path.join(sub, "heatmap_bias_pO2.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

        # 4. Uncertainty collapse (posterior std vs ensemble spread)
        fig = fig_uncertainty_collapse(sweep_name, sweep_values, res)
        fig.savefig(os.path.join(sub, "uncertainty_collapse.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

        # 5. pO2 absolute + relative reconstruction error
        fig = fig_pO2_both_errors(sweep_name, sweep_values, res)
        fig.savefig(os.path.join(sub, "pO2_errors.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"  → 5 figures + {len(res)} arrays → {sub}")

    # ────────────────────────────────────
    # B. 2D cross-sweeps
    # ────────────────────────────────────
    cross_results = []
    for (pA, valsA, pB, valsB, title) in CROSS_SWEEPS:
        pair_tag = f"{pA}_x_{pB}"
        print(f"\n{'='*62}")
        print(f"  CROSS  ·  {pair_tag}  ({len(valsA)}×{len(valsB)} grid)")
        print(f"{'='*62}")

        bias_g, std_g, pO2_g = run_cross_sweep(pA, valsA, pB, valsB)
        cross_results.append((pA, valsA, pB, valsB, title, bias_g, std_g, pO2_g))

        sub = os.path.join(SAVE_PATH, "cross", pair_tag)
        os.makedirs(sub, exist_ok=True)

        np.save(os.path.join(sub, "bias_grid.npy"),  bias_g)
        np.save(os.path.join(sub, "std_grid.npy"),   std_g)
        np.save(os.path.join(sub, "pO2_grid.npy"),   pO2_g)
        np.save(os.path.join(sub, "vals_A.npy"),     valsA)
        np.save(os.path.join(sub, "vals_B.npy"),     valsB)

        # 3-panel heatmap (bias / std / pO2)
        fig = fig_cross_heatmaps(pA, valsA, pB, valsB, bias_g, std_g, pO2_g, title)
        fig.savefig(os.path.join(sub, "heatmaps_3panel.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Bubble scatter (bias vs pO2, size=std)
        fig = fig_cross_scatter(pA, valsA, pB, valsB, bias_g, std_g, pO2_g, title)
        fig.savefig(os.path.join(sub, "scatter_bias_pO2.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"  → 2 figures + 5 arrays → {sub}")

    # ────────────────────────────────────
    # C. Summary figures
    # ────────────────────────────────────
    print("\n--- Generating summary figures ---")

    fig = fig_summary_bias(all_oat)
    fig.savefig(os.path.join(SAVE_PATH, "SUMMARY_bias_all_OAT.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = fig_summary_priority(all_oat)
    fig.savefig(os.path.join(SAVE_PATH, "SUMMARY_priority_metrics.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = fig_summary_cross(cross_results)
    fig.savefig(os.path.join(SAVE_PATH, "SUMMARY_cross_sweeps.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ────────────────────────────────────
    # D. CSV
    # ────────────────────────────────────
    df = save_csv(all_oat, cross_results,
                  os.path.join(SAVE_PATH, "sensitivity_summary.csv"))

    elapsed = time.time() - t0
    print(f"\n✅  Complete in {elapsed/60:.1f} min")
    print(f"   All output → {SAVE_PATH}")

    print("\n── OAT highlights (final iteration) ──")
    oat = df[df.section == "OAT"][
        ["sweep_param", "sweep_label", "cmro2_bias", "cmro2_std", "pO2_abs_err"]
    ].rename(columns={"cmro2_bias": "bias", "cmro2_std": "post_σ",
                      "pO2_abs_err": "pO2_abs"})
    print(oat.to_string(index=False))


if __name__ == "__main__":
    main()