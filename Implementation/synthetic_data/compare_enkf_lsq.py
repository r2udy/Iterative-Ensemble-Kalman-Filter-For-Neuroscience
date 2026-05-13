#!/usr/bin/env python3
"""
compare_enkf_lsq.py

Compare enKF sensitivity results (20-run Monte Carlo bootstrap)
against the analytical LSQ baseline (Po2Fitter_3 on Krogh-generated data).

No re-run of the enKF is needed — all enKF arrays are loaded from disk.
LSQ is run fresh (N_LSQ_BOOT=20 independent noise realizations per config).

Output:
  Data/enKF_plots/sensitivity_analysis_v4bootsrap/comparison_enkf_lsq/
  ├── README.txt
  └── <sweep>/
      ├── 01_convergence_panel.png
      ├── 02_final_accuracy.png
      ├── 03_uncertainty_comparison.png
      ├── 04_signed_bias.png
      ├── lsq_cmro2_all.npy   -- (n_vals, N_LSQ_BOOT)
      ├── lsq_cmro2_mean.npy  -- (n_vals,)
      └── lsq_cmro2_std.npy   -- (n_vals,)
"""

import sys, os, io, json, warnings
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT    = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/"
CLASSES = os.path.join(ROOT, "classes")
IMPL    = os.path.join(ROOT, "Implementation")
for p in [ROOT, CLASSES, IMPL]:
    if p not in sys.path:
        sys.path.insert(0, p)

# suppress the "Added to sys.path" prints from lsqnonlin_M_pvessel_rout
_buf = io.StringIO()
with redirect_stdout(_buf):
    from lsqnonlin_M_pvessel_rout import Po2Fitter_3

# FEM forward model — same imports as sensitivity_analysis_v3.py
from FEM_code.generateMesh_Solver_multiple_holes import SolverParameters, HoleGeometry
from MapGenerator import MapGenerator

# ── physical constants (must match sensitivity_analysis_v3.py) ────────────────
D        = 4.0e3
ALPHA    = 1.39e-15
C2M      = 60 * D * ALPHA * 1e12   # CMRO2 [umol/cm3/min] → M-space

RVES_DEF  = 17.5    # μm
PVES_DEF  = 80.0    # mmHg
CMRO2_DEF = 2.0     # umol/cm3/min
R0_DEF    = 120.0   # μm  — baseline R0 (physiological default for non-R0 sweeps)
SIG_NOISE = 2.0   # mmHg — noise used in the enKF sensitivity runs (fixed, do not change)
# Both enKF and LSQ now use the same obs_std (from metadata) for a fair comparison.
# LSQ data is FEM-generated (not Krogh) to expose model-mismatch effects.

# ── observation grid (must match sensitivity_analysis_v3.py) ─────────────────
GRID      = 20
CELL_RES  = 17.0                    # μm per pixel (2-photon imaging resolution)
R0_MAX    = GRID * CELL_RES / 2     # 170 μm — R0 cannot exceed half the FOV
DOMAIN    = 190.0
X_AX, Y_AX = np.meshgrid(
    np.linspace(-DOMAIN, DOMAIN, GRID),
    np.linspace(-DOMAIN, DOMAIN, GRID),
)

# ── directories ───────────────────────────────────────────────────────────────
DATA_ROOT = os.path.join(ROOT, "Data/enKF_plots/sensitivity_analysis_v4bootsrap")
OAT_ROOT  = os.path.join(DATA_ROOT, "oat")
BOOT_ROOT = os.path.join(DATA_ROOT, "bootstrap")
DATA_OUT      = os.path.join(DATA_ROOT, "comparison_enkf_lsq")
FEM_CACHE_DIR = os.path.join(DATA_ROOT, "fem_map_cache")
os.makedirs(DATA_OUT, exist_ok=True)
os.makedirs(FEM_CACHE_DIR, exist_ok=True)

N_LSQ_BOOT = 20   # matches enKF bootstrap count
ITER_IDX   = 0    # 0 = iteration 1 (equal-information comparison with LSQ);
                  # change to -1 to evaluate at the final enKF iteration instead

# ── ground-truth CMRO2 per (sweep_name, sweep_value) ─────────────────────────
# For prior-std sweeps the truth is fixed; LSQ is insensitive to prior width.
def gt_params(sweep_name, val):
    d = dict(cmro2=CMRO2_DEF, R0=R0_DEF, Pves=PVES_DEF, obs_std=SIG_NOISE)
    if   sweep_name == "cmro2_true":  d["cmro2"]   = val
    elif sweep_name == "R0_true":     d["R0"]      = val
    elif sweep_name == "obs_std":     d["obs_std"] = val
    # cmro2_std / R0_std / pvessel_std → only prior width changes, truth fixed
    return d

SWEEP_LABELS = {
    "cmro2_true":  "CMRO2 true [umol/cm³/min]",
    "R0_true":     "R0 true [μm]",
    "obs_std":     "Obs. noise std [mmHg]",
    "cmro2_std":   "CMRO2 prior std [umol/cm³/min]",
    "R0_std":      "R0 prior std [μm]",
    "pvessel_std": "Pvessel prior std [mmHg]",
}

# ── analytical Krogh-cylinder pO2 map ────────────────────────────────────────
def krogh_pO2(cmro2, R0, Pves, Rves, X, Y):
    """
    Unclipped Krogh cylinder pO2 — honest LSQ prediction.
    The no-flux condition is encoded in the log-term coefficient (zero derivative
    at r = R0); the formula is the analytic continuation beyond R0 without
    clamping.  This is what Po2Fitter_3._partial_pressure evaluates, so using
    the same function for evaluation gives a fair measure of what LSQ actually
    predicts vs the FEM true map.
    """
    M      = cmro2 / C2M
    R      = np.sqrt(X**2 + Y**2)
    R_safe = np.maximum(R, Rves)
    pO2    = Pves + (M / 4) * (R_safe**2 - Rves**2) - (M * R0**2 / 2) * np.log(R_safe / Rves)
    return np.where(R < Rves, Pves, pO2)

# ── FEM-based pO2 map (disk-cached) ──────────────────────────────────────────
def fem_pO2_map(cmro2, R0, Pves, Rves=RVES_DEF):
    """
    Generate pO2 map via FEniCS FEM — identical pipeline as sensitivity_analysis_v3.
    Results are cached to disk so each unique (cmro2, R0, Pves, Rves) is solved once.
    """
    key  = f"c{cmro2:.5f}_R{R0:.3f}_P{Pves:.3f}_r{Rves:.3f}"
    path = os.path.join(FEM_CACHE_DIR, key + ".npy")
    if os.path.exists(path):
        print(f"    [FEM cache hit] {key}")
        return np.load(path)
    print(f"    [FEM solve]     {key}")
    params = SolverParameters(filename="square_holes")
    holes  = [HoleGeometry(center=(0., 0., 0.), cmro2=cmro2,
                           Pves=Pves, radius_ves=Rves, radius_0=R0,
                           marker=params.marker)]
    pO2 = MapGenerator(holes=holes, params=params, X=X_AX, Y=Y_AX).pO2_array
    np.save(path, pO2)
    return pO2

def radial_profile(pO2_map, X, Y):
    """Sort all pixels by distance from origin → (r_sorted, pO2_sorted)."""
    R    = np.sqrt(X**2 + Y**2).flatten()
    pO2f = pO2_map.flatten()
    idx  = np.argsort(R)
    return R[idx], pO2f[idx]

# ── single LSQ run ────────────────────────────────────────────────────────────
def run_lsq_once(cmro2_true, R0_true, Pves_true, obs_std, seed,
                  cmro2_init=2.0, r0_init=R0_DEF, pvessel_init=80.0):
    """Returns (cmro2_est, pves_est, r0_est) or (nan, nan, nan) on failure.

    Data is FEM-generated (same forward model as enKF) so both methods face
    identical ground truth. Center is estimated from argmax (no oracle) to
    reflect the real experimental regime where vessel location is not known.
    """
    rng   = np.random.RandomState(seed)
    pO2   = fem_pO2_map(cmro2_true, R0_true, Pves_true)   # FEM, not Krogh
    pO2_n = pO2 + rng.normal(0, obs_std, pO2.shape)
    try:
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitter = Po2Fitter_3(pO2_array=pO2_n, Rves=RVES_DEF,
                                 X_axis=X_AX, Y_axis=Y_AX, center=None)   # no oracle center
            fitter.fit(cmro2_init=cmro2_init, r0_init=r0_init,
                       pvessel_init=pvessel_init, r0_max=R0_MAX)
        cmro2_est, _, pves_est, r0_est = fitter.get_results()
        return float(cmro2_est), float(pves_est), float(r0_est)
    except Exception:
        return float("nan"), float("nan"), float("nan")

def run_lsq_mc(cmro2_true, R0_true, Pves_true, obs_std,
                cmro2_init=2.0, r0_init=R0_DEF, pvessel_init=80.0,
                n=N_LSQ_BOOT, seed0=0):
    """Returns dict with keys 'cmro2', 'pves', 'r0', each shape (n,)."""
    results = [
        run_lsq_once(cmro2_true, R0_true, Pves_true, obs_std, seed=seed0 + i,
                     cmro2_init=cmro2_init, r0_init=r0_init,
                     pvessel_init=pvessel_init)
        for i in range(n)
    ]
    return {
        "cmro2": np.array([r[0] for r in results]),
        "pves":  np.array([r[1] for r in results]),
        "r0":    np.array([r[2] for r in results]),
    }

# ── colors ────────────────────────────────────────────────────────────────────
C_MC      = "#1565C0"   # blue   — enKF MC
C_POST    = "#E65100"   # orange — enKF internal posterior
C_LSQ     = "#B71C1C"   # red    — LSQ
C_TRUTH   = "#1B5E20"   # dark green — ground truth
ALPHA_RUN = 0.12        # individual MC run opacity

def _sv_label(sw, v):
    if sw in ("R0_true", "R0_std"): return f"{v:.0f}"
    return f"{v:.2g}"

# ── Figure 1: convergence panel ───────────────────────────────────────────────
def plot_convergence(sw, sv, all_cm, mean_cm, std_cm, mean_post,
                     lsq_by_val, gt_by_val, out_dir):
    """
    all_cm    : (n, B, T_boot)  — all MC runs
    mean_cm   : (n, T_boot)     — MC mean
    std_cm    : (n, T_boot)     — MC spread
    mean_post : (n, T_boot)     — mean posterior std
    """
    n      = len(sv)
    T_boot = all_cm.shape[2]
    it_b   = np.arange(1, T_boot + 1)

    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, val in enumerate(sv):
        ax  = axes_flat[idx]
        gt  = gt_by_val[idx]
        lsq = lsq_by_val[idx]
        lsq_m, lsq_s = np.nanmean(lsq), np.nanstd(lsq)

        # ── individual MC runs (light background)
        for b in range(all_cm.shape[1]):
            ax.plot(it_b, all_cm[idx, b, :], color="gray", alpha=ALPHA_RUN, lw=0.7)

        # ── MC spread band
        ax.fill_between(it_b, mean_cm[idx] - std_cm[idx], mean_cm[idx] + std_cm[idx],
                        color=C_MC, alpha=0.20)
        # ── MC mean
        ax.plot(it_b, mean_cm[idx], color=C_MC, lw=2.2, label="enKF MC mean")

        # ── filter internal uncertainty band (posterior std)
        ax.fill_between(it_b,
                        mean_cm[idx] - mean_post[idx],
                        mean_cm[idx] + mean_post[idx],
                        color=C_POST, alpha=0.18, label="enKF posterior std")

        # ── LSQ horizontal reference
        ax.axhline(lsq_m, color=C_LSQ, lw=2, ls="-.", label="LSQ mean (FEM data)")
        ax.axhspan(lsq_m - lsq_s, lsq_m + lsq_s, color=C_LSQ, alpha=0.12, label="LSQ ±1σ")

        # ── ground truth
        ax.axhline(gt, color=C_TRUTH, lw=1.6, ls=":", label="Ground truth")

        # ── mark the comparison iteration
        cmp_it = ITER_IDX + 1 if ITER_IDX >= 0 else T_boot
        ax.axvline(cmp_it, color="black", lw=1.2, ls="--", alpha=0.5,
                   label=f"Comparison point (iter {cmp_it})")

        ax.set_title(f"{SWEEP_LABELS[sw]} = {_sv_label(sw, val)}", fontsize=9)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("CMRO2 [umol/cm³/min]", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6.5, loc="best")

    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle(f"CMRO2 convergence  |  sweep: {sw}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_convergence_panel.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    saved 01_convergence_panel.png")

# ── Figure 2: final accuracy ──────────────────────────────────────────────────
def plot_final_accuracy(sw, sv, mean_cm, std_cm, lsq_by_val, gt_by_val, out_dir):
    n    = len(sv)
    x    = np.arange(n)
    w    = 0.3

    fe_m = mean_cm[:, ITER_IDX]
    fe_s = std_cm[:, ITER_IDX]
    lm   = np.array([np.nanmean(a) for a in lsq_by_val])
    ls   = np.array([np.nanstd(a)  for a in lsq_by_val])
    gt   = np.array(gt_by_val)

    iter_label = f"iter {ITER_IDX + 1 if ITER_IDX >= 0 else 'final'}"
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * n), 4))
    ax.bar(x - w/2, fe_m, w, yerr=fe_s, capsize=4, color=C_MC,  alpha=0.85,
           label=f"enKF (MC mean ± std, n={N_LSQ_BOOT}, {iter_label})")
    ax.bar(x + w/2, lm,   w, yerr=ls,   capsize=4, color=C_LSQ, alpha=0.85,
           label=f"LSQ  (MC mean ± std, n={N_LSQ_BOOT}, FEM data, 1 obs)")
    ax.plot(x, gt, "o--", color=C_TRUTH, lw=1.8, ms=6, label="Ground truth")

    ax.set_xticks(x)
    ax.set_xticklabels([_sv_label(sw, v) for v in sv])
    ax.set_xlabel(SWEEP_LABELS[sw])
    ax.set_ylabel("CMRO2 [umol/cm³/min]")
    ax.set_title(f"Accuracy at {iter_label}  |  sweep: {sw}", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_final_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    saved 02_final_accuracy.png")

# ── Figure 3: uncertainty comparison ─────────────────────────────────────────
def plot_uncertainty(sw, sv, std_cm, mean_post, lsq_by_val, out_dir):
    iter_label = f"iter {ITER_IDX + 1 if ITER_IDX >= 0 else 'final'}"
    mc_spread  = std_cm[:, ITER_IDX]
    post_std   = mean_post[:, ITER_IDX]
    lsq_spread = np.array([np.nanstd(a) for a in lsq_by_val])
    ratio      = np.where(mc_spread > 0, post_std / mc_spread, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(sv, mc_spread,  "o-", color=C_MC,   lw=2, ms=6,
                 label=f"enKF MC spread (empirical, std across 20 runs, {iter_label})")
    axes[0].plot(sv, post_std,   "s-", color=C_POST,  lw=2, ms=6,
                 label=f"enKF posterior std (filter self-reported, {iter_label})")
    axes[0].plot(sv, lsq_spread, "^-", color=C_LSQ,  lw=2, ms=6,
                 label="LSQ MC spread (std across 20 runs, 1 obs)")
    axes[0].set_xlabel(SWEEP_LABELS[sw])
    axes[0].set_ylabel("Std of CMRO2 [umol/cm³/min]")
    axes[0].set_title(f"Uncertainty at {iter_label}")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(sv, ratio, "D-", color="purple", lw=2, ms=6,
                 label="post. std / MC spread")
    axes[1].axhline(1.0, color="black", ls="--", lw=1.2, label="Well-calibrated (= 1)")
    axes[1].axhspan(0.8, 1.2, color="gray", alpha=0.10, label="±20% tolerance")
    axes[1].set_xlabel(SWEEP_LABELS[sw])
    axes[1].set_ylabel("Calibration ratio")
    axes[1].set_title("Filter calibration\n(ratio<1: overconfident, ratio>1: underconfident)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(f"Uncertainty analysis  |  sweep: {sw}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_uncertainty_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    saved 03_uncertainty_comparison.png")

# ── Figure 4: signed bias ─────────────────────────────────────────────────────
def plot_bias(sw, sv, mean_cm, std_cm, lsq_by_val, gt_by_val, out_dir):
    iter_label = f"iter {ITER_IDX + 1 if ITER_IDX >= 0 else 'final'}"
    gt    = np.array(gt_by_val)
    fe_m  = mean_cm[:, ITER_IDX]
    fe_s  = std_cm[:, ITER_IDX]
    lm    = np.array([np.nanmean(a) for a in lsq_by_val])
    ls    = np.array([np.nanstd(a)  for a in lsq_by_val])

    bias_e = fe_m - gt
    bias_l = lm   - gt

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_e  = np.where(gt != 0, 100 * bias_e / gt, np.nan)
        rel_l  = np.where(gt != 0, 100 * bias_l / gt, np.nan)
        rel_se = np.where(gt != 0, 100 * fe_s   / gt, np.nan)
        rel_sl = np.where(gt != 0, 100 * ls      / gt, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].errorbar(sv, bias_e, yerr=fe_s, fmt="o-",  color=C_MC,  capsize=4, lw=2, ms=6, label="enKF")
    axes[0].errorbar(sv, bias_l, yerr=ls,   fmt="s-.", color=C_LSQ, capsize=4, lw=2, ms=6, label="LSQ")
    axes[0].axhline(0, color="black", ls="--", lw=1.2)
    axes[0].set_xlabel(SWEEP_LABELS[sw])
    axes[0].set_ylabel("Signed bias [umol/cm³/min]")
    axes[0].set_title("Signed bias  (estimate − truth)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25)

    axes[1].errorbar(sv, rel_e, yerr=rel_se, fmt="o-",  color=C_MC,  capsize=4, lw=2, ms=6, label="enKF")
    axes[1].errorbar(sv, rel_l, yerr=rel_sl, fmt="s-.", color=C_LSQ, capsize=4, lw=2, ms=6, label="LSQ")
    axes[1].axhline(0, color="black", ls="--", lw=1.2)
    axes[1].set_xlabel(SWEEP_LABELS[sw])
    axes[1].set_ylabel("Relative bias [%]")
    axes[1].set_title("Relative bias  (100 × (est−truth)/truth)")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.25)

    fig.suptitle(f"Bias profile at {iter_label}  |  sweep: {sw}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_signed_bias.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    saved 04_signed_bias.png")

# ── Figure 5: 1D radial pO2 profiles ─────────────────────────────────────────
def plot_pO2_radial(sw, sv, meta, mean_cm,
                    lsq_cmro2_by_val, lsq_pves_by_val, lsq_r0_by_val, out_dir):
    """
    1D radial pO2 profile per sweep value.
    True (FEM)  : ground truth parameters.
    enKF (FEM)  : MC mean CMRO2 at final iteration, ground truth R0 and Pves.
                  Isolates CMRO2 estimation error in the pO2 map.
    LSQ (Krogh) : MC mean of all three LSQ estimates.
    """
    n     = len(sv)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    print(f"  [FEM] generating radial-profile maps for sweep: {sw}")
    for idx, val in enumerate(sv):
        ax = axes_flat[idx]
        gt = meta[idx]["ground_truth"]

        cmro2_t = gt["cmro2"];  R0_t = gt["R0"];  pves_t = gt["Pves"]
        cmro2_e = mean_cm[idx, ITER_IDX]
        cmro2_l = np.nanmean(lsq_cmro2_by_val[idx])
        pves_l  = np.nanmean(lsq_pves_by_val[idx])
        r0_l    = np.nanmean(lsq_r0_by_val[idx])

        iter_label = f"iter {ITER_IDX + 1 if ITER_IDX >= 0 else 'final'}"

        # True map — FEM
        map_true = fem_pO2_map(cmro2_t, R0_t, pves_t)
        r_t, pO2_t = radial_profile(map_true, X_AX, Y_AX)

        # enKF map — FEM with MC mean CMRO2 + true R0/Pves (isolates CMRO2 error)
        map_enkf = fem_pO2_map(cmro2_e, R0_t, pves_t)
        r_e, pO2_e = radial_profile(map_enkf, X_AX, Y_AX)

        # LSQ map — unclipped Krogh (honest prediction, same as _partial_pressure)
        map_lsq = krogh_pO2(cmro2_l, r0_l, pves_l, RVES_DEF, X_AX, Y_AX)
        r_l, pO2_l = radial_profile(map_lsq, X_AX, Y_AX)

        ax.plot(r_t, pO2_t, "o", color=C_TRUTH, ms=4, alpha=0.7,  label="True (FEM)")
        ax.plot(r_e, pO2_e, "s", color=C_MC,    ms=4, alpha=0.7,  label=f"enKF CMRO2 ({iter_label}), true R0/Pv (FEM)")
        ax.plot(r_l, pO2_l, "-", color=C_LSQ,   lw=2, alpha=0.85, label="LSQ (Krogh, unclipped)")

        ax.axvline(RVES_DEF, color="gray",  lw=1, ls=":", label="Rves")
        ax.axvline(R0_t,     color=C_TRUTH, lw=1, ls="--", alpha=0.5, label="R0 true")
        ax.axvline(r0_l,     color=C_LSQ,   lw=1, ls="--", alpha=0.5, label="R0 LSQ est")

        ax.set_title(
            f"{SWEEP_LABELS[sw]} = {_sv_label(sw, val)}\n"
            f"enKF CMRO2={cmro2_e:.3f}  (true R0={R0_t:.0f}, Pv={pves_t:.0f})\n"
            f"LSQ: C={cmro2_l:.3f}  R0={r0_l:.1f}  Pv={pves_l:.1f}",
            fontsize=7.5
        )
        ax.set_xlabel("Radius r [μm]", fontsize=8)
        ax.set_ylabel("pO2 [mmHg]", fontsize=8)
        ax.set_xlim(0, min(DOMAIN, 2.5 * R0_t))
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    iter_label = f"iter {ITER_IDX + 1 if ITER_IDX >= 0 else 'final'}"
    fig.suptitle(
        f"1D radial pO2 profile  |  sweep: {sw}  |  enKF at {iter_label}\n"
        f"True & enKF: FEM    LSQ: Krogh unclipped (honest prediction beyond R0)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_pO2_radial_profiles.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    saved 05_pO2_radial_profiles.png")


# ── Figure 6: 2D spatial pO2 maps (middle sweep value) ───────────────────────
def plot_pO2_maps(sw, sv, meta, mean_cm,
                  lsq_cmro2_by_val, lsq_pves_by_val, lsq_r0_by_val, out_dir):
    """
    2D pO2 maps for the middle sweep value.
    True (FEM)  : ground truth parameters.
    enKF (FEM)  : MC mean CMRO2 at final iteration, ground truth R0 and Pves.
    LSQ (Krogh) : MC mean of all three LSQ estimates.
    5 panels: True | enKF | LSQ | enKF residual | LSQ residual.
    """
    mid = len(sv) // 2
    val = sv[mid]
    gt  = meta[mid]["ground_truth"]

    cmro2_t = gt["cmro2"];  R0_t = gt["R0"];  pves_t = gt["Pves"]
    iter_label = f"iter {ITER_IDX + 1 if ITER_IDX >= 0 else 'final'}"
    cmro2_e = mean_cm[mid, ITER_IDX]
    cmro2_l = np.nanmean(lsq_cmro2_by_val[mid])
    pves_l  = np.nanmean(lsq_pves_by_val[mid])
    r0_l    = np.nanmean(lsq_r0_by_val[mid])

    print(f"  [FEM] generating 2D maps for {sw}={val:.3g}")
    map_true = fem_pO2_map(cmro2_t, R0_t, pves_t)
    map_enkf = fem_pO2_map(cmro2_e, R0_t, pves_t)   # MC mean CMRO2 + true R0/Pves
    map_lsq  = krogh_pO2(cmro2_l, r0_l, pves_l, RVES_DEF, X_AX, Y_AX)

    res_enkf = map_enkf - map_true
    res_lsq  = map_lsq  - map_true

    vmin, vmax = map_true.min(), map_true.max()
    res_abs    = max(abs(res_enkf).max(), abs(res_lsq).max(), 1e-6)

    theta = np.linspace(0, 2 * np.pi, 300)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    panels = [
        (map_true, "True pO2\n(FEM)",                         "jet",    vmin,     vmax,    "mmHg"),
        (map_enkf, "enKF CMRO2, true R0/Pv\n(FEM)",           "jet",    vmin,     vmax,    "mmHg"),
        (map_lsq,  "LSQ estimated\n(Krogh, unclipped)",        "jet",    vmin,     vmax,    "mmHg"),
        (res_enkf, "enKF residual\n(est − true)",             "RdBu_r", -res_abs, res_abs, "mmHg"),
        (res_lsq,  "LSQ residual\n(est − true)",              "RdBu_r", -res_abs, res_abs, "mmHg"),
    ]
    for ax, (data, title, cmap, vm0, vm1, unit) in zip(axes, panels):
        im = ax.pcolormesh(X_AX, Y_AX, data, cmap=cmap, vmin=vm0, vmax=vm1, shading="auto")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(unit, fontsize=7)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("x [μm]", fontsize=8)
        ax.set_ylabel("y [μm]", fontsize=8)
        ax.set_aspect("equal")
        ax.plot(RVES_DEF * np.cos(theta), RVES_DEF * np.sin(theta), "w--", lw=1.2, alpha=0.8)
        ax.plot(R0_t     * np.cos(theta), R0_t     * np.sin(theta), "w:",  lw=1.0, alpha=0.6)

    rmse_e = np.sqrt(np.mean(res_enkf**2))
    rmse_l = np.sqrt(np.mean(res_lsq**2))
    fig.suptitle(
        f"2D pO2 maps  |  sweep: {sw} = {_sv_label(sw, val)}  |  enKF at {iter_label}\n"
        f"enKF RMSE = {rmse_e:.3f} mmHg  (CMRO2 error only, true R0/Pv)     "
        f"LSQ RMSE = {rmse_l:.3f} mmHg  (Krogh unclipped vs FEM)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_pO2_2D_maps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    saved 06_pO2_2D_maps.png")


# ── Figure 7: pO2 MAE vs sweep parameter ─────────────────────────────────────
def plot_pO2_mae(sw, sv, enkf_pO2_mae, lsq_pO2_mae, out_dir):
    """
    enkf_pO2_mae : (n_vals,) — mean |pO2_est − pO2_true| at ITER_IDX (enKF)
    lsq_pO2_mae  : (n_vals,) — mean |pO2_krogh_unclipped − pO2_true|, MC mean (LSQ)
    """
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(sv)), 4))
    iter_label = f"iter {ITER_IDX + 1 if ITER_IDX >= 0 else 'final'}"
    ax.plot(sv, enkf_pO2_mae, "o-", color=C_MC,  lw=2, ms=7,
            label=f"enKF  (FEM estimated pO2  vs  FEM true pO2, {iter_label})")
    ax.plot(sv, lsq_pO2_mae,  "s-", color=C_LSQ, lw=2, ms=7,
            label="LSQ   (Krogh display pO2  vs  FEM true pO2)")
    ax.set_xlabel(SWEEP_LABELS[sw])
    ax.set_ylabel("Mean absolute error pO2 [mmHg]")
    ax.set_title(
        f"pO2 map MAE at {iter_label}  |  sweep: {sw}\n"
        f"enKF: MC-mean CMRO2 + true R0/Pves (FEM)     "
        f"LSQ: MC-mean estimates (Krogh unclipped)",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_pO2_mae.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    saved 07_pO2_mae.png")


# ── process one sweep ─────────────────────────────────────────────────────────
def process_sweep(sw):
    oat_dir  = os.path.join(OAT_ROOT,  sw)   # used only for metadata.json
    boot_dir = os.path.join(BOOT_ROOT, sw)
    out_dir  = os.path.join(DATA_OUT,  sw)

    if not os.path.isdir(boot_dir):
        print(f"  [skip] {sw}: missing bootstrap directory")
        return
    if not os.path.isdir(oat_dir):
        print(f"  [skip] {sw}: missing oat directory (needed for metadata.json)")
        return
    os.makedirs(out_dir, exist_ok=True)

    # ── load enKF bootstrap results
    sv        = np.load(os.path.join(boot_dir, "sweep_values.npy"))
    all_cm    = np.load(os.path.join(boot_dir, "all_cmro2_mean.npy"))   # (n, B, T_boot)
    mean_cm   = np.load(os.path.join(boot_dir, "mean_cmro2_mean.npy"))  # (n, T_boot)
    std_cm    = np.load(os.path.join(boot_dir, "std_cmro2_mean.npy"))   # (n, T_boot)

    mpost_path = os.path.join(boot_dir, "mean_cmro2_std.npy")
    mean_post  = np.load(mpost_path) if os.path.exists(mpost_path) else np.zeros_like(mean_cm)

    n_vals     = len(sv)

    # ── load enKF prior means and ground truth from OAT metadata (authoritative)
    meta = json.load(open(os.path.join(oat_dir, "metadata.json")))
    prior_means = [
        dict(cmro2 = m["prior"]["cmro2_mean"],
             R0    = m["prior"]["R0_mean"],
             pves  = m["prior"]["pvessel_mean"])
        for m in meta
    ]
    # Ground truth read from metadata — matches exactly what the enKF data was generated with
    gt_by_val = [m["ground_truth"]["cmro2"] for m in meta]

    # ── load enKF pO2 MAE (mean |pO2_est − pO2_true| per sweep value, at final iter)
    pO2_abs_path = os.path.join(boot_dir, "mean_pO2_abs.npy")
    enkf_pO2_mae = np.full(n_vals, np.nan)
    if os.path.exists(pO2_abs_path):
        mean_pO2_abs = np.load(pO2_abs_path)   # (n_vals, T_boot)
        enkf_pO2_mae = mean_pO2_abs[:, ITER_IDX]
    else:
        print(f"  [warn] mean_pO2_abs.npy not found in {boot_dir}, pO2 MAE skipped for enKF")

    # ── run LSQ MC with same initial guesses as enKF priors and same obs_std
    print(f"  Running LSQ ({N_LSQ_BOOT} runs × {n_vals} values)...")
    lsq_cmro2_by_val = []
    lsq_pves_by_val  = []
    lsq_r0_by_val    = []

    for i, val in enumerate(sv):
        gt        = meta[i]["ground_truth"]
        pr        = prior_means[i]
        lsq_noise = meta[i]["obs_std"]   # same noise as enKF (fair comparison)
        res = run_lsq_mc(gt["cmro2"], gt["R0"], gt["Pves"], lsq_noise,
                         cmro2_init=pr["cmro2"], r0_init=pr["R0"],
                         pvessel_init=pr["pves"],
                         n=N_LSQ_BOOT, seed0=i * 1000)
        lsq_cmro2_by_val.append(res["cmro2"])
        lsq_pves_by_val.append(res["pves"])
        lsq_r0_by_val.append(res["r0"])
        valid = res["cmro2"][~np.isnan(res["cmro2"])]
        print(f"    {sw}={val:.3g}: obs_std={lsq_noise:.1f}  "
              f"init=({pr['cmro2']:.2f},{pr['R0']:.0f},{pr['pves']:.0f})  "
              f"n_valid={len(valid)}/{N_LSQ_BOOT}  "
              f"cmro2={np.nanmean(res['cmro2']):.4f}  "
              f"R0={np.nanmean(res['r0']):.1f}  "
              f"Pves={np.nanmean(res['pves']):.1f}  "
              f"truth={gt['cmro2']:.4f}")

    # ── compute LSQ pO2 MAE (Krogh clipped vs FEM true)
    print(f"  Computing LSQ pO2 MAE...")
    lsq_pO2_mae = np.zeros(n_vals)
    for i in range(n_vals):
        gt_i     = meta[i]["ground_truth"]
        map_true = fem_pO2_map(gt_i["cmro2"], gt_i["R0"], gt_i["Pves"])
        cmro2_l  = np.nanmean(lsq_cmro2_by_val[i])
        pves_l   = np.nanmean(lsq_pves_by_val[i])
        r0_l     = np.nanmean(lsq_r0_by_val[i])
        if not (np.isnan(cmro2_l) or np.isnan(pves_l) or np.isnan(r0_l)):
            map_lsq        = krogh_pO2(cmro2_l, r0_l, pves_l, RVES_DEF, X_AX, Y_AX)
            lsq_pO2_mae[i] = np.mean(np.abs(map_lsq - map_true))
        else:
            lsq_pO2_mae[i] = np.nan

    # ── save LSQ arrays
    for key, lst in [("cmro2", lsq_cmro2_by_val),
                     ("pves",  lsq_pves_by_val),
                     ("r0",    lsq_r0_by_val)]:
        mat = np.stack(lst, axis=0)
        np.save(os.path.join(out_dir, f"lsq_{key}_all.npy"),  mat)
        np.save(os.path.join(out_dir, f"lsq_{key}_mean.npy"), np.nanmean(mat, axis=1))
        np.save(os.path.join(out_dir, f"lsq_{key}_std.npy"),  np.nanstd( mat, axis=1))

    # ── save LSQ MAE arrays
    np.save(os.path.join(out_dir, "lsq_pO2_mae.npy"),  lsq_pO2_mae)
    np.save(os.path.join(out_dir, "enkf_pO2_mae.npy"), enkf_pO2_mae)

    # ── figures
    plot_convergence(sw, sv, all_cm, mean_cm, std_cm, mean_post,
                     lsq_cmro2_by_val, gt_by_val, out_dir)
    plot_final_accuracy(sw, sv, mean_cm, std_cm, lsq_cmro2_by_val, gt_by_val, out_dir)
    plot_uncertainty(sw, sv, std_cm, mean_post, lsq_cmro2_by_val, out_dir)
    plot_bias(sw, sv, mean_cm, std_cm, lsq_cmro2_by_val, gt_by_val, out_dir)
    plot_pO2_radial(sw, sv, meta, mean_cm,
                    lsq_cmro2_by_val, lsq_pves_by_val, lsq_r0_by_val, out_dir)
    plot_pO2_maps(sw, sv, meta, mean_cm,
                  lsq_cmro2_by_val, lsq_pves_by_val, lsq_r0_by_val, out_dir)
    if not np.all(np.isnan(enkf_pO2_mae)):
        plot_pO2_mae(sw, sv, enkf_pO2_mae, lsq_pO2_mae, out_dir)

# ── README ────────────────────────────────────────────────────────────────────
README = """\
comparison_enkf_lsq/
====================
EnKF sensitivity results vs. LSQ (Po2Fitter_3) baseline.
Both methods use FEM-generated pO2 data and the same observation noise.

STRUCTURE
---------
  <sweep>/
    01_convergence_panel.png      -- CMRO2 vs iteration, per sweep value
    02_final_accuracy.png         -- final estimate vs sweep parameter (bar chart)
    03_uncertainty_comparison.png -- MC spread vs posterior std vs LSQ spread
    04_signed_bias.png            -- signed & relative bias vs sweep parameter
    05_pO2_radial_profiles.png    -- 1D radial pO2 profile per sweep value
    06_pO2_2D_maps.png            -- 2D pO2 maps for the middle sweep value
    07_pO2_mae.png                -- pO2 mean absolute error vs sweep parameter
    lsq_cmro2_all.npy             -- shape (n_vals, 20): raw LSQ CMRO2 estimates
    lsq_cmro2_mean.npy            -- shape (n_vals,): mean across 20 LSQ runs
    lsq_cmro2_std.npy             -- shape (n_vals,): std across 20 LSQ runs
    lsq_pO2_mae.npy               -- shape (n_vals,): LSQ pO2 MAE vs FEM true
    enkf_pO2_mae.npy              -- shape (n_vals,): enKF pO2 MAE at final iter

FIGURES
-------
01_convergence_panel.png
  One subplot per sweep value. x-axis = enKF iteration (1..10 from bootstrap).
  [gray thin]   : all 20 individual Monte-Carlo enKF runs
  [blue solid]  : MC mean (average over 20 runs)
  [blue band]   : MC mean ± 1σ MC spread (empirical variability)
  [orange band] : MC mean ± mean posterior std (filter self-reported uncertainty)
  [red dash-dot]: LSQ mean across 20 re-runs (horizontal — no iterations)
  [red band]    : LSQ ± 1σ (horizontal band)
  [green dot]   : Ground-truth CMRO2

02_final_accuracy.png
  Grouped bar chart at final iteration. x-axis = swept parameter value.
  Blue bars = enKF (MC mean ± MC std).
  Red bars  = LSQ  (MC mean ± MC std, n=20).
  Green line = ground truth (may be constant for prior-std sweeps).

03_uncertainty_comparison.png
  LEFT  — three uncertainty curves at final iteration vs sweep parameter:
    Blue   = enKF MC spread  (empirical: std of 20 independent estimates)
    Orange = enKF posterior std (internal: what the filter thinks its error is)
    Red    = LSQ MC spread   (empirical: std of 20 LSQ fits)
  RIGHT — calibration ratio = posterior_std / MC_spread
    = 1  : well-calibrated (internal uncertainty matches empirical variability)
    < 1  : overconfident   (filter underestimates uncertainty)
    > 1  : underconfident  (filter overestimates uncertainty)
  Gray band = ±20% tolerance zone around perfect calibration.

04_signed_bias.png
  LEFT  : Signed bias = estimate − truth [umol/cm³/min], error bars = MC std.
  RIGHT : Relative bias [%] = 100 × (estimate − truth) / truth.
  Blue = enKF, Red = LSQ.

07_pO2_mae.png
  Mean absolute pO2 error at final iteration vs sweep parameter.
  enKF: MAE = mean|pO2_FEM(cmro2_est, R0_true, Pves_true) − pO2_FEM_true|
         (isolates CMRO2 estimation error in pO2 space)
  LSQ:  MAE = mean|pO2_Krogh_display(MC-mean params) − pO2_FEM_true|
         (includes both parameter estimation error AND Krogh-FEM model mismatch)

NOTE ON DATA SOURCES
--------------------
enKF: loaded directly from sensitivity_analysis_v4bootsrap/ (no re-run needed).
  Ground truth was FEM-generated; filter used FEM as forward model.
  Observation noise: σ from metadata.json (= SIG_NOISE = 2.0 mmHg fixed,
  or the swept value for obs_std sweep).

LSQ:  freshly computed on FEM-generated synthetic data.
  Observation noise: same σ as enKF (from metadata.json) — fair comparison.
  Center estimated from argmax of noisy pO2 map (no oracle) — realistic regime.
  LSQ uses the Krogh forward model, which does NOT perfectly match FEM.
  Any residual bias is due to both noise and model mismatch (Krogh ≠ FEM).

INTERPRETATION GUIDE FOR PRIOR-STD SWEEPS (cmro2_std, R0_std, pvessel_std)
  The ground truth does not change across these sweeps — only the enKF prior
  width changes. LSQ has no prior, so its estimates (and bias) are identical
  across these sweeps. This reveals what is prior-dependent vs. data-dependent.
"""

# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open(os.path.join(DATA_OUT, "README.txt"), "w") as f:
        f.write(README.strip())
    print(f"README written to {DATA_OUT}")

    sweeps = ["cmro2_true", "R0_true", "obs_std", "cmro2_std", "R0_std", "pvessel_std"]
    for sw in sweeps:
        print(f"\n{'='*60}")
        print(f"  Sweep: {sw}")
        print(f"{'='*60}")
        try:
            process_sweep(sw)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {sw}: {e}")
            traceback.print_exc()

    print(f"\nDone. All results in:\n  {DATA_OUT}")
