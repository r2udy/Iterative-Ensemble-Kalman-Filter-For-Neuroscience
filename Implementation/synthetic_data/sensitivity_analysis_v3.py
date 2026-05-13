#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IEnKF Sensitivity Analysis  —  v3
===================================
Refinements over v2:
  1. All sweeps expressed in std (not variance); variance computed internally
  2. Ground-truth consistency: each experiment explicitly tracks its own ground
     truth; metadata saved as JSON per sweep for full traceability
  3. Physical validity checks: flags negative pO2, bias > BIAS_FLAG_THRESHOLD
  4. Tightened parameter ranges; extreme values avoided
  5. Log-scale on error / uncertainty plots where magnitude spans orders
  6. Bootstrap extension (optional, B=20, N=50, T=10) in separate folder
  7. Deterministic per-worker seeds for full reproducibility
  8. All parameter choices made explicit; no silent assumptions

Sweep axes (all expressed as std, variance computed internally)
--------------------------------------------------------------
  cmro2_true  : [1.0, 1.5, 2.0, 2.5]             µmol/cm3/min
  R0_true     : [80, 90, 100, 110, 120]            µm
  obs_std     : linspace(1.0, 10.0, 5)             mmHg
  cmro2_std   : linspace(0.1,  1.0, 5)             µmol/cm3/min (prior)
  R0_std      : linspace(2.0, 10.0, 5)             µm (prior)
  pvessel_std : linspace(1.0, 10.0, 5)             mmHg (prior)

2D cross-sweeps (final iteration, heatmap + scatter)
-----------------------------------------------------
  1. cmro2_true x obs_std      -- signal vs noise
  2. cmro2_true x cmro2_std    -- prior width vs ground truth
  3. R0_true    x obs_std      -- geometry vs noise
  4. obs_std    x cmro2_std    -- observation vs prior

Bootstrap (when RUN_BOOTSTRAP=True)
------------------------------------
  N_BOOTSTRAP   = 20  independent runs per configuration
  N_ENS_BOOT    = 50  ensemble members  (matched to OAT)
  ITER_MAX_BOOT = 15  iterations        (matched to OAT)
  Results -> <SAVE_PATH>/bootstrap/

Augmentation comparison (blind-spot baseline, when RUN_BOOTSTRAP=True)
----------------------------------------------------------------------
  Sweep: cmro2_true
  Runs with and without the R0 soft-constraint augmentation so the benefit
  of the augmented observation can be verified empirically.
  Results -> <SAVE_PATH>/augmentation_comparison/{aug,noaug}/

NOTE on prior centering
-----------------------
  cmro2_true sweep : prior mean stays at DEF["cmro2_mean0"] = 2.0 for ALL rows.
                     This intentionally tests recovery from a mis-centred prior.
  R0_true sweep    : prior mean tracks the true R0 (p["R0_mean0"] = R0_true).
  All other sweeps : prior means fixed at DEF values.
  These choices are recorded in the per-experiment metadata JSON.
"""

import sys, os, time, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 0.  PATHS
# =============================================================================
ROOT       = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/"
CLASSES    = os.path.join(ROOT, "classes")
IMPL       = os.path.join(ROOT, "Implementation")
CLUSTERING = os.path.join(IMPL, "Clustering")
SAVE_PATH  = os.path.join(ROOT, "Data/EnKF_plots/sensitivity_analysis_v4bootsrap/")

for _p in [ROOT, CLASSES, IMPL, CLUSTERING]:
    if _p not in sys.path:
        sys.path.append(_p)

from EnKF_FEM_3 import EnKF
from FEM_code.generateMesh_Solver_multiple_holes import SolverParameters, HoleGeometry
from MapGenerator import MapGenerator

for _sub in ["oat", "cross", "bootstrap"]:
    os.makedirs(os.path.join(SAVE_PATH, _sub), exist_ok=True)

# =============================================================================
# 1.  CONSTANTS & EXPERIMENT SETTINGS
# =============================================================================
D      = 4.0e3
ALPHA  = 1.39e-15
C2M    = 60 * D * ALPHA * 1e12   # CMRO2 [umol/cm3/min] -> M-space

GRID   = 20
X_AX, Y_AX = np.meshgrid(
    np.linspace(-190, 190, GRID),
    np.linspace(-190, 190, GRID),
)
OBS_DIM = GRID * GRID

# Main experiment settings
N_ENS       = 50
ITER_MAX    = 15
STATE_DIM   = 3
SIG_NOISE   = 2.0       # pO2 observation additive noise [mmHg]
MASTER_SEED = 42

# Bootstrap settings (separate from main)
RUN_BOOTSTRAP   = True
N_BOOTSTRAP     = 20    # independent repeats per config
N_ENS_BOOT      = 50    # ensemble members in bootstrap (matched to OAT)
ITER_MAX_BOOT   = 15    # iterations in bootstrap (matched to OAT)

# Physical validity threshold
BIAS_FLAG_THRESHOLD = .3   # umol/cm3/min -- flag if |bias| exceeds this

# Parallel workers

# Default parameter set (all prior widths in std, NOT variance)
DEF = dict(
    cmro2_true    = 2.0,    # ground-truth CMRO2  [umol/cm3/min]
    R0_true       = 100.0,  # ground-truth R0     [um]
    Pves_true     = 80.0,   # ground-truth Pves   [mmHg]
    Rves          = 17.5,   # vessel radius (fixed, not inferred) [um]
    obs_std       = 2.0,    # observation noise std  [mmHg]
    cmro2_std     = 0.5,    # CMRO2 prior std        [umol/cm3/min]
    R0_std        = 2.0,    # R0 prior std           [um]
    pvessel_std   = 2.0,    # pvessel prior std      [mmHg]
    cmro2_mean0   = 2.0,    # CMRO2 prior mean       [umol/cm3/min]
    R0_mean0      = 100.0,  # R0 prior mean          [um]
    pvessel_mean0 = 80.0,   # pvessel prior mean     [mmHg]
)

# OAT sweep grids (std for prior-width axes; actual values for true-value axes)
SWEEPS = {
    "cmro2_true"  : np.array([1.0, 1.5, 2.0, 2.5]),
    "R0_true"     : np.array([80., 90., 100., 110., 120.]),
    "obs_std"     : np.linspace(1.0, 10.0, 5),
    "cmro2_std"   : np.linspace(0.1,  1.0, 5),
    "R0_std"      : np.linspace(2.0, 10.0, 5),
    "pvessel_std" : np.linspace(1.0, 10.0, 5),
}

# (axis label, tick formatter)
SWEEP_META = {
    "cmro2_true"  : ("CMRO2 true [umol/cm3/min]",   lambda v: f"{v:.2f}"),
    "R0_true"     : ("R0 true [um]",                 lambda v: f"{v:.0f}"),
    "obs_std"     : ("Obs noise sigma [mmHg]",        lambda v: f"{v:.2f}"),
    "cmro2_std"   : ("CMRO2 prior sigma [umol/cm3/min]", lambda v: f"{v:.3f}"),
    "R0_std"      : ("R0 prior sigma [um]",          lambda v: f"{v:.1f}"),
    "pvessel_std" : ("pvessel prior sigma [mmHg]",    lambda v: f"{v:.2f}"),
}

# 2D cross-sweep definitions
CROSS_SWEEPS = [
    ("cmro2_true", np.array([1.0, 1.5, 2.0, 2.5]),
     "obs_std",    np.linspace(1.0, 10.0, 5),
     "CMRO2_true x obs_std -- signal vs noise"),

    ("cmro2_true", np.array([1.0, 1.5, 2.0, 2.5]),
     "cmro2_std",  np.linspace(0.1, 1.0, 5),
     "CMRO2_true x cmro2_std -- prior width vs ground truth"),

    ("R0_true",    np.array([80., 90., 100., 110., 120.]),
     "obs_std",    np.linspace(1.0, 10.0, 5),
     "R0_true x obs_std -- geometry vs noise"),

    ("obs_std",    np.linspace(1.0, 10.0, 5),
     "cmro2_std",  np.linspace(0.1, 1.0, 5),
     "obs_std x cmro2_std -- observation vs prior"),
]

# CROSS_SWEEPS = [  # keep only 1, the most informative one
#     ("cmro2_true", np.array([1.0, 1.5, 2.0, 2.5]),
#      "obs_std",    np.linspace(1.0, 10.0, 5),
#      "CMRO2_true x obs_std -- signal vs noise"),
#     # the other 3 commented out
# ]

# =============================================================================
# 2.  PHYSICAL VALIDITY CHECKER
# =============================================================================

def check_physical_validity(pO2_map, label, cmro2_est=None, cmro2_true=None):
    """
    Check a pO2 map for physical plausibility.
    Returns dict of boolean flags (True = problem detected).
    """
    flags = dict(negative_pO2=False, large_bias=False,
                 nan_in_map=False, inf_in_map=False)
    arr = np.asarray(pO2_map).flatten()

    if np.any(np.isnan(arr)):
        flags["nan_in_map"] = True
        print(f"  [VALIDITY] {label}: NaN in pO2 map")

    if np.any(np.isinf(arr)):
        flags["inf_in_map"] = True
        print(f"  [VALIDITY] {label}: Inf in pO2 map")

    if np.any(arr < 0):
        flags["negative_pO2"] = True
        print(f"  [VALIDITY] {label}: {int(np.sum(arr<0))} negative pO2 "
              f"(min={arr.min():.2f} mmHg)")

    if cmro2_est is not None and cmro2_true is not None:
        b = abs(cmro2_true - cmro2_est)
        if b > BIAS_FLAG_THRESHOLD:
            flags["large_bias"] = True
            print(f"  [VALIDITY] {label}: |bias|={b:.3f} > {BIAS_FLAG_THRESHOLD}")

    return flags


# =============================================================================
# 3.  CORE ENGINE
# =============================================================================

def _make_enkf(obs_std, cmro2_std_phys, R0_std, pvessel_std,
               cmro2_mean0, R0_mean0, pvessel_mean0,
               seed=None, n_ens=None, b_decay=0.0):
    """
    Build and initialise EnKF.
    All prior widths given as std; variance is computed here.

    b_decay : float
        Process-noise annealing rate.
        0.0 (default) = B=0 throughout, theoretically correct for static IEnKF.
        0.5           = geometric decay B_n = B * 0.5^n (practical fallback).
        1.0           = constant B (legacy behaviour, NOT recommended).
    """
    _n = n_ens if n_ens is not None else N_ENS
    obs_var     = obs_std ** 2
    cmro2_var_M = (cmro2_std_phys / C2M) ** 2
    R0_var      = R0_std ** 2
    pvessel_var = pvessel_std ** 2

    def dyn(x): return x

    enkf = EnKF(STATE_DIM, OBS_DIM, _n, dyn, seed=seed, b_decay=b_decay)
    a = np.array([cmro2_mean0 / C2M, R0_mean0, pvessel_mean0] * 3)
    b = np.array([cmro2_var_M, R0_var, pvessel_var] * 3)
    enkf.initialize_ensemble(a, b)
    enkf.set_process_noise(np.diag([cmro2_var_M, R0_var, pvessel_var]))
    enkf.set_observation_noise(obs_var * np.eye(OBS_DIM))
    return enkf



_TRUE_MAP_CACHE = {}   # (cmro2, Pves, Rves, R0) -> pO2_array

def _true_map(cmro2, Pves, Rves, R0):
    """
    FEM solve for ground-truth pO2. Cached by (cmro2, Pves, Rves, R0):
    bootstrap repeats and prior-sensitivity sweeps with the same ground truth
    skip the FEM solve entirely, saving ~14 min across the full experiment.
    """
    key = (round(cmro2, 6), round(Pves, 6), round(Rves, 6), round(R0, 6))
    if key in _TRUE_MAP_CACHE:
        return _TRUE_MAP_CACHE[key]
    params = SolverParameters(filename="square_holes")
    holes  = [HoleGeometry(center=(0., 0., 0.), cmro2=cmro2,
                           Pves=Pves, radius_ves=Rves, radius_0=R0,
                           marker=params.marker)]
    arr = MapGenerator(holes=holes, params=params, X=X_AX, Y=Y_AX).pO2_array
    _TRUE_MAP_CACHE[key] = arr
    return arr


def _est_map(cmro2, pves, Rves, R0):
    params = SolverParameters(filename="square_holes")
    holes  = [HoleGeometry(center=(0., 0., 0.), cmro2=cmro2,
                           Pves=pves, radius_ves=Rves, radius_0=R0,
                           marker=params.marker)]
    return MapGenerator(holes=holes, params=params, X=X_AX, Y=Y_AX).pO2_array


def run_single(p, seed=None, iter_max=None, n_ens=None,
               b_decay=0.0, use_augmentation=True):
    """
    Run one full IEnKF experiment.

    Parameters
    ----------
    p : dict
        Must contain (all in physical / std units):
        cmro2_true, R0_true, Pves_true, Rves,
        obs_std, cmro2_std, R0_std, pvessel_std,
        cmro2_mean0, R0_mean0, pvessel_mean0
    seed : int or None
    iter_max, n_ens : override global settings (used by bootstrap)
    b_decay : float
        Process-noise annealing rate passed to _make_enkf.
        0.0 (default) = no process noise, theoretically correct for static IEnKF.
    use_augmentation : bool
        If True (default), pass soft R0 prior constraint as an augmented
        observation at each update step.  Set False for the no-aug baseline.

    Returns
    -------
    result : dict of per-iteration np.arrays
    meta   : dict (ground truth, prior params, validity flags)
    """
    _iter = iter_max if iter_max is not None else ITER_MAX
    _n    = n_ens   if n_ens    is not None else N_ENS

    enkf = _make_enkf(
        obs_std        = p["obs_std"],
        cmro2_std_phys = p["cmro2_std"],
        R0_std         = p["R0_std"],
        pvessel_std    = p["pvessel_std"],
        cmro2_mean0    = p["cmro2_mean0"],
        R0_mean0       = p["R0_mean0"],
        pvessel_mean0  = p["pvessel_mean0"],
        seed           = seed,
        n_ens          = _n,
        b_decay        = b_decay,
    )

    enkf.set_rves(p["Rves"])  # Fix Bug 1: pass correct vessel radius to observation_operator

    # Generate TRUE pO2 map with THIS experiment's ground truth
    true_pO2 = _true_map(
        cmro2 = p["cmro2_true"],
        Pves  = p["Pves_true"],
        Rves  = p["Rves"],
        R0    = p["R0_true"],
    )

    val_true = check_physical_validity(
        true_pO2,
        label=f"true(cmro2={p['cmro2_true']:.2f},R0={p['R0_true']:.0f})"
    )

    out = {k: [] for k in [
        "cmro2_mean", "cmro2_std",
        "R0_mean",    "R0_std",
        "pves_mean",  "pves_std",
        "signed_bias",  # true - estimated  (+ = underestimate, - = overestimate)
        "bias",         # |signed_bias|
        "rel_bias",     # |bias| / true * 100  [%]
        "pO2_abs",      # mean |true - est| pO2  [mmHg]
        "pO2_rel",      # mean relative pO2 error  [%]
        "spread",       # ensemble std in CMRO2 space
        "gain_norm",    # Frobenius norm of Kalman gain
    ]}

    val_est_final = {}

    for it in range(_iter):
        obs = true_pO2.flatten() + np.random.normal(0, p["obs_std"], OBS_DIM)
        enkf.predict()
        enkf.update(obs, X_AX, Y_AX, use_augmentation=use_augmentation)
        mu, cov = enkf.get_state_estimate()

        cm  = mu[0] * C2M
        cs  = np.sqrt(max(cov[0, 0], 0)) * C2M
        R0m = mu[1];  R0s = np.sqrt(max(cov[1, 1], 0))
        pvm = mu[2];  pvs = np.sqrt(max(cov[2, 2], 0))

        ep    = _est_map(cm, pvm, p["Rves"], R0m)
        tf    = true_pO2.flatten()
        ef    = ep.flatten()
        denom = np.abs(tf) + 1e-9   # use true pO2, not noisy obs, for relative error
        sb    = p["cmro2_true"] - cm   # signed: + = under

        out["cmro2_mean"].append(cm)
        out["cmro2_std"].append(cs)
        out["R0_mean"].append(R0m)
        out["R0_std"].append(R0s)
        out["pves_mean"].append(pvm)
        out["pves_std"].append(pvs)
        out["signed_bias"].append(sb)
        out["bias"].append(abs(sb))
        out["rel_bias"].append(abs(sb) / (p["cmro2_true"] + 1e-9) * 100)
        out["pO2_abs"].append(float(np.mean(np.abs(tf - ef))))
        out["pO2_rel"].append(float(np.mean(np.abs(tf - ef) / denom) * 100))
        out["spread"].append(float(np.std(enkf.ensemble[0, :]) * C2M))
        try:
            out["gain_norm"].append(float(np.linalg.norm(enkf.K)))
        except Exception:
            out["gain_norm"].append(np.nan)

        if it == _iter - 1:
            val_est_final = check_physical_validity(
                ep, label=f"est(iter={it+1})",
                cmro2_est=cm, cmro2_true=p["cmro2_true"],
            )

    result = {k: np.array(v) for k, v in out.items()}

    meta = dict(
        ground_truth     = dict(cmro2=p["cmro2_true"], R0=p["R0_true"],
                                Pves=p["Pves_true"],  Rves=p["Rves"]),
        prior            = dict(cmro2_mean=p["cmro2_mean0"], cmro2_std=p["cmro2_std"],
                                R0_mean=p["R0_mean0"],       R0_std=p["R0_std"],
                                pvessel_mean=p["pvessel_mean0"],
                                pvessel_std=p["pvessel_std"]),
        obs_std          = p["obs_std"],
        n_ensembles      = _n,
        iter_max         = _iter,
        validity_true    = val_true,
        validity_est     = val_est_final,
        any_validity_flag= any(val_true.values()) or any(val_est_final.values()),
    )
    return result, meta


# =============================================================================
# 4.  OAT RUNNER  (serial)
# =============================================================================

def run_oat_sweep(sweep_name, sweep_values):
    keys  = ["cmro2_mean","cmro2_std","R0_mean","R0_std","pves_mean","pves_std",
             "signed_bias","bias","rel_bias","pO2_abs","pO2_rel","spread","gain_norm"]
    n     = len(sweep_values)
    res   = {k: np.zeros((n, ITER_MAX)) for k in keys}
    metas = [None] * n

    rng   = np.random.RandomState(MASTER_SEED)
    seeds = rng.randint(0, 2**31, size=n).tolist()

    for i, val in enumerate(sweep_values):
        np.random.seed(seeds[i])
        p = {k: v for k, v in DEF.items()}
        p[sweep_name] = val
        if sweep_name == "R0_true":
            p["R0_mean0"] = val
        result, meta = run_single(p, seed=seeds[i])
        for k in keys:
            res[k][i] = result[k]
        metas[i] = meta
        flag = " [!] validity" if meta["any_validity_flag"] else ""
        print(f"    [{sweep_name}={val:.4g}]  "
              f"CMRO2={res['cmro2_mean'][i,-1]:.3f}+/-{res['cmro2_std'][i,-1]:.3f}  "
              f"signed={res['signed_bias'][i,-1]:+.3f}  "
              f"pO2={res['pO2_abs'][i,-1]:.2f} mmHg{flag}")
    return res, metas


# =============================================================================
# 5.  2D CROSS-SWEEP RUNNER  (serial)
# =============================================================================

def run_cross_sweep(pA, valsA, pB, valsB):
    nA, nB      = len(valsA), len(valsB)
    signed_grid = np.zeros((nA, nB))
    bias_grid   = np.zeros((nA, nB))
    std_grid    = np.zeros((nA, nB))
    pO2_grid    = np.zeros((nA, nB))
    meta_grid   = [[None]*nB for _ in range(nA)]
    total       = nA * nB
    done        = 0

    rng = np.random.RandomState(MASTER_SEED + 1000)
    for i, va in enumerate(valsA):
        for j, vb in enumerate(valsB):
            seed = int(rng.randint(0, 2**31))
            np.random.seed(seed)
            p = {k: v for k, v in DEF.items()}
            p[pA] = va
            p[pB] = vb
            if pA == "R0_true": p["R0_mean0"] = va
            if pB == "R0_true": p["R0_mean0"] = vb
            result, meta = run_single(p, seed=seed)
            signed_grid[i,j] = result["signed_bias"][-1]
            bias_grid[i,j]   = result["bias"][-1]
            std_grid[i,j]    = result["cmro2_std"][-1]
            pO2_grid[i,j]    = result["pO2_abs"][-1]
            meta_grid[i][j]  = meta
            done += 1
            flag = " [!]" if meta["any_validity_flag"] else ""
            print(f"    ({i},{j}) signed={result['signed_bias'][-1]:+.3f}  "
                  f"|bias|={result['bias'][-1]:.3f}  "
                  f"pO2={result['pO2_abs'][-1]:.2f}  [{done}/{total}]{flag}")
    return signed_grid, bias_grid, std_grid, pO2_grid, meta_grid


# =============================================================================
# 6.  BOOTSTRAP RUNNER  (serial, only runs when RUN_BOOTSTRAP = True)
# =============================================================================

def run_bootstrap_sweep(sweep_name, sweep_values, use_augmentation=True):
    """
    N_BOOTSTRAP independent repeats per sweep value.
    Returns mean_res, std_res (n_vals x ITER_MAX_BOOT) and
            all_r (n_vals x N_BOOTSTRAP x ITER_MAX_BOOT).

    use_augmentation : bool
        Passed through to run_single/enkf.update.  Set False for no-aug baseline.
    """
    keys  = ["cmro2_mean", "cmro2_std", "signed_bias", "bias",
             "pO2_abs", "spread", "R0_mean", "R0_std", "pves_mean", "pves_std"]
    n     = len(sweep_values)
    all_r = {k: np.zeros((n, N_BOOTSTRAP, ITER_MAX_BOOT)) for k in keys}

    rng = np.random.RandomState(MASTER_SEED + 2000)
    for i, val in enumerate(sweep_values):
        for b in range(N_BOOTSTRAP):
            seed = int(rng.randint(0, 2**31))
            np.random.seed(seed)
            p = {k: v for k, v in DEF.items()}
            p[sweep_name] = val
            if sweep_name == "R0_true":
                p["R0_mean0"] = val
            result, _ = run_single(p, seed=seed,
                                   iter_max=ITER_MAX_BOOT, n_ens=N_ENS_BOOT,
                                   use_augmentation=use_augmentation)
            for k in keys:
                all_r[k][i, b, :] = result[k]
        print(f"    [{sweep_name}={val:.4g}]  {N_BOOTSTRAP} bootstrap runs done")

    mean_res = {k: all_r[k].mean(axis=1) for k in keys}
    std_res  = {k: all_r[k].std(axis=1)  for k in keys}
    return mean_res, std_res, all_r


# =============================================================================
# 7.  PLOTTING -- OAT
# =============================================================================

def _colors(n, cmap="plasma"):
    return plt.cm.get_cmap(cmap)(np.linspace(0.12, 0.92, n))


def _auto_log(ax, data):
    """Apply log y-scale if data spans > 1 order of magnitude."""
    flat = np.asarray(data).flatten()
    pos  = flat[flat > 0]
    if len(pos) > 0 and pos.max() / pos.min() > 10:
        ax.set_yscale("log")


def fig_priority_panel(sweep_name, sweep_values, res):
    meta   = SWEEP_META[sweep_name]
    colors = _colors(len(sweep_values))
    iters  = np.arange(1, ITER_MAX + 1)

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    configs = [
        ("signed_bias", "Signed CMRO2 bias\n(true-est) [umol/cm3/min]", False),
        ("cmro2_std",   "Posterior sigma (CMRO2)\n[umol/cm3/min]",       False),
        ("pO2_abs",     "Mean pO2 abs error\n[mmHg]",                    True),
    ]
    for ax, (metric, ylabel, try_log) in zip(axes, configs):
        for idx, val in enumerate(sweep_values):
            ax.plot(iters, res[metric][idx], color=colors[idx],
                    lw=2, marker="o", ms=3, label=meta[1](val))
        if metric == "signed_bias":
            ax.axhline(0, color="k", lw=1.4, ls="--", zorder=5)
        if try_log:
            _auto_log(ax, res[metric])
        ax.axvspan(ITER_MAX - 0.5, ITER_MAX + 0.5, color="gold", alpha=0.15)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7, ncol=3, loc="upper right")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Iteration", fontsize=10)
    fig.suptitle(f"Priority Metrics vs Iteration\n"
                 f"Sweep: {sweep_name}  ({meta[0]})",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    return fig


def fig_final_trio(sweep_name, sweep_values, res):
    meta   = SWEEP_META[sweep_name]
    labels = [meta[1](v) for v in sweep_values]
    x      = np.arange(len(sweep_values))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    configs = [
        ("signed_bias", "Signed CMRO2 bias\n[umol/cm3/min]",  "#2E75B6", False),
        ("cmro2_std",   "Posterior sigma (CMRO2)\n[umol/cm3/min]","#ED7D31", False),
        ("pO2_abs",     "Mean pO2 abs error\n[mmHg]",          "#70AD47", True),
    ]
    for ax, (metric, ylabel, clr, try_log) in zip(axes, configs):
        vals = res[metric][:, -1]
        bars = ax.bar(x, vals, color=clr, edgecolor="k", linewidth=0.6, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        if metric == "signed_bias":
            ax.axhline(0, color="k", lw=1.2, ls="--")
        if try_log:
            _auto_log(ax, vals)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel(meta[0], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Final-Iteration Priority Metrics  --  sweep: {sweep_name}",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    return fig


def fig_signed_heatmap(sweep_name, sweep_values, res):
    meta  = SWEEP_META[sweep_name]
    ylbls = [meta[1](v) for v in sweep_values]
    ext   = [1, ITER_MAX, len(sweep_values) - 0.5, -0.5]

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(sweep_values)*0.75+2)))

    sb   = res["signed_bias"]
    vmax = np.abs(sb).max() + 1e-9
    im0  = axes[0].imshow(sb, aspect="auto", cmap="RdBu",
                          norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                          extent=ext)
    axes[0].set_title("Signed CMRO2 bias  (+ = under, - = over)", fontsize=10)
    plt.colorbar(im0, ax=axes[0], label="[umol/cm3/min]")

    pO2  = res["pO2_abs"]
    im1  = axes[1].imshow(pO2, aspect="auto", cmap="RdYlGn_r",
                          norm=LogNorm(vmin=pO2.min()+1e-6, vmax=pO2.max()),
                          extent=ext)
    axes[1].set_title("pO2 abs error [mmHg]  (log scale)", fontsize=10)
    plt.colorbar(im1, ax=axes[1], label="[mmHg]")

    for ax in axes:
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel(meta[0], fontsize=9)
        ax.set_yticks(range(len(sweep_values)))
        ax.set_yticklabels(ylbls, fontsize=8)

    fig.suptitle(f"Signed Bias & pO2 Error Heatmaps  --  sweep: {sweep_name}",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    return fig


def fig_uncertainty_collapse(sweep_name, sweep_values, res):
    meta   = SWEEP_META[sweep_name]
    colors = _colors(len(sweep_values), "coolwarm")
    iters  = np.arange(1, ITER_MAX + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, val in enumerate(sweep_values):
        lbl = meta[1](val)
        axes[0].plot(iters, res["cmro2_std"][idx], color=colors[idx], lw=2, label=lbl)
        axes[1].plot(iters, res["spread"][idx],    color=colors[idx], lw=2, label=lbl)
        lo = np.minimum(res["cmro2_std"][idx], res["spread"][idx])
        hi = np.maximum(res["cmro2_std"][idx], res["spread"][idx])
        axes[0].fill_between(iters, lo, hi, alpha=0.08, color=colors[idx])

    for ax, title, yl in zip(
        axes,
        ["Posterior sigma (covariance)", "Ensemble spread (particles)"],
        ["sigma [umol/cm3/min]"] * 2,
    ):
        _auto_log(ax, res["cmro2_std"])
        ax.set_xlabel("Iteration"); ax.set_ylabel(yl, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.25)

    fig.suptitle(f"Uncertainty Collapse -- sweep: {sweep_name}",
                 fontweight="bold", fontsize=10)
    plt.tight_layout()
    return fig


def fig_pO2_both_errors(sweep_name, sweep_values, res):
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

    axes[0].set_ylabel("Mean |true - est| [mmHg]", fontsize=9)
    axes[1].set_ylabel("Mean relative error [%]", fontsize=9)
    for ax, title in zip(axes, ["pO2 Absolute Error", "pO2 Relative Error"]):
        _auto_log(ax, res["pO2_abs"])
        ax.set_xlabel("Iteration"); ax.set_title(title)
        ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.25)

    fig.suptitle(f"pO2 Reconstruction Error -- sweep: {sweep_name}",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    return fig


# =============================================================================
# 8.  PLOTTING -- 2D CROSS-SWEEPS
# =============================================================================

def fig_cross_heatmaps(pA, valsA, pB, valsB,
                       signed_g, bias_g, std_g, pO2_g, title):
    metaA = SWEEP_META[pA]; metaB = SWEEP_META[pB]
    xlbls = [metaB[1](v) for v in valsB]
    ylbls = [metaA[1](v) for v in valsA]

    fig, axes = plt.subplots(1, 4, figsize=(22, max(4, len(valsA)*0.9+2)))

    vmax_sb = np.abs(signed_g).max() + 1e-9
    panels = [
        (signed_g, "Signed bias\n[umol/cm3/min]",    "RdBu",
         TwoSlopeNorm(vmin=-vmax_sb, vcenter=0, vmax=vmax_sb)),
        (bias_g,   "|CMRO2 bias|\n[umol/cm3/min]",   "RdYlGn_r",
         LogNorm(vmin=bias_g.min()+1e-9, vmax=bias_g.max())),
        (std_g,    "Posterior sigma\n[umol/cm3/min]", "YlOrRd",  None),
        (pO2_g,    "pO2 abs error\n[mmHg]",           "RdYlGn_r", None),
    ]
    for ax, (data, cblabel, cmap, norm) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto", origin="upper")
        ax.set_xticks(range(len(valsB))); ax.set_yticks(range(len(valsA)))
        ax.set_xticklabels(xlbls, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(ylbls, fontsize=8)
        ax.set_xlabel(metaB[0], fontsize=9); ax.set_ylabel(metaA[0], fontsize=9)
        for r in range(len(valsA)):
            for c in range(len(valsB)):
                v   = data[r, c]
                fmt = f"{v:+.2f}" if cmap == "RdBu" else f"{v:.2f}"
                tc  = "white" if abs(v) > abs(data).max()*0.6 else "black"
                ax.text(c, r, fmt, ha="center", va="center",
                        fontsize=7, color=tc, fontweight="bold")
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.08)
        plt.colorbar(im, cax=cax, label=cblabel)
        ax.set_title(cblabel.replace("\n", " "), fontsize=9)

    fig.suptitle(f"2D Cross-sweep  --  {title}\n(final iteration)",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    return fig


def fig_cross_scatter(pA, valsA, pB, valsB, signed_g, std_g, pO2_g, title):
    metaA  = SWEEP_META[pA]; metaB = SWEEP_META[pB]
    colors = _colors(len(valsA), "tab10")

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, va in enumerate(valsA):
        ax.scatter(
            signed_g[i, :], pO2_g[i, :],
            s=np.clip(std_g[i, :] * 800, 20, 600),
            c=[colors[i]] * len(valsB),
            label=f"{pA}={metaA[1](va)}",
            alpha=0.78, edgecolors="k", linewidths=0.5,
        )
    for j, vb in enumerate(valsB):
        ax.annotate(f"{pB}={metaB[1](vb)}", (signed_g[0, j], pO2_g[0, j]),
                    fontsize=7, textcoords="offset points", xytext=(5, 4))
    ax.axvline(0, color="k", lw=1.2, ls="--", label="zero bias")
    ax.set_xlabel("Signed CMRO2 bias (true-est) [umol/cm3/min]", fontsize=10)
    ax.set_ylabel("Mean pO2 abs error [mmHg]", fontsize=10)
    ax.set_title(f"Signed bias vs pO2 error  --  {title}\n"
                 "Bubble size proportional to posterior sigma", fontsize=10)
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig


# =============================================================================
# 9.  PLOTTING -- BOOTSTRAP
# =============================================================================

def fig_bootstrap_panel(sweep_name, sweep_values, mean_res, std_res):
    meta   = SWEEP_META[sweep_name]
    colors = _colors(len(sweep_values))
    iters  = np.arange(1, ITER_MAX_BOOT + 1)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for idx, val in enumerate(sweep_values):
        c = colors[idx]; lbl = meta[1](val)
        for ax, metric, ylabel in zip(
            axes,
            ["signed_bias", "pO2_abs"],
            ["Signed CMRO2 bias [umol/cm3/min]",
             "Mean pO2 abs error [mmHg]"],
        ):
            mu  = mean_res[metric][idx]
            sig = std_res[metric][idx]
            ax.plot(iters, mu, color=c, lw=2, label=lbl)
            ax.fill_between(iters, mu - sig, mu + sig, color=c, alpha=0.15)

    axes[0].axhline(0, color="k", lw=1.2, ls="--")
    for ax, ylabel in zip(axes, ["Signed CMRO2 bias [umol/cm3/min]",
                                  "Mean pO2 abs error [mmHg]"]):
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Iteration", fontsize=10)

    fig.suptitle(
        f"Bootstrap (B={N_BOOTSTRAP}, N={N_ENS_BOOT}, T={ITER_MAX_BOOT})\n"
        f"Mean +/- std across runs  --  sweep: {sweep_name}  ({meta[0]})",
        fontweight="bold", fontsize=11)
    plt.tight_layout()
    return fig


# =============================================================================
# 10. SUMMARY FIGURES
# =============================================================================

def fig_summary_signed_bias(all_oat):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for idx, (sname, svals) in enumerate(SWEEPS.items()):
        ax     = axes[idx]
        res    = all_oat[sname]
        meta   = SWEEP_META[sname]
        x      = np.arange(len(svals))
        finals = res["signed_bias"][:, -1]
        std_f  = res["cmro2_std"][:, -1]
        clrs   = ["#D85A30" if v > 0 else "#378ADD" for v in finals]
        ax.bar(x, finals, color=clrs, yerr=std_f, capsize=4,
               edgecolor="k", alpha=0.82)
        ax.axhline(0, color="k", lw=1.2, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([meta[1](v) for v in svals],
                           rotation=28, ha="right", fontsize=7)
        ax.set_xlabel(meta[0], fontsize=8)
        ax.set_ylabel("Signed bias [umol/cm3/min]", fontsize=8)
        ax.set_title(sname, fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
    for idx in range(len(SWEEPS), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(
        "Signed CMRO2 Bias at Final Iteration  --  All OAT Sweeps\n"
        "Orange = underestimate  |  Blue = overestimate  |  Error bar = posterior sigma",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def fig_summary_priority(all_oat):
    snames  = list(SWEEPS.keys())
    metrics = [("signed_bias","Signed bias [umol/cm3/min]"),
               ("cmro2_std",  "Posterior sigma [umol/cm3/min]"),
               ("pO2_abs",    "pO2 abs error [mmHg]")]
    clrs = ["#2E75B6","#ED7D31","#70AD47"]

    fig, axes = plt.subplots(3, len(snames), figsize=(3.2*len(snames), 10),
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
            if metric == "signed_bias":
                ax.axhline(0, color="k", lw=1, ls="--")
            ax.set_xticks(x)
            ax.set_xticklabels([meta[1](v) for v in svals],
                               rotation=35, ha="right", fontsize=6)
            if col == 0: ax.set_ylabel(row_label, fontsize=8)
            if row == 0: ax.set_title(sname, fontsize=8, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Priority Metrics at Final Iteration -- All OAT Sweeps",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def fig_summary_cross(cross_results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for idx, (pA, valsA, pB, valsB, title,
               signed_g, bias_g, std_g, pO2_g, _) in enumerate(cross_results):
        ax    = axes[idx]
        metaA = SWEEP_META[pA]; metaB = SWEEP_META[pB]
        vmax  = np.abs(signed_g).max() + 1e-9
        im    = ax.imshow(signed_g, cmap="RdBu",
                          norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                          aspect="auto", origin="upper")
        ax.set_xticks(range(len(valsB))); ax.set_yticks(range(len(valsA)))
        ax.set_xticklabels([metaB[1](v) for v in valsB],
                           rotation=30, ha="right", fontsize=7)
        ax.set_yticklabels([metaA[1](v) for v in valsA], fontsize=7)
        ax.set_xlabel(metaB[0], fontsize=8); ax.set_ylabel(metaA[0], fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, label="signed bias [umol/cm3/min]")
    fig.suptitle("2D Cross-Sweep Summary  --  Signed CMRO2 Bias (final iteration)\n"
                 "Blue = overestimate  |  Red = underestimate  |  White = zero",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# =============================================================================
# 11. DATA MANAGEMENT
# =============================================================================

def save_metadata_json(metas, sweep_name, sweep_values, sub):
    records = []
    for i, val in enumerate(sweep_values):
        m = metas[i]
        if m is None:
            continue
        rec = dict(
            sweep_param  = sweep_name,
            sweep_value  = float(val),
            ground_truth = {k: float(v) for k, v in m["ground_truth"].items()},
            prior        = {k: float(v) for k, v in m["prior"].items()},
            obs_std      = float(m["obs_std"]),
            n_ensembles  = int(m["n_ensembles"]),
            iter_max     = int(m["iter_max"]),
            validity_true= {k: bool(v) for k, v in m["validity_true"].items()},
            validity_est = {k: bool(v) for k, v in m["validity_est"].items()},
            any_flag     = bool(m["any_validity_flag"]),
        )
        records.append(rec)
    with open(os.path.join(sub, "metadata.json"), "w") as f:
        json.dump(records, f, indent=2)


def save_csv(all_oat, cross_results, path):
    rows = []
    for sname, svals in SWEEPS.items():
        res  = all_oat[sname]
        meta = SWEEP_META[sname]
        for i, v in enumerate(svals):
            rows.append(dict(
                section="OAT", sweep_param=sname,
                sweep_value=float(v), sweep_label=meta[1](v),
                param_B="", value_B="",
                signed_bias=res["signed_bias"][i,-1],
                cmro2_bias=res["bias"][i,-1],
                cmro2_rel_bias=res["rel_bias"][i,-1],
                cmro2_mean=res["cmro2_mean"][i,-1],
                cmro2_std=res["cmro2_std"][i,-1],
                R0_mean=res["R0_mean"][i,-1],
                pves_mean=res["pves_mean"][i,-1],
                pO2_abs_err=res["pO2_abs"][i,-1],
                pO2_rel_err=res["pO2_rel"][i,-1],
                spread=res["spread"][i,-1],
            ))
    for (pA, valsA, pB, valsB, title,
         signed_g, bias_g, std_g, pO2_g, _) in cross_results:
        metaA = SWEEP_META[pA]; metaB = SWEEP_META[pB]
        for i, va in enumerate(valsA):
            for j, vb in enumerate(valsB):
                rows.append(dict(
                    section="CROSS", sweep_param=pA,
                    sweep_value=float(va), sweep_label=metaA[1](va),
                    param_B=pB, value_B=float(vb),
                    signed_bias=signed_g[i,j], cmro2_bias=bias_g[i,j],
                    cmro2_rel_bias=np.nan, cmro2_mean=np.nan,
                    cmro2_std=std_g[i,j], R0_mean=np.nan, pves_mean=np.nan,
                    pO2_abs_err=pO2_g[i,j], pO2_rel_err=np.nan, spread=np.nan,
                ))
    df = (pd.DataFrame(rows)
          .sort_values(["section","sweep_param","sweep_value"])
          .reset_index(drop=True))
    df.to_csv(path, index=False)
    print(f"\nCSV saved ({len(df)} rows, sorted) -> {path}")
    return df


# =============================================================================
# 11b. AUGMENTATION COMPARISON PLOT
# =============================================================================

def fig_augmentation_comparison(sweep_name, sweep_values,
                                 mean_aug, std_aug, all_aug,
                                 mean_noaug, std_noaug, all_noaug):
    """
    Three-panel figure comparing augmented vs. non-augmented EnKF bootstrap runs.

    Panel 1 – Signed CMRO2 bias (true-est): shows whether the R0 soft constraint
              pulls the filter towards the correct value.
    Panel 2 – Posterior sigma: whether augmentation changes uncertainty collapse.
    Panel 3 – Calibration ratio = mean(posterior_sigma) / std(cmro2_mean across B runs):
              ratio ~ 1 → well-calibrated; < 1 → overconfident; > 1 → underconfident.
              Both curves should be compared: augmentation should keep ratio closer to 1.
    """
    meta   = SWEEP_META[sweep_name]
    colors = _colors(len(sweep_values))
    iters  = np.arange(1, ITER_MAX_BOOT + 1)

    # Calibration ratio per sweep value, per iteration
    # MC spread = std of cmro2_mean estimates across the B bootstrap runs
    mc_spread_aug   = all_aug["cmro2_mean"].std(axis=1)    # (n_vals, ITER_MAX_BOOT)
    mc_spread_noaug = all_noaug["cmro2_mean"].std(axis=1)
    cal_aug   = mean_aug["cmro2_std"]   / (mc_spread_aug   + 1e-9)
    cal_noaug = mean_noaug["cmro2_std"] / (mc_spread_noaug + 1e-9)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for idx, val in enumerate(sweep_values):
        c   = colors[idx]
        lbl = meta[1](val)

        # --- signed bias ---
        mu_a  = mean_aug["signed_bias"][idx];    sig_a = std_aug["signed_bias"][idx]
        mu_n  = mean_noaug["signed_bias"][idx];  sig_n = std_noaug["signed_bias"][idx]
        axes[0].plot(iters, mu_a, color=c, lw=2, ls="-",  label=f"aug  {lbl}")
        axes[0].plot(iters, mu_n, color=c, lw=2, ls="--", label=f"no-aug {lbl}")
        axes[0].fill_between(iters, mu_a - sig_a, mu_a + sig_a, color=c, alpha=0.10)
        axes[0].fill_between(iters, mu_n - sig_n, mu_n + sig_n, color=c, alpha=0.06)

        # --- posterior sigma ---
        axes[1].plot(iters, mean_aug["cmro2_std"][idx],   color=c, lw=2, ls="-")
        axes[1].plot(iters, mean_noaug["cmro2_std"][idx], color=c, lw=2, ls="--")

        # --- calibration ratio ---
        axes[2].plot(iters, cal_aug[idx],   color=c, lw=2, ls="-")
        axes[2].plot(iters, cal_noaug[idx], color=c, lw=2, ls="--")

    axes[0].axhline(0, color="k", lw=1.2, ls=":", alpha=0.6)
    axes[2].axhline(1, color="k", lw=1.2, ls=":", alpha=0.6, label="ideal (ratio=1)")

    ylabels = [
        "Signed bias (true-est)\n[umol/cm3/min]",
        "Posterior sigma [umol/cm3/min]",
        "Calibration ratio\n(post_sigma / MC_spread)",
    ]
    for ax, yl in zip(axes, ylabels):
        ax.set_ylabel(yl, fontsize=9)
        ax.grid(True, alpha=0.25)

    # Single combined legend on the first panel only
    axes[0].legend(fontsize=6, ncol=4, loc="upper right")
    axes[2].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Iteration", fontsize=10)

    # Compact legend for line style meaning
    from matplotlib.lines import Line2D
    style_legend = [
        Line2D([0], [0], color="k", lw=2, ls="-",  label="with R0 augmentation"),
        Line2D([0], [0], color="k", lw=2, ls="--", label="without augmentation"),
    ]
    axes[1].legend(handles=style_legend, fontsize=8, loc="upper right")

    fig.suptitle(
        f"Augmentation comparison  (B={N_BOOTSTRAP}, N={N_ENS_BOOT}, T={ITER_MAX_BOOT})\n"
        f"Sweep: {sweep_name}  —  solid = R0-augmented  |  dashed = standard EnKF",
        fontweight="bold", fontsize=11,
    )
    plt.tight_layout()
    return fig


# =============================================================================
# 12. MAIN
# =============================================================================

def main():
    np.random.seed(MASTER_SEED)
    t0 = time.time()

    # A. OAT sweeps
    all_oat = {}
    for sweep_name, sweep_values in SWEEPS.items():
        print(f"\n{'='*64}")
        print(f"  OAT  {sweep_name}  ({len(sweep_values)} values)")
        print(f"{'='*64}")

        res, metas = run_oat_sweep(sweep_name, sweep_values)
        all_oat[sweep_name] = res

        sub = os.path.join(SAVE_PATH, "oat", sweep_name)
        os.makedirs(sub, exist_ok=True)

        for k, arr in res.items():
            np.save(os.path.join(sub, f"{k}.npy"), arr)
        np.save(os.path.join(sub, "sweep_values.npy"), sweep_values)
        save_metadata_json(metas, sweep_name, sweep_values, sub)

        for fname, figfn in [
            ("priority_metrics.png",   lambda: fig_priority_panel(sweep_name, sweep_values, res)),
            ("final_trio_bars.png",    lambda: fig_final_trio(sweep_name, sweep_values, res)),
            ("signed_heatmap.png",     lambda: fig_signed_heatmap(sweep_name, sweep_values, res)),
            ("uncertainty_collapse.png",lambda: fig_uncertainty_collapse(sweep_name, sweep_values, res)),
            ("pO2_errors.png",         lambda: fig_pO2_both_errors(sweep_name, sweep_values, res)),
        ]:
            fig = figfn()
            fig.savefig(os.path.join(sub, fname), dpi=200, bbox_inches="tight")
            plt.close(fig)

        print(f"  -> 5 figs + arrays + metadata.json -> {sub}")

    # B. 2D cross-sweeps
    cross_results = []
    for (pA, valsA, pB, valsB, title) in CROSS_SWEEPS:
        pair_tag = f"{pA}_x_{pB}"
        print(f"\n{'='*64}")
        print(f"  CROSS  {pair_tag}  ({len(valsA)}x{len(valsB)} grid)")
        print(f"{'='*64}")

        signed_g, bias_g, std_g, pO2_g, meta_grid = \
            run_cross_sweep(pA, valsA, pB, valsB)
        cross_results.append(
            (pA, valsA, pB, valsB, title, signed_g, bias_g, std_g, pO2_g, meta_grid))

        sub = os.path.join(SAVE_PATH, "cross", pair_tag)
        os.makedirs(sub, exist_ok=True)

        np.save(os.path.join(sub, "signed_bias_grid.npy"), signed_g)
        np.save(os.path.join(sub, "bias_grid.npy"),        bias_g)
        np.save(os.path.join(sub, "std_grid.npy"),         std_g)
        np.save(os.path.join(sub, "pO2_grid.npy"),         pO2_g)
        np.save(os.path.join(sub, "vals_A.npy"),           valsA)
        np.save(os.path.join(sub, "vals_B.npy"),           valsB)

        flat_metas = [meta_grid[i][j]
                      for i in range(len(valsA)) for j in range(len(valsB))]
        flat_vals  = [float(valsA[i])
                      for i in range(len(valsA)) for j in range(len(valsB))]
        save_metadata_json(flat_metas, pair_tag, flat_vals, sub)

        fig = fig_cross_heatmaps(pA, valsA, pB, valsB,
                                 signed_g, bias_g, std_g, pO2_g, title)
        fig.savefig(os.path.join(sub, "heatmaps_4panel.png"),
                    dpi=200, bbox_inches="tight"); plt.close(fig)

        fig = fig_cross_scatter(pA, valsA, pB, valsB,
                                signed_g, std_g, pO2_g, title)
        fig.savefig(os.path.join(sub, "scatter_signed_pO2.png"),
                    dpi=200, bbox_inches="tight"); plt.close(fig)

        print(f"  -> 2 figs + 6 arrays + metadata.json -> {sub}")

    # C. Summary figures
    print("\n--- Summary figures ---")
    for fname, figfn in [
        ("SUMMARY_signed_bias_OAT.png",  lambda: fig_summary_signed_bias(all_oat)),
        ("SUMMARY_priority_metrics.png", lambda: fig_summary_priority(all_oat)),
        ("SUMMARY_cross_sweeps.png",     lambda: fig_summary_cross(cross_results)),
    ]:
        fig = figfn()
        fig.savefig(os.path.join(SAVE_PATH, fname), dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> {fname}")

    # D. CSV
    df = save_csv(all_oat, cross_results,
                  os.path.join(SAVE_PATH, "sensitivity_summary.csv"))

    # E. Bootstrap (optional, additive -- does NOT replace A/B)
    if RUN_BOOTSTRAP:
        print("\n\n" + "="*64)
        print("  BOOTSTRAP  (B={}, N={}, T={})".format(
            N_BOOTSTRAP, N_ENS_BOOT, ITER_MAX_BOOT))
        print("="*64)
        boot_dir = os.path.join(SAVE_PATH, "bootstrap")

        for sweep_name, sweep_values in SWEEPS.items():
            print(f"\n  Bootstrap sweep: {sweep_name}")
            mean_res, std_res, all_r = run_bootstrap_sweep(
                sweep_name, sweep_values)

            sub = os.path.join(boot_dir, sweep_name)
            os.makedirs(sub, exist_ok=True)

            for k in mean_res:
                np.save(os.path.join(sub, f"mean_{k}.npy"),  mean_res[k])
                np.save(os.path.join(sub, f"std_{k}.npy"),   std_res[k])
                np.save(os.path.join(sub, f"all_{k}.npy"),   all_r[k])
            np.save(os.path.join(sub, "sweep_values.npy"), sweep_values)

            fig = fig_bootstrap_panel(sweep_name, sweep_values, mean_res, std_res)
            fig.savefig(os.path.join(sub, "bootstrap_panel.png"),
                        dpi=200, bbox_inches="tight"); plt.close(fig)
            print(f"    -> bootstrap_panel.png + arrays -> {sub}")

    # F. Augmentation comparison (cmro2_true sweep: aug vs standard EnKF)
    if RUN_BOOTSTRAP:
        print("\n\n" + "="*64)
        print("  AUGMENTATION COMPARISON  (cmro2_true sweep, blind-spot baseline)")
        print("="*64)
        aug_dir   = os.path.join(SAVE_PATH, "augmentation_comparison")
        os.makedirs(aug_dir, exist_ok=True)

        comp_sweep = "cmro2_true"
        comp_vals  = SWEEPS[comp_sweep]

        print("\n  Running with R0 augmentation:")
        m_aug, s_aug, a_aug = run_bootstrap_sweep(
            comp_sweep, comp_vals, use_augmentation=True)

        print("\n  Running WITHOUT augmentation (standard EnKF baseline):")
        m_noaug, s_noaug, a_noaug = run_bootstrap_sweep(
            comp_sweep, comp_vals, use_augmentation=False)

        for prefix, mr, sr, ar in [
            ("aug",   m_aug,   s_aug,   a_aug),
            ("noaug", m_noaug, s_noaug, a_noaug),
        ]:
            sub = os.path.join(aug_dir, prefix)
            os.makedirs(sub, exist_ok=True)
            for k in mr:
                np.save(os.path.join(sub, f"mean_{k}.npy"), mr[k])
                np.save(os.path.join(sub, f"std_{k}.npy"),  sr[k])
                np.save(os.path.join(sub, f"all_{k}.npy"),  ar[k])
            np.save(os.path.join(sub, "sweep_values.npy"), comp_vals)

        fig = fig_augmentation_comparison(
            comp_sweep, comp_vals,
            m_aug, s_aug, a_aug,
            m_noaug, s_noaug, a_noaug,
        )
        fig.savefig(os.path.join(aug_dir, "augmentation_comparison.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> augmentation_comparison.png + arrays -> {aug_dir}")

    elapsed = time.time() - t0
    print(f"\nAll done in {elapsed/60:.1f} min -> {SAVE_PATH}")

    print("\n-- OAT highlights (final iteration) --")
    oat_df = df[df.section == "OAT"][
        ["sweep_param","sweep_label","signed_bias","cmro2_bias","cmro2_std","pO2_abs_err"]
    ].rename(columns={"cmro2_bias":"abs_bias","cmro2_std":"post_s","pO2_abs_err":"pO2"})
    print(oat_df.to_string(index=False))


if __name__ == "__main__":
    main()