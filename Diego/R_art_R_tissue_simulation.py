import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
from scipy.signal import find_peaks
import matplotlib as mpl

# ----------------------------
# Constants and Parameters
# ----------------------------
D = 4e-5             # Diffusion coefficient in cm²/s
ALPHA = 1.39 / 1000  # Solubility coefficient in mLO2/mmHg/cm³
SIGMA_SMOOTH = 6     # Std dev for Gaussian smoothing
NOISE_STD = 2        # Std dev of additive Gaussian noise (mmHg)
SPACING = 1          # Microns per pixel — must be 1
assert SPACING == 1  # This analysis assumes 1 µm per pixel!! Other values will require code modifications.

# Reproducible noise
NOISE_SEED = None  # set to an int for reproducible runs; set to None for fresh randomness each run
rng = np.random.default_rng(NOISE_SEED) if NOISE_SEED is not None else np.random.default_rng()

# Radii grids used by the estimators
r_values_1 = np.arange(1, 27 + 2 + 2, 1)       # Radii for R_art estimation
r_values_2 = np.arange(1 + 64, 125 + 5, 1)     # Radii for R_tissue estimation

# Detection parameters
SMOOTH_CORRECTION   = 6 - 1
DERIV_THRESH        = 1.1e-2         # first-derivative threshold for R_tissue candidates
CLIP_THRESH_SINGLE  = 27                       # %; per-radius clipping threshold (full circle)
CLIP_THRESH_MULTI   = 27                       # %; per-radius clipping threshold (half circle)
R_ART_MIN_PEAK      = 0.33            # min required height of peak in -dpO2/dr
R_TISSUE_D2_THRESH  = 1.7e-2                   # |second derivative| must be <= this
TOLERANCE_FRAC = 0.1                            # 10% radial tolerance for half-sector containment
R_ART_CONTAIN_TOL_FRAC = 0.1                   # 10% radial tolerance for R_art containment
SHOW_DETECTION_PLOTS = True                   # Toggle diagnostic plots for R_art and R_tissue detection

# ----------------------------
# Krogh–Erlang 2D Model Function
# ----------------------------
def krogh_2d(x, y, source_params):
    x0, y0, CMRO2, Rt_um, pO2_art, R_art_um = source_params
    CMRO2_sec = CMRO2 / 60.0  # Convert from µmol/min to µmol/s
    r_um = np.sqrt((x - x0)**2 + (y - y0)**2)
    r_cm = r_um * 1e-4        # Convert to cm
    Rt_cm = Rt_um * 1e-4
    R_art_cm = R_art_um * 1e-4

    pO2 = np.full_like(r_cm, np.nan)
    inside = r_cm <= R_art_cm
    outside = r_cm > R_art_cm

    pO2[inside] = pO2_art
    pO2[outside] = pO2_art + (CMRO2_sec / (D * ALPHA)) * (
        (r_cm[outside]**2 - R_art_cm**2) / 4 -
        (Rt_cm**2 / 2) * np.log(r_cm[outside] / R_art_cm)
    )

    boundary_value = pO2_art + (CMRO2_sec / (D * ALPHA)) * (
        (Rt_cm**2 - R_art_cm**2) / 4 -
        (Rt_cm**2 / 2) * np.log(Rt_cm / R_art_cm)
    )
    pO2[r_um > Rt_um] = boundary_value
    return pO2, boundary_value

# ----------------------------
# Profiles
# ----------------------------
def circle_profile(P, X, Y, xc, yc, r_values):
    """Mean pO2 over full circles around (xc, yc). Uses dense, r-dependent sampling."""
    radii = np.array(r_values)
    means = np.full_like(radii, np.nan, dtype=float)
    for i, r in enumerate(radii):
        theta = np.linspace(0., 2*np.pi, 75*int(np.ceil(2*np.pi*r)), endpoint=False)
        x_circle = xc + r * np.cos(theta)
        y_circle = yc + r * np.sin(theta)
        x_idx = np.clip(np.round(x_circle).astype(int), 0, X.shape[1] - 1)
        y_idx = np.clip(np.round(y_circle).astype(int), 0, Y.shape[0] - 1)
        means[i] = np.nanmean(P[y_idx, x_idx])
    return radii, means

def half_circle_profile(P, X, Y, xc, yc, xc2, yc2, r_values):
    """
    Mean pO2 over half-circles centered at (xc, yc) and facing away from (xc2, yc2).
    Returns radii, means, per_clip (% OOB per radius).
    """
    dx, dy = xc - xc2, yc - yc2
    theta0 = np.arctan2(dy, dx)

    radii = np.array(r_values)
    means = np.full_like(radii, np.nan, dtype=float)
    per_clip = np.zeros_like(radii, dtype=float)

    max_y, max_x = P.shape[0] - 1, P.shape[1] - 1

    for i, r in enumerate(radii):
        theta = np.linspace(0., 2*np.pi, 75*int(np.ceil(2*np.pi*r)), endpoint=False)
        x_circle = xc + r * np.cos(theta)
        y_circle = yc + r * np.sin(theta)

        # half-circles away from (xc, yc)
        angles = np.arctan2(y_circle - yc, x_circle - xc)
        mask = np.cos(angles - theta0) > 0

        x_sel = x_circle[mask]
        y_sel = y_circle[mask]

        x_idx_raw = np.round(x_sel).astype(int)
        y_idx_raw = np.round(y_sel).astype(int)

        oob_mask = (x_idx_raw < 0) | (x_idx_raw > max_x) | (y_idx_raw < 0) | (y_idx_raw > max_y)
        oob_here = int(np.count_nonzero(oob_mask))
        per_clip[i] = 100.0 * oob_here / max(1, x_sel.size)

        x_idx = np.clip(x_idx_raw, 0, max_x)
        y_idx = np.clip(y_idx_raw, 0, max_y)
        means[i] = np.nanmean(P[y_idx, x_idx])

    return radii, means, per_clip

def full_circle_profile_for_tissue(P, X, Y, xc, yc, r_values):
    """
    Full-circle analogue for single-center R_tissue.
    Returns radii, means, per_clip (% OOB per radius).
    """
    radii = np.array(r_values)
    means = np.full_like(radii, np.nan, dtype=float)
    per_clip = np.zeros_like(radii, dtype=float)

    max_y, max_x = P.shape[0] - 1, P.shape[1] - 1

    for i, r in enumerate(radii):
        theta = np.linspace(0., 2*np.pi, 25*int(np.ceil(2*np.pi*r)), endpoint=False)
        x_circle = xc + r * np.cos(theta)
        y_circle = yc + r * np.sin(theta)

        x_idx_raw = np.round(x_circle).astype(int)
        y_idx_raw = np.round(y_circle).astype(int)

        oob_mask = (x_idx_raw < 0) | (x_idx_raw > max_x) | (y_idx_raw < 0) | (y_idx_raw > max_y)
        oob_here = int(np.count_nonzero(oob_mask))
        per_clip[i] = 100.0 * oob_here / max(1, theta.size)

        x_idx = np.clip(x_idx_raw, 0, max_x)
        y_idx = np.clip(y_idx_raw, 0, max_y)
        means[i] = np.nanmean(P[y_idx, x_idx])

    return radii, means, per_clip

# ----------------------------
# Half-sector occupancy (only considers allowed indices)
# ----------------------------
def half_sector_contains_other_centers(coords, i, r_um, theta0, allowed_indices, tol_frac=TOLERANCE_FRAC, eps=0.0):
    """
    Check if any allowed center lies within the away-facing half-sector of center i
    at radius r_um expanded by a fractional tolerance (tol_frac).
    """
    xi, yi = coords[i]
    r_eff = float(r_um) * (1.0 + float(tol_frac))  # apply tolerance
    for k in allowed_indices:
        if k == i:
            continue
        xk, yk = coords[k]
        dx, dy = xk - xi, yk - yi
        rk = float(np.hypot(dx, dy))
        if rk <= r_eff + eps:
            ang = np.arctan2(dy, dx)
            # away-facing half-sector test
            if np.cos(ang - theta0) > 0:
                return True, k
    return False, None

def center_tag(i, coords):
    x, y = coords[i]
    return f"Center {i+1} ({x:.0f}, {y:.0f})"

# ----------------------------
# Simulation setup (multiple sources supported)
# ----------------------------
grid_size = 400
x = np.arange(0, grid_size, SPACING)
y = np.arange(0, grid_size, SPACING)
X, Y = np.meshgrid(x, y)

# Define any number of oxygen sources:
# (x0, y0, CMRO2 [µmol/min], Rt_um, pO2_art [mmHg], R_art_um)

# sources = [
#     (160, 150, 1.5, 107.0, 70, 6)
# ]

# sources = [
#     (160, 150, 1.5, 106.0, 70, 6),
#     (245, 238, 2.1,  80.0, 50, 10),
# ]

sources = [
    (160, 150, 1.5, 106.0, 70, 6),
    (245, 238, 2.1,  80.0, 50, 10),
    (115, 260, 2.0,  90.0, 60, 8)
]

# noisy map
maps = []
boundaries = []
for sp in sources:
    m, b = krogh_2d(X, Y, sp)
    m = m + rng.normal(0, NOISE_STD, m.shape)
    maps.append(m); boundaries.append(b)
Z_total = np.sum(maps, axis=0)
baseline = np.min(boundaries)
Z_corrected = Z_total - baseline
Z_smooth = gaussian_filter(Z_corrected.copy(), sigma=SIGMA_SMOOTH)

# ----------------------------
# Peak Detection
# ----------------------------
min_distance_px = int(10 / SPACING)
threshold_abs = (np.max(Z_smooth) + np.min(Z_smooth)) / (2)
coordinates = peak_local_max(
    Z_smooth,
    min_distance=min_distance_px,
    threshold_abs=threshold_abs,
    exclude_border=5
)
peak_coords = [(X[row, col], Y[row, col]) for row, col in coordinates]

# ----------------------------
# Analysis (R_art and R_tissue) + containment filter + prints
# ----------------------------
def run_analysis(P, X, Y, peak_coords):
    """
    Pass 1: R_art detection with prints (+ optional plots) + suppression on weak minima.
    Containment filter: discard centers whose R_art*(1+tol) contains another retained center.
    Pass 2: R_tissue detection among retained centers with 1st-deriv candidates,
            per-radius clipping gate, then 2nd-deriv test; half-sector rule in multi-center case.
    Returns:
      R_art_v:    list of [i, R_art_radius_um or None]
      R_tissue_v: list of [(i, j or None), index_in_r_values_2 or None]
    """
    n_all = len(peak_coords)

    R_art_v = []
    suppressed = np.zeros(n_all, dtype=bool)

    # ---------- PASS 1: R_art ----------
    for i, (xc, yc) in enumerate(peak_coords):
        tag_i = center_tag(i, peak_coords)

        radii1, means1 = circle_profile(P, X, Y, xc, yc, r_values_1)
        d1 = np.diff(means1)
        inv = -d1
        peaks_all, _ = find_peaks(inv, prominence=0.002)

        chosen_peak_idx = None
        corrected_idx = None
        r_art_radius = None
        suppress_reason = None

        if len(peaks_all) == 0:
            dmin, dmax = float(np.min(d1)), float(np.max(d1))
            trend = "monotonic increasing" if dmin > 0 else ("monotonic decreasing" if dmax < 0 else "no distinct minimum")
            suppress_reason = f"no minimum of dpO2/dr (range [{dmin:.3e}, {dmax:.3e}], {trend})"
        else:
            heights_all = inv[peaks_all]
            good_mask = heights_all >= R_ART_MIN_PEAK
            if not np.any(good_mask):
                strongest_idx = int(peaks_all[int(np.argmax(heights_all))])
                strongest_r = float(radii1[strongest_idx])
                suppress_reason = (f"strongest minimum height {float(np.max(heights_all)):.3f} "
                                   f"at r≈{strongest_r:.0f} µm < {R_ART_MIN_PEAK:.3f}")
            else:
                chosen_peak_idx = int(peaks_all[np.where(good_mask)[0][0]])
                corrected_idx = chosen_peak_idx - SMOOTH_CORRECTION
                corrected_idx = np.max([1, corrected_idx])
                if corrected_idx <= 0:
                    suppress_reason = (f"minimum at r≈{float(radii1[chosen_peak_idx]):.0f} µm but "
                                       f"correction -{SMOOTH_CORRECTION} → non-positive radius")
                else:
                    r_art_radius = float(radii1[corrected_idx])

        # ---- R_art diagnostics plots (optional) ----
        if SHOW_DETECTION_PLOTS:
            fig, axs = plt.subplots(1, 2, figsize=(14, 5))
            axs[0].plot(radii1, means1, marker='o', markersize=3, label="pO2 Profile")
            if r_art_radius is not None:
                axs[0].axvline(r_art_radius, color='gray', linestyle='--',
                               label=f"R_art ≈ {r_art_radius:.0f} µm (corrected)")
            elif chosen_peak_idx is not None:
                axs[0].axvline(float(radii1[chosen_peak_idx]), color='gray', linestyle='--',
                               label=f"Min(d pO2/dr) at r≈{float(radii1[chosen_peak_idx]):.0f} µm (uncorrected)")
            axs[0].set_title(f"Full Circle pO2 Profile\n{tag_i}")
            axs[0].set_xlabel("Radius (µm)"); axs[0].set_ylabel("pO2 (mmHg)")
            axs[0].grid(); axs[0].legend()

            axs[1].plot(radii1[:-1], d1, marker='x', color='black', label="dpO2/dr")
            axs[1].axhline(-R_ART_MIN_PEAK, color='orange', linestyle='--', label=f"threshold: dpO2/dr ≤ {-R_ART_MIN_PEAK:.2f}")
            if len(peaks_all) > 0:
                ok_mask = inv[peaks_all] >= R_ART_MIN_PEAK
                if np.any(~ok_mask):
                    axs[1].scatter(radii1[peaks_all[~ok_mask]], d1[peaks_all[~ok_mask]], s=50, color='red', label="minima (weak)")
                if np.any(ok_mask):
                    axs[1].scatter(radii1[peaks_all[ok_mask]], d1[peaks_all[ok_mask]], s=50, color='green', label="minima (≥ threshold)")
                if chosen_peak_idx is not None and inv[chosen_peak_idx] >= R_ART_MIN_PEAK:
                    axs[1].scatter([radii1[chosen_peak_idx]], [d1[chosen_peak_idx]],
                                   s=120, marker='*', color='gold', edgecolor='black', zorder=5, label="chosen minimum")
            axs[1].set_title(
                f"dpO2/dr vs r (ACCEPTED R_art ≈ {r_art_radius:.0f} µm)" if r_art_radius is not None
                else f"dpO2/dr vs r (SUPPRESSED: {suppress_reason if suppress_reason is not None else 'unknown reason'})"
            )
            axs[1].set_xlabel("Radius (µm)"); axs[1].set_ylabel("dpO2/dr")
            axs[1].grid(); axs[1].legend()
            plt.tight_layout(); plt.show()

        # Final prints
        if r_art_radius is None:
            print(f"[R_art] {tag_i}: {suppress_reason} → NOT DETECTED. CENTER SUPPRESSED.")
            R_art_v.append([i, None])
            suppressed[i] = True
        else:
            peak_height = float(inv[chosen_peak_idx]) if chosen_peak_idx is not None else float('nan')
            print(f"[R_art] {tag_i}: accepted at r≈{r_art_radius:.0f} µm (peak height {peak_height:.3f} ≥ {R_ART_MIN_PEAK:.3f}).")
            R_art_v.append([i, r_art_radius])

    # ---------- Containment filter (with tolerance, among retained) ----------
    r_art_lookup = {i: r for i, r in R_art_v}
    kept_indices = [i for i in range(n_all) if (not suppressed[i]) and (r_art_lookup.get(i) is not None)]

    if len(kept_indices) >= 2:
        for i in list(kept_indices):
            ri = r_art_lookup.get(i, None)
            if ri is None:
                continue
            xi, yi = peak_coords[i]
            offenders = []
            ri_eff = float(ri) * (1.0 + float(R_ART_CONTAIN_TOL_FRAC))  # apply tolerance

            for j in kept_indices:
                if j == i:
                    continue
                xj, yj = peak_coords[j]
                d = float(np.hypot(xj - xi, yj - yi))
                if d <= ri_eff:
                    offenders.append((j, d))

            if offenders:
                suppressed[i] = True
                for k in range(len(R_art_v)):
                    if R_art_v[k][0] == i:
                        R_art_v[k][1] = None
                        break
                offenders_txt = ", ".join([
                    f"{center_tag(j, peak_coords)} (d={dist:.1f} µm)" for j, dist in offenders
                ])
                print(
                    f"[R_art filter] {center_tag(i, peak_coords)}: "
                    f"R_art={ri:.0f} µm with {R_ART_CONTAIN_TOL_FRAC*100:.0f}% tol "
                    f"(effective {ri_eff:.0f} µm) contains {offenders_txt} → CENTER DISCARDED."
                )
            else:
                print(
                    f"[R_art filter] {center_tag(i, peak_coords)}: "
                    f"no other centers within effective radius R_art*(1+tol)={ri_eff:.0f} µm "
                    f"(R_art={ri:.0f} µm, tol={R_ART_CONTAIN_TOL_FRAC*100:.0f}%) → RETAINED."
                )
    else:
        for i in kept_indices:
            ri = r_art_lookup.get(i, None)
            if ri is not None:
                ri_eff = float(ri) * (1.0 + float(R_ART_CONTAIN_TOL_FRAC))
                print(
                    f"[R_art filter] {center_tag(i, peak_coords)}: only center considered; "
                    f"R_art={ri:.0f} µm, tol={R_ART_CONTAIN_TOL_FRAC*100:.0f}% "
                    f"(effective {ri_eff:.0f} µm) → RETAINED."
                )

    kept_indices = [i for i in range(n_all) if (not suppressed[i]) and (r_art_lookup.get(i) is not None)]
    if len(kept_indices) == 0:
        print("[R_tissue] All centers suppressed (R_art weakness and/or containment) → skipping R_tissue estimation.")
        return R_art_v, []

    # ---------- PASS 2: R_tissue ----------
    R_tissue_v = []
    if len(kept_indices) == 1:
        # Single retained center → full-circle tissue estimation
        i = kept_indices[0]
        xc, yc = peak_coords[i]
        tag_i = center_tag(i, peak_coords)

        radii2, means2, per_clip = full_circle_profile_for_tissue(P, X, Y, xc, yc, r_values_2)
        d = np.diff(means2)      # first derivative
        d2 = np.diff(d)          # second derivative

        try:
            cand = np.where(np.abs(d) <= DERIV_THRESH)[0]
            if len(cand) == 0:
                dmin, dmax = float(np.min(d)), float(np.max(d))
                print(f"[R_tissue] {tag_i}: no radius where |dpO2/dr| <= {DERIV_THRESH:.2e} "
                      f"(range [{dmin:.3e}, {dmax:.3e}]) → NOT DETECTED.")
                R_tissue_idx = None
            else:
                print(f"[R_tissue] {tag_i}: {len(cand)} first-derivative candidates → per-radius clipping & 2nd-derivative tests …")
                R_tissue_idx = None
                rejected_by_clip = 0
                rejected_by_d2 = 0

                for idx in cand:
                    idx = int(idx)

                    # Per-radius clipping gate (single-center)
                    clip_here = float(per_clip[idx])
                    if clip_here > CLIP_THRESH_SINGLE:
                        rejected_by_clip += 1
                        print(f"[R_tissue] {tag_i}: candidate r≈{r_values_2[idx]:.0f} µm "
                              f"rejected by per-radius clipping ({clip_here:.1f}% > {CLIP_THRESH_SINGLE:.1f}%).")
                        continue

                    # Second-derivative test
                    idx2 = int(np.clip(idx - 1, 0, len(d2) - 1))
                    sec_val = float(d2[idx2]) if len(d2) > 0 else 0.0
                    if abs(sec_val) > R_TISSUE_D2_THRESH:
                        rejected_by_d2 += 1
                        print(f"[R_tissue] {tag_i}: candidate r≈{r_values_2[idx]:.0f} µm "
                              f"rejected by 2nd-derivative test (|d²pO2/dr²|={abs(sec_val):.3e} > {R_TISSUE_D2_THRESH:.3e}).")
                        continue

                    # Accept first candidate that passes
                    R_tissue_idx = idx
                    print(f"[R_tissue] {tag_i}: accepted at r≈{r_values_2[R_tissue_idx]:.0f} µm "
                          f"( |d²pO2/dr²|={abs(sec_val):.3e} ≤ {R_TISSUE_D2_THRESH:.3e}; "
                          f"per-radius clip {clip_here:.1f}% ).")
                    break

                if R_tissue_idx is None:
                    if rejected_by_clip > 0 and rejected_by_d2 == 0:
                        print(f"[R_tissue] {tag_i}: all {rejected_by_clip} candidate(s) rejected by per-radius clipping → NOT DETECTED.")
                    elif rejected_by_d2 > 0 and rejected_by_clip == 0:
                        print(f"[R_tissue] {tag_i}: all {rejected_by_d2} candidate(s) rejected by 2nd-derivative test → NOT DETECTED.")
                    else:
                        print(f"[R_tissue] {tag_i}: all candidates rejected (clipping={rejected_by_clip}, d²={rejected_by_d2}) → NOT DETECTED.")
        except Exception as e:
            print(f"[R_tissue] {tag_i}: error while detecting candidate: {e} → NOT DETECTED.")
            R_tissue_idx = None

        R_tissue_v.append([(i, None), R_tissue_idx])

        # Diagnostics for R_tissue (single-center) (optional)
        if SHOW_DETECTION_PLOTS:
            fig, axs = plt.subplots(1, 2, figsize=(14, 5))
            axs[0].plot(radii2, means2, marker='o', markersize=3, label="pO2 Profile (full)")
            if R_tissue_idx is not None:
                axs[0].axvline(radii2[R_tissue_idx], color='red', linestyle='--',
                               label=f"R_tissue ≈ {radii2[R_tissue_idx]:.0f} µm")
            axs[0].set_title(f"Full Circle pO2 for R_tissue\n{tag_i}")
            axs[0].set_xlabel("Radius (µm)"); axs[0].set_ylabel("pO2 (mmHg)")
            axs[0].grid(); axs[0].legend()

            axs[1].plot(radii2[:-1], d, marker='x', color='black')
            if R_tissue_idx is not None:
                axs[1].axvline(radii2[R_tissue_idx], color='red', linestyle='--', label="Estimated R_tissue")
                axs[1].set_title("dpO2/dr vs r"); axs[1].legend()
            else:
                axs[1].set_title("dpO2/dr vs r\nINVALID DETECTION")
            axs[1].set_xlabel("Radius (µm)"); axs[1].set_ylabel("dpO2/dr")
            axs[1].grid()
            plt.tight_layout(); plt.show()

    else:
        # Multiple retained centers -> half-circle tissue estimation for every ordered pair (i,j)
        for i in kept_indices:
            xi, yi = peak_coords[i]
            tag_i = center_tag(i, peak_coords)
            for j in kept_indices:
                if i == j:
                    continue
                xj, yj = peak_coords[j]
                tag_j = center_tag(j, peak_coords)

                radii2, means2, per_clip = half_circle_profile(
                    P, X, Y, xi, yi, xj, yj, r_values_2,
                )
                d = np.diff(means2)     # first derivative
                d2 = np.diff(d)         # second derivative

                try:
                    cand = np.where(np.abs(d) <= DERIV_THRESH)[0]
                    if len(cand) == 0:
                        dmin, dmax = float(np.min(d)), float(np.max(d))
                        print(f"[R_tissue] {tag_i} away from {tag_j}: no radius where |dpO2/dr| <= {DERIV_THRESH:.2e} "
                              f"(range [{dmin:.3e}, {dmax:.3e}]) → NOT DETECTED.")
                        R_tissue_idx = None
                    else:
                        print(f"[R_tissue] {tag_i} away from {tag_j}: {len(cand)} first-derivative candidates → per-radius clipping, d²-test & half-sector …")
                        R_tissue_idx = None

                        # Track precise rejection causes
                        rejected_by_clip = 0
                        rejected_by_d2 = 0
                        rejected_by_sector = 0

                        # Allowed indices for half-sector occupancy: retained centers excluding i and j
                        allowed_for_sector = [k for k in kept_indices if k not in (i, j)]

                        for idx in cand:
                            idx = int(idx)

                            # Per-radius clipping gate (multi-center)
                            clip_here = float(per_clip[idx])
                            if clip_here > CLIP_THRESH_MULTI:
                                rejected_by_clip += 1
                                print(f"[R_tissue] {tag_i} away from {tag_j}: candidate r≈{r_values_2[idx]:.0f} µm "
                                      f"rejected by per-radius clipping ({clip_here:.1f}% > {CLIP_THRESH_MULTI:.1f}%).")
                                continue

                            # 2nd-derivative test
                            idx2 = int(np.clip(idx - 1, 0, len(d2) - 1))
                            sec_val = float(d2[idx2]) if len(d2) > 0 else 0.0
                            if abs(sec_val) > R_TISSUE_D2_THRESH:
                                rejected_by_d2 += 1
                                print(f"[R_tissue] {tag_i} away from {tag_j}: candidate r≈{r_values_2[idx]:.0f} µm "
                                      f"rejected by 2nd-derivative test (|d²pO2/dr²|={abs(sec_val):.3e} > {R_TISSUE_D2_THRESH:.3e}).")
                                continue

                            # Half-sector occupancy test (only against retained centers)
                            r_um = r_values_2[idx]
                            theta0 = np.arctan2(yi - yj, xi - xj)  # away-from-j direction
                            contains, offender = half_sector_contains_other_centers(
                                peak_coords, i, r_um, theta0, allowed_for_sector, eps=0.0
                            )
                            if contains:
                                rejected_by_sector += 1
                                print(f"[R_tissue] {tag_i} away from {tag_j}: candidate r≈{r_um:.0f} µm "
                                      f"rejected by half-sector rule (contains {center_tag(offender, peak_coords)}; "
                                      f"tolerance {TOLERANCE_FRAC*100:.0f}%).")
                                continue

                            # Accept this candidate
                            R_tissue_idx = idx
                            print(
                                f"[R_tissue] {tag_i} away from {tag_j}: accepted at r≈{r_values_2[R_tissue_idx]:.0f} µm "
                                f"( |d²pO2/dr²|={abs(sec_val):.3e} ≤ {R_TISSUE_D2_THRESH:.3e}; "
                                f"per-radius clip {clip_here:.1f}% ).")
                            break

                        # Precise summary if none accepted
                        if R_tissue_idx is None:
                            if (rejected_by_clip > 0) and (rejected_by_d2 == 0) and (rejected_by_sector == 0):
                                print(f"[R_tissue] {tag_i} away from {tag_j}: all {rejected_by_clip} candidate(s) "
                                      f"rejected by per-radius clipping → NOT DETECTED.")
                            elif (rejected_by_d2 > 0) and (rejected_by_clip == 0) and (rejected_by_sector == 0):
                                print(f"[R_tissue] {tag_i} away from {tag_j}: all {rejected_by_d2} candidate(s) "
                                      f"rejected by 2nd-derivative test → NOT DETECTED.")
                            elif (rejected_by_sector > 0) and (rejected_by_clip == 0) and (rejected_by_d2 == 0):
                                print(f"[R_tissue] {tag_i} away from {tag_j}: all {rejected_by_sector} candidate(s) "
                                      f"rejected by half-sector rule → NOT DETECTED.")
                            else:
                                print(f"[R_tissue] {tag_i} away from {tag_j}: all candidates rejected "
                                      f"(clip={rejected_by_clip}, d²={rejected_by_d2}, sector={rejected_by_sector}) → NOT DETECTED.")
                except Exception as e:
                    print(f"[R_tissue] {tag_i} away from {tag_j}: error while detecting candidate: {e} → NOT DETECTED.")
                    R_tissue_idx = None

                R_tissue_v.append([(i, j), R_tissue_idx])

                # Diagnostics for R_tissue (multi-center) (optional)
                if SHOW_DETECTION_PLOTS:
                    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
                    axs[0].plot(radii2, means2, marker='o', markersize=3, label="pO2 Profile (half)")
                    if R_tissue_idx is not None:
                        axs[0].axvline(radii2[R_tissue_idx], color='red', linestyle='--',
                                       label=f"R_tissue ≈ {radii2[R_tissue_idx]:.0f} µm")
                    title_ok = f"Half Circle pO2\n{tag_i} away from {tag_j}"
                    title_bad = title_ok + "\nINVALID DETECTION"
                    axs[0].set_title(title_ok if R_tissue_idx is not None else title_bad)
                    axs[0].set_xlabel("Radius (µm)"); axs[0].set_ylabel("pO2 (mmHg)")
                    axs[0].grid(); axs[0].legend()

                    axs[1].plot(radii2[:-1], d, marker='x', color='black')
                    if R_tissue_idx is not None:
                        axs[1].axvline(radii2[R_tissue_idx], color='red', linestyle='--', label="Estimated R_tissue")
                        axs[1].set_title("dpO2/dr vs r"); axs[1].legend()
                    else:
                        axs[1].set_title("dpO2/dr vs r\nINVALID DETECTION")
                    axs[1].set_xlabel("Radius (µm)"); axs[1].set_ylabel("dpO2/dr")
                    axs[1].grid()
                    plt.tight_layout(); plt.show()

    return R_art_v, R_tissue_v

# ----------------------------
# Main
# ----------------------------
def main():
    print(f"Noise seed: {NOISE_SEED}")
    # 2D visualization
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].imshow(Z_corrected, cmap='jet')
    axs[0].set_title("Combined pO₂ Map (Noisy)")
    axs[1].imshow(Z_smooth, cmap='jet')
    axs[1].set_title("Smoothed pO₂ Map")
    plt.tight_layout(); plt.show()

    # 3D plot with detected maxima
    fig3 = plt.figure(figsize=(8, 6))
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot_surface(X, Y, Z_smooth, cmap='jet', edgecolor='none')
    for x_peak, y_peak in peak_coords:
        z_peak = Z_smooth[int(y_peak), int(x_peak)] + 3
        ax3.scatter(x_peak, y_peak, z_peak, color='yellow', edgecolor='black', s=120, marker='^', linewidth=1.5)
    ax3.set_title("Smoothed pO₂ Profiles with Detected Local Maxima")
    ax3.set_xlabel("x (µm)"); ax3.set_ylabel("y (µm)"); ax3.set_zlabel("pO₂ (mmHg)")
    plt.tight_layout(); plt.show()

    # Run analysis
    print("Detected centers:", peak_coords)
    if len(peak_coords) >= 1:
        R_art_v, R_tissue_v = run_analysis(Z_smooth, X, Y, peak_coords)
    else:
        print(f"Expected at least 1 peak for analysis, but found {len(peak_coords)}.")
        return

    print("R_art_v:", R_art_v)
    print("R_tissue_v:", R_tissue_v)

    # -------- Final overlay (R_art circles + R_tissue arcs/circles) --------
    from matplotlib.patches import Circle
    import matplotlib.patheffects as pe
    pe_outline = [pe.Stroke(linewidth=3, foreground='black'), pe.Normal()]

    fig_final, ax_final = plt.subplots(1, 1, figsize=(8, 7), layout="tight")
    ax_final.imshow(Z_smooth, cmap='jet')
    ax_final.set_title("Smoothed pO₂ with Estimated R_art and R_tissue")

    # Re-draw peak markers
    for x_peak, y_peak in peak_coords:
        ax_final.scatter(x_peak, y_peak, marker='x', s=160, c='white', linewidths=2.5, path_effects=pe_outline)

    # R_art circles
    for idx, r_art_um in R_art_v:
        if r_art_um is None:
            continue
        x_peak, y_peak = peak_coords[idx]
        circ = Circle((x_peak, y_peak), r_art_um, fill=False, linewidth=2.0, edgecolor='white')
        ax_final.add_patch(circ)
        ax_final.text(x_peak + r_art_um + 3, y_peak, f"R_art ≈ {r_art_um:.0f} µm",
                      color='white', fontsize=9, va='center', path_effects=pe_outline)

    # Helper to keep text inside the image
    h, w = Z_smooth.shape
    PAD = 8
    def clamp_point(xp, yp, pad=PAD):
        return float(np.clip(xp, pad, w - pad)), float(np.clip(yp, pad, h - pad))

    tab10_colors = list(mpl.colormaps['tab10'].colors)

    # R_tissue overlay
    for (i_j, R_tissue_idx) in R_tissue_v:
        i, j = i_j
        if R_tissue_idx is None:
            continue
        if not (0 <= R_tissue_idx < len(r_values_2)):
            print(f"[Overlay] {center_tag(i, peak_coords)} (j={None if j is None else j+1}): "
                  f"R_tissue index {R_tissue_idx} out of range (len={len(r_values_2)}) → DISCARD FROM OVERLAY.")
            continue

        xi, yi = peak_coords[i]
        r_um = r_values_2[R_tissue_idx]
        color_i = tab10_colors[i % len(tab10_colors)]

        if j is None:
            circ_t = Circle((xi, yi), r_um, fill=False, linewidth=2.0, edgecolor=color_i)
            ax_final.add_patch(circ_t)
            lx, ly = clamp_point(xi + r_um + 3, yi)
            ax_final.text(lx, ly, f"R_tissue ≈ {r_um:.0f} µm", color=color_i, fontsize=8, va='center', path_effects=pe_outline)
        else:
            xj, yj = peak_coords[j]
            theta0 = np.arctan2(yi - yj, xi - xj)
            angles = np.linspace(theta0 - np.pi / 2, theta0 + np.pi / 2, 256)
            x_arc = xi + r_um * np.cos(angles)
            y_arc = yi + r_um * np.sin(angles)
            ax_final.plot(x_arc, y_arc, linewidth=2.0, color=color_i)
            mid = len(angles) // 2
            lx, ly = clamp_point(x_arc[mid], y_arc[mid] - 6)
            ax_final.text(lx, ly, f"R_tissue ≈ {r_um:.0f} µm", color=color_i, fontsize=8, ha='center', va='bottom', path_effects=pe_outline)

    # Legend for centers that have some valid R_tissue
    from matplotlib.lines import Line2D
    valid_centers = set()
    for (i_j, idx) in R_tissue_v:
        i, j = i_j
        if idx is not None:
            valid_centers.add(i)
    handles = []
    for i in sorted(valid_centers):
        xp, yp = peak_coords[i]
        handles.append(Line2D([0],[0], color=tab10_colors[i % len(tab10_colors)], lw=2,
                              label=f"Center {i+1} ({xp:.0f}, {yp:.0f})"))
    if handles:
        ax_final.legend(handles=handles, loc='upper left', fontsize=9, frameon=True)

    ax_final.set_xlabel("x (µm)"); ax_final.set_ylabel("y (µm)")
    ax_final.set_aspect('equal')

    # Ground truth printout
    print("\nGround Truth:")
    for sidx, sp in enumerate(sources, start=1):
        print(f"  Source {sidx}: center=({sp[0]}, {sp[1]}), R_art={sp[5]}, R_tissue={sp[3]}")

    plt.show()

if __name__ == "__main__":
    main()
