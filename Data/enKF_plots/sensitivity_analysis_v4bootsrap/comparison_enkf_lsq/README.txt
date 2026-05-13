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