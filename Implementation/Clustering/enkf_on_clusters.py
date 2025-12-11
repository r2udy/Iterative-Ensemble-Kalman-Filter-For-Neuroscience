import sys
import os

py_data_location = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/Synthetic Dataset/"
py_file_location = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/classes/"
sys.path.append(os.path.abspath(py_file_location))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as P
from synthetic_data_generation import load_synthetic_data

from enkf_cluster_runner import EnKFClusterRunner
from EnKF_FEM import build_obs_covariance_radial, build_obs_covariance_diagonal
from EnKF_FEM_3 import EnKF

# ---------------------------
# Save generated data
filename = "mulitple_sources_R0"

# ---------------------------
# Load generated data
maps, df_meta = load_synthetic_data(filename)

data_path = py_data_location + "synthetic_po2_clustered_" + filename + ".npz"
runner = EnKFClusterRunner(data_path, EnKF)

cluster_id = 2

state_dim = 3     # [M_ratio, R0, Pvessel]
obs_dim = 400     # 20×20 pO2 grid
n_ensemble = 50

X_axis, Y_axis = np.meshgrid(np.linspace(-190, 190, 20), np.linspace(-190, 190, 20))

# Constants initial #
D = 4.0e3
alpha = 1.39e-15
cmro2_by_M = (60 * D * alpha * 1e12)
grid_size = 20 # data size

# Prior associated with cmro2
cmro2_lower, cmro2_upper = 1.0, 3.0
cmro2_var = (cmro2_upper - cmro2_lower)**2 / 12 # variance of uniform distribution
M_var = cmro2_var / cmro2_by_M**2 # model uncertainty scaled

# Prior associated with R0
R0_lower, R0_upper = 80., 120.
R0_var = 5.0**2 # prior uncertainty of caparilary-free space radius

# Prior associated with pvessel
pvessel_lower, pvessel_upper = 70., 90.
pvessel_var = 10.0**2 # prior uncertainty of Neumann boundary condition

obs_var_constant = 3.**2   # constant uncertainty of measurements in the observation covariance matrix R
sigma = 2.0  # noise level in synthetic data
obs_var_high = 5.**2    # high uncertainty of measurements
obs_var_low = 1.**2     # low uncertainty of measurements


# Update the the background noise
B   = np.array([[M_var,    0.0,        0.0],
                [0.0,      R0_var,     0.0],
                [0.0,   0.0,           pvessel_var]])   # Background covariance matrix


# R = build_obs_covariance_radial(
# origin = center,
# obs_var_high = obs_var_high,
# obs_var_low = obs_var_low,
# mode='exponential'
# )

R = obs_var_constant * np.eye(obs_dim) # Observation covariance matrix


enkf_config = dict(
    state_dim=state_dim,
    obs_dim=obs_dim,
    n_ensembles=n_ensemble,
    dynamics_model=lambda x: x,  # you have no dynamics
    R = R,
    B = B,
)

a = np.array([cmro2_lower / cmro2_by_M, R0_lower, pvessel_lower])
b = np.array([cmro2_upper / cmro2_by_M, R0_upper, pvessel_upper])
prior_bounds = (
    a,   # lower bounds
    b    # upper bounds
)

results = runner.run_individual(
    cluster_id,
    X_axis, Y_axis,
    enkf_config,
    prior_bounds
)

centroid_result = runner.run_on_centroid(
    cluster_id,
    X_axis, Y_axis,
    enkf_config,
    prior_bounds
)

results["estimates"]      # shape (#maps_in_cluster, state_dim)
results["estimates_cov"]
results["posteriors"]     # list of ensembles
results["meta"]           # param ground-truth for comparison

centroid_result["estimate"]     # posterior mean (3,)
centroid_result["estimate_cov"]
centroid_result["posterior"]    # ensemble (3×50)
centroid_result["mean_map"]     # mean Po2 map for that cluster

import numpy as np
import matplotlib.pyplot as plt

state_labels=["CMRO2", "R0", "Pvessel"]

# ===============================
# 1. Posterior mean per map
# ===============================
est = results["estimates"]                  # shape (Nmaps, 3)
est_cov = results["estimates_cov"]          # shape (Nmaps, 3, 3)
post = np.array(results["posteriors"])                # list of (3×Nens) arrays
meta = results["meta"]                      # ground truth dict

Nmaps, state_dim = est.shape


# --------- Plots the results ---------
# -----------------------
# CMRO2 Stats
# -----------------------
observations_id = runner.get_cluster(cluster_id=cluster_id)[-1]
x = np.arange(1, len(observations_id) + 1)
# Stats

data = post[:, 0, :].T * cmro2_by_M # Define data
numBoxes = len(observations_id) # Define numBoxes
names = [f'obs {i}' for i in observations_id]

P.figure()
bp = P.boxplot(data, labels=names)
for i in range(numBoxes):
    y = data[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value CMRO2 (umol /cm^3 /min)')
P.title('EnKF State Estimation with Uncertainty')
P.grid(True)
# P.savefig(path + 'enkf_state_estimation_test.png', dpi=300, bbox_inches='tight')
P.show()


# Simulated time steps
indx_sorted = np.argsort(meta["cmro2_sampled"])
cmro2_true_values = meta["cmro2_sampled"][indx_sorted]
cmro2_est_enkf = est[:, 0][indx_sorted] * cmro2_by_M
cmro2_cov_est_enkf = est_cov[:, 0, 0][indx_sorted] * cmro2_by_M**2
x_obs = np.arange(1, numBoxes + 1)
# Create figure
plt.figure(figsize=(10, 6))
# Plot mean +/- 1 standard deviation (sqrt of variance)
plt.plot(x_obs, cmro2_true_values, '-x', color='black', label='True paramter (CMRO2)')
plt.plot(x_obs, cmro2_est_enkf, '-x', color='green', label='State EnKF estimate (CMRO2)')
plt.fill_between(
    x_obs,
    cmro2_est_enkf - np.sqrt(cmro2_cov_est_enkf),  # Lower bound (mean - σ)
    cmro2_est_enkf + np.sqrt(cmro2_cov_est_enkf),  # Upper bound (mean + σ)
    color='blue',
    alpha=0.2,
    label='Uncertainty (+/- sigma)'
)
# Labels and title
plt.ylabel('Estimated CMRO2 (umol /cm^3 /min)')
plt.xlabel('Input CMRO2 (umol /cm^3 /min)')
plt.title('EnKF State Estimation with Uncertainty using Krogh-Erlang Cylinder Model')
plt.legend()
plt.grid(True)
plt.xticks(x_obs, [f'Obs{i}' for i in observations_id])
# plt.savefig(path + 'enkf_state_estimation_time_steps.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------
# R0 Stats
# -----------------------
data = post[:, 1, :].T
numBoxes = len(observations_id) # Define numBoxes
names = [f'obs {i}' for i in observations_id]

P.figure()
bp = P.boxplot(data, labels=names)
for i in range(numBoxes):
    y = data[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value R0 (um)')
P.title('EnKF R0 State Estimation with Uncertainty')
P.grid(True)
# P.savefig(path + 'enkf_R0_state_estimation.png', dpi=300, bbox_inches='tight')
P.show()


# Simulated time steps
indx_sorted = np.argsort(meta["R0"])
R0_true_values = meta["R0"][indx_sorted]
R0_est_enkf = est[:, 1][indx_sorted]
R0_cov_est_enkf = est_cov[:, 1, 1][indx_sorted]
x_obs = np.arange(1, numBoxes + 1)
# Create figure
plt.figure(figsize=(10, 6))
# Plot mean +/- 1 standard deviation (sqrt of variance)
plt.plot(x_obs, R0_true_values, '-x', color='black', label='True paramter (CMRO2)')
plt.plot(x_obs, R0_est_enkf, '-x', color='green', label='State EnKF estimate (CMRO2)')
plt.fill_between(
    x_obs,
    R0_est_enkf - np.sqrt(R0_cov_est_enkf),  # Lower bound (mean - σ)
    R0_est_enkf + np.sqrt(R0_cov_est_enkf),  # Upper bound (mean + σ)
    color='blue',
    alpha=0.2,
    label='Uncertainty (+/- sigma)'
)
# Labels and title
plt.ylabel('State value R0 (um)')
plt.xlabel('Observation Id Number')
plt.title('EnKF State Estimation with Uncertainty using Krogh-Erlang Cylinder Model')
plt.legend()
plt.grid(True)
plt.xticks(x_obs, [f'Obs{i}' for i in observations_id])
# plt.savefig(path + 'enkf_state_estimation_time_steps.png', dpi=300, bbox_inches='tight')
plt.show()


# -----------------------
# pvessel Stats
# -----------------------
data = post[:, 2, :].T
numBoxes = len(observations_id) # Define numBoxes
names = [f'obs {i}' for i in observations_id]

P.figure()
bp = P.boxplot(data, labels=names)
for i in range(numBoxes):
    y = data[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('Observation Id Number')
P.ylabel('State values pvessel (mmHg)')
P.title('EnKF Partial Pressure at the vessel wall\n State Estimation with Uncertainty')
P.grid(True)
# P.savefig(path + 'enkf_pvessel_state_estimation.png', dpi=300, bbox_inches='tight')
P.show()

# Simulated time steps
indx_sorted = np.argsort(meta["pvessel_sampled"])
pvessel_true_values = meta["pvessel_sampled"][indx_sorted]
pvessel_est_enkf = est[:, 2][indx_sorted]
pvessel_cov_est_enkf = est_cov[:, 2, 2][indx_sorted]
x_obs = np.arange(1, numBoxes + 1)
# Create figure
plt.figure(figsize=(10, 6))
# Plot mean +/- 1 standard deviation (sqrt of variance)
plt.plot(x_obs, pvessel_true_values, '-x', color='black', label='True paramter (CMRO2)')
plt.plot(x_obs, pvessel_est_enkf, '-x', color='green', label='State EnKF estimate (CMRO2)')
plt.fill_between(
    x_obs,
    pvessel_est_enkf - np.sqrt(pvessel_cov_est_enkf),  # Lower bound (mean - σ)
    pvessel_est_enkf + np.sqrt(pvessel_cov_est_enkf),  # Upper bound (mean + σ)
    color='blue',
    alpha=0.2,
    label='Uncertainty (+/- sigma)'
)
# Labels and title
plt.ylabel('State values pvessel (mmHg)')
plt.xlabel('Observation Id Number')
plt.title('EnKF State Estimation with Uncertainty using Krogh-Erlang Cylinder Model')
plt.legend()
plt.grid(True)
plt.xticks(x_obs, [f'Obs{i}' for i in observations_id])
# plt.savefig(path + 'enkf_state_estimation_time_steps.png', dpi=300, bbox_inches='tight')
plt.show()


# -----------------------
# Uncertainty associated to estimation
# -----------------------
data = est_cov[:, 0, 0] * cmro2_by_M**2 # Define data
numBoxes = data.shape[0]  # now robust
x_obs = np.arange(1, numBoxes + 1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_obs, data, '-o', color='blue', label='Uncertainty in CMRO2 estimation (StD)')
plt.ylabel('Estimated CMRO2 Uncertainty (umol /cm^3 /min)')
plt.xlabel('$PO_{2}$ Map ID')
plt.title('EnKF Uncertainty')
plt.grid(True)
plt.xticks(x_obs, [f'Obs{i}' for i in observations_id])
plt.legend()
plt.tight_layout()
# plt.savefig(path + 'enkf_uncertainty.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(15, 5))
plt.plot(est[:, i], 'o-', label='Posterior mean')

# Overlay ground-truth if available
key = list(meta.keys())[i] if i < len(meta) else None
if key in meta:
    plt.hlines(1.5, 0, Nmaps-1,
                colors='r', linestyles='--', label='Ground truth')

plt.title(f"Posterior mean: {state_labels[i]}")
plt.xlabel("Map index")
plt.legend()

plt.suptitle("Posterior State Estimates Across Maps")
plt.tight_layout()
plt.show()