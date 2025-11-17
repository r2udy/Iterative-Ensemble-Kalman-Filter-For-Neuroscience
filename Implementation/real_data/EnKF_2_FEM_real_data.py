#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 14:35:46 2025

@author: ruudybayonne

This code performs the Ensemble Kalman filter procedure on the a subset of the dataset 
depecting homegeneous oxygen consumption.
"""

import sys
import os

py_data_location = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/"
py_file_location = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/classes/"
sys.path.append(os.path.abspath(py_file_location))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from EnKF_FEM import EnKF, build_obs_covariance_radial, build_obs_covariance_diagonal
from EnKF_FEM_2 import EnKF
from circlesearch import Po2Analyzer
from MapGenerator import MapGenerator
from Po2Dataset import load_data
import pylab as P

# --------- Load data --------- #
df = pd.read_pickle(py_data_location + "dataset.pkl")
df_copy = df.copy()
df_copy['pO2Value'] = df_copy['pO2Value'].apply(lambda x: x.flatten())
df_copy.keys()
uniform_dataset = load_data(py_data_location + 'uniform_dataset.txt')

# --------------------------
# Constants initial #
D = 4.0e3
alpha = 1.39e-15
cmro2_by_M = (60 * D * alpha * 1e12)
grid_size = 20 # data size

cmro2_lower, cmro2_upper = 1.0, 3.0
cmro2_var = (cmro2_upper - cmro2_lower)**2 / 12.0  # Uniform distribution variance
M_var = cmro2_var / cmro2_by_M**2  # model uncertainty scaled

# Prior associated with R0
R0_lower, R0_upper = 80., 120.
R0_var = 5.0**2 # prior uncertainty of caparilary-free space radius

# Prior associated with pvessel
pvessel_lower, pvessel_upper = 70., 90.
pvessel_var = 10.0**2 # prior uncertainty of Neumann boundary condition

# Observation variance parameters
obs_var_constant = 5.**2   # constant uncertainty of measurements in the observation covariance matrix R
sigma = 2.0  # noise level in synthetic data
obs_var_high = 5.**2    # high uncertainty of measurements
obs_var_low = 1.**2     # low uncertainty of measurements

# --------------------------
# EnKF Parameters
seed = np.random.seed(1)
state_dim = 2
obs_dim = 400
n_ensembles = 50

# Initialize the ensemble
a = np.array([cmro2_lower / cmro2_by_M, R0_lower])
b = np.array([cmro2_upper / cmro2_by_M, R0_upper])

# No dynamic model
def dynamics_model(x):
    return x

# -------------------------
# Create the EnKF object
enkf = EnKF(state_dim, obs_dim, n_ensembles, dynamics_model, seed)

# Initialize the EnKF method
enkf.initialize_ensemble(a, b)
    
# Update the the background noise
B   = np.array([[M_var,    0.0],
                [0.0,   R0_var]])   # Background covariance matrix
enkf.set_process_noise(B)

# -------------------------
# Initialization of Arrays
observations_id = [entry for entry in uniform_dataset]
observations = []
cmro2_est_enkf = []
cmro2_cov_est_enkf = []
R0_est_enkf = []
R0_cov_est_enkf = []
pvessel_est_enkf = []
errors_enkf_relative = []
errors_enkf_absolute = []
state_ensembles = []
stats_overall = []
corrections = []
state_ensembles = np.zeros((state_dim, len(uniform_dataset), n_ensembles))


# --------------------------
# Simulate a sequence with observation for the uniform case
for i, entry in enumerate(uniform_dataset):
    art_id = entry[0][0]
    dth_id = entry[0][1]

    angles_1 = entry[1]
    angles_2 = entry[2]

    min_radius = entry[3][0]
        
    # Observations
    mask = (df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id)
    obs = df_copy[mask]['pO2Value'].tolist()[0]
    observations.append(obs)

    X = df_copy[mask]['pointsX'].tolist()[0]
    X = X - X[0, :].mean(axis=0) # Centering the coordinates X
    Y = df_copy[mask]['pointsY'].tolist()[0]
    Y = Y - Y[:, 0].mean(axis=0)

    pO2_array = obs.reshape((grid_size, grid_size), order='F')
    # Find Geometric parameters such as Rves and R0
    analyzer = Po2Analyzer(pO2_array, X, Y)
    analyzer.find_circles()
    Rves = analyzer.rin
    pvessel_est = analyzer.p_vessel
    center = analyzer.center_ij
    center_coordinates = analyzer.center


    # ---------- Target Cells -----------
    # Adjust the observation covariance matrix to account very uncertain measurement
    # R = build_obs_covariance_diagonal(
    #     grid_size=grid_size,
    #     origin=center,
    #     angle_ranges=[angles_1, angles_2],
    #     min_radius=min_radius,
    #     obs_var_high=obs_var_high,
    #     obs_var_low=obs_var_low
    # )
    # R = build_obs_covariance_radial(
    # origin = center,
    # obs_var_high = obs_var_high,
    # obs_var_low = obs_var_low,
    # mode='exponential'
    # )
    R = obs_var_constant * np.eye(obs_dim) # Observation covariance matrix

    enkf.set_observation_noise(R)

    # ----------------------
    # EnKF steps
    enkf.predict()
    enkf.update(obs, X, Y)

    # Get current estimate
    mean, cov = enkf.get_state_estimate()

    # Means and Covariances
    cmro2_mean  = mean[0] * cmro2_by_M
    R0_mean    = mean[1]

    cmro2_cov   = cov[0, 0] * (cmro2_by_M)**2
    R0_cov     = cov[1, 1]


    correction = np.abs(np.mean(enkf.length_scale * enkf.K @ enkf.innovation)) * cmro2_by_M

    # Compute the absolute error
    print(f'CMRO_2: {cmro2_mean}')
    print(f'R0: {R0_mean}')
    print(f"p_vessel: {pvessel_est}")
    print(f'Rves: {Rves}')
    generator_enkf = MapGenerator(cmro2=cmro2_mean, 
                        pvessel=pvessel_est, 
                        Rves=Rves, 
                        R0=R0_mean, 
                        Rt=R0_mean,
                        X=X,
                        Y=Y)
    error_enkf_relative = np.abs(obs - generator_enkf.pO2_array.flatten()) * 100 / np.abs(obs) 
    error_enkf_absolute = np.abs(obs - generator_enkf.pO2_array.flatten())

    # --- Quick visualization: diagonal (variance) map ---
    fig = plt.figure(figsize=(12, 8))
    # ----------------------
    # (1) Radial pO₂ map with circles
    ax = fig.add_subplot(2, 3, 1)
    theta = np.linspace(0, 2 * np.pi, 100) # angles
    circle_in_x = center_coordinates[0] + Rves*np.cos(theta)
    circle_in_y = center_coordinates[1] + Rves*np.sin(theta)
    circle_out_x = center_coordinates[0] + R0_mean*np.cos(theta)
    circle_out_y = center_coordinates[1] + R0_mean*np.sin(theta)
    ax.plot(circle_in_x, circle_in_y, "m-", lw=2, label=f"Inner r={Rves:.1f}")
    ax.plot(circle_out_x, circle_out_y, "c--", lw=2, label=f"Outer r={R0_mean:.1f}")
    ax.plot(center_coordinates[0], center_coordinates[1], "kx", ms=8, label="Center")

    c = ax.pcolormesh(X, Y, pO2_array, shading='auto', cmap='jet')
    fig.colorbar(c, label='pO₂')
    ax.set_title(f"Radial pO₂ Map | Arteriole {art_id} | Depth {dth_id}")
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.axis('equal')
    ax.legend()
    ax.set_aspect('equal', 'box')
    # ----------------------
    # (2) Diagonal variance map
    ax = fig.add_subplot(2, 3, 2)
    uncertainty_map = np.diag(enkf.R).reshape((grid_size, grid_size))
    c = ax.pcolormesh(X, Y, uncertainty_map, cmap='viridis')
    fig.colorbar(c, label='Variance')
    ax.set_title("Diagonal Variance Map of PO2 Covariance Matrix")
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.axis('equal')
    ax.set_aspect('equal', 'box')

    # ----------------------
    # (3) Relative error 3D surface
    ax = fig.add_subplot(2, 3, 3, projection='3d')

    error_enkf_relative_array = error_enkf_relative.reshape((grid_size, grid_size), order='F')
    sc = ax.plot_surface(X, Y, error_enkf_relative_array, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('pO2 (mmHg)')
    ax.set_title(f"Map of relative error, \nmean relative error:{error_enkf_relative.mean():.2f}; \nCMRO2:{cmro2_mean:.2f} umol /cm^3 /min\n R0:{R0_mean:.2f} um")
    plt.colorbar(sc, ax=ax, shrink=0.3, aspect=10, label='Color scale: Relative Error (%)')
    # ----------------------
    # (4) Absolute error 3D surface
    ax = fig.add_subplot(2, 3, 4, projection='3d')

    error_enkf_absolute_array = error_enkf_absolute.reshape((grid_size, grid_size), order='F')
    sc = ax.plot_surface(X, Y, error_enkf_absolute_array, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('pO2 (mmHg)')
    ax.set_title(f"Map of relative error, \nmean relative error:{error_enkf_relative.mean():.2f}; \nCMRO2:{cmro2_mean:.2f} umol /cm^3 /min")
    plt.colorbar(sc, ax=ax, shrink=0.3, aspect=10, label='Color scale: Relative Error (%)')
    # ----------------------
    # (5) Approximated vs. True map
    ax = fig.add_subplot(2, 3, 5, projection="3d")

    sc = ax.plot_surface(X, Y, generator_enkf.pO2_array, cmap='viridis', edgecolor='none')
    ax.plot_surface(X, Y, pO2_array, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('pO2 (mmHg)')
    ax.set_title(f"Display of the approximated map: \nCMRO2:{cmro2_mean:.2f} umol /cm^3 /min\n R0:{R0_mean:.2f} um")
    plt.colorbar(sc, ax=ax, shrink=0.3, aspect=10, label='Color scale: Relative Error (%)')
    plt.tight_layout()
    plt.show()


    # ----------------------
    # Print the results
    print(f"Correction: {correction}")
    print(f"Mean Relative Error: {error_enkf_relative.mean()}")   
    print(f"Mean Absolute Error: {error_enkf_absolute.mean()}") 

    # Save stats
    stats_overall.append((cmro2_mean, cmro2_cov))
    errors_enkf_relative.append(np.abs(error_enkf_relative)) # Save the relative errors
    errors_enkf_absolute.append(np.abs(error_enkf_absolute)) # Save the absolute errors
    pvessel_est_enkf.append(pvessel_est)
    corrections.append(correction)
    cmro2_est_enkf.append(cmro2_mean)
    cmro2_cov_est_enkf.append(cmro2_cov)
    R0_est_enkf.append(R0_mean)
    R0_cov_est_enkf.append(R0_cov)
    state_ensembles[0, i, :] = enkf.ensemble[0, :]  # CMRO2 ensembles
    state_ensembles[1, i, :] = enkf.ensemble[1, :]  # R0 ensembles


    # Print results in the terminal
    print(f"\n\n Ensemble Kalman Filter paramaters estimation")
    print("-"*65)
    print(f"Observation ID: {art_id}, Depth ID: {dth_id}")
    print(f"\nCMRO2 Mean: {cmro2_mean}, CMRO2 √(Cov): {np.sqrt(cmro2_cov)}, cmro2 Covariance: {cmro2_cov}")
    print("-"*25)
    print(f"R0 Mean: {R0_mean}, R0 √(Cov): {np.sqrt(R0_cov)}, R0 Covariance: {R0_cov}\n")
    print(f"Mean Relative Error: {error_enkf_relative.mean()}")
    print(f"Mean Absolute Error: {error_enkf_absolute.mean()}")
    print(f"Correction: {correction}")

cmro2_est_enkf = np.array(cmro2_est_enkf)
cmro2_cov_est_enkf = np.array(cmro2_cov_est_enkf)
pvessel_est_enkf = np.array(pvessel_est_enkf)
errors_enkf_relative = np.array(errors_enkf_relative)
errors_enkf_absolute = np.array(errors_enkf_absolute)
state_ensembles = np.array(state_ensembles)
stats_overall = np.array(stats_overall)
corrections = np.array(corrections)


# ------------------------------------------------------------------
# ----------------------+ Plots the results +----------------------#
# Path for saving the data
path = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/JuliPresentation/"

x_obs = np.arange(1, len(observations_id) + 1) # Simulated iteration steps

# -----------------------
# CMRO_2 Stats for converged iterations
observations_id = [i for i in range(1, len(observations) + 1)]
observations = np.array(observations)
x = np.arange(1, len(observations_id) + 1)
# Stats
overall_mean = cmro2_est_enkf.mean()

data = state_ensembles[0].T * cmro2_by_M # Define data
numBoxes = len(observations) # Define numBoxes
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

# -----------------------
# CMRO_2 Stats for overall for converged iterations
data = np.mean(state_ensembles[0].T * cmro2_by_M, axis=0)
P.figure()
bp = P.boxplot(data, labels=['Overall Stats'])
y = data
x = np.random.normal(1, 0.04, size=len(y))
P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID') 
P.ylabel('State value CMRO2 (umol /cm^3 /min)')
P.title('EnKF State Estimation with Uncertainty - Overall')
P.grid(True)
P.show()    

# -----------------------
# R0 Stats
# -----------------------
observations_id = [i for i in range(1, len(observations) + 1)]
observations = np.array(observations)

data = state_ensembles[1].T
numBoxes = len(observations) # Define numBoxes
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


# -----------------------
# P_vessel Stats
data = pvessel_est_enkf
# -----------------------
# Corrections associated to estimation overall
numBoxes = data.shape[0]  # now robust
x_obs = np.arange(1, numBoxes + 1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_obs, data)
# Labels and title
plt.ylabel('$P_{vessel}$ (mmHg)')
plt.xlabel('$PO_{2}$ Map ID')
plt.title('Estimated Partial Pressure at Vessel Wall')
plt.grid(True)
plt.xticks(x_obs, [f'Obs{i}' for i in x_obs])
plt.axhline(y=np.mean(data), color='r', linestyle='--', label='Mean $P_{vessel}$')
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------
# Relative Error Stats
# Stats
data = errors_enkf_relative.T # Define data
numBoxes = data.shape[1]  # now robust
names = [f'obs{i}' for i in range(1, numBoxes + 1)]
P.figure()
bp = P.boxplot(data, labels=names)
for i in range(numBoxes):
    y = data[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('Relative Partial Pressure Error')
P.title('Relative Errors distributions - EnKF')
P.grid(True)
P.show()

# -----------------------
# Absolute Error Stats
# Stats
data = errors_enkf_absolute.T # Define data
numBoxes = data.shape[1]  # now robust
names = [f'obs{i}' for i in range(1, numBoxes + 1)]
P.figure()
bp = P.boxplot(data, labels=names)
for i in range(numBoxes):
    y = data[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('Absolute Partial Pressure Error')
P.title('Absolute Errors distributions - EnKF')
P.grid(True)
P.show()


# -----------------------
# Uncertainty associated to estimation
data = state_ensembles[0] * cmro2_by_M
# -----------------------
# Corrections associated to estimation overall
numBoxes = data.shape[0]  # now robust
x_obs = np.arange(1, numBoxes + 1)

fig, ax = plt.subplots(figsize=(10, 6))
cov_track = np.array([np.std(array) for array in data])
ax.plot(x_obs, cov_track)
# Labels and title
plt.ylabel('Estimated CMRO2 Uncertainty (umol /cm^3 /min)')
plt.xlabel('$PO_{2}$ Map ID')
plt.title('EnKF Uncertainty')
plt.grid(True)
plt.xticks(x_obs, [f'Obs{i}' for i in x_obs])
plt.axhline(y=np.mean(cov_track), color='r', linestyle='--', label='Mean Uncertainty')
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------
# Corrections associated to estimation
numBoxes = corrections.shape[0]  # now robust
x_obs = np.arange(1, numBoxes + 1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_obs, corrections, marker='o', color='orange')
# Labels and title
plt.ylabel('Estimated CMRO2 Corrections (umol /cm^3 /min)')
plt.xlabel('$PO_{2}$ Map ID')
plt.title('EnKF Corrections "Kd" - Iterative Estimation')
plt.grid(True)
plt.xticks(x_obs, [f'Obs{i}' for i in x_obs])
plt.show()

# -----------------------
# Uncertainty associated to estimation
# Create figure
plt.figure(figsize=(10, 6))
cmro2_mean_ = stats_overall[:, 0]
cmro2_cov_ = stats_overall[:, 1]
# Plot mean +/- 1 standard deviation (sqrt of variance)
plt.plot(x_obs, cmro2_mean_, '-o', color='green', label='State EnKF estimate (CMRO2)')
plt.fill_between(
    x_obs,
    cmro2_mean_ - np.sqrt(cmro2_cov_),  # Lower bound (mean - σ)
    cmro2_mean_ + np.sqrt(cmro2_cov_),  # Upper bound (mean + σ)
    color='blue',
    alpha=0.2,
    label='Uncertainty (+/- 1 StD)'
)
plt.xlabel('$PO_{2}$ Map ID')
plt.ylabel('CMRO2 (umol /cm^3 /min)')
plt.title('EnKF CMRO2 Estimation with Uncertainty')
plt.xticks(x_obs, [f'Obs{i}' for i in x_obs])
plt.axhline(y=np.mean(cmro2_mean_), color='r', linestyle='--', label='Mean CMRO2')
plt.axhline(y=cmro2_lower, color='orange', linestyle='--', label='CMRO2 Lower Bound (Prior)')
plt.axhline(y=cmro2_upper, color='orange', linestyle='--', label='CMRO2 Upper Bound (Prior)')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------
# Posterior distribution through the iteration
# Sample data
data = np.array(state_ensembles[0].T) * cmro2_by_M
pdf_matrix = np.zeros(data.shape) # rows: u, cols: t

# Create a grid for the x and y axes
iterations = np.arange(data.shape[1])  # 9 iterations
points = np.linspace(data.min(), data.max(), data.shape[0]) # 100 points
X, Y = np.meshgrid(iterations, points)

for i in range(data.shape[1]):
    kde = gaussian_kde(data[:, i], bw_method='scott')
    pdf_matrix[:, i] = kde(points)

# 3D surface plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, pdf_matrix, cmap='viridis', edgecolor='none')

ax.set_xlabel('Artificial Time Step n')
ax.set_ylabel('CMRO_2(n) (umol /cm^3 /min)')
ax.set_zlabel('PDF f(U,t) of CMRO_2')
ax.set_title('PDF of Oxygen consumption over the artificial time n.')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.show()

# # Save the data
# np.save(path + f"state_ensembles_{n_ensembles}.npy", state_ensembles)
# np.save(path + f"cmro2_means_{n_ensembles}.npy",stats_overall[:, 0])
# np.save(path + f"cmro2_covs_{n_ensembles}.npy", stats_overall[:, 1])
# np.save(path + f"R0_means_{n_ensembles}.npy", R0_est_enkf)
# np.save(path + f"R0_covs_{n_ensembles}.npy", R0_cov_est_enkf)
# np.save(path + f"p_vessel_estimates_{n_ensembles}.npy", pvessel_est_enkf)
# np.save(path + f"errors_enkf_relative_{n_ensembles}.npy", errors_enkf_relative)
# np.save(path + f"errors_enkf_absolute_{n_ensembles}.npy", errors_enkf_absolute)
