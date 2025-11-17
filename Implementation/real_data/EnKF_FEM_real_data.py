#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:29:03 2025

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
cmro2_var = (cmro2_upper - cmro2_lower)**2
M_var = cmro2_var / cmro2_by_M**2 / 10  # model uncertainty scaled
obs_var_high = 5.**2
obs_var_low = 1.**2 # measurement uncertainty

# --------------------------
# EnKF Parameters
seed = np.random.seed(1)
state_dim = 1
obs_dim = 400
n_ensembles = 10

# Initialize the ensemble
a = np.array([cmro2_lower / cmro2_by_M])
b = np.array([cmro2_upper / cmro2_by_M])

# No dynamic model
def dynamics_model(x):
    return x

# -------------------------
# Create the EnKF object
enkf = EnKF(state_dim, obs_dim, n_ensembles, dynamics_model, seed)

# Initialize the EnKF method
enkf.initialize_ensemble(a, b)
    
# Update the the background noise
B = np.array([[M_var]])         # Background covariance matrix
enkf.set_process_noise(B)

# -------------------------
# Initialization of Arrays
observations_id = [entry for entry in uniform_dataset]
observations = []
cmro2_est_enkf = []
cmro2_cov_est_enkf = []
p_vessel_est = []
errors_enkf_relative = []
errors_enkf_absolute = []
state_ensembles = []
state_ensembles_overall = []
stats_overall = []
corrections_overall = []
corrections = []

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
    Y = df_copy[mask]['pointsY'].tolist()[0]

    pO2_array = obs.reshape((grid_size, grid_size), order='F')
    # Find Geometric parameters such as Rves and R0
    analyzer = Po2Analyzer(pO2_array, X, Y)
    analyzer.find_circles()
    Rves = analyzer.rin
    R0 = analyzer.rout
    p_vessel = analyzer.p_vessel
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
    R = build_obs_covariance_radial(
    origin = center,
    obs_var_high = obs_var_high,
    obs_var_low = obs_var_low,
    mode='exponential'
    )
    # R = 15. * np.eye(obs_dim) # Observation covariance matrix

    enkf.set_observation_noise(R)

    # --- Quick visualization: diagonal (variance) map ---
    fig = plt.figure(figsize=(12, 8))
    # ----------------------
    # (1) Radial pO₂ map with circles
    ax = fig.add_subplot(2, 3, 1)
    
    theta = np.linspace(0, 2 * np.pi, 100) # angles
    circle_in_x = center_coordinates[0] + Rves*np.cos(theta)
    circle_in_y = center_coordinates[1] + Rves*np.sin(theta)
    circle_out_x = center_coordinates[0] + R0*np.cos(theta)
    circle_out_y = center_coordinates[1] + R0*np.sin(theta)
    ax.plot(circle_in_x, circle_in_y, "m-", lw=2, label=f"Inner r={Rves:.1f}")
    ax.plot(circle_out_x, circle_out_y, "c--", lw=2, label=f"Outer r={R0:.1f}")
    ax.plot(center_coordinates[0], center_coordinates[1], "kx", ms=8, label="Center")

    c = ax.pcolormesh(X, Y, pO2_array, shading='auto', cmap='jet')
    fig.colorbar(c, label='pO₂')
    ax.set_title(f"Radial pO₂ Map | Arteriole {art_id} | Depth {dth_id}")
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.axis('equal')
    ax.legend()
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
    ax.legend()



    # ----------------------
    # EnKF steps
    enkf.predict()
    enkf.update(obs, X, Y)

    # Get current estimate
    mean, cov = enkf.get_state_estimate()

    # Means and Covariances
    cmro2_mean  = mean[0] * cmro2_by_M
    cmro2_cov   = cov * (cmro2_by_M)**2
    correction = np.abs(np.mean(enkf.length_scale * enkf.K @ enkf.innovation) * cmro2_by_M)
    p_vessel_est.append(p_vessel)

    # Compute the absolute error
    print(f'CMRO_2: {cmro2_mean}')
    print(f"p_vessel: {p_vessel}")
    print(f'Rves: {Rves}')
    print(f'R0: {R0}')
    generator_enkf = MapGenerator(cmro2=cmro2_mean, 
                        pvessel=p_vessel, 
                        Rves=Rves, 
                        R0=R0, 
                        Rt=R0,
                        X=X,
                        Y=Y)
    error_enkf_relative = np.abs(obs - generator_enkf.pO2_array.flatten()) * 100 / np.abs(obs) 
    error_enkf_absolute = np.abs(obs - generator_enkf.pO2_array.flatten())


    # ----------------------
    # (3) Relative error 3D surface
    ax = fig.add_subplot(2, 3, 3, projection='3d')

    error_enkf_relative_array = error_enkf_relative.reshape((grid_size, grid_size), order='F')
    sc = ax.plot_surface(X, Y, error_enkf_relative_array, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('pO2 (mmHg)')
    ax.set_title(f"Map of relative error, \nmean relative error:{error_enkf_relative.mean():.2f}; \nCMRO2:{cmro2_mean:.2f} umol /cm^3 /min")
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
    ax.set_title(f"Display of the approximated map: {cmro2_mean:.2f} umol /cm^3 /min")
    plt.colorbar(sc, ax=ax, shrink=0.3, aspect=10, label='Color scale: Relative Error (%)')
    plt.tight_layout()
    plt.show()


    # ----------------------
    # Print the results
    print(f"Correction: {correction}")
    print(f"Mean Relative Error: {error_enkf_relative.mean()}")   
    print(f"Mean Absolute Error: {error_enkf_absolute.mean()}") 

    # Results tracking overall iterations
    state_ensembles_overall.append(enkf.ensemble.copy())
    stats_overall.append((cmro2_mean, cmro2_cov))
    corrections_overall.append(correction) # Save the correction term

    cmro2_est_enkf.append(cmro2_mean)
    cmro2_cov_est_enkf.append(cmro2_cov)
    state_ensembles.append(enkf.ensemble.copy()) # Save the ensemble distribution for uncertainty quatitfication
    errors_enkf_relative.append(np.abs(error_enkf_relative)) # Save the relative errors
    errors_enkf_absolute.append(np.abs(error_enkf_absolute)) # Save the absolute errors
    corrections.append(correction) # Save the correction term

    # Print results in the terminal
    print(f"\n\n Ensemble Kalman Filter paramaters estimation")
    print("-"*65)
    print(f"Observation ID: {art_id}, Depth ID: {dth_id}")
    print(f"\nCMRO2 Mean: {cmro2_mean}, Rves: {Rves}, R0: {R0}, P_vessel: {p_vessel}, CMRO2 √(Cov): {np.sqrt(cmro2_cov)}\n")
    print(f"Mean Relative Error: {error_enkf_relative.mean()}")
    print(f"Mean Absolute Error: {error_enkf_absolute.mean()}")
    print(f"Correction: {correction}")

cmro2_est_enkf = np.array(cmro2_est_enkf)
cmro2_cov_est_enkf = np.array(cmro2_cov_est_enkf)
p_vessel_est = np.array(p_vessel_est)
errors_enkf_relative = np.array(errors_enkf_relative)
errors_enkf_absolute = np.array(errors_enkf_absolute)
state_ensembles = np.array(state_ensembles)
stats_overall = np.array(stats_overall)
corrections = np.array(corrections)
corrections_overall = np.array(corrections_overall)
state_ensembles_overall = np.array(state_ensembles_overall)



# ------------------------------------------------------------------
# ----------------------+ Plots the results +----------------------#
x_obs = np.arange(1, len(observations_id) + 1) # Simulated iteration steps

# -----------------------
# CMRO_2 Stats for converged iterations
state_ensembles = np.squeeze(state_ensembles, 1) # remove a dimension
# Apply cmro2_by_M across the last dimension (broadcasting)
data = state_ensembles.T * cmro2_by_M # shape: (n_ensembles, n_iterations)
numBoxes = data.shape[1]  # now robust

names = [f'obs{i}' for i in range(1, numBoxes + 1)]

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
P.show()
# P.savefig('enkf_state_estimation_test.png', dpi=300, bbox_inches='tight')


# # -----------------------
# # CMRO_2 Stats for overall iterations
# # Apply cmro2_by_M across the last dimension (broadcasting)
# state_ensembles_overall = np.squeeze(state_ensembles_overall, 1) # remove a dimension
# data = state_ensembles_overall.T * cmro2_by_M # shape: (n_ensembles, n_iterations)
# numBoxes = data.shape[1]  # now robust
# x_obs = np.arange(1, numBoxes + 1)
# names = [f'obs{i}' for i in range(1, numBoxes + 1)]
# P.figure()
# bp = P.boxplot(data, labels=names)

# for i in range(numBoxes):
#     y = data[:, i]
#     x = np.random.normal(1+i, 0.04, size=len(y))
#     P.plot(x, y, 'r.', alpha=0.2)
# P.xlabel('$PO_{2}$ Map ID')
# P.ylabel('State value CMRO2 (umol /cm^3 /min)')
# P.title('EnKF State Estimation with Uncertainty')
# P.grid(True)
# P.show()

# -----------------------
# CMRO_2 Stats for overall for converged iterations
data = np.mean(state_ensembles.T * cmro2_by_M, axis=0)
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
# P_vessel Stats
data = p_vessel_est
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
# plt.savefig('enkf_uncertainty.png', dpi=300, bbox_inches='tight')

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
# P.savefig('enkf_relative_error.png', dpi=300, bbox_inches='tight')

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
# P.savefig('enkf_absolute_error.png', dpi=300, bbox_inches='tight')


# -----------------------
# Uncertainty associated to estimation
data = state_ensembles * cmro2_by_M
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
# plt.savefig('enkf_uncertainty.png', dpi=300, bbox_inches='tight')

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
# plt.savefig('enkf_corrections.png', dpi=300, bbox_inches='tight')

# # -----------------------
# # Corrections associated to estimation overall
# numBoxes = corrections_overall.shape[0]  # now robust
# x_obs = np.arange(1, numBoxes + 1)
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(x_obs, corrections_overall, marker='o', color='orange')
# # Labels and title
# plt.ylabel('Estimated CMRO2 Corrections (umol /cm^3 /min)')
# plt.xlabel('$PO_{2}$ Map ID')
# plt.title('EnKF Corrections "Kd" - Iterative Estimation')
# plt.grid(True)
# plt.xticks(x_obs, [f'Obs{i}' for i in x_obs])
# plt.show()

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
# plt.savefig('enkf_cmro2_estimation_and_uncertainty.png', dpi=300, bbox_inches='tight')

# -----------------------
# Posterior distribution through the iteration
# Sample data
data = np.array(state_ensembles.T) * cmro2_by_M
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
# plt.savefig('pdf_surface_plot.png', dpi=300, bbox_inches='tight')

# Save the data
# path = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/EnKF_real_data_iterative/"
# np.save(path + f"state_ensembles_{n_ensembles}.npy", state_ensembles)
# np.save(path + f"cmro2_means_{n_ensembles}.npy", cmro2_mean_)
# np.save(path + f"cmro2_covs_{n_ensembles}.npy", cmro2_cov_)