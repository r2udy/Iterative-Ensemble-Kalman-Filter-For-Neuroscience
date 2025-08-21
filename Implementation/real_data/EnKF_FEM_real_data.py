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

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from EnKF_FEM import EnKF, build_obs_covariance, build_obs_covariance_diagonal
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

cmro2_lower, cmro2_upper = 1.0, 3.0
cmro2_var = (cmro2_upper - cmro2_lower)**2 / 12
M_var = cmro2_var / cmro2_by_M**2 / 5
M_std = np.sqrt(cmro2_var) / cmro2_by_M # model uncertainty
obs_var_high = 15.**2
obs_var_low = 1.**2 # measurement uncertainty
acceptance_rate = 1.
count = 0
max_inner_iterations = 10

n = 20 # data size
pixel_size = 10

# Create coordinate grids in physical units (microns)
X, Y = np.meshgrid(np.arange(n), np.arange(n))
X = X * pixel_size
Y = Y * pixel_size

# --------------------------
# EnKF Parameters
seed = np.random.seed(1)
state_dim = 1
obs_dim = 400
n_ensembles = 100

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
means = []
covariances = []
errors_enkf = []
state_ensembles = []
state_ensembles_overall = []
stats_overall = []
corrections_overall = []
N_it = []
corrections = []

# --------------------------
# Simulate a sequence with observation for the uniform case
# --------------------------
for i, entry in enumerate(uniform_dataset):
    art_id = entry[0][0]
    dth_id = entry[0][1]

    angles_1 = entry[1]
    angles_2 = entry[2]

    min_radius = entry[3][0]
        
    # Observations
    obs = df_copy[(df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id)]['pO2Value'].tolist()[0]
    observations.append(obs)

    pO2_array = obs.reshape((n, n), order='F')
    # Find Geometric parameters such as Rves and R0
    analyzer = Po2Analyzer(pO2_array)
    analyzer.find_circles()
    Rves = analyzer.rin
    R0 = analyzer.rout
    p_vessel = analyzer.p_vessel
    center = analyzer.center


    # ---------- Target Cells -----------
    # Adjust the observation covariance matrix to account very uncertain measurement
    grid_size = 20
    origin = center
    angle_ranges = [angles_1, angles_2]
    min_radius = min_radius

    C_diag = build_obs_covariance_diagonal(
        grid_size=grid_size,
        origin=origin,
        angle_ranges=angle_ranges,
        min_radius=min_radius,
        obs_var_high=obs_var_high,
        obs_var_low=obs_var_low
    )
    R = C_diag
    enkf.set_observation_noise(R)

    for inner_iteration in range(max_inner_iterations):
        # ----------------------
        # EnKF steps
        enkf.predict()
        enkf.update(obs)

        # Get current estimate
        mean, cov = enkf.get_state_estimate()

        # Means and Covariances
        cmro2_mean  = mean[0] * cmro2_by_M
        cmro2_cov   = cov * (cmro2_by_M)**2
        correction = np.abs(np.mean(enkf.length_scale * enkf.K @ enkf.innovation) * cmro2_by_M)

        # Compute the absolute error
        generator_enkf = MapGenerator(cmro2=cmro2_mean, 
                            pvessel=p_vessel, 
                            Rves=Rves, 
                            R0=R0, 
                            Rt=R0)
        error_enkf = np.abs(obs - generator_enkf.pO2_array.flatten())

        # Print the results
        print(f"Correction: {correction}")
        print(f"Mean Absolute Error: {error_enkf.mean()}")    

        # Results tracking overall iterations
        state_ensembles_overall.append(enkf.ensemble.copy())
        stats_overall.append((cmro2_mean, cmro2_cov))
        corrections_overall.append(correction) # Save the correction term

        # If the samples are accepted
        if np.abs(correction) < acceptance_rate:
            count += 1

            B = B / 2
            enkf.set_process_noise(B)  # Reset the process noise for the next iteration
            acceptance_rate = acceptance_rate * 0.5
            enkf.length_scale = (.5)**(count)  # Set the length scale for the covariance matrix

            if count > 2:
                cmro2_est_enkf.append(cmro2_mean)
                cmro2_cov_est_enkf.append(cmro2_cov)
                state_ensembles.append(enkf.ensemble.copy()) # Save the ensemble distribution for uncertainty quatitfication
                errors_enkf.append(np.abs(error_enkf)) # Save the absolute errors
                corrections.append(correction) # Save the correction term
                break
                    
        elif inner_iteration == max_inner_iterations - 1:
            print(f"Inner iteration {inner_iteration + 1} reached maximum iterations without convergence.")
            cmro2_est_enkf.append(cmro2_mean)
            cmro2_cov_est_enkf.append(cmro2_cov)
            state_ensembles.append(enkf.ensemble.copy())

    
    
    N_it.append(count)
    count = 0
    # Reset the EnKF parameter
    enkf.length_scale = 1.0  # Reset the length scale for each observation
    B = np.array([[M_var]])  # Reset the background covariance matrix
    enkf.set_process_noise(B)

    # Print results in the terminal
    print(f"\n\n Ensemble Kalman Filter paramaters estimation")
    print("-"*65)
    print(f"Observation ID: {art_id}, Depth ID: {dth_id}")
    print(f"\nCMRO2 Mean: {cmro2_mean}, Rves: {Rves}, R0: {R0}, CMRO2 √(Cov): {np.sqrt(cmro2_cov)}\n")
    print(f"Mean Absolute Error: {error_enkf.mean()}")
    print(f"Correction: {correction}")

cmro2_est_enkf = np.array(cmro2_est_enkf)
cmro2_cov_est_enkf = np.array(cmro2_cov_est_enkf)
errors_enkf = np.array(errors_enkf)
state_ensembles = np.array(state_ensembles)
stats_overall = np.array(stats_overall)
corrections_overall = np.array(corrections_overall)
state_ensembles_overall = np.array(state_ensembles_overall)



# ------------------------------------------------------------------
# ----------------------+ Plots the results +----------------------#

# Simulated iteration steps
x_obs = np.arange(1, len(observations_id) + 1)

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


# -----------------------
# CMRO_2 Stats for overall iterations
# Apply cmro2_by_M across the last dimension (broadcasting)
state_ensembles_overall = np.squeeze(state_ensembles_overall, 1) # remove a dimension
data = state_ensembles_overall.T * cmro2_by_M # shape: (n_ensembles, n_iterations)
numBoxes = data.shape[1]  # now robust
x_obs = np.arange(1, numBoxes + 1)
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
# P.savefig('enkf_state_estimation_overall_test.png', dpi=300, bbox_inches='tight')


# -----------------------
# Absolute Error Stats
# Stats
data = errors_enkf.T # Define data
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
# P.savefig('enkf_absolute_error_test.png', dpi=300, bbox_inches='tight')

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
plt.xlabel('Id DataPoint')
plt.title('EnKF Uncertainty')
plt.grid(True)
plt.xticks(x_obs, [f'Obs{i}' for i in x_obs])
plt.axhline(y=np.mean(cov_track), color='r', linestyle='--', label='Mean Uncertainty')
plt.legend()
plt.tight_layout()
plt.show()
# plt.savefig('enkf_uncertainty_test.png', dpi=300, bbox_inches='tight')

# -----------------------
# Corrections associated to estimation
numBoxes = corrections.shape[0]  # now robust
x_obs = np.arange(1, numBoxes + 1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_obs, corrections, marker='o', color='orange')
# Labels and title
plt.ylabel('Estimated CMRO2 Corrections (umol /cm^3 /min)')
plt.xlabel('Id DataPoint')
plt.title('EnKF Corrections "Kd" - Iterative Estimation')
plt.grid(True)
plt.xticks(x_obs, [f'Obs{i}' for i in x_obs])
plt.show()
# plt.savefig('enkf_corrections_test.png', dpi=300, bbox_inches='tight')

# -----------------------
# Corrections associated to estimation overall
numBoxes = corrections_overall.shape[0]  # now robust
x_obs = np.arange(1, numBoxes + 1)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_obs, corrections_overall, marker='o', color='orange')
# Labels and title
plt.ylabel('Estimated CMRO2 Corrections (umol /cm^3 /min)')
plt.xlabel('Id DataPoint')
plt.title('EnKF Corrections "Kd" - Iterative Estimation')
plt.grid(True)
plt.xticks(x_obs, [f'Obs{i}' for i in x_obs])
plt.show()
# plt.savefig('enkf_corrections_test.png', dpi=300, bbox_inches='tight')

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
plt.xlabel('Id DataPoint')
plt.ylabel('CMRO2 (umol /cm^3 /min)')
plt.title('EnKF CMRO2 Estimation *OVERALL* with Uncertainty')
plt.xticks(x_obs, [f'Obs{i}' for i in x_obs])
plt.axhline(y=np.mean(cmro2_mean_), color='r', linestyle='--', label='Mean CMRO2')
plt.axhline(y=cmro2_lower, color='orange', linestyle='--', label='CMRO2 Lower Bound (Prior)')
plt.axhline(y=cmro2_upper, color='orange', linestyle='--', label='CMRO2 Upper Bound (Prior)')
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig('enkf_cmro2_estimation_overall_test.png', dpi=300, bbox_inches='tight')

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
# plt.savefig('pdf_surface_plot_test.png', dpi=300, bbox_inches='tight')

# Save the data
# path = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/EnKF_real_data_iterative/"
# np.save(path + f"state_ensembles_{n_ensembles}.npy", state_ensembles)
# np.save(path + f"cmro2_means_{n_ensembles}.npy", cmro2_mean_)
# np.save(path + f"cmro2_covs_{n_ensembles}.npy", cmro2_cov_)