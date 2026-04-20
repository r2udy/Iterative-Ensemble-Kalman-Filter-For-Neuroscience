#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 11:36:47 2025

@author: ruudybayonne
"""

import sys
import os

py_data_location = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/"
ROOT = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/"
CLASSES = os.path.join(ROOT, "classes")
IMPL = os.path.join(ROOT, "Implementation")
CLUSTERING = os.path.join(IMPL, "Clustering")

# Add paths to Python import system
print(">>> Added to sys.path:")
sys.path.append(ROOT)
sys.path.append(CLASSES)
sys.path.append(IMPL)
sys.path.append(CLUSTERING)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from EnKF_FEM_3 import EnKF
from FEM_code.generateMesh_Solver_multiple_holes import DiffusionSolver, SolverParameters, HoleGeometry

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
# --
file_id_saving = "constant_01_aug"  # ID for saving the data


# --------------------------
# Constants initial #   
D = 4.0e3
alpha = 1.39e-15
cmro2_by_M = (60 * D * alpha * 1e12)
grid_size = 20 # data size

cmro2_mean_inital = 2.0  # initial mean
cmro2_var_initial = .5**2 
cmro2_var = .5**2  # Uniform distribution variance
M_var = cmro2_var / cmro2_by_M**2  # model uncertainty scaled

# Prior associated with R0
# R0_lower, R0_upper = 70., 100.
R0_mean_initial = 100.
R0_var_initial = 2.**2
R0_var = 2.**2 # prior uncertainty of capillary-free space radius

# Prior associated with pvessel
# pvessel_lower, pvessel_upper = 85., 95.
pvessel_mean_initial = 80.
pvessel_var_initial = 5.**2 
pvessel_var = 5.**2 # prior uncertainty of Neumann boundary condition

# Observation variance parameters
obs_var_constant = 20.**2   # constant uncertainty of measurements in the observation covariance matrix R
sigma = 2.0  # Gaussian filter sigma for observation noise smoothing
obs_var_high = 5.**2    # high uncertainty of measurements
obs_var_low = 1.**2     # low uncertainty of measurements


X_axis, Y_axis = np.meshgrid(np.linspace(-190, 190, 20), np.linspace(-190, 190, 20))

# --------------------------
# EnKF Parameters
seed = np.random.seed(1)
state_dim = 3
obs_dim = 400
n_ensembles = 50
iterations_max = 10
cmro2_lower, cmro2_upper = 1.0, 3.0
cmro2_true_values = np.linspace(cmro2_lower, cmro2_upper, 5) # True CMRO2 values for synthetic data generation


# Initialize the ensemble
a = np.array([cmro2_mean_inital / cmro2_by_M,       R0_mean_initial, pvessel_mean_initial, 
              cmro2_mean_inital / cmro2_by_M,       R0_mean_initial, pvessel_mean_initial,
              cmro2_mean_inital / cmro2_by_M,       R0_mean_initial, pvessel_mean_initial])

b = np.array([cmro2_var_initial / (cmro2_by_M)**2,  R0_var_initial,  pvessel_var_initial,
              cmro2_var_initial / (cmro2_by_M)**2,  R0_var_initial,  pvessel_var_initial,
              cmro2_var_initial / (cmro2_by_M)**2,  R0_var_initial,  pvessel_var_initial])

# No dynamic model
def dynamics_model(x):
    return x

# -------------------------
# Create the EnKF object
enkf = EnKF(state_dim, obs_dim, n_ensembles, dynamics_model, seed)

# Initialize the EnKF method
enkf.initialize_ensemble(a, b)
    
# Update the the background noise
B   = np.array([[M_var,    0.0,        0.0],
                [0.0,      R0_var,     0.0],
                [0.0,   0.0,           pvessel_var]])   # Background covariance matrix
enkf.set_process_noise(B)

# Initialize lists to store meaningful quantities
state_ensembles = np.zeros((state_dim, len(cmro2_true_values), n_ensembles, iterations_max))  # To store state ensembles for all iterations
observations = np.zeros((obs_dim, len(cmro2_true_values), iterations_max))  # Store observations for all iterations
stats_overall = np.zeros((2, len(cmro2_true_values), iterations_max))  # To store mean and covariance of CMRO2 estimates for all iterations
errors_enkf_relative = np.zeros((obs_dim, len(cmro2_true_values), iterations_max))  # To store relative errors for all iterations
errors_enkf_absolute = np.zeros((obs_dim, len(cmro2_true_values), iterations_max))  # To store absolute errors for all iterations
corrections = np.zeros((len(cmro2_true_values), iterations_max))  # To store corrections for all iterations
cmro2_est_enkf = np.zeros((len(cmro2_true_values), iterations_max))
cmro2_cov_est_enkf = np.zeros((len(cmro2_true_values), iterations_max))
R0_est_enkf = np.zeros((len(cmro2_true_values), iterations_max))
R0_cov_est_enkf = np.zeros((len(cmro2_true_values), iterations_max))
pvessel_est_enkf = np.zeros((len(cmro2_true_values), iterations_max))
pvessel_cov_est_enkf = np.zeros((len(cmro2_true_values), iterations_max))

# ------------------+ Synthetic Data +------------------ #
for i, cmro2_true in enumerate(cmro2_true_values):
    
    # Hole 1:
    cmro2_1     = 2.0
    Pves_1      = 80.
    Rves_1      = 17.
    R0_1        = 100.
    center_1    = (0., 0., 0.)

    # Hole 2:
    cmro2_2     = .5
    Pves_2      = Pves_1 * 0.8
    Rves_2      = 17.
    R0_2        = 80.
    position_x  = np.random.uniform(-190, -130)
    position_y  = np.random.uniform(-190, -130)
    center_2    = (position_x, position_y, 0.)

    # Hole 3:
    cmro2_3     = .5
    Pves_3      = Pves_1 * 0.8
    Rves_3      = 17.
    R0_3        = 80.
    position_x  = np.random.uniform(190, 130)
    position_y  = np.random.uniform(-190, 190)
    center_3    = (position_x, position_y, 0.)

    # Create solver parameters
    params = SolverParameters(filename="square_holes")

    # Define holes
    holes1 = [
        HoleGeometry(center=center_1, cmro2=cmro2_1, Pves=Pves_1, radius_ves=Rves_1, radius_0=R0_1, marker=params.marker),
        ]
    
    # holes2 = [
    #     HoleGeometry(center=center_1, cmro2=cmro2_1, Pves=Pves_1, radius_ves=Rves_1, radius_0=R0_1, marker=params.marker),
    #     HoleGeometry(center=center_2, cmro2=cmro2_2, Pves=Pves_2, radius_ves=Rves_2, radius_0=R0_2, marker=params.marker + 1),
    #     ]

    # holes3 = [
    #     HoleGeometry(center=center_1, cmro2=cmro2_1, Pves=Pves_1, radius_ves=Rves_1, radius_0=R0_1, marker=params.marker),
    #     HoleGeometry(center=center_2, cmro2=cmro2_2, Pves=Pves_2, radius_ves=Rves_2, radius_0=R0_2, marker=params.marker + 1),
    #     HoleGeometry(center=center_3, cmro2=cmro2_3, Pves=Pves_3, radius_ves=Rves_3, radius_0=R0_3, marker=params.marker + 2)
    #     ]
    
    generator = MapGenerator(
        holes=holes1,
        params=params,
        X=X_axis,
        Y=Y_axis)
    profile = generator.pO2_array
    
    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(projection='3d')
    # sc = ax.plot_surface(X_axis, Y_axis, obs_perturbated_array, cmap='viridis', edgecolor='none')
    # ax.set_xlabel('X (nm)')
    # ax.set_ylabel('Y (nm)')
    # ax.set_zlabel('pO2 (mmHg)')
    # plt.colorbar(sc, ax=ax, shrink=0.3, aspect=10, label='pO2 (mmHg)')
    # ax.set_title(f'Synthetic pO2 Data with Noise')
    # plt.show()

    # Find Geometric parameters such as Rves and R0
    center = (0.0, 0.0)
    
    # ----------------------+ EnKF +----------------------#
    R = obs_var_constant * np.eye(obs_dim) # Observation covariance matrix

    # Update the the background and observation noise
    enkf.set_observation_noise(R)
    
    for j in range(iterations_max):

        obs_perturbated = profile.flatten() + np.random.normal(np.zeros(grid_size*grid_size), scale=sigma)
        obs_perturbated_array = obs_perturbated.reshape((grid_size, grid_size), order='F')

        # Simulate a sequence with observation for the uniform case
        # Observations
        obs = obs_perturbated
        observations[:, i, j] = obs

        # EnKF steps
        enkf.predict()
        enkf.update(obs, X_axis, Y_axis)
    
        # Get current estimate
        mean, cov = enkf.get_state_estimate()


        cmro2_mean = mean[0] * cmro2_by_M
        R0_mean = mean[1]
        pvessel_mean = mean[2]

        cmro2_cov = cov[0, 0] * (cmro2_by_M**2)
        R0_cov = cov[1, 1]
        pvessel_cov = cov[2, 2]

        correction = np.abs(np.mean(enkf.length_scale * enkf.K @ enkf.innovation))
    
        params = SolverParameters(filename="square_holes")
        hole_estimated = [HoleGeometry(center=(*center, 0.0), cmro2=cmro2_mean, Pves=pvessel_mean, radius_ves=Rves_1, radius_0=R0_mean)]

        generator_enkf = MapGenerator(
                            holes=hole_estimated,
                            params=params,
                            X=X_axis,
                            Y=Y_axis)
        obs_estimation = generator_enkf.pO2_array

        # fig = plt.figure(figsize=(12, 8))
        # ax = fig.add_subplot(projection='3d')
        # sc = ax.plot_surface(X_axis, Y_axis, obs_perturbated_array, cmap='viridis', edgecolor='none')
        # ax.plot_surface(X_axis, Y_axis, obs_estimation, cmap='plasma', alpha=0.6, edgecolor='none')
        # ax.set_xlabel('X (nm)')
        # ax.set_ylabel('Y (nm)')
        # ax.set_zlabel('pO2 (mmHg)')
        # plt.colorbar(sc, ax=ax, shrink=0.3, aspect=10, label='pO2 (mmHg)')
        # ax.set_title(f'Synthetic pO2 Data with Noise')
        # plt.show()

        error_enkf_relative = np.abs(profile.flatten() - obs_estimation.flatten()) * 100 / np.abs(obs) 
        error_enkf_absolute = np.abs(profile.flatten() - obs_estimation.flatten())

        # Save stats
        stats_overall[0, i, j] = cmro2_mean
        stats_overall[1, i, j] = cmro2_cov

        errors_enkf_relative[:, i, j] = np.abs(error_enkf_relative) # Save the relative errors
        errors_enkf_absolute[:, i, j] = np.abs(error_enkf_absolute) # Save the absolute errors
        
        corrections[i, j] = correction

        cmro2_est_enkf[i, j] = cmro2_mean
        cmro2_cov_est_enkf[i, j] = cmro2_cov
        R0_est_enkf[i, j] = R0_mean
        R0_cov_est_enkf[i, j] = R0_cov
        pvessel_est_enkf[i, j] = pvessel_mean
        pvessel_cov_est_enkf[i, j] = pvessel_cov
        state_ensembles[0, i, :, j] = enkf.ensemble[0, :]  # CMRO2 ensembles
        state_ensembles[1, i, :, j] = enkf.ensemble[1, :]  # R0 ensembles
        state_ensembles[2, i, :, j] = enkf.ensemble[2, :]  # pvessel ensembles

    print("\n\n Ensemble Kalman Filter paramaters estimation:")
    cmro2_displayed_mean    = cmro2_est_enkf[i, -1]
    cmro2_displayed_cov     = cmro2_cov_est_enkf[i, -1]
    R0_displayed_mean       = R0_est_enkf[i, -1]
    R0_displayed_cov        = R0_cov_est_enkf[i, -1]
    pvessel_displayed_mean  = pvessel_est_enkf[i, -1]
    pvessel_displayed_cov   = pvessel_cov_est_enkf[i, -1]
    print("-"*65)
    print(f"\nCMRO2 Mean: {cmro2_displayed_mean}, CMRO2 √(Cov): {np.sqrt(cmro2_displayed_cov)}, cmro2 Covariance: {cmro2_displayed_cov}")
    print("-"*25)
    print(f"R0 Mean: {R0_displayed_mean}, R0 √(Cov): {np.sqrt(R0_displayed_cov)}, R0 Covariance: {R0_displayed_cov}")
    print("-"*25)
    print(f"pvessel Mean: {pvessel_displayed_mean}, pvessel √(Cov): {np.sqrt(pvessel_displayed_cov)}, pvessel Covariance: {pvessel_displayed_cov}\n")

# Turn list into numpy arrays
cmro2_est_enkf = np.array(cmro2_est_enkf)
cmro2_cov_est_enkf = np.array(cmro2_cov_est_enkf)
R0_est_enkf = np.array(R0_est_enkf)
R0_cov_est_enkf = np.array(R0_cov_est_enkf)
pvessel_est_enkf = np.array(pvessel_est_enkf)
pvessel_cov_est_enkf = np.array(pvessel_cov_est_enkf)
corrections =  np.array(corrections)
errors_enkf_relative = np.array(errors_enkf_relative)
errors_enkf_absolute = np.array(errors_enkf_absolute)
observations = np.array(observations)
stats_overall = np.array(stats_overall)


# Path for saving the data
path = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/" + file_id_saving + "/"

# --------- Plots the results ---------
# -----------------------
# CMRO2 Stats
# -----------------------
observations_id = [i for i in range(1, observations.shape[1] + 1)]
observations = np.array(observations[:, :, -1])  # Get the last iteration observations for plotting
x = np.arange(1, len(observations_id) + 1)

# Stats

data = state_ensembles[0, :, :, -1].T * cmro2_by_M # Define data
numBoxes = observations.shape[1] # Define numBoxes
names = [f' cmro2 {e}' for e in cmro2_true_values]

P.figure()
bp = P.boxplot(data, labels=names)
for i in range(numBoxes):
    y = data[:, i].T
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value CMRO2 (umol /cm^3 /min)')
P.title('EnKF State Estimation with Uncertainty')
P.grid(True)
P.savefig(path + 'enkf_state_estimation_test.png', dpi=300, bbox_inches='tight')
P.show()


# Simulated time steps
x = cmro2_true_values  
# Create figure
plt.figure(figsize=(10, 6))
# Plot mean +/- 1 standard deviation (sqrt of variance)
plt.plot(x, cmro2_true_values, '-x', color='black', label='True paramter (CMRO2)')
plt.plot(x, cmro2_est_enkf[:, -1], '-x', color='green', label='State EnKF estimate (CMRO2)')
plt.fill_between(
    x,
    cmro2_est_enkf[:, -1] - np.sqrt(cmro2_cov_est_enkf[:, -1]),  # Lower bound (mean - σ)
    cmro2_est_enkf[:, -1] + np.sqrt(cmro2_cov_est_enkf[:, -1]),  # Upper bound (mean + σ)
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
plt.savefig(path + 'enkf_state_estimation_time_steps.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------
# R0 Stats
# -----------------------
data        = state_ensembles[1, :, :, -1].T # Define data
numBoxes    = observations.shape[1] # Define numBoxes
names = [f' cmro2 {e}' for e in cmro2_true_values]

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
P.savefig(path + 'enkf_R0_state_estimation.png', dpi=300, bbox_inches='tight')
P.show()


# -----------------------
# pvessel Stats
# -----------------------
data        = state_ensembles[2, :, :, -1].T # Define data
numBoxes    = observations.shape[1] # Define numBoxes
names = [f' cmro2 {e}' for e in cmro2_true_values]

P.figure()
bp = P.boxplot(data, labels=names)
for i in range(numBoxes):
    y = data[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State values pvessel (mmHg)')
P.title('EnKF Partial Pressure at the vessel wall\n State Estimation with Uncertainty')
P.grid(True)
P.savefig(path + 'enkf_pvessel_state_estimation.png', dpi=300, bbox_inches='tight')
P.show()


# -----------------------
# Relative Error Stats
# -----------------------
data        = errors_enkf_relative[:, :, -1] # Define data
numBoxes    = data.shape[1]  # now robust
names = [f' cmro2 {e}' for e in cmro2_true_values]

P.figure()
bp = P.boxplot(data, labels=names)
for i in range(numBoxes):
    y = data[:, i].T
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('Relative Partial Pressure Error')
P.title('Relative Errors distributions - EnKF')
P.grid(True)
P.savefig(path + 'enkf_relative_error.png', dpi=300, bbox_inches='tight')
P.show()

# -----------------------
# Absolute Error Stats
# -----------------------
data        = errors_enkf_absolute[:, :, -1] # Define data
numBoxes    = data.shape[1]  # now robust
names = [f' cmro2 {e}' for e in cmro2_true_values]

P.figure()
bp = P.boxplot(data, labels=names)
for i in range(numBoxes):
    y = data[:, i].T
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('Absolute Partial Pressure Error')
P.title('Absolute Errors distributions - EnKF')
P.grid(True)
P.savefig(path + 'enkf_absolute_error.png', dpi=300, bbox_inches='tight')
P.show()

# --------------------------------------
# Error of the mean estimation
# ---------------------------------------
data        = state_ensembles[0, :, :, -1] * cmro2_by_M # Define data
numBoxes    = data.shape[0]  # now robust
x_obs       = cmro2_true_values

plt.figure(figsize=(10, 6))
cmro2_mean_ = stats_overall[0, :, -1]
error_abs_cmro2 = np.abs(cmro2_true_values - cmro2_mean_)
plt.plot(x_obs, error_abs_cmro2, '-o', color='blue', label='Absolute Error in CMRO2 estimation')
plt.xlabel('$PO_{2}$ Map ID')
plt.ylabel('Absolute Error in CMRO2 (umol /cm^3 /min)')
plt.title('Absolute Error of CMRO2 Estimation from Synthetic Data')
plt.xticks(x_obs, names)
plt.legend()
plt.grid(True)
plt.savefig(path + 'enkf_cmro2_abs_error.png', dpi=300, bbox_inches='tight')
plt.show()

# ----------------------
# Uncertainty associated to estimation
# -----------------------

fig, ax = plt.subplots(figsize=(10, 6))
cov_track = np.array([np.std(array) for array in data])
ax.plot(x_obs, cov_track, '-o', color='blue', label='Uncertainty in CMRO2 estimation (StD)')
plt.ylabel('Estimated CMRO2 Uncertainty (umol /cm^3 /min)')
plt.xlabel('$PO_{2}$ Map ID')
plt.title('EnKF Uncertainty')
plt.grid(True)
plt.xticks(x_obs, [f'Obs{i}' for i in x_obs])
plt.legend()
plt.tight_layout()
plt.savefig(path + 'enkf_uncertainty.png', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------
# Uncertainty associated to estimation
# -----------------------
# Create figure
plt.figure(figsize=(10, 6))
cmro2_mean_ = stats_overall[0, :, -1]
cmro2_cov_ = stats_overall[1, :, -1]
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
plt.legend()
plt.grid(True)
plt.savefig(path + 'enkf_cmro2_estimation_and_uncertainty.png', dpi=300, bbox_inches='tight')
plt.show()


# Save the data
np.save(path + f"state_ensembles_{n_ensembles}.npy", state_ensembles)
np.save(path + f"cmro2_means_{n_ensembles}.npy",stats_overall[:, 0])
np.save(path + f"cmro2_covs_{n_ensembles}.npy", stats_overall[:, 1])
np.save(path + f"R0_means_{n_ensembles}.npy", R0_est_enkf)
np.save(path + f"R0_covs_{n_ensembles}.npy", R0_cov_est_enkf)
np.save(path + f"pvessel_means_{n_ensembles}.npy", pvessel_est_enkf)
np.save(path + f"pvessel_covs_{n_ensembles}.npy", pvessel_cov_est_enkf)
np.save(path + f"errors_enkf_relative_{n_ensembles}.npy", errors_enkf_relative)
np.save(path + f"errors_enkf_absolute_{n_ensembles}.npy", errors_enkf_absolute)
