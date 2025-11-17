#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 11:36:47 2025

@author: ruudybayonne
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
from scipy.ndimage import gaussian_filter
from EnKF_FEM import build_obs_covariance_radial, build_obs_covariance_diagonal
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
# --
file_id_saving = "2states_varying_alternative_source_02"  # ID for saving the data

# --------------------------
# Constants initial #
D = 4.0e3
alpha = 1.39e-15
cmro2_by_M = (60 * D * alpha * 1e12)
grid_size = 20 # data size

cmro2_lower, cmro2_upper = 1.0, 3.0
R0_lower, R0_upper = 80., 120.
cmro2_var = (cmro2_upper - cmro2_lower)**2 / 12 # variance of uniform distribution
M_var = cmro2_var / cmro2_by_M**2 # model uncertainty scaled
R0_var = 5.0**2 # prior uncertainty of caparilary-free space radius
obs_var_constant = 5.**2   # constant uncertainty of measurements in the observation covariance matrix R
sigma = 2.0  # noise level in synthetic data
obs_var_high = 5.**2    # high uncertainty of measurements
obs_var_low = 1.**2     # low uncertainty of measurements

# Grid configuration
# mask = (df_copy["arteriole_id"] == 2) & (df_copy['depth_id'] == 3)
# X = df_copy[mask]['pointsX'].tolist()[0]
# Y = df_copy[mask]['pointsY'].tolist()[0]

X, Y = np.meshgrid(np.linspace(-190, 190, 20), np.linspace(-190, 190, 20))

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

# Initialize lists to store meaningful quantities
observations = []
cmro2_est_enkf = []
cmro2_cov_est_enkf = []
R0_est_enkf = []
R0_cov_est_enkf = []
pvessel_est_enkf = []
pvessel_cov_est_enkf = []
state_ensembles = []
errors_enkf_relative = []
errors_enkf_absolute = []
means = []
covs = []
stats_overall = []
corrections = []
cmro2_true_values = np.linspace(1.0, 3.0, 3)
state_ensembles = np.zeros((state_dim, len(cmro2_true_values), n_ensembles))


# ------------------+ Synthetic Data +------------------ #
for i, cmro2_true in enumerate(cmro2_true_values):
    
    # First vessel
    pvessel = 80.0
    M = cmro2_true / cmro2_by_M
    Rves = 11.
    R0=100.
    Rt=100.

    generator = MapGenerator(
        cmro2=cmro2_true,
        pvessel=pvessel,
        Rves=Rves,
        R0=R0,
        Rt=Rt,
        X=X,
        Y=Y)
    profile_main = generator.pO2_array.reshape((grid_size, grid_size), order='F')

    # Second vessel
    pvessel2 = (profile_main.max() - profile_main.min())
    center_secondary = (-150.0, -150.0)
    generator2 = MapGenerator(
        cmro2=2.,
        pvessel=pvessel2,
        Rves=10.,
        R0=60.,
        Rt=60.,
        X=X,
        Y=Y,
        center=center_secondary)
    profile_secondary = generator2.pO2_array.reshape((grid_size, grid_size), order='F')
    profile_secondary = profile_secondary - profile_secondary.min()
    profile_secondary[profile_secondary < 0.0] = 0.0
    profile_secondary = gaussian_filter(profile_secondary.flatten(), sigma=2.).reshape((grid_size, grid_size), order='F')

    # Add noise to the generated data
    true_obs = profile_main + profile_secondary
    offset_random = np.random.normal(0, 5.) * np.ones(true_obs.flatten().shape[0])
    obs_perturbated = true_obs.flatten() + np.random.normal(np.zeros(true_obs.flatten().shape[0]), scale=sigma) + offset_random
    obs_perturbated_array = obs_perturbated.reshape((grid_size, grid_size), order='F')

    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(projection='3d')
    # sc = ax.plot_surface(X, Y, obs_perturbated_array, cmap='viridis', edgecolor='none')
    # ax.plot_surface(X, Y, true_obs, cmap='plasma', alpha=0.6, edgecolor='none')
    # ax.plot_surface(X, Y, profile_secondary, cmap='viridis', edgecolor='none')
    # ax.set_xlabel('X (nm)')
    # ax.set_ylabel('Y (nm)')
    # ax.set_zlabel('pO2 (mmHg)')
    # plt.colorbar(sc, ax=ax, shrink=0.3, aspect=10, label='pO2 (mmHg)')
    # ax.set_title(f'Synthetic pO2 Data with Noise\n offset: {offset_random[0]}')
    # plt.show()

    # Find Geometric parameters such as Rves and R0
    analyzer = Po2Analyzer(obs_perturbated_array, X, Y)
    analyzer.find_circles()
    Rves_est = analyzer.rin
    pvessel_est = analyzer.p_vessel
    center = analyzer.center_ij
    center_coordinates = analyzer.center
    
    # ----------------------+ EnKF +----------------------#
    R = build_obs_covariance_radial(
    origin = center,
    obs_var_high = obs_var_high,
    obs_var_low = obs_var_low,
    mode='exponential'
    )

    # R = build_obs_covariance_diagonal(
    #     grid_size=grid_size,
    #     origin=center,
    #     angle_ranges=[angles_1, angles_2],
    #     min_radius=min_radius,
    #     obs_var_high=obs_var_high,
    #     obs_var_low=obs_var_low
    # )

    # R = obs_var_constant * np.eye(obs_dim) # Observation covariance matrix

    # Update the the background and observation noise
    enkf.set_observation_noise(R)
    
    # Simulate a sequence with observation for the uniform case
    # Observations
    obs = obs_perturbated
    observations.append(obs)

    # EnKF steps
    enkf.predict()
    enkf.update(obs, X, Y)
    
    # Get current estimate
    mean, cov = enkf.get_state_estimate()

    cmro2_mean = mean[0] * cmro2_by_M
    R0_mean = mean[1]

    cmro2_cov = cov[0, 0] * (cmro2_by_M**2)
    R0_cov = cov[1, 1]

    correction = np.abs(np.mean(enkf.length_scale * enkf.K @ enkf.innovation))
    
    generator_enkf = MapGenerator(cmro2=cmro2_mean, 
                        pvessel=pvessel, 
                        Rves=Rves, 
                        R0=R0_mean, 
                        Rt=R0_mean,
                        X=X,
                        Y=Y)
    obs_estimation = generator_enkf.pO2_array

    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(projection='3d')
    # sc = ax.plot_surface(X, Y, obs_perturbated_array, cmap='viridis', edgecolor='none')
    # ax.plot_surface(X, Y, obs_estimation, cmap='plasma', alpha=0.6, edgecolor='none')
    # ax.set_xlabel('X (nm)')
    # ax.set_ylabel('Y (nm)')
    # ax.set_zlabel('pO2 (mmHg)')
    # plt.colorbar(sc, ax=ax, shrink=0.3, aspect=10, label='pO2 (mmHg)')
    # ax.set_title(f'Synthetic pO2 Data with Noise')
    # plt.show()

    error_enkf_relative = np.abs(true_obs.flatten() - obs_estimation.flatten()) * 100 / np.abs(obs) 
    error_enkf_absolute = np.abs(true_obs.flatten() - obs_estimation.flatten())

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

    print("\n\n Ensemble Kalman Filter paramaters estimation:")
    print("-"*65)
    print(f"\nCMRO2 Mean: {cmro2_mean}, CMRO2 √(Cov): {np.sqrt(cmro2_cov)}, cmro2 Covariance: {cmro2_cov}")
    print("-"*25)
    print(f"R0 Mean: {R0_mean}, R0 √(Cov): {np.sqrt(R0_cov)}, R0 Covariance: {R0_cov}\n")

cmro2_est_enkf = np.array(cmro2_est_enkf)
cmro2_cov_est_enkf = np.array(cmro2_cov_est_enkf)
R0_est_enkf = np.array(R0_est_enkf)
R0_cov_est_enkf = np.array(R0_cov_est_enkf)
pvessel_est_enkf = np.array(pvessel_est_enkf)
corrections =  np.array(corrections)
errors_enkf_relative = np.array(errors_enkf_relative)
errors_enkf_absolute = np.array(errors_enkf_absolute)
observations = np.array(observations)
stats_overall = np.array(stats_overall)



# Path for saving the data
path = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/" + file_id_saving + "/"

# --------- Plots the results ---------
# Simulated iteration steps
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
P.savefig(path + 'enkf_state_estimation_test.png', dpi=300, bbox_inches='tight')
# P.show()


# Simulated time steps
x = cmro2_true_values  
# Create figure
plt.figure(figsize=(10, 6))
# Plot mean +/- 1 standard deviation (sqrt of variance)
plt.plot(x, cmro2_true_values, '-x', color='black', label='True paramter (CMRO2)')
plt.plot(x, cmro2_est_enkf, '-x', color='green', label='State EnKF estimate (CMRO2)')
plt.fill_between(
    x,
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
plt.savefig(path + 'enkf_state_estimation_time_steps.png', dpi=300, bbox_inches='tight')
# plt.show()

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
P.savefig(path + 'enkf_R0_state_estimation.png', dpi=300, bbox_inches='tight')
# P.show()

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
P.savefig(path + 'enkf_relative_error.png', dpi=300, bbox_inches='tight')
# P.show()

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
P.savefig(path + 'enkf_absolute_error.png', dpi=300, bbox_inches='tight')
# P.show()

# --------------------------------------
# Error of the mean estimation
# ---------------------------------------
data = state_ensembles[0] * cmro2_by_M # Define data
numBoxes = data.shape[0] 
x_obs = np.arange(1, numBoxes + 1)

plt.figure(figsize=(10, 6))
cmro2_mean_ = stats_overall[:, 0]
error_abs_cmro2 = np.abs(cmro2_true_values - cmro2_mean_)
plt.plot(x_obs, error_abs_cmro2, '-o', color='blue', label='Absolute Error in CMRO2 estimation')
plt.xlabel('$PO_{2}$ Map ID')
plt.ylabel('Absolute Error in CMRO2 (umol /cm^3 /min)')
plt.title('Absolute Error of CMRO2 Estimation from Synthetic Data')
plt.xticks(x_obs, [f'Obs{i}' for i in x_obs])
plt.legend()
plt.grid(True)
plt.savefig(path + 'enkf_cmro2_abs_error.png', dpi=300, bbox_inches='tight')
# plt.show()

# -----------------------
# Uncertainty associated to estimation
data = state_ensembles[0] * cmro2_by_M # Define data
# -----------------------
numBoxes = data.shape[0]  # now robust
x_obs = np.arange(1, numBoxes + 1)

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
plt.savefig(path + 'enkf_cmro2_estimation_and_uncertainty.png', dpi=300, bbox_inches='tight')
# plt.show()


# Save the data
np.save(path + f"state_ensembles_{n_ensembles}.npy", state_ensembles)
np.save(path + f"cmro2_means_{n_ensembles}.npy",stats_overall[:, 0])
np.save(path + f"cmro2_covs_{n_ensembles}.npy", stats_overall[:, 1])
np.save(path + f"R0_means_{n_ensembles}.npy", R0_est_enkf)
np.save(path + f"R0_covs_{n_ensembles}.npy", R0_cov_est_enkf)
np.save(path + f"p_vessel_estimates_{n_ensembles}.npy", pvessel_est_enkf)
np.save(path + f"errors_enkf_relative_{n_ensembles}.npy", errors_enkf_relative)
np.save(path + f"errors_enkf_absolute_{n_ensembles}.npy", errors_enkf_absolute)
