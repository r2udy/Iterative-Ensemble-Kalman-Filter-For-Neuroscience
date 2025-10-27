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
cmro2_var = (cmro2_upper - cmro2_lower)**2  # variance of uniform distribution
M_var = cmro2_var / cmro2_by_M**2 # model uncertainty scaled
obs_var_constant = 20.**2
obs_var_high = 20.**2    # high uncertainty of measurements
obs_var_low = 1.**2     # low uncertainty of measurements

# Grid configuration
mask = (df_copy["arteriole_id"] == 2) & (df_copy['depth_id'] == 3)
X = df_copy[mask]['pointsX'].tolist()[0]
Y = df_copy[mask]['pointsY'].tolist()[0]

# --------------------------
# EnKF Parameters
seed = np.random.seed(1)
state_dim = 1
obs_dim = 400
n_ensembles = 50

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


# ------------------+ Synthetic Data +------------------ #

observations = []
cmro2_est_lsqnonlin = []
cmro2_est_enkf = []
cmro2_cov_est_enkf = []
p_vessel_est = []

state_ensembles = []

errors_enkf_relative = []
errors_enkf_absolute = []

means = []
covs = []
stats_overall = []

corrections = []

# PDF for the ensemble initialization
f_alpha = enkf.ensemble.flatten() * cmro2_by_M
# PDF for the ensemble initialization (KDE + histogram + uniform prior)
from scipy.stats import gaussian_kde
samples = f_alpha  # CMRO2 in umol/cm^3/min

kde = gaussian_kde(samples)
x_grid = np.linspace(samples.min()*(1 - 0.05), samples.max()**(1 + 0.05), 300)
pdf_kde = kde(x_grid)

# analytical uniform prior between cmro2_lower and cmro2_upper
prior_pdf = np.where((x_grid >= cmro2_lower) & (x_grid <= cmro2_upper),
                     1.0 / (cmro2_upper - cmro2_lower), 0.0)

plt.figure(figsize=(6,4))
plt.hist(samples, bins=30, density=True, alpha=0.4, label='Ensemble (hist)')
plt.plot(x_grid, pdf_kde, label='KDE', lw=2)
plt.plot(x_grid, prior_pdf, '--', label='Uniform prior', lw=2)
plt.axvline(np.mean(samples), color='k', linestyle=':', label='Ensemble mean')
plt.xlabel('CMRO2 (umol /cm^3 /min)')
plt.ylabel('Density')
plt.title('PDF of Ensemble Initialization')
plt.legend()
plt.tight_layout()
plt.show()


cmro2_true_values = np.linspace(1., 3., 5)
state_ensembles = np.zeros((len(cmro2_true_values), n_ensembles))
for i, cmro2_true in enumerate(cmro2_true_values):
    
    # ------------------+ Synthetic Data Generation +------------------ #
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
    center_secondary = (-50.0, -50.0)

    generator2 = MapGenerator(
        cmro2=2.,
        pvessel=pvessel2,
        Rves=10.,
        R0=70,
        Rt=70.,
        X=X,
        Y=Y,
        center=center_secondary)
    profile_secondary = generator2.pO2_array.reshape((grid_size, grid_size), order='F')
    profile_secondary[profile_secondary < 0] = 0
    profile_secondary = gaussian_filter(profile_secondary.flatten(), sigma=2.).reshape((grid_size, grid_size), order='F')

    # Add noise to the generated data
    true_obs = profile_main 
    sigma = 2.0
    obs_perturbated = np.random.normal(true_obs.flatten(), scale=sigma)
    obs_perturbated_array = obs_perturbated.reshape((grid_size, grid_size), order='F')

    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(projection='3d')
    # sc = ax.plot_surface(X, Y, obs_perturbated_array, cmap='viridis', edgecolor='none')
    # # ax.plot_surface(X, Y, profile_main, cmap='viridis', alpha=0.3, edgecolor='none')
    # # ax.plot_surface(X, Y, profile_secondary, cmap='viridis', alpha=0.3, edgecolor='none')
    # ax.set_xlabel('X (nm)')
    # ax.set_ylabel('Y (nm)')
    # ax.set_zlabel('pO2 (mmHg)')
    # plt.colorbar(sc, ax=ax, shrink=0.3, aspect=10, label='pO2 (mmHg)')
    # ax.set_title(f'Synthetic pO2 Data with Noise\n (True: CMRO2={cmro2_true} umol/cm^3/min, Pvessel={pvessel} mmHg)')
    # plt.show()

    # Find Geometric parameters such as Rves and R0
    analyzer = Po2Analyzer(obs_perturbated_array, X, Y)
    analyzer.find_circles()
    Rves_est = analyzer.rin
    R0_est = analyzer.rout
    pvessel_est = analyzer.p_vessel
    center = analyzer.center_ij
    center_coordinates = analyzer.center


    # ----------------------+ EnKF +----------------------#    
    # R = build_obs_covariance_radial(
    # origin = center,
    # obs_var_high = obs_var_high,
    # obs_var_low = obs_var_low,
    # mode='linear'
    # )

    # R = build_obs_covariance_diagonal(
    #     grid_size=grid_size,
    #     origin=center,
    #     angle_ranges=[angles_1, angles_2],
    #     min_radius=min_radius,
    #     obs_var_high=obs_var_high,
    #     obs_var_low=obs_var_low
    # )

    R = obs_var_constant * np.eye(obs_dim) # Observation covariance matrix
    
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
    cmro2_cov = cov * (cmro2_by_M**2)

    correction = np.abs(np.mean(enkf.length_scale * enkf.K @ enkf.innovation))
    
    generator_enkf = MapGenerator(cmro2=cmro2_mean, 
                        pvessel=pvessel, 
                        Rves=Rves, 
                        R0=R0, 
                        Rt=R0,
                        X=X,
                        Y=Y)
    obs_estimation = generator_enkf.pO2_array

    # # -------------------
    # f_alpha = enkf.ensemble.flatten() * cmro2_by_M
    # # PDF for the ensemble initialization (KDE + histogram + uniform prior)
    # from scipy.stats import gaussian_kde
    # samples = f_alpha  # CMRO2 in umol/cm^3/min

    # kde = gaussian_kde(samples)
    # x_grid = np.linspace(samples.min(), samples.max(), 300)
    # pdf_kde = kde(x_grid)
    # # analytical uniform prior between cmro2_lower and cmro2_upper
    # prior_pdf = np.where((x_grid >= cmro2_lower) & (x_grid <= cmro2_upper),
    #                     1.0 / (cmro2_upper - cmro2_lower), 0.0)
    # plt.figure(figsize=(6,4))
    # plt.hist(samples, bins=30, density=True, alpha=0.4, label='Ensemble (hist)')
    # plt.plot(x_grid, pdf_kde, label='KDE', lw=2)
    # plt.plot(x_grid, prior_pdf, '--', label='Uniform prior', lw=2)
    # plt.axvline(np.mean(samples), color='k', linestyle=':', label='Ensemble mean')
    # plt.xlabel('CMRO2 (umol /cm^3 /min)')
    # plt.ylabel('Density')
    # plt.title(f'PDF of Ensemble Step {i+1}')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # # -------------------

    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(projection='3d')
    # sc = ax.plot_surface(X, Y, obs_estimation, cmap='viridis', edgecolor='none')
    # ax.plot_surface(X, Y, obs.reshape((grid_size, grid_size), order='F'), cmap='viridis', alpha=0.3, edgecolor='none')
    # ax.set_xlabel('X (nm)')
    # ax.set_ylabel('Y (nm)')
    # ax.set_zlabel('pO2 (mmHg)')
    # plt.colorbar(sc, ax=ax, shrink=0.3, aspect=10, label='pO2 (mmHg)')
    # ax.set_title(f'Synthetic pO2 Data with Noise\n (True: CMRO2={cmro2_true} umol/cm^3/min, Pvessel={pvessel} mmHg)\n (EnKF Estimate: CMRO2={cmro2_mean:.2f} umol/cm^3/min, Pvessel={pvessel_est:.2f} mmHg)')
    # plt.show()

    error_enkf_relative = np.abs(obs - obs_estimation.flatten()) * 100 / np.abs(obs) 
    error_enkf_absolute = np.abs(obs - obs_estimation.flatten())

    # Save stats
    stats_overall.append((cmro2_mean, cmro2_cov))
    errors_enkf_relative.append(np.abs(error_enkf_relative)) # Save the relative errors
    errors_enkf_absolute.append(np.abs(error_enkf_absolute)) # Save the absolute errors
    p_vessel_est.append(pvessel_est)
    corrections.append(correction)
    cmro2_est_enkf.append(cmro2_mean)
    cmro2_cov_est_enkf.append(cmro2_cov)
    state_ensembles[i, :] = enkf.ensemble * cmro2_by_M

    print("\n\n Ensemble Kalman Filter paramaters estimation:")
    print("-"*65)
    print(f"\nCMRO2 Mean: {cmro2_mean}, CMRO2 √(Cov): {np.sqrt(cmro2_cov)}, cmro2 Covariance: {cmro2_cov}\n")
    
cmro2_est_enkf = np.array(cmro2_est_enkf)
cmro2_cov_est_enkf = np.array(cmro2_cov_est_enkf)
p_vessel_est = np.array(p_vessel_est)
corrections =  np.array(corrections)
errors_enkf_relative = np.array(errors_enkf_relative)
errors_enkf_absolute = np.array(errors_enkf_absolute)
observations = np.array(observations)
stats_overall = np.array(stats_overall)



# Path for saving the data
path = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/constant_02/"

# --------- Plots the results ---------
# Simulated iteration steps
observations_id = [i for i in range(1, len(observations) + 1)]
observations = np.array(observations)
x = np.arange(1, len(observations_id) + 1)
print(observations.shape)
print(state_ensembles.shape)
# Stats
overall_mean = cmro2_est_enkf.mean()

data = state_ensembles.T # Define data
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
# plt.plot(x, cmro2_est_lsqnonlin, '-x', label='LSQNonLin estimate (CMRO2 + pvessel + $R_{0}$)')
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
plt.savefig(path + 'enkf_partial_pressure_estimated.png', dpi=300, bbox_inches='tight')
# plt.show()

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

# -----------------------
# Uncertainty associated to estimation
data = state_ensembles
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
np.save(path + f"p_vessel_estimates_{n_ensembles}.npy", p_vessel_est)
np.save(path + f"errors_enkf_relative_{n_ensembles}.npy", errors_enkf_relative)
np.save(path + f"errors_enkf_absolute_{n_ensembles}.npy", errors_enkf_absolute)
