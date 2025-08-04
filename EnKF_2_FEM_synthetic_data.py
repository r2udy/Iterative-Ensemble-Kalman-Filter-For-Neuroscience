#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

py_file_location = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/classes/"
sys.path.append(os.path.abspath(py_file_location))

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from MapGenerator import MapGenerator
from circlesearch import Po2Analyzer
from EnKF_FEM_2 import EnKF
import pylab as P


def interpolation_grid(grid_refined, grid_coarse):
    def find_closest_point(given_point, array_of_points):
        # Use Euclidean distance for multi-dimensional points
        distances = np.abs(array_of_points - given_point)
        closest_index = np.argmin(distances)
        return closest_index

    idx_list = []
    for point_coarse in grid_coarse:
        closest_idx = find_closest_point(point_coarse, grid_refined)
        idx_list.append(closest_idx)
    
    return np.array(idx_list)


# --------- Perform the estimation ---------

# Initial Constants
D = 4.0e3
alpha = 1.39e-15
cmro2_low, cmro2_high = 1, 3 # umol/cm3/min
cmro2_by_M = (60 * D * alpha * 1e12)

cmro2_lower, cmro2_upper = 1.0, 3.0
cmro2_var = (cmro2_upper - cmro2_lower)**2 / 12
M_std = np.sqrt(cmro2_var) / cmro2_by_M
obs_uncertainty = 3.0 # measurement uncertainty
R0_uncertainty = 10.0 # prior uncertainty of caparilary-free space radius

n = 20 # data size
pixel_size = 10.0

# EnKF Parameters
seed = np.random.seed(1)
state_dim = 2
obs_dim = 400
n_ensembles = 100

# Covariance Matrix
Q   = np.array([[M_std,            0.0],
                [0.0,   R0_uncertainty]])   # Background covariance matrix
R   = obs_uncertainty * np.eye(obs_dim)    # Observation covariance matrix

# Create coordinate grids in physical units (microns)
cols, rows = 20, 20
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
X = X * pixel_size
Y = Y * pixel_size

# No dynamic model
def dynamics_model(x):
  return x

# Initialization of Arrays
errors = []
observations = []
cmro2_est_lsqnonlin = []
cmro2_est_enkf = []
cmro2_cov_est_enkf = []
R0_est_enkf = []
R0_cov_est_enkf = []
means = []
covs = []
cmro2_true_values = np.linspace(1., 3., 5)
state_ensembles_M = np.zeros((len(cmro2_true_values), n_ensembles))
state_ensembles_R0 = np.zeros((len(cmro2_true_values), n_ensembles))

for i, cmro2_true in enumerate(cmro2_true_values):
    
    # ------------------+ Synthetic Data +------------------ #
    # Add noise to the generated data
    data_one_hole_axis  = np.load(f"/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/FEM_dataset/square_one_hole_id{i+1}_coordinates.npy")
    data_one_hole       = np.load(f"/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/FEM_dataset/square_one_hole_id{i+1}_solution.npy")

    # Interpolate to observation grid
    # Assign the data
    z = data_one_hole
    x = data_one_hole_axis[:, 0]
    y = data_one_hole_axis[:, 1]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    
    # Create observation grid points
    x_obs = np.linspace(x_min, x_max, n)
    y_obs = np.linspace(y_min, y_max, n)
    
    # Create simulation grid
    x_idx_domain = interpolation_grid(x, x_obs)
    y_idx_domain = interpolation_grid(y, y_obs)
    x_domain = x[x_idx_domain]
    y_domain = y[y_idx_domain]
    X_domain, Y_domain = np.meshgrid(x_domain, y_domain)
    points = np.column_stack((X_domain.ravel(), Y_domain.ravel()))
    obs = griddata((x, y), z, points, method='linear')
    sigma = 2.0
    obs_perturbated = np.random.normal(obs, scale=sigma)
    observations.append(obs_perturbated)

    ##
    # ----------------------+ EnKF +----------------------#
    enkf = EnKF(state_dim, obs_dim, n_ensembles, dynamics_model, seed)
    
    ##
    #Initialize the ensemble
    cmro2_pertubed = np.random.normal(cmro2_true, scale=M_std * cmro2_by_M)
    a = np.array([cmro2_pertubed / cmro2_by_M, 80.])
    b = np.array([M_std, R0_uncertainty])
    enkf.initialize_ensemble(a, b)
    
    # Update the the background and observation noise
    enkf.set_process_noise(Q)
    enkf.set_observation_noise(R)
    
    # EnKF steps
    enkf.predict()
    enkf.update(obs)
    
    # Get current estimate
    mean, cov = enkf.get_state_estimate()
    state_ensembles_M[i, :] = enkf.get_ensemble()[0, :]
    state_ensembles_R0[i, :] = enkf.get_ensemble()[1, :]

    cmro2_mean = mean[0] * cmro2_by_M
    R0_mean    = mean[1]
    cmro2_est_enkf.append(cmro2_mean)
    R0_est_enkf.append(R0_mean)

    cmro2_cov   = cov[0 ,0] * (cmro2_by_M)**2
    R0_cov      = cov[1, 1]
    cmro2_cov_est_enkf.append(cmro2_cov)
    R0_cov_est_enkf.append(cmro2_cov)
    
    means.append(mean)
    covs.append(cov)

    pO2_array = obs_perturbated.reshape((n, n), order='F')
    cmro2 = cmro2_mean
    M = cmro2_mean / cmro2_by_M
    pvessel = obs.max()

    # Find circles
    analyzer = Po2Analyzer(pO2_array, pixel_size)
    analyzer.find_circles()
    Rves = analyzer.rin
    R0 = R0_mean
    Rt = R0_mean
    generator = MapGenerator(cmro2=cmro2, pvessel=pvessel, Rves=Rves, R0=R0, Rt=Rt)
    error = obs - generator.pO2_array.flatten()
    errors.append(np.abs(error))

    print("\n\n Ensemble Kalman Filter paramaters estimation:")
    print("-"*65)
    print(f"\nCMRO2 Mean: {cmro2_mean}, CMRO2 √(Cov): {np.sqrt(cmro2_cov)}, cmro2 Covariance: {cmro2_cov}")
    print("-"*25)
    print(f"R0 Mean: {R0_mean}, R0 √(Cov): {np.sqrt(R0_cov)}, R0 Covariance: {R0_cov}\n")
    
    # # Create the plot1
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    
    # # Create coordinate grids in physical units (microns)
    # X = np.meshgrid(np.arange(n), np.arange(n))[0] * pixel_size
    # Y = np.meshgrid(np.arange(n), np.arange(n))[1] * pixel_size

    # # Plot surface
    # surf = ax.plot_surface(X, Y, pO2_array, 
    #                         cmap='jet',
    #                         rstride=1, cstride=1,
    #                         linewidth=0, 
    #                         antialiased=True,
    #                         alpha=0.3)
    
    # surf = ax.plot_surface(X, Y, generator.pO2_array, 
    #                         cmap='viridis',
    #                         rstride=1, cstride=1,
    #                         linewidth=0, 
    #                         antialiased=True)

    # # Add labels and colorbar
    # ax.set_xlabel('X (µm)')
    # ax.set_ylabel('Y (µm)')
    # ax.set_zlabel('Partial Pressure (mmHg)')
    # ax.set_title(f'The predicted CMRO2={cmro2:0.3f}', fontsize=18)
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    # plt.tight_layout()
    # plt.show()

cmro2_est_enkf = np.array(cmro2_est_enkf)
R0_est_enkf = np.array(R0_est_enkf)
cmro2_cov_est_enkf = np.array(cmro2_cov_est_enkf)
R0_cov_est_enkf = np.array(R0_cov_est_enkf)
means = np.array(means)
covs = np.array(covs)
observations = np.array(observations)
errors = np.array(errors)


# ----------------------+ Plots the results +----------------------#

# Simulated iteration steps
observations_id = [1, 2, 3, 4, 5]
observations = np.array(observations)
x = np.arange(1, len(observations_id) + 1)

# Stats
overall_mean = cmro2_est_enkf.mean()

# -----------------------
# Ratio M Stats
data = state_ensembles_M.T * cmro2_by_M # Define data
numBoxes = len(observations) # Define numBoxes
names = [f'obs {i}' for i in range(1, numBoxes + 1)]

P.figure()
bp = P.boxplot(data, labels=names)

for i in range(numBoxes):
    y = data[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value CMRO2 (umol /cm^3 /min)')
P.title('EnKF State Estimation with Uncertainty')
P.grid(True)
P.show()

# -----------------------
# Radius R0 Stats
data = state_ensembles_R0.T # Define data
numBoxes = len(observations) # Define numBoxes
names = [f'obs {i}' for i in range(1, numBoxes + 1)]

P.figure()
bp = P.boxplot(data, labels=names)

for i in range(numBoxes):
    y = data[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value CMRO2 (umol /cm^3 /min)')
P.title('EnKF State Estimation with Uncertainty')
P.grid(True)
P.show()

# -----------------------
# Absolute Error Stats
# Simulated iteration steps
x = np.arange(1, len(observations_id) + 1)

# Stats
data = errors.T # Define data

P.figure()

bp = P.boxplot(data, labels=names)

for i in range(numBoxes):
    y = data[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('Absolute Partial Pressure Error')
P.title('Errors repartition')
P.grid(True)
P.show()



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
    label='Uncertainty (+/- 1 StD)'
)

# Labels and title
plt.ylabel('Estimated CMRO2 (umol /cm^3 /min)')
plt.xlabel('Input CMRO2 (umol /cm^3 /min)')
plt.title('EnKF State Estimation with Uncertainty using Krogh-Erlang Cylinder Model')
plt.legend()
plt.grid(True)
plt.show()