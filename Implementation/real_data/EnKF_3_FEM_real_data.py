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

print("Loaded paths:")
for p in sys.path[-4:]:
    print("  ", p)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from EnKF_FEM import EnKF, build_obs_covariance_radial, build_obs_covariance_diagonal
from EnKF_FEM_3 import EnKF
from FEM_code.generateMesh_Solver_multiple_holes import DiffusionSolver, SolverParameters, HoleGeometry
from circlesearch import Po2Analyzer
from MapGenerator import MapGenerator
from Po2Dataset import load_data
from lsqnonlin_M_pvessel_rout import Po2Fitter_3
import pylab as P
import time

# --------- Load data --------- #
df = pd.read_pickle(py_data_location + "dataset.pkl")
df_copy = df.copy()
df_copy['pO2Value'] = df_copy['pO2Value']#.apply(lambda x: x.flatten())
# df_copy.keys()

path_metadata = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/TODEsource/dbase/"
dict_meta = sio.loadmat(path_metadata + 'database.mat')
metadata = dict_meta['main']
# print(f"Metadata keys: {metadata.dtype.names}")
uniform_dataset = load_data(py_data_location + 'uniform_dataset.txt')

# --------------------------
# Path for saving the data
file_id_saving = "Test_stats/71_72_73_74_filtered_test"  # ID for saving the data
path = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/" + file_id_saving + "/"

# --------------------------
# Constants initial #   
D = 4.0e3
alpha = 1.39e-15
cmro2_by_M = (60 * D * alpha * 1e12)
grid_size = 20 # data size

cmro2_lower, cmro2_upper = 1.0, 3.0
cmro2_mean_inital = 2.0  # initial mean
cmro2_var_initial = .5**2 
cmro2_var = .2**2  # Uniform distribution variance
M_var = cmro2_var / cmro2_by_M**2  # model uncertainty scaled

# Prior associated with R0
# R0_lower, R0_upper = 70., 100.
R0_mean_initial = 90.
R0_var_initial = 1.**2
R0_var = 1.**2 # prior uncertainty of capillary-free space radius

# Prior associated with pvessel
# pvessel_lower, pvessel_upper = 85., 95.
pvessel_mean_initial = 90.
pvessel_var_initial = 5.**2 
pvessel_var = 5.**2 # prior uncertainty of Neumann boundary condition

# Observation variance parameters
obs_var_constant = 5.**2   # constant uncertainty of measurements in the observation covariance matrix R
sigma = 2.0  # Gaussian filter sigma for observation noise smoothing
obs_var_high = 5.**2    # high uncertainty of measurements
obs_var_low = 1.**2     # low uncertainty of measurements

# --------------------------
# EnKF Parameters
seed = np.random.seed(1)
state_dim = 3
obs_dim = 400
n_ensembles = 50

# Initialize the ensemble
# a = np.array([cmro2_lower / cmro2_by_M, R0_lower, pvessel_lower])
# b = np.array([cmro2_upper / cmro2_by_M, R0_upper, pvessel_upper])

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

# -------------------------
# Initialization of Arrays
observations_id = [entry for entry in uniform_dataset]
observations = []
cmro2_est_enkf = []
cmro2_est_lsq = []
cmro2_cov_est_enkf = []
R0_est_enkf = []
R0_cov_est_enkf = []
pvessel_est_enkf = []
pvessel_cov_est_enkf = []
errors_enkf_relative = []
errors_enkf_absolute = []
erros_lsq_relative = []
erros_lsq_absolute = []
state_ensembles = []
stats_overall = []
corrections = []
state_ensembles = np.zeros((state_dim, len(uniform_dataset), n_ensembles))

time_start = time.time()
# --------------------------
# Simulate a sequence with observation for the uniform case
for i, entry in enumerate(uniform_dataset):

    print(f"\n--- Processing observation {i+1} / {len(uniform_dataset)} ---")

    art_id = entry[0][0]
    dth_id = entry[0][1]

    angles_1 = entry[1]
    angles_2 = entry[2]

    min_radius = entry[3][0]
    
    depth = metadata['data_depth'][art_id-1][0][0][dth_id-1]
    meta_sexe =  metadata['meta_sex'][art_id-1][0][0]
    # Observations
    mask = (df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id)
    obs_ = df_copy[mask]['pO2Value'].tolist()[0].flatten().copy()
    obs_filtered = gaussian_filter(obs_, sigma=sigma)  # Smooth the observation with a Gaussian filter
    obs_filtered_array = obs_filtered.reshape((grid_size, grid_size), order='F')

    obs = obs_filtered.copy()
    obs_array = obs.reshape((grid_size, grid_size), order='F')

    observations.append(obs)

    X_axis = df_copy[mask]['pointsX'].tolist()[0]
    Y_axis = df_copy[mask]['pointsY'].tolist()[0]

    # ------------------------
    # Find Geometric parameters such as Rves and the center
    analyzer = Po2Analyzer(obs_array, X_axis, Y_axis)
    analyzer.find_circles()
    Rves = np.diff(X_axis[0])[0]
    center = analyzer.center_ij
    center_coordinates = analyzer.center

    # ------------------------
    # Non - Linear Least Square Fitting
    print(f"\n--- Observation {i+1} / {len(uniform_dataset)}: Arteriole ID {art_id}, Depth ID {dth_id} ---")
    fitter = Po2Fitter_3(pO2_array=obs_array, Rves=Rves, X_axis=X_axis, Y_axis=Y_axis)
    fitter.fit()
    fitter.plot_estimated_parameters()
    cmro2_lsq, _, pvessel_lsq, R0_lsq = fitter.get_results()

    params = SolverParameters(filename="square_holes")
    hole_estimated = [HoleGeometry(center=(*center, 0.0), cmro2=cmro2_lsq, Pves=pvessel_lsq, radius_ves=Rves, radius_0=R0_lsq)]
    generator_lsq = MapGenerator(
                        holes=hole_estimated,
                        params=params,
                        X=X_axis,
                        Y=Y_axis
                    )
    obs_estimation_lsq = generator_lsq.pO2_array
    # Compute the errors
    error_lsq_relative = np.abs(obs_ - obs_estimation_lsq.flatten()) * 100 / np.abs(obs_) 
    error_lsq_absolute = np.abs(obs_ - obs_estimation_lsq.flatten())


    # ----------------------
    # EnKF steps

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
    enkf.length_scale = 1.   # length scale for localization
    enkf.predict()
    enkf.update(obs, X_axis, Y_axis)

    # Get current estimate
    mean, cov = enkf.get_state_estimate()

    # Means and Covariances
    cmro2_mean  = mean[0] * cmro2_by_M
    R0_mean    = mean[1]
    pvessel_mean = mean[2]

    cmro2_cov   = cov[0, 0] * (cmro2_by_M)**2
    R0_cov     = cov[1, 1]
    pvessel_cov = cov[2, 2]

    correction = np.abs(np.mean(enkf.length_scale * enkf.K @ enkf.innovation))

    # Compute the absolute error
    print(f'CMRO_2: {cmro2_mean}')
    print(f'R0: {R0_mean}')
    print(f"p_vessel: {pvessel_mean}")
    print(f'Rves: {Rves}')

    hole_estimated = [HoleGeometry(center=(*center, 0.0), cmro2=cmro2_mean, Pves=pvessel_mean, radius_ves=Rves, radius_0=R0_mean)]

    generator_enkf = MapGenerator(
                        holes=hole_estimated,
                        params=params,
                        X=X_axis,
                        Y=Y_axis
                    )
    obs_estimation = generator_enkf.pO2_array
    # Compute the errors
    error_enkf_relative = np.abs(obs_ - obs_estimation.flatten()) * 100 / np.abs(obs_) 
    error_enkf_absolute = np.abs(obs_ - obs_estimation.flatten())

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

    c = ax.pcolormesh(X_axis, Y_axis, obs_array.T, shading='auto', cmap='jet')
    fig.colorbar(c, label='pO₂')
    ax.set_title(f"Radial pO₂ Map\n {meta_sexe} | Arteriole {art_id} | Depth {dth_id}: {depth} um", fontsize=10)
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.axis('equal')
    ax.legend()
    ax.set_aspect('equal', 'box')
    # # ----------------------
    # # (2) Diagonal variance map
    # ax = fig.add_subplot(2, 3, 2)
    # uncertainty_map = np.diag(enkf.R).reshape((grid_size, grid_size))
    # c = ax.pcolormesh(X_axis, Y_axis, uncertainty_map, cmap='viridis')
    # fig.colorbar(c, label='Variance')
    # ax.set_title(f"Diagonal Variance Map of PO2\n Covariance Matrix R: {uncertainty_map.mean():.2f} mmHg²")
    # ax.set_xlabel('X (nm)')
    # ax.set_ylabel('Y (nm)')
    # ax.axis('equal')
    # ax.set_aspect('equal', 'box')
    # ----------------------
    # (2) Smoothed map vs. True map
    ax = fig.add_subplot(2, 3, 2, projection="3d")
    sc = ax.plot_surface(X_axis, Y_axis, obs_.reshape((grid_size, grid_size), order='F'), cmap='viridis', edgecolor='none')
    ax.plot_surface(X_axis, Y_axis, obs_filtered_array, cmap='plasma', edgecolor='none')
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('pO2 (mmHg)')
    ax.set_title(f"Display of the approximated map: \nCMRO2:{cmro2_mean:.2f} umol /cm^3 /min\n R0:{R0_mean:.2f} um;\n P_vessel:{pvessel_mean:.2f} mmHg")
    plt.colorbar(sc, ax=ax, shrink=0.3, aspect=10, label='Color scale: Partial Pressure (mmHg)')
    # ----------------------
    # (3) Absolute error 3D surface
    ax = fig.add_subplot(2, 3, 3, projection='3d')

    error_lsq_absolute_array = error_lsq_absolute.reshape((grid_size, grid_size), order='F')
    sc = ax.plot_surface(X_axis, Y_axis, error_lsq_absolute_array, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('pO2 (mmHg)')
    ax.set_title(f"Map of Absolute error LSQ, \nmean absolute error: {error_lsq_absolute.mean():.2f}; \nCMRO2: {cmro2_lsq:.2f} umol /cm^3 /min\n R0: {R0_lsq:.2f} um;\n P_vessel: {pvessel_lsq:.2f} mmHg")
    plt.colorbar(sc, ax=ax, shrink=0.3, aspect=10)#, label='Absolute Error')
    # ----------------------
    # (4) Absolute error 3D surface
    ax = fig.add_subplot(2, 3, 4, projection='3d')

    error_enkf_absolute_array = error_enkf_absolute.reshape((grid_size, grid_size), order='F')
    sc = ax.plot_surface(X_axis, Y_axis, error_enkf_absolute_array, cmap='viridis', edgecolor='none')
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('pO2 (mmHg)')
    ax.set_title(f"Map of Absolute error EnKF, \nmean absolute error: {error_enkf_absolute.mean():.2f}; \nCMRO2: {cmro2_mean:.2f} umol /cm^3 /min\n R0: {R0_mean:.2f} um;\n P_vessel: {pvessel_mean:.2f} mmHg", fontsize=9)
    plt.colorbar(sc, ax=ax, shrink=0.3, aspect=10)#, label='Absolute Error')
    # ----------------------
    # (5) Approximated vs. True map
    ax = fig.add_subplot(2, 3, 5, projection="3d")

    sc = ax.plot_surface(X_axis, Y_axis, obs_estimation, cmap='plasma', edgecolor='none')
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('pO2 (mmHg)')
    ax.set_title(f"Display of the approximated map: \nCMRO2:{cmro2_mean:.2f} umol /cm^3 /min\n R0:{R0_mean:.2f} um | sigma:{np.sqrt(R0_cov):.2f};\n Pvessel:{pvessel_mean:.2f} mmHg | sigma:{np.sqrt(pvessel_cov):.2f} mmHg\n NIS: {enkf.NIS}", fontsize=10)
    plt.colorbar(sc, ax=ax, shrink=0.3, aspect=10, label='Color scale: Partial Pressure (mmHg)')

    # ----------------------
    # (6) Non-Linear Least Square fitting
    ax = fig.add_subplot(2, 3, 6, projection="3d")

    sc = ax.plot_surface(X_axis, Y_axis, obs_estimation_lsq, cmap='plasma', edgecolor='none')
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    ax.set_zlabel('pO2 (mmHg)')
    ax.set_title(f"Display of the approximated map: \nCMRO2:{cmro2_lsq:.2f} umol /cm^3 /min\n R0:{R0_lsq:.2f} um;\n Pvessel:{pvessel_lsq:.2f} mmHg", fontsize=10)
    plt.colorbar(sc, ax=ax, shrink=0.3, aspect=10, label='Color scale: Partial Pressure (mmHg)')
    plt.tight_layout()
    plt.savefig(path + f'graphs_{art_id}{dth_id}_{i}.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # X = enkf.ensemble                    # (S, Ne)
    # # recompute obs_model_ens if not stored:
    # Ne = X.shape[1]
    # obs_model_ens = np.zeros((enkf.obs_dim, Ne))
    # for l in range(Ne):
    #     obs_model_ens[:, l] = enkf.observation_operator(X[:, l], X_axis, Y_axis)

    # x_mean = X.mean(axis=1)[:,None]
    # y_mean = obs_model_ens.mean(axis=1)[:,None]
    # X_dev = X - x_mean
    # Y_dev = obs_model_ens - y_mean

    # # cross covariance
    # C_xy = (X_dev @ Y_dev.T) / (Ne - 1)   # shape (state_dim, obs_dim)

    # ----------------------
    # Print the results
    print(f"Correction: {correction}")
    print(f"EnKF Mean Relative Error: {error_enkf_relative.mean()}")   
    print(f"EnKF Mean Absolute Error: {error_enkf_absolute.mean()}") 
    print(f"LSQ Mean Relative Error: {error_lsq_relative.mean()}")
    print(f"LSQ Mean Absolute Error: {error_lsq_absolute.mean()}")

    # Save stats
    stats_overall.append((cmro2_mean, cmro2_cov))
    errors_enkf_relative.append(np.abs(error_enkf_relative)) # Save the relative errors
    errors_enkf_absolute.append(np.abs(error_enkf_absolute)) # Save the absolute errors
    erros_lsq_relative.append(np.abs(error_lsq_relative)) # Save the relative errors
    erros_lsq_absolute.append(np.abs(error_lsq_absolute)) # Save the absolute errors
    corrections.append(correction)
    cmro2_est_enkf.append(cmro2_mean)
    cmro2_est_lsq.append(cmro2_lsq)
    cmro2_cov_est_enkf.append(cmro2_cov)
    R0_est_enkf.append(R0_mean)
    R0_cov_est_enkf.append(R0_cov)
    pvessel_est_enkf.append(pvessel_mean)
    pvessel_cov_est_enkf.append(pvessel_cov)
    state_ensembles[0, i, :] = enkf.ensemble[0, :]  # CMRO2 ensembles
    state_ensembles[1, i, :] = enkf.ensemble[1, :]  # R0 ensembles
    state_ensembles[2, i, :] = enkf.ensemble[2, :]  # pvessel ensembles


    # Print results in the terminal
    print(f"\n\n Ensemble Kalman Filter paramaters estimation")
    print("-"*65)
    print(f"Observation ID: {art_id}, Depth ID: {dth_id}")
    print(f"\nCMRO2 Mean: {cmro2_mean}, CMRO2 √(Cov): {np.sqrt(cmro2_cov)}, cmro2 Covariance: {cmro2_cov}")
    print("-"*25)
    print(f"R0 Mean: {R0_mean}, R0 √(Cov): {np.sqrt(R0_cov)}, R0 Covariance: {R0_cov}")
    print("-"*25)
    print(f"Pvessel Mean: {pvessel_mean}, Pvessel √(Cov): {np.sqrt(pvessel_cov)}, R0 Covariance: {pvessel_cov}\n")
    print(f"Mean Relative Error: {error_enkf_relative.mean()}")
    print(f"Mean Absolute Error: {error_enkf_absolute.mean()}")
    print(f"Correction: {correction}")

time_end = time.time()
print(f"\n--- Total time for EnKF estimation over {len(uniform_dataset)} observations: {time_end - time_start:.2f} seconds ---")
cmro2_est_enkf = np.array(cmro2_est_enkf)
cmro2_cov_est_enkf = np.array(cmro2_cov_est_enkf)
R0_est_enkf = np.array(R0_est_enkf)
R0_cov_est_enkf = np.array(R0_cov_est_enkf)
pvessel_est_enkf = np.array(pvessel_est_enkf)
pvessel_cov_est_enkf = np.array(pvessel_cov_est_enkf)
corrections =  np.array(corrections)
errors_enkf_relative = np.array(errors_enkf_relative)
errors_enkf_absolute = np.array(errors_enkf_absolute)
erros_lsq_relative = np.array(erros_lsq_relative)
erros_lsq_absolute = np.array(erros_lsq_absolute)
state_ensembles = np.array(state_ensembles)
stats_overall = np.array(stats_overall)
corrections = np.array(corrections)

# sys.exit(0)
x_obs = np.arange(1, len(observations_id) + 1) # Simulated iteration steps

# --------- Plots the results ---------
# -----------------------
# # CMRO_2 Stats
observations_id = [i for i in range(1, len(observations) + 1)]
observations = np.array(observations)
x = np.arange(1, len(observations_id) + 1)
# Stats

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
P.savefig(path + 'enkf_state_estimation.png', dpi=300, bbox_inches='tight')
# P.show()


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
# pvessel Stats
# -----------------------
observations_id = [i for i in range(1, len(observations) + 1)]
observations = np.array(observations)

data = state_ensembles[2].T
numBoxes = len(observations) # Define numBoxes
names = [f'obs {i}' for i in observations_id]

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
# P.show()

# -----------------------
# Relative Error Stats
# -----------------------
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
# -----------------------
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
plt.savefig(path + 'enkf_correction.png', dpi=300, bbox_inches='tight')
# plt.show()

# -----------------------
# Uncertainty associated to estimation
# -----------------------
fig = plt.figure(figsize=(12, 14))

ax = fig.add_subplot(3, 1, 1)
data = state_ensembles[0] * cmro2_by_M # Define data
numBoxes = data.shape[0]  # now robust
x_obs = np.arange(1, numBoxes + 1) 
cov_track = np.array([np.std(array) for array in data])
ax.plot(x_obs, cov_track, '-o', color='blue', label='Uncertainty in CMRO2 estimation (StD)')
plt.ylabel('Estimated CMRO2 Uncertainty (umol /cm^3 /min)')
plt.xlabel('$PO_{2}$ Map ID')
plt.title('EnKF Uncertainty')
plt.grid(True)
plt.xticks(x_obs, [f'Obs{i}' for i in x_obs])
plt.legend()

ax = fig.add_subplot(3, 1, 2)
data = state_ensembles[1] # Define data
numBoxes = data.shape[0]  # now robust
x_obs = np.arange(1, numBoxes + 1) 
cov_track = np.array([np.std(array) for array in data])
ax.plot(x_obs, cov_track, '-o', color='blue', label='Uncertainty in CMRO2 estimation (StD)')
plt.ylabel('Estimated R0 Uncertainty (um)')
plt.xlabel('$PO_{2}$ Map ID')
plt.title('EnKF Uncertainty')
plt.grid(True)
plt.xticks(x_obs, [f'Obs{i}' for i in x_obs])
plt.legend()

ax = fig.add_subplot(3, 1, 3)
data = state_ensembles[2] # Define data
numBoxes = data.shape[0]  # now robust
x_obs = np.arange(1, numBoxes + 1) 
cov_track = np.array([np.std(array) for array in data])
ax.plot(x_obs, cov_track, '-o', color='blue', label='Uncertainty in CMRO2 estimation (StD)')
plt.ylabel('Estimated Pvessel Uncertainty (mmHg)')
plt.xlabel('$PO_{2}$ Map ID')
plt.title('EnKF Uncertainty')
plt.grid(True)
plt.xticks(x_obs, [f'Obs{i}' for i in x_obs])
plt.legend()
plt.tight_layout()
plt.savefig(path + 'enkf_uncertainty.png', dpi=300, bbox_inches='tight')
# plt.show()

# -----------------------
# Estimation + Uncertainty of CMRO2
# -----------------------
# Create figure
plt.figure(figsize=(10, 6))
cmro2_mean_ = stats_overall[:, 0]
cmro2_cov_ = stats_overall[:, 1]
# Plot mean +/- 1 standard deviation (sqrt of variance)
plt.plot(x_obs, cmro2_mean_, '-o', color='green', label='State EnKF estimate (CMRO2)')
plt.plot(x_obs, cmro2_est_lsq, '-o', color='green', label='NonLinear LS estimate (CMRO2)')
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

# -----------------------
# Data
labels = [f'Obs{i}' for i in x_obs]
control_means = cmro2_est_lsq
control_err = np.zeros_like(control_means)  # No error bars for control

experimental_means = cmro2_mean_
experimental_err = np.sqrt(cmro2_cov_)

width = 0.35

# Plot
plt.figure(figsize=(8, 5))
plt.bar(x_obs - width/2, control_means, width,
        yerr=control_err, capsize=5,
        label="Control")

plt.bar(x_obs + width/2, experimental_means, width,
        yerr=experimental_err, capsize=5,
        label="Experimental")
plt.title("CMRO2 Estimation: EnKF vs Non-Linear LSQ")
plt.xticks(x_obs, labels)
plt.ylabel('CMRO2 (umol /cm^3 /min)')
plt.legend()
plt.tight_layout()
plt.savefig(path + 'enkf_cmro2_estimation_and_uncertainty_barplot.png', dpi=300, bbox_inches='tight')
# plt.show()


# -----------------------
# Posterior distribution through the iteration
# Sample data
data = np.array(state_ensembles[0].T) * cmro2_by_M
pdf_matrix = np.zeros(data.shape) # rows: u, cols: t

# Create a grid for the x and y axes
iterations = np.arange(data.shape[1])  # 9 iterations
points = np.linspace(data.min()*.9, data.max()*1.1, data.shape[0]) # 100 points
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
plt.savefig(path + 'enkf_cmro2_estimation_and_uncertainty.png', dpi=300, bbox_inches='tight')
# plt.show()

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
np.save(path + f"maps.npy", observations)
