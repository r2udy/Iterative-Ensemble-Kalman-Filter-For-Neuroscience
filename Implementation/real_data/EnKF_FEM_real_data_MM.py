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
from scipy.interpolate import griddata
from EnKF_FEM_MM import EnKF
from circlesearch import Po2Analyzer
from MapGenerator import MapGenerator
from FEM_code.generateMesh_Solver_one_hole_MM import HoleGeometry, DiffusionSolver, SolverParameters
import pylab as P
from mpi4py import MPI

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

# --------- Load data --------- #
df = pd.read_pickle(py_data_location + "dataset.pkl")
df_copy = df.copy()
df_copy['pO2Value'] = df_copy['pO2Value'].apply(lambda x: x.flatten())
df_copy.keys()

def load_data(filepath):
    with open(filepath, 'r') as f:
        # Read lines and remove empty lines
        lines = [line.strip() for line in f if line.strip()]

        # Process each line into tuples
        data = [
            tuple(tuple(map(int, pair.strip('()').split(',')))
            for pair in line.split('), ('))
            for line in lines
        ]
    return data

uniform_dataset = load_data(py_data_location + 'uniform_dataset.txt')


# --------------------------
# Constants initial #
# --------------------------
D = 4.0e3
alpha = 1.39e-15
cmro2_low, cmro2_high = 1, 3 # umol/cm3/min
cmro2_by_M = (60 * D * alpha * 1e12)

cmro2_lower, cmro2_upper = 1.0, 3.0
cmro2_var = (cmro2_upper - cmro2_lower)**2 / 12
M_std = np.sqrt(cmro2_var) / cmro2_by_M
sigma = 2.

n = 20 # data size
pixel_size = 10

# EnKF Parameters
seed = np.random.seed(1)
state_dim = 2
obs_dim = 400
n_ensembles = 100

# Create coordinate grids in physical units (microns)
cols, rows = 20, 20
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
X = X * pixel_size
Y = Y * pixel_size

# Initialize the ensemble
a = np.array([cmro2_lower / cmro2_by_M, 10.0])
b = np.array([cmro2_upper / cmro2_by_M, 4.0])

# Initialization of the covariance matrix
Q   = np.array([[M_std**2, 0.0],
                [0.0,      4.0]]) # Background covariance matrix
R   = sigma**2 * np.eye(obs_dim) # Observation covariance matrix

# No dynamic model
def dynamics_model(x):
  return x

# Initialization of Arrays
observations_id = [entry for entry in uniform_dataset]
observations = []
cmro2_est_enkf = []
cmro2_cov_est_enkf = []
p50_est_enkf = []
p50_cov_est_enkf = []
means = []
covariances = []
errors = []
state_ensembles_M = np.zeros((len(observations_id), n_ensembles))
state_ensembles_p50 = np.zeros((len(observations_id), n_ensembles))

# --------------------------
# Create the EnKF object
enkf = EnKF(state_dim, obs_dim, n_ensembles, dynamics_model, seed)

# --------------------------
# Initialize the Ensemble Kalman Filter
enkf.initialize_ensemble(a, b)

# --------------------------
# Simulate a sequence with observation for the uniform case
for i, entry in enumerate(uniform_dataset):
    art_id = entry[0][0]
    dth_id = entry[0][1]
    
    # Observations
    obs = df_copy[(df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id)]['pO2Value'].tolist()[0]
    obs = np.array(obs)
    observations.append(obs)
    
    # Update the the background and observation noise
    enkf.set_process_noise(Q)
    enkf.set_observation_noise(R)
    
    # EnKF steps
    enkf.predict()
    enkf.update(obs)
    
    # Get current estimate
    mean, cov = enkf.get_state_estimate()
    state_ensembles_M[i, :] = enkf.get_ensemble()[0, :]
    state_ensembles_p50[i, :] = enkf.get_ensemble()[1, :]

    cmro2_mean  = mean[0] * cmro2_by_M
    p50_mean    = mean[1]
    cmro2_est_enkf.append(cmro2_mean)
    p50_est_enkf.append(p50_mean)

    cmro2_cov   = cov[0 ,0] * (cmro2_by_M)**2
    p50_cov     = cov[1, 1]
    cmro2_cov_est_enkf.append(cmro2_cov)
    p50_cov_est_enkf.append(cmro2_cov)
    
    means.append(mean)
    covariances.append(cov)

    pO2_array = obs.reshape((n, n), order='F')
    cmro2   = cmro2_mean
    p50     = p50_mean
    M       = cmro2_mean / cmro2_by_M
    pvessel = obs.max()

    # Find circles
    analyzer = Po2Analyzer(pO2_array, pixel_size, M)
    analyzer.find_circles()

    Rves = analyzer.rin
    R0 = analyzer.rout
    Rt = analyzer.rout

    # Initialize MPI
    comm = MPI.COMM_WORLD
    
    # Create solver parameters
    params = SolverParameters(filename="square_one_hole_MM", cmro2=-cmro2, Pves=pO2_array.max(), p50=p50, Rves=Rves, R0=R0)
    
    # Create solver instance
    solver = DiffusionSolver(comm)
    
    # Define holes
    holes = [
        HoleGeometry(center=(0, 0, 0), radius_ves=params.Rves, radius_0=params.R0, marker=params.marker),
        ]
    
    # Generate mesh
    solver.generate_mesh(holes)
    
    # Set up and solve problem
    solver.setup_problem(params, holes)
    solver.solve()

    # Interpolate to observation grid
    # Assign the data
    data_one_hole_axis = np.array(solver.domain.geometry.x)
    data_one_hole = np.array(solver.uh.x.array)
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
    obs_FEM = griddata((x, y), z, points, method='linear')

    error = obs - obs_FEM.flatten()
    errors.append(np.abs(error))


    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create coordinate grids in physical units (microns)
    X = np.meshgrid(np.arange(n), np.arange(n))[0] * pixel_size
    Y = np.meshgrid(np.arange(n), np.arange(n))[1] * pixel_size

    # Plot surface
    surf = ax.plot_surface(X, Y, pO2_array, 
                            cmap='jet',
                            rstride=1, cstride=1,
                            linewidth=0, 
                            antialiased=True,
                            alpha=0.3)
    
    surf = ax.plot_surface(X, Y, obs_FEM.reshape(n, n), 
                            cmap='viridis',
                            rstride=1, cstride=1,
                            linewidth=0, 
                            antialiased=True)

    # Add labels and colorbar
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_zlabel('Partial Pressure (mmHg)')
    ax.set_title(f'The predicted CMRO2={cmro2:0.3f}', fontsize=18)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()

    print("\n\n Ensemble Kalman Filter paramaters estimation:")
    print("-"*65)
    print(f"\nCMRO2 Mean: {cmro2_mean}, CMRO2 √(Cov): {np.sqrt(cmro2_cov)}, cmro2 Covariance: {cmro2_cov}\n")
    print(f"\P50 Mean: {p50_mean}, P50 √(Cov): {np.sqrt(p50_cov)}, P50 Covariance: {p50_cov}\n")
    
cmro2_est_enkf = np.array(cmro2_est_enkf)
cmro2_cov_est_enkf = np.array(cmro2_cov_est_enkf)
means = np.array(means)
covariances = np.array(covariances)
errors = np.array(errors)

# ----------------------+ Plots the results +----------------------#

# Simulated iteration steps
x = np.arange(1, len(observations_id) + 1)

# Stats
overall_mean = cmro2_est_enkf.mean()

data = state_ensembles_M.T * cmro2_by_M # Define data
numBoxes = len(uniform_dataset) # Define numBoxes
names = [f'obs {i}' for i in range(1, numBoxes + 1)]

P.figure()
bp = P.boxplot(data, labels=names)

for i in range(numBoxes):
    y = data[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value CMRO2 (umol /cm^3 /min)')
P.title('EnKF State Estimation of CMRO2 with Uncertainty')
P.grid(True)
P.show()


# Simulated iteration steps
x = np.arange(1, len(observations_id) + 1)

# Stats
overall_mean = cmro2_est_enkf.mean()

data = state_ensembles_p50.T # Define data
numBoxes = len(uniform_dataset) # Define numBoxes
names = [f'obs {i}' for i in range(1, numBoxes + 1)]

P.figure()

bp = P.boxplot(data, labels=names)

for i in range(numBoxes):
    y = data[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value CMRO2 (umol /cm^3 /min)')
P.title('EnKF State Estimation of p50 with Uncertainty')
P.grid(True)
P.show()


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