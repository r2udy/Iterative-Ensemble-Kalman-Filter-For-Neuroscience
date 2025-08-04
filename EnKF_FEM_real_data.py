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
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde, norm
from EnKF_FEM import EnKF
from circlesearch import Po2Analyzer
from MapGenerator import MapGenerator
from lsqnonlin import Po2Fitter
import pylab as P

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

# ---------- Target Cells ----------- #
def get_cells_by_angle(grid_size, origin, angle_ranges, distance_range=None):
    x0, y0 = origin
    selected_cells = []

    for y in range(grid_size):
        for x in range(grid_size):
            dx = x - x0
            dy = y0 - y  # reverse y if needed (grid coordinates)
            
            angle = math.degrees(math.atan2(dy, dx)) % 360
            distance = math.hypot(dx, dy)

            # Check angle ranges
            in_angle = any(
                start <= angle <= end if start <= end else angle >= start or angle <= end
                for (start, end) in angle_ranges
            )
            
            # Check distance range if given
            in_distance = True
            if distance_range:
                min_d, max_d = distance_range
                in_distance = min_d <= distance <= max_d

            if in_angle and in_distance:
                selected_cells.append((x, y))
    
    return selected_cells


# --------------------------
# Constants initial #
D = 4.0e3
alpha = 1.39e-15
cmro2_low, cmro2_high = 1, 3 # umol/cm3/min
cmro2_by_M = (60 * D * alpha * 1e12)

cmro2_lower, cmro2_upper = 1.0, 3.0
cmro2_var = (cmro2_upper - cmro2_lower)**2 / 12
M_var = cmro2_var / cmro2_by_M**2 / 5
M_std = np.sqrt(cmro2_var) / cmro2_by_M # model uncertainty
obs_var_uncertain = 15.**2
obs_var = 1.0**2 # measurement uncertainty

n = 20 # data size
pixel_size = 10

tol_cov = 1e-2
max_inner_iterations = 10

# --------------------------
# EnKF Parameters
seed = np.random.seed(1)
state_dim = 1
obs_dim = 400
n_ensembles = 100

# Initialize the ensemble
a = np.array([cmro2_lower / cmro2_by_M])
b = np.array([cmro2_upper / cmro2_by_M])

# Covariance Matrix
Q = np.array([[M_var]])         # Background covariance matrix
R = obs_var * np.eye(obs_dim)   # Observation covariance matrixservation covariance matrix

# -------------------------
# Create coordinate grids in physical units (microns)
X, Y = np.meshgrid(np.arange(n), np.arange(n))
X = X * pixel_size
Y = Y * pixel_size

# No dynamic model
def dynamics_model(x):
    return x

# Create the EnKF object
enkf = EnKF(state_dim, obs_dim, n_ensembles, dynamics_model, seed)

# Initialize the EnKF method
enkf.initialize_ensemble(a, b)
    
# Update the the background and observation noise1
enkf.set_process_noise(Q)
enkf.set_observation_noise(R)

# Initialization of Arrays
observations_id = [entry for entry in uniform_dataset]
observations = []
cmro2_est_enkf = []
cmro2_est_lsqnonlin = []
cmro2_cov_est_enkf = []
means = []
covariances = []
errors_enkf = []
state_ensembles = [] #np.zeros((len(observations_id), n_ensembles))
obs_covariances_matrices = []
N_it = []

# --------------------------
# Simulate a sequence with observation for the uniform case
# --------------------------
for i, entry in enumerate(uniform_dataset):
    art_id = entry[0][0]
    dth_id = entry[0][1]

    angles_1 = entry[1]
    angles_2 = entry[2]
    
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


    # ----------------------
    # Adjust the observation covariance matrix to account very uncertain measurement
    min_radius = 9
    max_radius = math.hypot(n, n)  # furthest possible distance in the grid

    # Get updated selected cells
    selected_cells_border = get_cells_by_angle(
        n,
        center,
        [angles_1, angles_2],
        distance_range=(min_radius, max_radius)
    )

    # Create a 400x400 matrix initialized to zeros
    matrix_size = 400
    matrix = np.zeros((matrix_size, matrix_size))

    # Set higher values for targeted cells and lower for non-targeted ones
    high_value = obs_var_uncertain
    low_value = obs_var

    # Create a 400x400 matrix for diagonals representing each cell of the 20x20 grid
    matrix_diag = np.zeros((matrix_size, matrix_size))

    # Flatten the 20x20 grid into a 1D list of 400 positions corresponding to diagonals
    grid_cells = [(x, y) for y in range(n) for x in range(n)]

    # Assign higher or lower values to each diagonal cell based on whether it was targeted
    for k, (x, y) in enumerate(grid_cells):
        matrix_diag[k, k] = high_value if (x, y) in selected_cells_border else low_value

    R = matrix_diag
    enkf.set_observation_noise(R)
    power_factor_it = 0

    # Initialize stopping loop
    for _ in range(max_inner_iterations):
        # ----------------------
        # Save the previous stats properties to compare later
        prev_mean = enkf.get_state_estimate()[0] * cmro2_by_M
        prev_cov = enkf.get_state_estimate()[1] * (cmro2_by_M)**2

        # EnKF steps
        enkf.predict(obs)
        enkf.update()

        # Get current estimate
        mean, cov = enkf.get_state_estimate()

        # Means and Covariances
        cmro2_mean  = mean[0] * cmro2_by_M
        cmro2_cov = cov * (cmro2_by_M)**2

        # Save the the updated stats propreties to compare
        new_mean = cmro2_mean
        new_cov = cmro2_cov

        # Early stop quantities
        mean_diff = np.abs(new_mean - prev_mean)
        cov_diff = np.abs(new_cov - prev_cov)

        # Errors
        generator_enkf = MapGenerator(cmro2=cmro2_mean, 
                            pvessel=p_vessel, 
                            Rves=Rves, 
                            R0=R0, 
                            Rt=R0)

        # Compute the absolute error
        error_enkf = np.abs(obs - generator_enkf.pO2_array.flatten())
        print(f"Absolute Mean and Cov Difference of the EnKF: \nmean: {mean_diff}\ncovariance: {cov_diff}")

        # Early stop criteria
        if mean_diff < np.sqrt(tol_cov)*2 and cov_diff < tol_cov:
            # Save estimates, uncertainties and errors
            cmro2_est_enkf.append(cmro2_mean)
            cmro2_cov_est_enkf.append(cmro2_cov)
            errors_enkf.append(np.abs(error_enkf)) # Save the absolute errors
            state_ensembles.append(enkf.ensemble.copy()) # Save the ensemble distribution for uncertainty quatitfication
            
            print(f"Converged on data point, \narteriole:{art_id}\ndepth_id:{dth_id}")
            break
        else:
            power_factor_it += 1
            enkf.set_observation_noise(R)

    # If it didn't converge
    cmro2_est_enkf.append(cmro2_mean)
    cmro2_cov_est_enkf.append(cmro2_cov)
    errors_enkf.append(np.abs(error_enkf)) # Save the absolute errors
    state_ensembles.append(enkf.ensemble.copy()) # Save the ensemble distribution for uncertainty quatitfication
    obs_covariances_matrices.append(power_factor_it)
    print(f"Didn't converged on data point, \narteriole:{art_id}\ndepth_id:{dth_id}")
    
    # Print results in the terminal
    print(f"\n\n Ensemble Kalman Filter paramaters estimation")
    print("-"*65)
    print(f"\nCMRO2 Mean: {cmro2_mean}, Rves: {Rves}, R0: {R0}, CMRO2 √(Cov): {np.sqrt(cmro2_cov)}\n")
    

cmro2_est_enkf = np.array(cmro2_est_enkf)
cmro2_cov_est_enkf = np.array(cmro2_cov_est_enkf)
errors_enkf = np.array(errors_enkf)
state_ensembles = np.array(state_ensembles)


# ----------------------+ Plots the results +----------------------#
# Simulated iteration steps
x_obs = np.arange(1, len(observations_id) + 1)

# Stats
overall_mean = cmro2_est_enkf.mean()

# -----------------------
# Ratio M Stats
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


data = np.mean(data, axis=0)
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
# Absolute Error Stats
# Stats
data = errors_enkf.T # Define data

P.figure()

bp = P.boxplot(data, labels=names)

for i in range(numBoxes):
    y = data[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('Absolute Partial Pressure Error')
P.title('Absolute Errors Rpartition - EnKF')
P.grid(True)
P.show()

# -----------------------
# Uncertainty associated to estimation
data = state_ensembles * cmro2_by_M
fig, ax = plt.subplots(figsize=(10, 6))
cov_track = np.array([np.std(array) for array in data])
ax.plot(cov_track)
# Labels and title
plt.ylabel('Estimated CMRO2 Uncertainty (umol /cm^3 /min)')
plt.xlabel('Id DataPoint')
plt.title('EnKF Uncertainty')
plt.show()

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
ax.set_ylabel('U')
ax.set_zlabel('PDF f(U,t)')
ax.set_title('PDF of U(t) over time via KDE')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.show()