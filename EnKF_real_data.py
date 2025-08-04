#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 11:58:05 2025

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
from EnKF import EnKF
from circlesearch import Po2Analyzer
from MapGenerator import MapGenerator
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
# Create a set of all (art_id, dth_id) pairs for O(1) lookups
pair_set = {entry[0] for entry in uniform_dataset}

# --------- Perform the estimation ---------

# Constants initial
D = 4.0e3
alpha = 1.39e-15
cmro2_low, cmro2_high = 1, 3 # umol/cm3/min
cmro2_by_M = (60 * D * alpha * 1e12)

cmro2_lower, cmro2_upper = 1.0, 3.0
cmro2_var = (cmro2_upper - cmro2_lower)**2 / 12
M_std = np.sqrt(cmro2_var) / cmro2_by_M

n = 20 # data size
pixel_size = 10

# EnKF Parameters
seed = np.random.seed(0)
state_dim = 1
obs_dim = 400
n_ensembles = 100

# Create coordinate grids in physical units (microns)
cols, rows = 20, 20
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
X = X * pixel_size
Y = Y * pixel_size

# No dynamic model
def dynamics_model(x):
    return x

# Initialization of Arrays
observations_id = [entry for entry in uniform_dataset]
errors = []
observations = []
cmro2_est_enkf = []
cmro2_cov_est_enkf = []
means = []
covariances = []

# Data Subset 
for entry in uniform_dataset:
    art_id = entry[0][0]
    dth_id = entry[0][1]
    print(f'art_id: {art_id}, dth_id: {dth_id}')

# Create the EnKF
enkf = EnKF(state_dim, obs_dim, n_ensembles, dynamics_model, seed)

##
#Initialize the ensemble
cmro2_lower, cmro2_upper = 1.0, 3.0
cmro2_var = (cmro2_upper - cmro2_lower)**2 / 12
M_std = np.sqrt(cmro2_var) * 2 / cmro2_by_M

a = np.array([cmro2_lower / cmro2_by_M])
b = np.array([cmro2_upper / cmro2_by_M])
enkf.initialize_ensemble(a, b)

# Covariance matrix
Q = np.array([[M_std**2]])     # Background Covariance Matrix
R = 9. * np.eye(obs_dim)       # Observation Covariance Matrix

##
# Simulate a sequence with observation for the uniform case
cmro2_est_lsqnonlin = []
state_ensembles = np.zeros((len(uniform_dataset), n_ensembles))
for i, entry in enumerate(uniform_dataset):
    art_id = entry[0][0]
    dth_id = entry[0][1]
    
    # Observations
    obs = df_copy[(df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id)]['pO2Value'].tolist()[0]
    observations.append(obs)
    
    # Update the the background and observation noise
    enkf.set_process_noise(Q)
    enkf.set_observation_noise(R)
    
    # EnKF steps
    enkf.predict()
    enkf.update(obs)
    
    # Get current estimate
    mean, cov = enkf.get_state_estimate()
    state_ensembles[i, :] = enkf.ensemble
    
    cmro2_mean = mean[0] * cmro2_by_M
    cmro2_cov = cov * (cmro2_by_M)**2
    
    cmro2_est_enkf.append(cmro2_mean)
    cmro2_cov_est_enkf.append(cmro2_cov)
    
    means.append(mean)
    covariances.append(cov)
    
    pO2_array = obs.reshape((n, n), order='F')
    cmro2 = cmro2_mean
    M = cmro2_mean / cmro2_by_M
    pvessel = obs.max()

    # Find circles
    analyzer = Po2Analyzer(pO2_array, pixel_size, M)
    analyzer.find_circles()

    Rves = analyzer.rin
    R0 = analyzer.rout
    Rt = analyzer.rout
    generator = MapGenerator(cmro2=cmro2, pvessel=pvessel, Rves=Rves, R0=R0, Rt=Rt)
    error = obs - generator.pO2_array.flatten()
    errors.append(np.abs(error))


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create coordinate grids in physical units (microns)
    X = np.meshgrid(np.arange(n), np.arange(n))[0] * pixel_size
    Y = np.meshgrid(np.arange(n), np.arange(n))[1] * pixel_size

    # Plot surface
    surf = ax.plot_surface(X, Y, pO2_array, 
                            cmap='viridis',
                            rstride=1, cstride=1,
                            linewidth=0, 
                            antialiased=True,
                            alpha=0.3)
    
    surf = ax.plot_surface(X, Y, generator.pO2_array, 
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
    print(f"Label         | {'CMRO2'} (umol /cm^3 /min)")
    print("-"*65)
    print(f"Means         | {cmro2_mean:0.3f} umol /cm^3 /min ")
    print(f"Covariances   | {cmro2_cov:0.3f} umol /cm^3 /min ")
    print(f"Std Deviation | {np.sqrt(cmro2_cov):0.3f} umol /cm^3 /min ")

cmro2_est_enkf = np.array(cmro2_est_enkf)
cmro2_cov_est_enkf = np.array(cmro2_cov_est_enkf)
means = np.array(means)
covariances = np.array(covariances)
errors = np.array(errors)
print("\n\nTotal mean: ")
print(f"Means | {np.mean(means, axis=0)[0]*cmro2_by_M:0.3f} umol /cm^3 /min ")
std_cmro2 = np.sqrt(np.mean(covariances)) * cmro2_by_M
print(f"Std Deviation | {std_cmro2:0.3f} umol /cm^3 /min ")


# --------- Plots the results ---------

# Simulated iteration steps
x = np.arange(1, len(observations_id) + 1)

# Stats
overall_mean = cmro2_est_enkf.mean()

data = state_ensembles.T * cmro2_by_M # Define data
numBoxes = len(uniform_dataset) # Define numBoxes
names = ['observation 1', 'observation 2', 'observation 3', 'observation 4']

P.figure()

bp = P.boxplot(data, labels=names)

for i in range(numBoxes):
    y = data[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value CMRO2 (umol /cm^3 /min)')
P.title('EnKF State Estimation with Uncertainty')
P.show()



# Simulated iteration steps
x = np.arange(1, len(observations_id) + 1)

# Stats
data = errors.T # Define data
numBoxes = len(uniform_dataset) # Define numBoxes
names = ['observation 1', 'observation 2', 'observation 3', 'observation 4']

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