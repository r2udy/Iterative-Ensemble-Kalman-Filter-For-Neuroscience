#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 14:35:46 2025

@author: ruudybayonne

@description: This code performs the Ensemble Kalman filter procedure on the a subset of the dataset 
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
from EnKF_FEM_2 import EnKF
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


# --------------------------
# Initial Constants #
D = 4.0e3
alpha = 1.39e-15
cmro2_by_M = (60 * D * alpha * 1e12)

cmro2_lower, cmro2_upper = 1.0, 3.0
cmro2_var = (cmro2_upper - cmro2_lower)**2 / 12
prior_cmro2_mean = (cmro2_lower + cmro2_upper) / 2
prior_R0_mean = 90.
prior_M_var = cmro2_var / cmro2_by_M**2
M_std = np.sqrt(cmro2_var) / cmro2_by_M # model uncertainty
obs_var = 3.0**2 # measurement uncertainty
R0_var = 5.0**2 # prior uncertainty of caparilary-free space radius

n = 20 # data size
pixel_size = 10

# --------------------------
# EnKF Parameters
seed = np.random.seed(1)
state_dim = 2
obs_dim = 400
n_ensembles = 100

# Covariance Matrix
Q   = np.array([[prior_M_var,     0.0],
                [0.0,          R0_var]])   # Background covariance matrix
R   = obs_var * np.eye(obs_dim)    # Observation covariance matrix

# Create coordinate grids in physical units (microns)
cols, rows = 20, 20
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
X = X * pixel_size
Y = Y * pixel_size

# No dynamic model
def dynamics_model(x):
    return x

# --------------------------
# Initialization of Arrays
observations_id = [entry for entry in uniform_dataset]
observations = []
cmro2_est_enkf = []
cmro2_est_lsqnonlin = []
cmro2_cov_est_enkf = []
R0_est_enkf = []
R0_cov_est_enkf = []
means = []
covariances = []
errors_enkf = []
errors_lsq = []
state_ensembles_M = np.zeros((len(observations_id), n_ensembles))
state_ensembles_R0 = np.zeros((len(observations_id), n_ensembles))

# Create the EnKF object
enkf = EnKF(state_dim, obs_dim, n_ensembles, dynamics_model, seed)

# Initialize the ensemble
a = np.array([ prior_cmro2_mean / cmro2_by_M, prior_R0_mean])
b = np.array([np.sqrt(prior_M_var), np.sqrt(R0_var)])
enkf.initialize_ensemble(a, b)
    
# Update the the background and observation noise1
enkf.set_process_noise(Q)
enkf.set_observation_noise(R)

# --------------------------
# Simulate a sequence with observation for the uniform case
# --------------------------
for i, entry in enumerate(uniform_dataset):
    art_id = entry[0][0]
    dth_id = entry[0][1]
    
    # Observations
    obs = df_copy[(df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id)]['pO2Value'].tolist()[0]
    obs = np.array(obs)
    sigma = 1.5
    obs_perturbated = np.random.normal(obs, scale=sigma)
    observations.append(obs)
    
    # ----------------------+ LSQNonLin +----------------------#
    pO2_array = obs_perturbated.reshape((n, n), order='F')
    # Find circles
    analyzer = Po2Analyzer(pO2_array)
    analyzer.find_circles()
    Rves = analyzer.rin
    R0 = analyzer.rout
    ##
    ## CMRO2 + pvessel + rout fitting ##
    lsqnonlin_fitter = Po2Fitter(pO2_array=pO2_array, Rves=Rves)
    lsqnonlin_fitter.fit()
    
    cmro2_lsqnonlin = lsqnonlin_fitter.get_results()[0]
    cmro2_est_lsqnonlin.append(cmro2_lsqnonlin)
    
    ##
    # ----------------------+ EnKF +----------------------#
    # EnKF steps
    enkf.predict()
    enkf.update(obs_perturbated)
    
    # Get current estimate
    mean, cov = enkf.get_state_estimate()
    state_ensembles_M[i, :] = enkf.get_ensemble()[0, :]
    state_ensembles_R0[i, :] = enkf.get_ensemble()[1, :]

    # Means
    cmro2_mean = mean[0] * cmro2_by_M
    R0_mean    = mean[1]
    cmro2_est_enkf.append(cmro2_mean)
    R0_est_enkf.append(R0_mean)
    
    # Covariances
    cmro2_cov   = cov[0 ,0] * (cmro2_by_M)**2
    R0_cov      = cov[1, 1]
    cmro2_cov_est_enkf.append(cmro2_cov)
    R0_cov_est_enkf.append(cmro2_cov)
    
    means.append(mean)
    covariances.append(cov)
    
    print("\n\n Ensemble Kalman Filter paramaters estimation:")
    print("-"*65)
    print(f"\nCMRO2 Mean: {cmro2_mean}, CMRO2 √(Cov): {np.sqrt(cmro2_cov)}, cmro2 Covariance: {cmro2_cov}")
    print("-"*25)
    print(f"R0 Mean: {R0_mean}, R0 √(Cov): {np.sqrt(R0_cov)}, R0 Covariance: {R0_cov}\n")
    lsqnonlin_fitter.plot_estimated_parameters() # print estimated parameters from nonlinear least square fitting

    # Generate the maps
    generator_enkf = MapGenerator(cmro2=cmro2_mean, 
                                  pvessel=obs.max(),
                                  Rves=Rves, 
                                  R0=R0_mean, 
                                  Rt=R0_mean)
    generator_lsq = MapGenerator(cmro2=cmro2_lsqnonlin, 
                                 pvessel=lsqnonlin_fitter.pvessel, 
                                 Rves=lsqnonlin_fitter.rin, 
                                 R0=lsqnonlin_fitter.rout, 
                                 Rt=lsqnonlin_fitter.rout)
    
    # Compute the absolute errors
    error_enkf  = obs - generator_enkf.pO2_array.flatten()
    error_lsq   = obs - generator_lsq.pO2_array.flatten()
    errors_enkf.append(np.abs(error_enkf))
    errors_lsq.append(np.abs(error_lsq))

    # # Plot the predictions
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')

    # # Plot surface
    # surf = ax.plot_surface(X, Y, pO2_array, 
    #                         cmap='jet',
    #                         rstride=1, cstride=1,
    #                         linewidth=0, 
    #                         antialiased=True)
    
    # surf = ax.plot_surface(X, Y, generator_enkf.pO2_array, 
    #                         cmap='viridis',
    #                         rstride=1, cstride=1,
    #                         linewidth=0, 
    #                         antialiased=True,
    #                         alpha=0.8)
    
    # surf = ax.plot_surface(X, Y, generator_lsq.pO2_array, 
    #                         cmap='magma',
    #                         rstride=1, cstride=1,
    #                         linewidth=0, 
    #                         antialiased=True,
    #                         alpha=0.3)

    # # Add labels and colorbar
    # ax.set_xlabel('X (µm)')
    # ax.set_ylabel('Y (µm)')
    # ax.set_zlabel('Partial Pressure (mmHg)')
    # ax.set_title(f'The predicted CMRO2_enkf={cmro2_mean:0.3f}, CMRO2_lsq={cmro2_lsqnonlin:0.3f}', fontsize=18)
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    # plt.tight_layout()
    # plt.show()
    
cmro2_est_enkf = np.array(cmro2_est_enkf)
cmro2_cov_est_enkf = np.array(cmro2_cov_est_enkf)
means = np.array(means)
covariances = np.array(covariances)
errors_enkf = np.array(errors_enkf)
errors_lsq = np.array(errors_lsq)

# ----------------------+ Plots the results +----------------------#
# Simulated iteration steps
x = np.arange(1, len(observations_id) + 1)

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

data = np.mean(data, axis=0)
P.figure()
bp = P.boxplot(data, labels=['Overall Stats'])

y = data
x = np.random.normal(1+i, 0.04, size=len(y))
P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value CMRO2 (umol /cm^3 /min)')
P.title('EnKF State Estimation with Uncertainty - Overall')
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
data = errors_enkf.T # Define data

P.figure()

bp = P.boxplot(data, labels=names)

for i in range(numBoxes):
    y = data[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('Absolute Partial Pressure Error - EnKF')
P.title('Errors repartition')
P.grid(True)
P.show()


# Stats
data = errors_lsq.T # Define data

P.figure()

bp = P.boxplot(data, labels=names)

for i in range(numBoxes):
    y = data[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('Absolute Partial Pressure Error - LSQ')
P.title('Errors repartition')
P.grid(True)
P.show()
