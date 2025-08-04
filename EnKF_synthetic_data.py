#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 11:36:47 2025

@author: ruudybayonne
"""

import sys
import os

py_file_location = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/classes/"
sys.path.append(os.path.abspath(py_file_location))

import numpy as np
import matplotlib.pyplot as plt
from MapGenerator import MapGenerator
from EnKF import EnKF
import pylab as P

# --------- Perform the estimation ---------

#####################
# Constants initial #
#####################
D = 4.0e3
alpha = 1.39e-15
cmro2_low, cmro2_high = 1, 3 # umol/cm3/min
cmro2_by_M = (60 * D * alpha * 1e12)

cmro2_lower, cmro2_upper = 1.0, 3.0
cmro2_var = (cmro2_upper - cmro2_lower)**2 / 12
M_std = np.sqrt(cmro2_var) / cmro2_by_M

pixel_size = 10.0

# Create coordinate grids in physical units (microns)
cols, rows = 20, 20
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
X = X * pixel_size
Y = Y * pixel_size

# EnKF Parameters
state_dim = 1
obs_dim = 400
n_ensembles = 10

#######################################

# ------------------+ Synthetic Data +------------------ #

# No dynamic model
def dynamics_model(x):
  return x

seed = np.random.seed(1)

observations = []
cmro2_est_lsqnonlin = []
cmro2_est_enkf = []
cmro2_cov_est_enkf = []
means = []
covs = []
cmro2_true_values = np.linspace(1., 3., 5)
state_ensembles = np.zeros((len(cmro2_true_values), n_ensembles))
for i, cmro2_true in enumerate(cmro2_true_values):
    
    # True observation
    pvessel = 100.0
    M = cmro2_true / cmro2_by_M
    Rves = 10.0
    R0=80.
    Rt=80.
    model = 'KE'

    generator = MapGenerator(
        cmro2=cmro2_true,
        pvessel=pvessel,
        Rves=Rves,
        R0=R0,
        Rt=Rt,
        model=model
    )
    
    # Add noise to the generated data
    true_obs = generator.pO2_array
    sigma = 2.0
    obs_perturbated = np.random.normal(true_obs.flatten(), scale=sigma)
    n = 20 # data size
    obs_ = obs_perturbated.reshape((n, n), order='F')
    
    # # ----------------------+ LSQNonLin +----------------------#
    # ##
    # ## CMRO2 + pvessel + rout fitting ##
    # lsqnonlin_fitter = Po2Fitter_3(pO2_array=obs_, Rves=Rves, Rt=Rt, pixel_size=10.0)
    # lsqnonlin_fitter.fit()
    # lsqnonlin_fitter.plot_estimated_parameters()
    
    # cmro2_lsqnonlin = lsqnonlin_fitter.get_results()[0]
    # cmro2_est_lsqnonlin.append(cmro2_lsqnonlin)
    
    ##
    # ----------------------+ EnKF +----------------------#
    enkf = EnKF(state_dim, obs_dim, n_ensembles, dynamics_model, seed)
    
    ##
    #Initialize the ensemble
    a = np.array([cmro2_lower / cmro2_by_M])
    b = np.array([cmro2_upper / cmro2_by_M])
    enkf.initialize_ensemble(a, b)
    Q = np.array([[M_std**2]]) # Background covariance matrix
    R = sigma**2 * np.eye(obs_dim) # Observation covariance matrix
    
    # Simulate a sequence with observation for the uniform case
    # Observations
    obs = obs_perturbated
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
    cmro2_est_enkf.append(cmro2_mean)
    cmro2_cov = cov * (cmro2_by_M)**2
    cmro2_cov_est_enkf.append(cmro2_cov)
    
    means.append(mean)
    covs.append(cov)
    print("\n\n Ensemble Kalman Filter paramaters estimation:")
    print("-"*65)
    print(f"\nCMRO2 Mean: {cmro2_mean}, CMRO2 √(Cov): {np.sqrt(cmro2_cov)}, cmro2 Covariance: {cmro2_cov}\n")
    
cmro2_est_enkf = np.array(cmro2_est_enkf)
cmro2_cov_est_enkf = np.array(cmro2_cov_est_enkf)
means = np.array(means)
covs = np.array(covs)
# cmro2_est_lsqnonlin = np.array(cmro2_est_lsqnonlin)
observations = np.array(observations)


# --------- Plots the results ---------
# Simulated iteration steps
observations_id = [1, 2, 3, 4, 5]
observations = np.array(observations)
x = np.arange(1, len(observations_id) + 1)
print(observations.shape)
print(state_ensembles.shape)
# Stats
overall_mean = cmro2_est_enkf.mean()

data = state_ensembles.T * cmro2_by_M # Define data
numBoxes = len(observations) # Define numBoxes
names = ['obs 1', 'obs 2', 'obs 3', 'obs 4', 'obs 5']

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
    label='Uncertainty (±1σ)'
)

# Labels and title
plt.ylabel('Estimated CMRO2 (umol /cm^3 /min)')
plt.xlabel('Input CMRO2 (umol /cm^3 /min)')
plt.title('EnKF State Estimation with Uncertainty using Krogh-Erlang Cylinder Model')
plt.legend()
plt.grid(True)
plt.show()

# abs_error_lsqnonlin = np.abs(cmro2_true_values - cmro2_est_lsqnonlin)
abs_error_enkf = np.abs(cmro2_true_values - cmro2_est_enkf)

# Create figure
plt.figure(figsize=(10, 6))
# plt.plot(x, abs_error_lsqnonlin, '-x', label='LSQNonLin Absolute Error')
plt.plot(x, abs_error_enkf, '-x', label='EnsKF Absolute Error')

# Labels and title
plt.ylabel('Absolute error of CMRO2')
plt.xlabel('Input CMRO2 (umol /cm^3 /min)')
plt.title('EnKF State Error Estimation using Krogh-Erlang Cylinder Model')
plt.legend()
plt.grid(True)
plt.show()