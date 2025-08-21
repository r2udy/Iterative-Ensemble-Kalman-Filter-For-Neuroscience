# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 16:05:49 2025

@author: ruudy
"""

import sys
import os

py_file_location = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/classes/"
sys.path.append(os.path.abspath(py_file_location))

import numpy as np
import matplotlib.pyplot as plt
from MapGenerator import MapGenerator
from EnKF_3 import EnKF
from lsqnonlin_M_pvessel_rout import Po2Fitter_3

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

#######################################
#############
# KF / EnKF #
#############

# ------------------+ Synthetic Data +------------------ #

# No dynamic model
def dynamics_model(x):
  return x

seed = np.random.seed(1)


cmro2_est_lsqnonlin = []
cmro2_est_enkf = []
cmro2_cov_est_enkf = []
means = []
covs = []
cmro2_true_values = np.linspace(1., 3., 5)
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
    generator.plot_3d(array=obs_)
    
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
    # EnKF Parameters
    state_dim = 3
    obs_dim = 400
    n_ensembles = 1000
    enkf = EnKF(state_dim, obs_dim, n_ensembles, dynamics_model, seed)
    
    ##
    #Initialize the ensemble
    a = np.array([cmro2_lower / cmro2_by_M, np.max(obs_perturbated), 70.0])
    b = np.array([cmro2_upper / cmro2_by_M, sigma**2, 10.**2])
    enkf.initialize_ensemble(a, b)
    Q = np.array([[M_std**2, 0., 0.], # Background covariance matrix
                  [0.,       2.**2, 0.],
                  [0.,       0., 10.**2]])
    R = sigma**2 * np.eye(obs_dim)
    # Simulate a sequence with observation for the uniform case
    # Observations
    obs = obs_perturbated
    
    # Update the the background and observation noise
    enkf.set_process_noise(Q)
    enkf.set_observation_noise(R)
    
    # EnKF steps
    enkf.predict()
    enkf.update(obs)
    
    # Get current estimate
    mean, cov = enkf.get_state_estimate()
    cmro2_mean = mean[0] * cmro2_by_M
    cmro2_est_enkf.append(cmro2_mean)
    cmro2_cov = cov[0, 0] * (cmro2_by_M)**2
    cmro2_cov_est_enkf.append(cmro2_cov)
    
    means.append(mean)
    covs.append(cov)
    print("\n\n Ensemble Kalman Filter paramaters estimation:")
    print("-"*65)
    print(f"\nCMRO2 Mean: {cmro2_mean}, CMRO2 √(Cov): {np.sqrt(cmro2_cov)}, cmro2 Covariance: {cmro2_cov}\n")
    print(f"{'Max Pressure'} (mmHg)     | {'R0'} (um)")
    print("-"*65)
    print(f"{mean[1]:0.3f} mmHg         | {mean[2]:0.3f} (um)")
    
    generator_est = MapGenerator(
        cmro2=mean[0]*cmro2_by_M,
        pvessel=mean[1],
        Rves=Rves,
        R0=80.,
        Rt=80.,
        model=model
    )
    
    # Plot surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, obs_, 
                           cmap='viridis',
                           rstride=1, cstride=1,
                           linewidth=0, 
                           antialiased=True)
    ax.plot_surface(X, Y, generator_est.pO2_array, 
                           cmap='viridis',
                           rstride=1, cstride=1,
                           linewidth=0, 
                           antialiased=True, alpha=0.5)
    # Add labels and colorbar
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_zlabel('Partial Pressure (mmHg)')
    ax.set_title('Estimation of the map', fontsize=22)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.show()

cmro2_est_enkf = np.array(cmro2_est_enkf)
cmro2_cov_est_enkf = np.array(cmro2_cov_est_enkf)
means = np.array(means)
covs = np.array(covs)
# cmro2_est_lsqnonlin = np.array(cmro2_est_lsqnonlin)


#################################################################### PLOTS

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