#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 11:55:54 2025

@author: ruudybayonne
"""

import os
import sys

py_data_location = '/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/'
py_file_location = '/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/classes'
sys.path.append(py_file_location)

import numpy as np
import pandas as pd
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from circlesearch import Po2Analyzer
from Po2Dataset import load_data, get_cells_by_angle
import pylab as P


# --------- Load data --------- #
df = pd.read_pickle(py_data_location + "dataset.pkl")
df_copy = df.copy()
df_copy['pO2Value'] = df_copy['pO2Value'].apply(lambda x: x.flatten())
df_copy.keys()
uniform_dataset = load_data(py_data_location + 'uniform_dataset.txt')
observations_id = [entry for entry in uniform_dataset]
state_ensembles = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/uncertainty_radial_spatially_varying_1st_set/exponential/state_ensembles_100.npy")


cmro2_means_radialexp = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/uncertainty_radial_spatially_varying_1st_set/exponential/cmro2_means_100.npy")
cmro2_covs_radialexp = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/uncertainty_radial_spatially_varying_1st_set/exponential/cmro2_covs_100.npy")
cmro2_means_radiallinear = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/uncertainty_radial_spatially_varying_1st_set/linear/cmro2_means_100.npy")
cmro2_covs_radiallinear = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/uncertainty_radial_spatially_varying_1st_set/linear/cmro2_covs_100.npy")
cmro2_means_constant = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/uncertainty_radial_spatially_varying_1st_set/constant/cmro2_means_100.npy")
cmro2_covs_constant = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/uncertainty_radial_spatially_varying_1st_set/constant/cmro2_covs_100.npy")
cmro2_means_constant2areas = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/uncertainty_radial_spatially_varying_1st_set/constant_2areas/cmro2_means_100.npy")
cmro2_covs_constant2areas = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/uncertainty_radial_spatially_varying_1st_set/constant_2areas/cmro2_covs_100.npy")


corrections = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/uncertainty_radial_spatially_varying_1st_set/exponential/corrections_100.npy")
errors_enkf_relative = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/uncertainty_radial_spatially_varying_1st_set/exponential/errors_enkf_relative_100.npy")
errors_enkf_absolute = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/uncertainty_radial_spatially_varying_1st_set/exponential/errors_enkf_absolute_100.npy")
p_vessel_est = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/uncertainty_radial_spatially_varying_1st_set/exponential/p_vessel_est_100.npy")
p_vessel_true = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/uncertainty_radial_spatially_varying_1st_set/exponential/p_vessel_true_100.npy") 





# ------------------------------------------------------------------
# ----------------------+ Plots the results +----------------------#
# Constants initial #
D = 4.0e3
alpha = 1.39e-15
cmro2_by_M = (60 * D * alpha * 1e12)
cmro2_lower, cmro2_upper = 1., 3.
# -----------------------

x_obs = np.arange(1, len(observations_id) + 1) # Simulated iteration steps
# -----------------------
# CMRO_2 Stats for converged iterations
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

# -----------------------
# Uncertainty associated to estimation
# Create figure
plt.figure(figsize=(10, 6))
cmro2_mean_ = cmro2_means_radialexp
cmro2_cov_ = cmro2_covs_radialexp
# Plot mean +/- 1 standard deviation (sqrt of variance)
plt.plot(x_obs, cmro2_means_radialexp, '-o', color='green', label='State EnKF estimate (CMRO2)')
plt.fill_between(
    x_obs,
    cmro2_mean_ - np.sqrt(cmro2_cov_),  # Lower bound (mean - σ)
    cmro2_mean_ + np.sqrt(cmro2_cov_),  # Upper bound (mean + σ)
    color='blue',
    alpha=0.2,
    label='Uncertainty (+/- 1 StD)'
)

plt.plot(x_obs, cmro2_means_radiallinear, '-o', color='purple', label='State EnKF estimate (CMRO2) - linear')
plt.fill_between(
    x_obs,
    cmro2_means_radiallinear - np.sqrt(cmro2_covs_radiallinear),  # Lower bound (mean - σ)
    cmro2_means_radiallinear + np.sqrt(cmro2_covs_radiallinear),  # Upper bound (mean + σ)
    color='orange',
    alpha=0.2,
    label='Uncertainty (+/- 1 StD) - linear'
)

plt.plot(x_obs, cmro2_means_constant, '-o', color='brown', label='State EnKF estimate (CMRO2) - constant')
plt.fill_between(
    x_obs,
    cmro2_means_constant - np.sqrt(cmro2_covs_constant),  # Lower bound (mean - σ)
    cmro2_means_constant + np.sqrt(cmro2_covs_constant),  # Upper bound (mean + σ)
    color='pink',
    alpha=0.2,
    label='Uncertainty (+/- 1 StD) - constant'
)


plt.xlabel('$PO_{2}$ Map ID')
plt.ylabel('CMRO2 (umol /cm^3 /min)')
plt.title('EnKF CMRO2 Estimation with Uncertainty')
plt.xticks(x_obs, [f'Obs{i}' for i in x_obs])
# plt.axhline(y=np.mean(cmro2_mean_), color='r', linestyle='--', label='Mean CMRO2')
plt.axhline(y=cmro2_lower, color='orange', linestyle='--', label='CMRO2 Lower Bound (Prior)')
plt.axhline(y=cmro2_upper, color='orange', linestyle='--', label='CMRO2 Upper Bound (Prior)')
plt.legend()
plt.grid(True)
plt.show()


