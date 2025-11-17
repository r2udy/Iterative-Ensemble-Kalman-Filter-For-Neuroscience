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
from scipy.stats import gaussian_kde
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from circlesearch import Po2Analyzer
from Po2Dataset import load_data, get_cells_by_angle
import pylab as P


# --------- Load data --------- #
state_ensembles_01 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/constant_01/state_ensembles_50.npy")
state_ensembles_02 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/constant_02/state_ensembles_50.npy")
state_ensembles_03 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/constant_03/state_ensembles_50.npy")
state_ensembles_04 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/constant_04/state_ensembles_50.npy")

state_ensembles_0x = np.array([state_ensembles_01, state_ensembles_02, state_ensembles_03, state_ensembles_04])
M_vars_0x = np.array([0.33, 4.0, 0.004, 4.0e-8])
colors_0x = ['orange', 'green', 'red', 'cyan']

#
state_ensembles_21 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/constant_21/state_ensembles_50.npy")
state_ensembles_22 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/constant_22/state_ensembles_50.npy")
state_ensembles_23 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/constant_23/state_ensembles_50.npy")
state_ensembles_24 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/constant_24/state_ensembles_50.npy")
state_ensembles_25 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/constant_25/state_ensembles_50.npy")
state_ensembles_2x = np.array([state_ensembles_21, state_ensembles_22, state_ensembles_23, state_ensembles_24, state_ensembles_25])
obs_vars_2x = np.array([1., 25.0, 100., 2500., 10000.])
colors_2x = ['orange', 'green', 'red', 'cyan', 'blue']

#
state_ensembles_31 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/constant_31/state_ensembles_100.npy")
state_ensembles_32 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/constant_32/state_ensembles_100.npy")
state_ensembles_33 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/constant_33/state_ensembles_100.npy")
state_ensembles_34 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/constant_34/state_ensembles_100.npy")
state_ensembles_3x = np.array([state_ensembles_31, state_ensembles_32, state_ensembles_33, state_ensembles_34])
simga_3x = np.array([2., 5., 10., 20.])
colors_3x = ['orange', 'green', 'red', 'cyan']

#
state_ensembles_2states_02 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/constant_2states_02/state_ensembles_50.npy")
state_ensembles_2states_03 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/synthetic_data/constant_2states_03/state_ensembles_50.npy")
state_ensembles_2states_0x = np.array([state_ensembles_2states_02, state_ensembles_2states_02])
simga_20x = np.array([[20., 150.], [90., 110.]])
colors_20x = ['orange', 'green', 'red', 'cyan']
labels = [f"Posterior {i+1}th iteration" for i in range(state_ensembles_2states_02[0].shape[0])]


# ------------------------------------------------------------------
# ----------------------+ Plots the results +----------------------#
# Constants initial #
D = 4.0e3
alpha = 1.39e-15
cmro2_by_M = (60 * D * alpha * 1e12)
cmro2_lower, cmro2_upper = 1., 3.
# -----------------------

# # -----------------------
# # CMRO_2 Stats 
# x_obs = np.arange(1, state_ensembles_01.shape[0] + 1) # Simulated iteration steps
# for i, (state, var, color) in enumerate(zip(state_ensembles_0x, M_vars_0x, colors_0x), start=1):
    
#     data = state.T # shape: (n_ensembles, n_iterations)
#     numBoxes = data.shape[1]  # now robust

#     names = [f'obs{i}' for i in range(1, numBoxes + 1)]

#     P.figure()
#     bp = P.boxplot(data, labels=names)

#     for i in range(numBoxes):
#         y = data[:, i]
#         x = np.random.normal(1+i, 0.04, size=len(y))
#         P.plot(x, y, 'r.', alpha=0.2)
#     P.xlabel('$PO_{2}$ Map ID')
#     P.ylabel('State value CMRO2 (umol /cm^3 /min)')
#     P.title(f'EnKF State Estimation with Uncertainty for cmro2_var= {var}')
#     P.grid(True)

#     P.show()

# # -----------------------
# # Uncertainty associated to estimation for varying model error covariance matrix
# # Create figure
# x_obs = np.arange(1, state_ensembles_01.shape[0] + 1) # Simulated iteration steps
# plt.figure(figsize=(10, 6))

# for i, (state, var, color) in enumerate(zip(state_ensembles_0x, M_vars_0x, colors_0x), start=1):

#     cmro2_mean_0x = np.mean(state, 1)
#     cmro2_var_0x = np.var(state, 1)
#     # Plot mean +/- 1 standard deviation (sqrt of variance)
#     plt.plot(x_obs, cmro2_mean_0x, '-o', label=f'cmro2_var={var}', color=color)
#     plt.fill_between(
#         x_obs,
#         cmro2_mean_0x - np.sqrt(cmro2_var_0x),  # Lower bound (mean - σ)
#         cmro2_mean_0x + np.sqrt(cmro2_var_0x),  # Upper bound (mean + σ)
#         color=color,
#         alpha=0.2,
#         label='Uncertainty (+/- 1 StD)'
#     )
# plt.xlabel('$PO_{2}$ Map ID')
# plt.ylabel('CMRO2 (umol /cm^3 /min)')
# plt.title('EnKF CMRO2 Estimation with Uncertainty')
# plt.xticks(x_obs, [f'Obs{i}' for i in x_obs])
# plt.axhline(y=cmro2_lower, color='orange', linestyle='--', label='CMRO2 Lower Bound (Prior)')
# plt.axhline(y=cmro2_upper, color='orange', linestyle='--', label='CMRO2 Upper Bound (Prior)')
# plt.legend()
# plt.grid(True)
# plt.show()


# # -----------------------
# # CMRO_2 Stats for converged iterations 
# x_obs = np.arange(1, state_ensembles_21.shape[0] + 1) # Simulated iteration steps
# for i, (state, var, color) in enumerate(zip(state_ensembles_0x, M_vars_0x, colors_0x), start=1):
    
#     data = state.T # shape: (n_ensembles, n_iterations)
#     numBoxes = data.shape[1]  # now robust

#     names = [f'obs{i}' for i in range(1, numBoxes + 1)]

#     P.figure()
#     bp = P.boxplot(data, labels=names)

#     for i in range(numBoxes):
#         y = data[:, i]
#         x = np.random.normal(1+i, 0.04, size=len(y))
#         P.plot(x, y, 'r.', alpha=0.2)
#     P.xlabel('$PO_{2}$ Map ID')
#     P.ylabel('State value CMRO2 (umol /cm^3 /min)')
#     P.title(f'EnKF State Estimation with Uncertainty for cmro2_var= {var}')
#     P.grid(True)

#     P.show()

# # -----------------------
# # Uncertainty associated to estimation for varying observation error covariance matrix
# # Create figure
# plt.figure(figsize=(10, 6))

# for i, (state, var, color) in enumerate(zip(state_ensembles_2x, obs_vars_2x, colors_2x), start=1):

#     cmro2_var_2x = np.var(state, 1)
#     # Plot mean +/- 1 standard deviation (sqrt of variance)
#     plt.plot(x_obs, cmro2_var_2x, '-o', label=f'obs_var={var}', color=color)

# plt.xlabel('$PO_{2}$ Map ID')
# plt.ylabel('CMRO2 Variance (umol /cm^3 /min)')
# plt.title('EnKF CMRO2 Uncertainty')
# plt.xticks(x_obs, [f'Obs{i}' for i in x_obs])
# plt.yscale('log')
# # plt.axhline(y=np.mean(cmro2_mean_), color='r', linestyle='--', label='Mean CMRO2')
# plt.legend()
# plt.grid(True)
# plt.show()


# Priors Study for two-state synthetic data for CMRO2 estimation
# large prior
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(1, 2, 1)
for i, (data, color, label) in enumerate(zip(state_ensembles_2states_02[0], colors_20x, labels)):
    # Kernel Density Estimation (KDE) plot
    cmro2_ = data * cmro2_by_M
    kde = gaussian_kde(cmro2_)
    x_range = np.linspace(np.min(cmro2_)*(0.9) , np.max(cmro2_)*(1.1), 300)
    ax.plot(x_range, kde(x_range), color=color, label=label, lw=2, alpha=0.9)
    ax.fill_between(x_range, kde(x_range), color=color, alpha=0.3)
ax.set_title('Posterior Distributions of CMRO2 Estimates\n (large prior R0)', fontsize=14) # for a priors range: $\Delta R_0^{initial}$ = 130um & $\Delta CMRO_2^{initial}$ = 2 umol/cm³/min
ax.set_xlabel('CMRO2 (umol /cm^3 /min)')
ax.set_ylabel('Density')
ax.legend(title='Posteriors Distributions')
ax.grid(True, linestyle='--', alpha=0.5)


# small prior
ax = fig.add_subplot(1, 2, 2)
for i, (data, color, label) in enumerate(zip(state_ensembles_2states_03[0], colors_20x, labels)):
    # Kernel Density Estimation (KDE) plot
    cmro2_ = data * cmro2_by_M
    kde = gaussian_kde(cmro2_)
    x_range = np.linspace(np.min(cmro2_)*(0.9) , np.max(cmro2_)*(1.1), 300)
    ax.plot(x_range, kde(x_range), color=color, label=label, lw=2, alpha=0.9)
    ax.fill_between(x_range, kde(x_range), color=color, alpha=0.3)
ax.set_title('Posterior Distributions of CMRO2 Estimates\n (small prior R0)', fontsize=14) # for a priors range: $\Delta R_0^{initial}$ = 130um & $\Delta CMRO_2^{initial}$ = 2 umol/cm³/min
ax.set_xlabel('CMRO2 (umol /cm^3 /min)')
ax.set_ylabel('Density')
ax.legend(title='Posteriors Distributions')
ax.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Comparison plot
fig = plt.figure(figsize=(10, 6))
cmro2_large = state_ensembles_2states_02[0][-1] * cmro2_by_M
cmro2_small = state_ensembles_2states_03[0][-1] * cmro2_by_M
kde_large = gaussian_kde(cmro2_large)
kde_small = gaussian_kde(cmro2_small)
x_range = np.linspace(min(np.min(cmro2_large), np.min(cmro2_small))*(0.9) , max(np.max(cmro2_large), np.max(cmro2_small))*(1.1), 300)
plt.plot(x_range, kde_large(x_range), color='blue', label='Large Prior R0 = $\Delta R_0$=130um', lw=2, alpha=0.9)
plt.fill_between(x_range, kde_large(x_range), color='blue', alpha=0.3)
plt.plot(x_range, kde_small(x_range), color='green', label='Small Prior R0 = $\Delta R_0$=20um', lw=2, alpha=0.9)
plt.fill_between(x_range, kde_small(x_range), color='green', alpha=0.3)
plt.title('Comparison of Posterior Distributions of CMRO2 Estimates', fontsize=14)
plt.xlabel('CMRO2 (umol /cm^3 /min)')
plt.ylabel('Density')
plt.legend(title='Posteriors Distributions')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()




# Priors Study for two-state synthetic data for R0 estimation
# large prior
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(1, 2, 1)
for i, (data, color, label) in enumerate(zip(state_ensembles_2states_02[1], colors_20x, labels)):
    # Kernel Density Estimation (KDE) plot
    r0_ = data
    kde = gaussian_kde(r0_)
    x_range = np.linspace(np.min(r0_)*(0.9) , np.max(r0_)*(1.1), 300)
    ax.plot(x_range, kde(x_range), color=color, label=label, lw=2, alpha=0.9)
    ax.fill_between(x_range, kde(x_range), color=color, alpha=0.3)
ax.set_title('Posterior Distributions of R0 Estimates\n (large prior R0)', fontsize=14) # for a priors range: $\Delta R_0^{initial}$ = 130um & $\Delta CMRO_2^{initial}$ = 2 umol/cm³/min
ax.set_xlabel('R0 (um)')
ax.set_ylabel('Density')
ax.legend(title='Posteriors Distributions')
ax.grid(True, linestyle='--', alpha=0.5)


# small prior
ax = fig.add_subplot(1, 2, 2)
for i, (data, color, label) in enumerate(zip(state_ensembles_2states_03[1], colors_20x, labels)):
    # Kernel Density Estimation (KDE) plot
    r0_ = data
    kde = gaussian_kde(r0_)
    x_range = np.linspace(np.min(r0_)*(0.9) , np.max(r0_)*(1.1), 300)
    ax.plot(x_range, kde(x_range), color=color, label=label, lw=2, alpha=0.9)
    ax.fill_between(x_range, kde(x_range), color=color, alpha=0.3)
ax.set_title('Posterior Distributions of R0 Estimates\n (small prior R0)', fontsize=14) # for a priors range: $\Delta R_0^{initial}$ = 130um & $\Delta CMRO_2^{initial}$ = 2 umol/cm³/min
ax.set_xlabel('R0 (um)')
ax.set_ylabel('Density')
ax.legend(title='Posteriors Distributions')
ax.grid(True, linestyle='--', alpha=0.5)
plt.show()


# Comparison plot
fig = plt.figure(figsize=(10, 6))
r0_large = state_ensembles_2states_02[1][-1]
r0_small = state_ensembles_2states_03[1][-1]
kde_large = gaussian_kde(r0_large)
kde_small = gaussian_kde(r0_small)
x_range = np.linspace(min(np.min(r0_large), np.min(r0_small))*(0.9) , max(np.max(r0_large), np.max(r0_small))*(1.1), 300)
plt.plot(x_range, kde_large(x_range), color='blue', label='Large Prior R0 = $\Delta R_0$=130um', lw=2, alpha=0.9)
plt.fill_between(x_range, kde_large(x_range), color='blue', alpha=0.3)

plt.plot(x_range, kde_small(x_range), color='green', label='Small Prior R0 = $\Delta R_0$=20um', lw=2, alpha=0.9)
plt.fill_between(x_range, kde_small(x_range), color='green', alpha=0.3)
plt.title('Comparison of Posterior Distributions of R0 Estimates', fontsize=14)
plt.xlabel('R0 (um)')
plt.ylabel('Density')
plt.legend(title='Posteriors Distributions')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
