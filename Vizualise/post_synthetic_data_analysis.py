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


# 21
state_ensembles_21 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/21_aug/state_ensembles_50.npy")
error_ensembles_21 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/21_aug/errors_enkf_absolute_50.npy")
# 22
state_ensembles_22 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/22_aug/state_ensembles_50.npy") 
error_ensembles_22 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/22_aug/errors_enkf_absolute_50.npy")
# 23
state_ensembles_23 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/23_aug/state_ensembles_50.npy") 
error_ensembles_23 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/23_aug/errors_enkf_absolute_50.npy")
# 24
state_ensembles_24 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/24_aug/state_ensembles_50.npy")
error_ensembles_24 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/24_aug/errors_enkf_absolute_50.npy")


# 31
state_ensembles_31 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/31_aug/state_ensembles_50.npy")
error_ensembles_31 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/31_aug/errors_enkf_absolute_50.npy")
# 32
state_ensembles_32 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/32_aug/state_ensembles_50.npy") 
error_ensembles_32 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/32_aug/errors_enkf_absolute_50.npy")
# 33
state_ensembles_33 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/33_aug/state_ensembles_50.npy") 
error_ensembles_33 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/33_aug/errors_enkf_absolute_50.npy")
# 34
state_ensembles_34 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/34_aug/state_ensembles_50.npy")
error_ensembles_34 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/34_aug/errors_enkf_absolute_50.npy")
# 35
state_ensembles_35 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/35_aug/state_ensembles_50.npy")
error_ensembles_35 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/35_aug/errors_enkf_absolute_50.npy")
# 36
state_ensembles_36 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/36_aug/state_ensembles_50.npy")
error_ensembles_36 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/36_aug/errors_enkf_absolute_50.npy")

# 52
state_ensembles_52 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/52_aug/state_ensembles_50.npy")
error_ensembles_52 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/52_aug/errors_enkf_absolute_50.npy")
# 53
state_ensembles_53 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/53_aug/state_ensembles_50.npy")
error_ensembles_53 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/53_aug/errors_enkf_absolute_50.npy")
# 54
state_ensembles_54 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/54_aug/state_ensembles_50.npy")
error_ensembles_54 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/54_aug/errors_enkf_absolute_50.npy")
# 55
state_ensembles_55 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/55_aug/state_ensembles_50.npy")
error_ensembles_55 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/55_aug/errors_enkf_absolute_50.npy")
# 56
state_ensembles_56 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/56_aug/state_ensembles_50.npy")
error_ensembles_56 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/56_aug/errors_enkf_absolute_50.npy")

# 71
state_ensembles_71 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/71_aug_1/state_ensembles_50.npy") 
error_ensembles_71 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/71_aug_1/errors_enkf_absolute_50.npy")
# 72
state_ensembles_72 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/72_aug_0/state_ensembles_50.npy")
error_ensembles_72 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/72_aug_0/errors_enkf_absolute_50.npy")
# 73
state_ensembles_73 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/73_aug/state_ensembles_50.npy")
error_ensembles_73 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/73_aug/errors_enkf_absolute_50.npy")
# 74
state_ensembles_74 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/74_aug/state_ensembles_50.npy")
error_ensembles_74 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/74_aug/errors_enkf_absolute_50.npy")
# 75
state_ensembles_75 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/75_aug/state_ensembles_50.npy")
error_ensembles_75 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/75_aug/errors_enkf_absolute_50.npy")

# 81
state_ensembles_81 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/81_aug/state_ensembles_50.npy")
error_ensembles_81 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/81_aug/errors_enkf_absolute_50.npy")
# 82
state_ensembles_82 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/82_aug/state_ensembles_50.npy")
error_ensembles_82 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/82_aug/errors_enkf_absolute_50.npy")
# 83
state_ensembles_83 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/83_aug/state_ensembles_50.npy")
error_ensembles_83 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/83_aug/errors_enkf_absolute_50.npy")

# 91
state_ensembles_91 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/91_aug_0/state_ensembles_50.npy")
error_ensembles_91 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/91_aug_0/errors_enkf_absolute_50.npy")
# 92
state_ensembles_92 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/92_aug/state_ensembles_50.npy")
error_ensembles_92 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/92_aug/errors_enkf_absolute_50.npy")
# 93
state_ensembles_93 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/93_aug/state_ensembles_50.npy")
error_ensembles_93 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/93_aug/errors_enkf_absolute_50.npy")
# 94
state_ensembles_94 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/94_aug/state_ensembles_50.npy")
error_ensembles_94 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/94_aug/errors_enkf_absolute_50.npy")
# 95
state_ensembles_95 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/95_aug/state_ensembles_50.npy")
error_ensembles_95 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/95_aug/errors_enkf_absolute_50.npy")

# 101
state_ensembles_101 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/101_aug/state_ensembles_50.npy")
error_ensembles_101 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/101_aug/errors_enkf_absolute_50.npy")
# 102
state_ensembles_102 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/102_aug/state_ensembles_50.npy")
error_ensembles_102 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/102_aug/errors_enkf_absolute_50.npy")
# 104
state_ensembles_104 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/104_aug/state_ensembles_50.npy")
error_ensembles_104 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/104_aug/errors_enkf_absolute_50.npy")
# 106
state_ensembles_106 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/106_aug/state_ensembles_50.npy")
error_ensembles_106 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/106_aug/errors_enkf_absolute_50.npy")

# 111
state_ensembles_111 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/111_aug/state_ensembles_50.npy")
error_ensembles_111 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/111_aug/errors_enkf_absolute_50.npy")
# 112
state_ensembles_112 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/112_aug/state_ensembles_50.npy")
error_ensembles_112 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/112_aug/errors_enkf_absolute_50.npy")
# 115
state_ensembles_115 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/115_aug/state_ensembles_50.npy")
error_ensembles_115 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/Test_stats/115_aug/errors_enkf_absolute_50.npy")

# ----------------------+ Plots the results +---------------------- #
# Constants initial #
D = 4.0e3
alpha = 1.39e-15
cmro2_by_M = (60 * D * alpha * 1e12)

# -----------------------------
# 22, 23 and 24

cmro2_21 = state_ensembles_21[0, -1, :]
R0_21 = state_ensembles_21[1, -1, :]
pvessel_21 = state_ensembles_21[2, -1, :]
error_21 = error_ensembles_21[-1, :]

cmro2_22 = state_ensembles_22[0, -1, :]
R0_22 = state_ensembles_22[1, -1, :]
pvessel_22 = state_ensembles_22[2, -1, :]
error_22 = error_ensembles_22[-1, :]

cmro2_23 = state_ensembles_23[0, -1, :]
R0_23 = state_ensembles_23[1, -1, :]
pvessel_23 = state_ensembles_23[2, -1, :]
error_23 = error_ensembles_23[-1, :]

cmro2_24 = state_ensembles_24[0, -1, :]
R0_24 = state_ensembles_24[1, -1, :]
pvessel_24 = state_ensembles_24[2, -1, :]
error_24 = error_ensembles_24[-1, :]

cmro2_2 = np.vstack([cmro2_21, cmro2_22, cmro2_23, cmro2_24]).T * cmro2_by_M
numBoxes = cmro2_2.shape[1]
names = [f'obs 2{i}' for i in range(1, numBoxes + 1)]
P.figure()
bp = P.boxplot(cmro2_2, labels=names)
for i in range(numBoxes):
    y = cmro2_2[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value CMRO2 (umol /cm^3 /min)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs21-24')
P.grid(True)
P.show()

R0_2 = np.vstack([R0_21, R0_22, R0_23, R0_24]).T
numBoxes = R0_2.shape[1]
P.figure()
bp = P.boxplot(R0_2, labels=names)
for i in range(numBoxes):
    y = R0_2[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value R0 (um)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs21-24')
P.grid(True)
P.show()

pvessel_2 = np.vstack([pvessel_21, pvessel_22, pvessel_23, pvessel_24]).T
numBoxes = pvessel_2.shape[1]
P.figure()
bp = P.boxplot(pvessel_2, labels=names)
for i in range(numBoxes):
    y = pvessel_2[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value Pvessel (mmHg)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs21-24')
P.grid(True)
P.show()   

errors_2 = np.vstack([error_21, error_22, error_23, error_24]).T
numBoxes = errors_2.shape[1]
P.figure()
bp = P.boxplot(errors_2, labels=names)
for i in range(numBoxes):
    y = errors_2[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('Relative Partial Pressure Error')
P.title('Relative Errors distributions - EnKF')
P.grid(True)
P.show()

# Data
x_obs = np.arange(1, 5)  # 4 observations: 21, 22, 23 and 24
labels = [f'Obs 2{i}' for i in x_obs]

experimental_means = cmro2_2.mean(axis=0)
experimental_err = cmro2_2.std(axis=0)
width = 0.35

# Plot
plt.figure(figsize=(8, 5))
plt.bar(x_obs + width/2, experimental_means, width,
        yerr=experimental_err, capsize=5,
        label="Experimental")
plt.title("CMRO2 Estimation: EnKF with 3 States (Obs 21-24)")
plt.xticks(x_obs, labels)
plt.ylabel('CMRO2 (umol /cm^3 /min)')
plt.legend()
plt.tight_layout()
plt.show()



# -----------------------------
# 31, 32, 33, 34, 35 and 36

cmro2_31 = state_ensembles_31[0, -1, :]
R0_31 = state_ensembles_31[1, -1, :]
pvessel_31 = state_ensembles_31[2, -1, :]
error_31 = error_ensembles_31[-1, :]

cmro2_32 = state_ensembles_32[0, -1, :]
R0_32 = state_ensembles_32[1, -1, :]
pvessel_32 = state_ensembles_32[2, -1, :]
error_32 = error_ensembles_32[-1, :]

cmro2_33 = state_ensembles_33[0, -1, :]
R0_33 = state_ensembles_33[1, -1, :]
pvessel_33 = state_ensembles_33[2, -1, :]
error_33 = error_ensembles_33[-1, :]

cmro2_34 = state_ensembles_34[0, -1, :]
R0_34 = state_ensembles_34[1, -1, :]
pvessel_34 = state_ensembles_34[2, -1, :]
error_34 = error_ensembles_34[-1, :]

cmro2_35 = state_ensembles_35[0, -1, :]
R0_35 = state_ensembles_35[1, -1, :]
pvessel_35 = state_ensembles_35[2, -1, :]
error_35 = error_ensembles_35[-1, :]

cmro2_36 = state_ensembles_36[0, -1, :]
R0_36 = state_ensembles_36[1, -1, :]
pvessel_36 = state_ensembles_36[2, -1, :]
error_36 = error_ensembles_36[-1, :]

cmro2_3 = np.vstack([cmro2_31, cmro2_32, cmro2_33, cmro2_34, cmro2_35]).T * cmro2_by_M
numBoxes = cmro2_3.shape[1]
names = [f'obs 3{i}' for i in range(1, numBoxes + 1)]
P.figure()
bp = P.boxplot(cmro2_3, labels=names)
for i in range(numBoxes):
    y = cmro2_3[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value CMRO2 (umol /cm^3 /min)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs31-35')
P.grid(True)
P.show()

R0_3 = np.vstack([R0_31, R0_32, R0_33, R0_34, R0_35]).T
numBoxes = R0_3.shape[1]
P.figure()
bp = P.boxplot(R0_3, labels=names)
for i in range(numBoxes):
    y = R0_3[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value R0 (um)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs31-35')
P.grid(True)
P.show()

pvessel_3 = np.vstack([pvessel_31, pvessel_32, pvessel_33, pvessel_34, pvessel_35]).T
numBoxes = pvessel_3.shape[1]
P.figure()
bp = P.boxplot(pvessel_3, labels=names)
for i in range(numBoxes):
    y = pvessel_3[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value Pvessel (mmHg)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs31-35')
P.grid(True)
P.show()   

errors_3 = np.vstack([error_31, error_32, error_33, error_34, error_35]).T
numBoxes = errors_3.shape[1]
P.figure()
bp = P.boxplot(errors_3, labels=names)
for i in range(numBoxes):
    y = errors_3[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('Relative Partial Pressure Error')
P.title('Relative Errors distributions - EnKF')
P.grid(True)
P.show()

# Data
x_obs = np.arange(1, 6)  # 5 observations: 31, 32, 33, 34, 35
labels = [f'Obs 3{i}' for i in x_obs]

experimental_means = cmro2_3.mean(axis=0)
experimental_err = cmro2_3.std(axis=0)
width = 0.35

# Plot
plt.figure(figsize=(8, 5))
plt.bar(x_obs + width/2, experimental_means, width,
        yerr=experimental_err, capsize=5,
        label="Experimental")
plt.title("CMRO2 Estimation: EnKF with 3 States (Obs 31-35)")
plt.xticks(x_obs, labels)
plt.ylabel('CMRO2 (umol /cm^3 /min)')
plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------
# 52, 53, 54, 55 and 56

cmro2_52 = state_ensembles_52[0, -1, :]
R0_52 = state_ensembles_52[1, -1, :]
pvessel_52 = state_ensembles_52[2, -1, :]
error_52 = error_ensembles_52[-1, :]

cmro2_53 = state_ensembles_53[0, -1, :]
R0_53 = state_ensembles_53[1, -1, :]
pvessel_53 = state_ensembles_53[2, -1, :]
error_53 = error_ensembles_53[-1, :]

cmro2_54 = state_ensembles_54[0, -1, :]
R0_54 = state_ensembles_54[1, -1, :]
pvessel_54 = state_ensembles_54[2, -1, :]
error_54 = error_ensembles_54[-1, :]

cmro2_55 = state_ensembles_55[0, -1, :]
R0_55 = state_ensembles_55[1, -1, :]
pvessel_55 = state_ensembles_55[2, -1, :]
error_55 = error_ensembles_55[-1, :]

cmro2_56 = state_ensembles_56[0, -1, :]
R0_56 = state_ensembles_56[1, -1, :]
pvessel_56 = state_ensembles_56[2, -1, :]
error_56 = error_ensembles_56[-1, :]

cmro2_5 = np.vstack([cmro2_52, cmro2_53, cmro2_54, cmro2_55, cmro2_56]).T * cmro2_by_M
numBoxes = cmro2_5.shape[1]
names = [f'obs 5{i}' for i in range(2, numBoxes + 2)]
P.figure()
bp = P.boxplot(cmro2_5, labels=names)
for i in range(numBoxes):
    y = cmro2_5[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value CMRO2 (umol /cm^3 /min)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs52-54')
P.grid(True)
P.show()

R0_5 = np.vstack([R0_52, R0_53, R0_54, R0_55, R0_56]).T
numBoxes = R0_5.shape[1]
P.figure()
bp = P.boxplot(R0_5, labels=names)
for i in range(numBoxes):
    y = R0_5[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value R0 (um)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs52-54')
P.grid(True)
P.show()

pvessel_5 = np.vstack([pvessel_52, pvessel_53, pvessel_54, pvessel_55, pvessel_56]).T
numBoxes = pvessel_5.shape[1]
P.figure()
bp = P.boxplot(pvessel_5, labels=names)
for i in range(numBoxes):
    y = pvessel_5[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value Pvessel (mmHg)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs52-54')
P.grid(True)
P.show()   

errors_5 = np.vstack([error_52, error_53, error_54, error_55, error_56]).T
numBoxes = errors_5.shape[1]
P.figure()
bp = P.boxplot(errors_5, labels=names)
for i in range(numBoxes):
    y = errors_5[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('Relative Partial Pressure Error')
P.title('Relative Errors distributions - EnKF')
P.grid(True)
P.show()

# Data
x_obs = np.arange(1, 6)  # 5 observations: 52, 53, 54, 55, 56
labels = [f'Obs 5{i}' for i in x_obs]

experimental_means = cmro2_5.mean(axis=0)
experimental_err = cmro2_5.std(axis=0)

width = 0.35

# Plot
plt.figure(figsize=(8, 5))
plt.bar(x_obs + width/2, experimental_means, width,
        yerr=experimental_err, capsize=5,
        label="Experimental")
plt.title("CMRO2 Estimation: EnKF vs Non-Linear LSQ")
plt.xticks(x_obs, labels)
plt.ylabel('CMRO2 (umol /cm^3 /min)')
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 71, 72, 73, 74 and 75

cmro2_71 = state_ensembles_71[0, -1, :]
R0_71 = state_ensembles_71[1, -1, :]
pvessel_71 = state_ensembles_71[2, -1, :]
error_71 = error_ensembles_71[-1, :]

cmro2_72 = state_ensembles_72[0, -1, :]
R0_72 = state_ensembles_72[1, -1, :]
pvessel_72 = state_ensembles_72[2, -1, :]
error_72 = error_ensembles_72[-1, :]

cmro2_73 = state_ensembles_73[0, -1, :]
R0_73 = state_ensembles_73[1, -1, :]
pvessel_73 = state_ensembles_73[2, -1, :]
error_73 = error_ensembles_73[-1, :]

cmro2_74 = state_ensembles_74[0, -1, :]
R0_74 = state_ensembles_74[1, -1, :]
pvessel_74 = state_ensembles_74[2, -1, :]
error_74 = error_ensembles_74[-1, :]

cmro2_75 = state_ensembles_75[0, -1, :]
R0_75 = state_ensembles_75[1, -1, :]
pvessel_75 = state_ensembles_75[2, -1, :]
error_75 = error_ensembles_75[-1, :]

cmro2_7 = np.vstack([cmro2_71, cmro2_72, cmro2_73, cmro2_74, cmro2_75]).T * cmro2_by_M

numBoxes = cmro2_7.shape[1]
names = [f'obs 7{i}' for i in range(1, numBoxes + 1)]
P.figure()
bp = P.boxplot(cmro2_7, labels=names)

for i in range(numBoxes):
    y = cmro2_7[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value CMRO2 (umol /cm^3 /min)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs71-74')
P.grid(True)
P.show()


R0_7 = np.vstack([R0_71, R0_72, R0_73, R0_74, R0_75]).T

numBoxes = R0_7.shape[1]
P.figure()
bp = P.boxplot(R0_7, labels=names)
for i in range(numBoxes):
    y = R0_7[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value R0 (um)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs71-74')
P.grid(True)
P.show()

pvessel_7 = np.vstack([pvessel_71, pvessel_72, pvessel_73, pvessel_74, pvessel_75]).T
numBoxes = pvessel_7.shape[1]
P.figure()
bp = P.boxplot(pvessel_7, labels=names)
for i in range(numBoxes):
    y = pvessel_7[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value Pvessel (mmHg)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs71-74')
P.grid(True)
P.show()

errors_7 = np.vstack([error_71, error_72, error_73, error_74, error_75]).T
numBoxes = errors_7.shape[1]
P.figure()
bp = P.boxplot(errors_7, labels=names)
for i in range(numBoxes):
    y = errors_7[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('Relative Partial Pressure Error')
P.title('Relative Errors distributions - EnKF')
P.grid(True)
P.show()

# Data
x_obs = np.arange(1, 6)  # 5 observations: 71, 72, 73, 74 and 75
labels = [f'Obs 7{i}' for i in x_obs]

experimental_means = cmro2_7.mean(axis=0)
experimental_err = cmro2_7.std(axis=0)

width = 0.35

# Plot
plt.figure(figsize=(8, 5))
plt.bar(x_obs + width/2, experimental_means, width,
        yerr=experimental_err, capsize=5,
        label="Experimental")
plt.title("CMRO2 Estimation: EnKF: ")
plt.xticks(x_obs, labels)
plt.ylabel('CMRO2 (umol /cm^3 /min)')
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 81, 82 and 83

cmro2_81 = state_ensembles_81[0, -1, :]
R0_81 = state_ensembles_81[1, -1, :]
pvessel_81 = state_ensembles_81[2, -1, :]
error_81 = error_ensembles_81[-1, :]

cmro2_82 = state_ensembles_82[0, -1, :]
R0_82 = state_ensembles_82[1, -1, :]
pvessel_82 = state_ensembles_82[2, -1, :]
error_82 = error_ensembles_82[-1, :]

cmro2_83 = state_ensembles_83[0, -1, :]
R0_83 = state_ensembles_83[1, -1, :]
pvessel_83 = state_ensembles_83[2, -1, :]
error_83 = error_ensembles_83[-1, :]

cmro2_8 = np.vstack([cmro2_81, cmro2_82, cmro2_83]).T * cmro2_by_M
numBoxes = cmro2_8.shape[1]
names = [f'obs 8{i}' for i in range(1, numBoxes + 1)]
P.figure()
bp = P.boxplot(cmro2_8, labels=names)
for i in range(numBoxes):
    y = cmro2_8[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value CMRO2 (umol /cm^3 /min)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs81-83')
P.grid(True)
P.show()

R0_8 = np.vstack([R0_81, R0_82, R0_83]).T
numBoxes = R0_8.shape[1]
P.figure()
bp = P.boxplot(R0_8, labels=names)
for i in range(numBoxes):
    y = R0_8[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value R0 (um)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs81-83')
P.grid(True)
P.show()

pvessel_8 = np.vstack([pvessel_81, pvessel_82, pvessel_83]).T
numBoxes = pvessel_8.shape[1]
P.figure()
bp = P.boxplot(pvessel_8, labels=names)
for i in range(numBoxes):
    y = pvessel_8[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value Pvessel (mmHg)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs81-83')
P.grid(True)
P.show()

error_ensembles_8 = np.vstack([error_81, error_82, error_83]).T 
numBoxes = error_ensembles_8.shape[1]
P.figure()
bp = P.boxplot(error_ensembles_8, labels=names)
for i in range(numBoxes):
    y = error_ensembles_8[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('Relative Partial Pressure Error')
P.title('Relative Errors distributions - EnKF')
P.grid(True)
P.show()

# Data
x_obs = np.arange(1, 4)  # 3 observations: 81, 82, 83
labels = [f'Obs 8{i}' for i in x_obs]

experimental_means_8 = cmro2_8.mean(axis=0)
experimental_err_8 = cmro2_8.std(axis=0)

width = 0.35

# Plot
plt.figure(figsize=(8, 5))
plt.bar(x_obs + width/2, experimental_means_8, width,
        yerr=experimental_err_8, capsize=5,
        label="Experimental")
plt.title("CMRO2 Estimation: EnKF vs Non-Linear LSQ")
plt.xticks(x_obs, labels)
plt.ylabel('CMRO2 (umol /cm^3 /min)')
plt.legend()
plt.tight_layout()

# -----------------------------
# 91, 92, 93, 94 and 95

cmro2_91 = state_ensembles_91[0, -1, :]
R0_91 = state_ensembles_91[1, -1, :]
pvessel_91 = state_ensembles_91[2, -1, :]
error_91 = error_ensembles_91[-1, :]

cmro2_92 = state_ensembles_92[0, -1, :]
R0_92 = state_ensembles_92[1, -1, :]
pvessel_92 = state_ensembles_92[2, -1, :]
error_92 = error_ensembles_92[-1, :]

cmro2_93 = state_ensembles_93[0, -1, :]
R0_93 = state_ensembles_93[1, -1, :]
pvessel_93 = state_ensembles_93[2, -1, :]
error_93 = error_ensembles_93[-1, :]

cmro2_94 = state_ensembles_94[0, -1, :]
R0_94 = state_ensembles_94[1, -1, :]
pvessel_94 = state_ensembles_94[2, -1, :]
error_94 = error_ensembles_94[-1, :]

cmro2_95 = state_ensembles_95[0, -1, :]
R0_95 = state_ensembles_95[1, -1, :]
pvessel_95 = state_ensembles_95[2, -1, :]
error_95 = error_ensembles_95[-1, :]

cmro2_9 = np.vstack([cmro2_91, cmro2_92, cmro2_93, cmro2_94, cmro2_95]).T * cmro2_by_M
numBoxes = cmro2_9.shape[1]
names = [f'obs 9{i}' for i in range(1, numBoxes + 1)]
P.figure()
bp = P.boxplot(cmro2_9, labels=names)
for i in range(numBoxes):
    y = cmro2_9[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value CMRO2 (umol /cm^3 /min)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs92-94')
P.grid(True)
P.show()

R0_9 = np.vstack([R0_91, R0_92, R0_93, R0_94, R0_95]).T
numBoxes = R0_9.shape[1]
P.figure()
bp = P.boxplot(R0_9, labels=names)
for i in range(numBoxes):
    y = R0_9[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value R0 (um)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs92-94')
P.grid(True)
P.show()

pvessel_9 = np.vstack([pvessel_91, pvessel_92, pvessel_93, pvessel_94, pvessel_95]).T
numBoxes = pvessel_9.shape[1]
P.figure()
bp = P.boxplot(pvessel_9, labels=names)
for i in range(numBoxes):
    y = pvessel_9[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value Pvessel (mmHg)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs92-94')
P.grid(True)
P.show()

errors_9 = np.vstack([error_91, error_92, error_93, error_94, error_95]).T 
numBoxes = errors_9.shape[1]
P.figure()
bp = P.boxplot(errors_9, labels=names)
for i in range(numBoxes):
    y = errors_9[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('Relative Partial Pressure Error')
P.title('Relative Errors distributions - EnKF')
P.grid(True)
P.show()

# Data
x_obs = np.arange(1, 6)  # 4 observations: 91, 92, 93, 94 and 95
labels = [f'Obs 9{i}' for i in x_obs]

experimental_means_9 = cmro2_9.mean(axis=0)
experimental_err_9 = cmro2_9.std(axis=0)

width = 0.35

# Plot
plt.figure(figsize=(8, 5))
plt.bar(x_obs + width/2, experimental_means_9, width,
        yerr=experimental_err_9, capsize=5,
        label="Experimental")
plt.title("CMRO2 Estimation: EnKF vs Non-Linear LSQ")
plt.xticks(x_obs, labels)
plt.ylabel('CMRO2 (umol /cm^3 /min)')
plt.legend()
plt.tight_layout()


# -----------------------------
# 101, 102, 104 and 106

cmro2_101 = state_ensembles_101[0, -1, :]
R0_101 = state_ensembles_101[1, -1, :]
pvessel_101 = state_ensembles_101[2, -1, :]
error_101 = error_ensembles_101[-1, :]

cmro2_102 = state_ensembles_102[0, -1, :]
R0_102 = state_ensembles_102[1, -1, :]
pvessel_102 = state_ensembles_102[2, -1, :]
error_102 = error_ensembles_102[-1, :]

cmro2_104 = state_ensembles_104[0, -1, :]
R0_104 = state_ensembles_104[1, -1, :]
pvessel_104 = state_ensembles_104[2, -1, :]
error_104 = error_ensembles_104[-1, :]

cmro2_106 = state_ensembles_106[0, -1, :]
R0_106 = state_ensembles_106[1, -1, :]
pvessel_106 = state_ensembles_106[2, -1, :]
error_106 = error_ensembles_106[-1, :]

cmro2_10 = np.vstack([cmro2_101, cmro2_102, cmro2_104, cmro2_106]).T * cmro2_by_M
numBoxes = cmro2_10.shape[1]
names = [f'obs 10{i}' for i in range(1, numBoxes + 1)]
P.figure()
bp = P.boxplot(cmro2_10, labels=names)
for i in range(numBoxes):
    y = cmro2_10[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value CMRO2 (umol /cm^3 /min)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs101-106')
P.grid(True)
P.show()

R0_10 = np.vstack([R0_101, R0_102, R0_104, R0_106]).T
numBoxes = R0_10.shape[1]
P.figure()
bp = P.boxplot(R0_10, labels=names)
for i in range(numBoxes):
    y = R0_10[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value R0 (um)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs101-106')
P.grid(True)
P.show()

pvessel_10 = np.vstack([pvessel_101, pvessel_102, pvessel_104, pvessel_106]).T
numBoxes = pvessel_10.shape[1]
P.figure()
bp = P.boxplot(pvessel_10, labels=names)
for i in range(numBoxes):
    y = pvessel_10[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value Pvessel (mmHg)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs101-106')
P.grid(True)
P.show()

errors_10 = np.vstack([error_101, error_102, error_104, error_106]).T 
numBoxes = errors_10.shape[1]
P.figure()
bp = P.boxplot(errors_10, labels=names)
for i in range(numBoxes):
    y = errors_10[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('Relative Partial Pressure Error')
P.title('Relative Errors distributions - EnKF')
P.grid(True)
P.show()

# Data
x_obs = np.arange(1, 5)  # 4 observations: 101, 102, 104 and 106
labels = [f'Obs 10{i}' for i in x_obs]

experimental_means_10 = cmro2_10.mean(axis=0)
experimental_err_10 = cmro2_10.std(axis=0)
width = 0.35

# Plot
plt.figure(figsize=(8, 5))
plt.bar(x_obs + width/2, experimental_means_10, width,
        yerr=experimental_err_10, capsize=5,
        label="Experimental")
plt.title("CMRO2 Estimation: EnKF vs Non-Linear LSQ")
plt.xticks(x_obs, labels)
plt.ylabel('CMRO2 (umol /cm^3 /min)')
plt.legend()
plt.tight_layout()


# -----------------------------
# 111, 112 and 115

cmro2_111 = state_ensembles_111[0, -1, :]
R0_111 = state_ensembles_111[1, -1, :]
pvessel_111 = state_ensembles_111[2, -1, :]
error_111 = error_ensembles_111[-1, :]

cmro2_112 = state_ensembles_112[0, -1, :]
R0_112 = state_ensembles_112[1, -1, :]
pvessel_112 = state_ensembles_112[2, -1, :]
error_112 = error_ensembles_112[-1, :]

cmro2_115 = state_ensembles_115[0, -1, :]
R0_115 = state_ensembles_115[1, -1, :]
pvessel_115 = state_ensembles_115[2, -1, :]
error_115 = error_ensembles_115[-1, :]

cmro2_11 = np.vstack([cmro2_111, cmro2_112, cmro2_115]).T * cmro2_by_M
numBoxes = cmro2_11.shape[1]
names = [f'obs 11{i}' for i in range(1, numBoxes + 1)]
P.figure()
bp = P.boxplot(cmro2_11, labels=names)
for i in range(numBoxes):
    y = cmro2_11[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value CMRO2 (umol /cm^3 /min)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs111-115')
P.grid(True)
P.show()

R0_11 = np.vstack([R0_111, R0_112, R0_115]).T
numBoxes = R0_11.shape[1]
P.figure()
bp = P.boxplot(R0_11, labels=names)
for i in range(numBoxes):
    y = R0_11[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value R0 (um)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs111-115')
P.grid(True)
P.show()

pvessel_11 = np.vstack([pvessel_111, pvessel_112, pvessel_115]).T
numBoxes = pvessel_11.shape[1]
P.figure()
bp = P.boxplot(pvessel_11, labels=names)
for i in range(numBoxes):
    y = pvessel_11[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('State value Pvessel (mmHg)')
P.title(f'EnKF State Estimation with Uncertainty\n for 3 states - obs111-115')
P.grid(True)
P.show()

errors_11 = np.vstack([error_111, error_112, error_115]).T 
numBoxes = errors_11.shape[1]
P.figure()
bp = P.boxplot(errors_11, labels=names)
for i in range(numBoxes):
    y = errors_11[:, i]
    x = np.random.normal(1+i, 0.04, size=len(y))
    P.plot(x, y, 'r.', alpha=0.2)
P.xlabel('$PO_{2}$ Map ID')
P.ylabel('Relative Partial Pressure Error')
P.title('Relative Errors distributions - EnKF')
P.grid(True)
P.show()

# Data
x_obs = np.arange(1, 4)  # 4 observations: 111, 112, 115
labels = [f'Obs 11{i}' for i in x_obs]

experimental_means_11 = cmro2_11.mean(axis=0)
experimental_err_11 = cmro2_11.std(axis=0)
width = 0.35

# Plot
plt.figure(figsize=(8, 5))
plt.bar(x_obs + width/2, experimental_means_11, width,
        yerr=experimental_err_11, capsize=5,
        label="Experimental")
plt.title("CMRO2 Estimation: EnKF vs Non-Linear LSQ")
plt.xticks(x_obs, labels)
plt.ylabel('CMRO2 (umol /cm^3 /min)')
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------

# # Mean CMRO2 vs Depth Plotting
import scipy.io as sio
path_metadata = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/TODEsource/dbase/"
dict_meta = sio.loadmat(path_metadata + 'database.mat')
metadata = dict_meta['main']
# Plot Mean CMRO2 vs Depth for arteriol IDs 2, 3, 5, 7, 8, and 9
art_id_2 = 2  # Arteriol ID
depth_2 = metadata['data_depth'][art_id_2-1][0][0][:4]
cmro2_2_mean = cmro2_2.mean(axis=0)
art_id_3 = 3  # Arteriol ID
depth_3 = metadata['data_depth'][art_id_3-1][0][0][:5]
cmro2_3_mean = cmro2_3.mean(axis=0)
art_id_5 = 5  # Arteriol ID
depth_5 = metadata['data_depth'][art_id_5-1][0][0][:5]
cmro2_5_mean = cmro2_5.mean(axis=0)
art_id_7 = 7  # Arteriol ID
depth_7 = metadata['data_depth'][art_id_7-1][0][0][:5]
cmro2_7_mean = cmro2_7.mean(axis=0)
art_id_8 = 8  # Arteriol ID
depth_8 = metadata['data_depth'][art_id_8-1][0][0][:3]
cmro2_8_mean = cmro2_8.mean(axis=0)
art_id_9 = 9  # Arteriol ID
depth_9 = metadata['data_depth'][art_id_9-1][0][0][:5]
cmro2_9_mean = cmro2_9.mean(axis=0)
art_10 = 10  # Arteriol ID
depth_10 = metadata['data_depth'][art_10-1][0][0][:4]
cmro2_10_mean = cmro2_10.mean(axis=0)
art_11 = 11  # Arteriol ID
depth_11 = metadata['data_depth'][art_11-1][0][0][:3]
cmro2_11_mean = cmro2_11.mean(axis=0)

# Plotting
fig = plt.figure(figsize=(10, 6))
plt.plot(depth_2, cmro2_2_mean, 'o-', label='Arteriol 2')
plt.plot(depth_3, cmro2_3_mean, 'o-', label='Arteriol 3')
plt.plot(depth_5, cmro2_5_mean, 'o-', label='Arteriol 5')
plt.plot(depth_7, cmro2_7_mean, 'o-', label='Arteriol 7')
plt.plot(depth_8, cmro2_8_mean, 'o-', label='Arteriol 8')
plt.plot(depth_9, cmro2_9_mean, 'o-', label='Arteriol 9')
plt.plot(depth_10, cmro2_10_mean, 'o-', label='Arteriol 10')
plt.plot(depth_11, cmro2_11_mean, 'o-', label='Arteriol 11')
plt.xlabel('Depth (μm)')
plt.ylabel('Mean CMRO2 (μmol/cm³/min)')
plt.title('Mean CMRO2 vs Depth')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------

depth_all = np.concatenate([depth_2, depth_3, depth_5, depth_7, depth_8, depth_9, depth_10, depth_11])
cmro2_all = np.concatenate([cmro2_2_mean, cmro2_3_mean, cmro2_5_mean, cmro2_7_mean, cmro2_8_mean, cmro2_9_mean, cmro2_10_mean, cmro2_11_mean])
cmro2_std_all = np.concatenate([cmro2_2.std(axis=0), cmro2_3.std(axis=0), cmro2_5.std(axis=0), cmro2_7.std(axis=0), cmro2_8.std(axis=0), cmro2_9.std(axis=0), cmro2_10.std(axis=0), cmro2_11.std(axis=0)])

dictionary_depth_cmro2 = {'Depth (um)': depth_all, 'Mean CMRO2 (umol/cm3/min)': cmro2_all, 'CMRO2 Std Dev': cmro2_std_all}

bin_edges = np.arange(0, 501, 100)  # 0–500 μm
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_labels = [f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" 
              for i in range(len(bin_edges)-1)]

binned_mean = []
binned_err = []

for i in range(len(bin_edges)-1):
    mask = (dictionary_depth_cmro2["Depth (um)"] >= bin_edges[i]) & (dictionary_depth_cmro2["Depth (um)"] < bin_edges[i+1])
    
    if np.any(mask):
        mean_val = np.mean(dictionary_depth_cmro2["Mean CMRO2 (umol/cm3/min)"][mask])
        
        # Combine uncertainties conservatively
        err_val = np.sqrt(np.sum(dictionary_depth_cmro2["CMRO2 Std Dev"][mask]**2)) / np.sum(mask)

        binned_mean.append(mean_val)
        binned_err.append(err_val)
    else:
        binned_mean.append(np.nan)
        binned_err.append(np.nan)

binned_mean = np.array(binned_mean)
binned_err = np.array(binned_err)
plt.figure(figsize=(8, 5))
plt.bar(
    bin_centers,
    binned_mean,
    yerr=binned_err,
    width=80,
    capsize=6,
    color="steelblue",
    edgecolor="black",
    alpha=0.8
)

plt.xticks(bin_centers, bin_labels, rotation=45)
plt.ylabel("CMRO₂ (μmol/cm³/min)")
plt.xlabel("Depth (μm)")
plt.title("CMRO₂ vs Cortical Depth")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()