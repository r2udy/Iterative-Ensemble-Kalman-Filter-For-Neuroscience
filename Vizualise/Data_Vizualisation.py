# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 16:43:41 2025

@author: ruudy
"""

import os
import sys

py_data_location = '/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data'
py_file_location = '/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/classes'
sys.path.append(py_file_location)

import math
import numpy as np
import pandas as pd
import matplotlib.tri as tri
import matplotlib.pyplot as plt
from circlesearch import Po2Analyzer
from MapGenerator import MapGenerator
from Po2Dataset import load_data, get_cells_by_angle
from EnKF_FEM import build_obs_covariance, build_obs_covariance_diagonal, build_obs_covariance_radial
import pylab as P

# --------- Load data --------- #
df = pd.read_pickle(py_data_location + "/dataset.pkl")
df_copy = df.copy()
df_copy['pO2Value'] = df_copy['pO2Value'].apply(lambda x: x.flatten())
df_copy.keys()

uniform_dataset = load_data(py_data_location + '/uniform_dataset.txt')
# Create a set of all (art_id, dth_id) pairs for O(1) lookups
pair_set = {entry[0] for entry in uniform_dataset}




# ------------------- 
# Constants initial 
D = 4.0e3
alpha = 1.39e-15
cmro2_by_M = (60 * D * alpha * 1e12)

# -------------------
# Data Vizualation #
art_id, dth_id = (5, 2)
# Angle ranges: from 0 to 90 degrees and from 270 to 360 degrees
angle_ranges = [(170, 290), (170, 290)]
min_radius = 5
grid_size = 20 # data sizes



# Load the map
mask    = (df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id)
array   = df_copy[mask]['pO2Value'].tolist()[0]
X       = df_copy[mask]['pointsX'].tolist()[0]
Y       = df_copy[mask]['pointsY'].tolist()[0]
Z       = array
# Reshape the array to 20x20
Z_array = np.array(Z).reshape((grid_size, grid_size), order='F')

# Find circles
analyzer = Po2Analyzer(Z_array, X, Y)
analyzer.find_circles()

r_in = analyzer.rin
r_out = analyzer.rout
center = analyzer.center
center_ij = analyzer.center_ij

print(f"Center (ij): {center_ij}")
print(f"r_in = {r_in:.2f}, r_out = {r_out:.2f}, center = {center}")



# ---------------
# Figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

theta = np.linspace(0, 2 * np.pi, 100) # angles
circle_in_x = center[0] + r_in*np.cos(theta)
circle_in_y = center[1] + r_in*np.sin(theta)
circle_out_x = center[0] + r_out*np.cos(theta)
circle_out_y = center[1] + r_out*np.sin(theta)

ax = axes[0]
c = ax.pcolormesh(X, Y, Z_array, shading='auto', cmap='jet')

fig.colorbar(c, label='pO₂')
ax.plot(circle_in_x, circle_in_y, "m-", lw=2, label=f"Inner r={r_in:.1f}")
ax.plot(circle_out_x, circle_out_y, "c--", lw=2, label=f"Outer r={r_out:.1f}")
ax.plot(center[0], center[1], "kx", ms=8, label="Center")
ax.set_title(f"Radial pO₂ Map | Arteriole {art_id} | Depth {dth_id}")
ax.set_xlabel('X (nm)')
ax.set_ylabel('Y (nm)')
ax.axis('equal')
ax.legend()



# --------------------------------
# Example usage
C = build_obs_covariance_radial(
    origin = center_ij,
    obs_var_high = 15.0**2,
    obs_var_low = 1.0**2,
    mode="exponential"
)

# --- Quick visualization: diagonal (variance) map ---
uncertainty_map = np.diag(C).reshape((grid_size, grid_size))
ax = axes[1]
c = ax.pcolormesh(X, Y, uncertainty_map, cmap='viridis')
fig.colorbar(c, label='Variance')
ax.set_title("Diagonal Variance Map of PO2 Covariance Matrix")
ax.set_xlabel('X (pixels)')
ax.set_ylabel('Y (pixels)')
ax.axis('equal')
plt.show()



# --------- Plot Original + Circles ---------
triang = tri.Triangulation(X.flatten(), Y.flatten())

plt.figure(figsize=(7, 6))
plt.tripcolor(triang, Z, shading="flat", cmap="jet")
plt.colorbar(label="pO₂")

theta = np.linspace(0, 2 * np.pi, 100) # angles
circle_in_x = center[0] + r_in*np.cos(theta)
circle_in_y = center[1] + r_in*np.sin(theta)
circle_out_x = center[0] + r_out*np.cos(theta)
circle_out_y = center[1] + r_out*np.sin(theta)

plt.plot(circle_in_x, circle_in_y, "m-", lw=2, label=f"Inner r={r_in:.1f}")
plt.plot(circle_out_x, circle_out_y, "c--", lw=2, label=f"Outer r={r_out:.1f}")
plt.plot(center[0], center[1], "kx", ms=8, label="Center")

plt.axis("equal")
plt.title(f"Radial pO₂ Map | Arteriole {art_id} | Depth {dth_id}")
plt.legend()
plt.show()

# --------------------------------
