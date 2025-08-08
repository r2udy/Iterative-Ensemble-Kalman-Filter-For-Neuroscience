# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 16:43:41 2025

@author: ruudy
"""

import os
import sys

py_data_location = os.path.join(os.getcwd(), "Data")
py_file_location = os.path.join(os.getcwd(), "classes")
sys.path.append(py_file_location)

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from circlesearch import Po2Analyzer
from scipy.ndimage import gaussian_filter
from Po2Dataset import load_data

# --------- Load data --------- #
df = pd.read_pickle(py_data_location + "/dataset.pkl")
df_copy = df.copy()
df_copy['pO2Value'] = df_copy['pO2Value'].apply(lambda x: x.flatten())
df_copy.keys()

uniform_dataset = load_data(py_data_location + '/uniform_dataset.txt')
# Create a set of all (art_id, dth_id) pairs for O(1) lookups
pair_set = {entry[0] for entry in uniform_dataset}

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

pixel_size = 10

####################
# Data Vizualation #
####################

# Select your the data
art_id, dth_id = (5, 2)
n = 20 # data size

# Load the map
array = df_copy[(df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id )]['pO2Value'].tolist()[0]
X = df_copy[(df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id )]['pointsX'].tolist()[0]
Y = df_copy[(df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id )]['pointsY'].tolist()[0]
grid_shape = (n, n)
pO2_value = array.reshape(grid_shape, order='F')

# Constants
D = 4.0e3
alpha = 1.39e-15
cmro2_true = 2.0 # umol/cm3/min
cmro2_by_M = (60 * D * alpha * 1e12)
M = cmro2_true / cmro2_by_M
pixel_size = 10.0

plt.pcolor(pO2_value, cmap='jet', shading='auto')
plt.axis('equal')
plt.colorbar()
plt.title("Inner and Outer radius search")
plt.legend()
plt.show()

# --------------------------------
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

grid_size = 20
origin = (10, 10)

# Angle ranges: from 0 to 90 degrees and from 270 to 360 degrees
angle_ranges = [(75, 80), (80, 180)]

# Modify distance range to target from a given radius (e.g., 8 units) to the edge of the map
min_radius = 5
max_radius = math.hypot(grid_size, grid_size)  # furthest possible distance in the grid

# Get updated selected cells
selected_cells_border = get_cells_by_angle(
    grid_size,
    origin,
    angle_ranges,
    distance_range=(min_radius, max_radius)
)

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal')
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)

# Draw grid
for x in range(grid_size + 1):
    ax.axvline(x, color='lightgray', linewidth=0.5)
for y in range(grid_size + 1):
    ax.axhline(y, color='lightgray', linewidth=0.5)

# Highlight selected cells
for x, y in selected_cells_border:
    rect = plt.Rectangle((x, y), 1, 1, color='lightgreen')
    ax.add_patch(rect)

# Mark origin
ox, oy = origin
origin_rect = plt.Rectangle((ox, oy), 1, 1, color='red')
ax.add_patch(origin_rect)

# Axis labels
ax.set_xticks(range(grid_size))
ax.set_yticks(range(grid_size))
ax.grid(False)

plt.title("Cells from Radius to Border in Given Angle Ranges")
plt.gca().invert_yaxis()
plt.show()



# Create a 400x400 matrix initialized to zeros
matrix_size = 400
matrix = np.zeros((matrix_size, matrix_size))

# Set higher values for targeted cells and lower for non-targeted ones
high_value = 20.
low_value = -20.

# Create a 400x400 matrix for diagonals representing each cell of the 20x20 grid
matrix_diag = np.zeros((matrix_size, matrix_size))

# Flatten the 20x20 grid into a 1D list of 400 positions corresponding to diagonals
grid_cells = [(x, y) for y in range(grid_size) for x in range(grid_size)]

# Assign higher or lower values to each diagonal cell based on whether it was targeted
for i, (x, y) in enumerate(grid_cells):
    value = high_value if (x, y) in selected_cells_border else low_value
    matrix_diag[i, i] = value

# Show the matrix
plt.figure(figsize=(6, 6))
plt.imshow(matrix_diag, cmap='viridis', origin='upper')
plt.title("Diagonal Matrix: 2D Map Cell Encoding")
plt.colorbar(label='Value')
plt.show()

