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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from circlesearch import Po2Analyzer
from scipy.ndimage import gaussian_filter

# --------- Load data --------- #
df = pd.read_pickle(py_data_location + "/dataset.pkl")
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
art_id = 7
dth_id = 2
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

# # Find circles
# analyzer = Po2Analyzer(pO2_value, M, r0=60.0)
# analyzer.find_circles()
# analytic_map = analyzer.compute_analytical_maps()[0]
# analytic_map_perturbated = np.random.normal(analytic_map.flatten(), scale=1.0)
# analytic_map_perturbated = analytic_map_perturbated.reshape(grid_shape, order='F')

# r_in = analyzer.rin
# r_out = analyzer.rout
# center = analyzer.center
# mask_outer = analyzer.mask_outer
# mask_inner = analyzer.mask_inner
# mask_angle = analyzer.mask_angle
# circumference_out = analyzer.circumference_out

# print(f'The ratio M is: {M * 1e3} * 1e-3')
# print(f'r_in is: {r_in}')
# print(f'r_out is: {r_out}')

# # --------- Plot Original + Circles ---------
# theta = np.linspace(0, 2 * np.pi, 100) # angles
# rin = r_in / pixel_size
# rout = r_out / pixel_size
# # Outer circle
# circle_outer_x = rout * np.cos(theta) + center[0]
# circle_outer_y = rout * np.sin(theta) + center[1]
# # Inner circle
# circle_inner_x = rin * np.cos(theta) + center[0]
# circle_inner_y = rin * np.sin(theta) + center[1]

plt.pcolor(pO2_value, cmap='jet', shading='auto')
# plt.plot(circle_outer_x, circle_outer_y, '--', linewidth=2, color='cyan', label=f'Outer | radius = {r_out} μm')
# plt.plot(circle_inner_x, circle_inner_y, '-', linewidth=2, color='magenta', label=f'Inner | radius = {r_in} μm ')
# plt.plot(center[0], center[1], 'x', color='black', label='Center')
plt.axis('equal')
plt.colorbar()
plt.title("Inner and Outer radius search")
plt.legend()
plt.show()

# ###################################################################################################

# # 3D Vizualition of the Laplacian of the Analytical solution Vs. the measurement
# x = np.linspace(-10, 10, 20) * pixel_size
# y = np.linspace(-10, 10, 20) * pixel_size

# # Create meshgrid 3D plot
# X, Y = np.meshgrid(x, y)

# # Michaelis-Menten Enzyme kinetics
# """
# Modeling oxygen or glucose consumption in tissue
# """
# def cmro2(p, M0, p50): 
#     return M0 * p / (p50 + p) * cmro2_by_M


# # True observation
# pvessel = 60.0
# cmro2_true = 2.0
# M0 = cmro2_true / cmro2_by_M
# Rves = 10.0
# R0 = 80.0
# Rt = 80.0

# # Generate the 1D / 2D Map
# generator = MapGenerator(
#     cmro2=cmro2_true,
#     pvessel=pvessel,
#     Rves=Rves,
#     R0=R0,
#     Rt=Rt,
# )

# # Set my arrays
# r_data = np.linspace(10., 100, 200)
# p_data = generator._partial_pressure(r_data)
# r_values = generator.r_values

# # Define the coefficient p50
# tolerance = 0.02
# idx = np.where((p_data >= pvessel/2 - tolerance) & (p_data <= pvessel/2 + tolerance))
# p50 = p_data[idx[0][0]]

# x = r_data
# y = cmro2(p_data, M0, p50) 
# plt.figure(figsize=(10, 6))
# plt.plot(x, y)
# plt.xlabel('Radius (um)')
# plt.ylabel('CMRO2 (umol /cm^3 /min)')
# plt.show()

# --------------------------------


# # Data
# array = analytic_map

# from scipy.ndimage import convolve

# def cylindrical_laplacian_2d(f, dr, r_values):

#   """
#   Compute the cylindrical Laplacian of a 2D axisymmetric array.

#   Args:
#       f: 2D numpy array (axisymmetric, so ∂f/∂θ = 0)
#       dr: Radial step size
#       r_values: 2D array of radial distances from center

#   Returns:
#       2D array of Laplacian values
#   """
#   # Avoid division by zero at r=0
#   r_values = np.where(r_values == 0, 1e-10, r_values)

#   # First derivative using central differences
#   df_dr = np.gradient(f, dr, axis=0)

#   # Multiply by r
#   r_df_dr = r_values * df_dr

#   # Second derivative using central differences
#   d2f_dr2 = np.gradient(r_df_dr, dr, axis=0)

#   # Final Laplacian
#   laplacian = d2f_dr2 / r_values

#   return laplacian

# rows, cols = 20, 20
# dr = 10.0 # radial step size

# # Create radial coordinate grid
# X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

# # Center the outer circle in the middle of the pixel cells
# dx = (X - (2*center[0] - 1) / 2) * dr
# dy = (Y - (2*center[1] - 1) / 2) * dr
# r = np.sqrt(dx**2 + dy**2)
# f = generator.pO2_array
# laplacian_ = - cylindrical_laplacian_2d(f, dr, r) * cmro2_by_M

# laplacian = np.where(r<20., np.nan, laplacian_)

# # Quantity(ies) of interest
# Z = laplacian * cmro2_by_M

# Z_tild = 1.44 * np.ones((20, 20))
# Z_plot = np.where(Z > Z_tild, np.nan, Z_tild)

# Z_modes = 5.0 * np.sin(2*np.pi*X / 200) * np.sin(2*np.pi*Y / 200)

# fig = plt.figure(figsize=(10, 6))
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(X, Y, Z, cmap='viridis')
# surf2 = ax.plot_surface(X, Y, Z_plot, cmap='jet', edgecolor='none', alpha=0.3)
# ax.view_init(elev=30, azim=-45) # Change the viewing angle
# ax.set_ylabel('Y')
# ax.set_xlabel('X')
# ax.set_zlabel('Lplacian')
# ax.set_title('3D PDF Surface of Laplacian of the partial Pressure')
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
# plt.tight_layout()
# plt.show()

# --------------------------------
import math

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
angle_ranges = [(110, 160), (265, 315)]

# Modify distance range to target from a given radius (e.g., 8 units) to the edge of the map
min_radius = 4
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

