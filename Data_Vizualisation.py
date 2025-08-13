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
import pylab as P


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
observations = []
for i, entry in enumerate(uniform_dataset):
    art_id, dth_id = entry[0][0], entry[0][1]
    array = df_copy[(df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id )]['pO2Value'].tolist()[0].copy()
    observations.append(array - 14.049259355414593)
    plt.show()
observations = np.array(observations)
print(f"Total observations: {observations.shape}")

observations_min = np.zeros(observations.shape[0])
for i, map in enumerate(observations):
   observations_min[i] = np.min(map)
   print(f"Map min: {map.min()}, Map max: {map.max()}, Map mean: {map.mean()}, Map std: {map.std()}")
print(f"The means of of all map mins: {observations_min.mean()}")

# Normalize the observations
observations_normalized = (observations - np.min(observations)) / (np.max(observations) - np.min(observations))

# Select your the data
art_id, dth_id = (7, 2)
# Angle ranges: from 0 to 90 degrees and from 270 to 360 degrees
angle_ranges = [(75, 80), (80, 180)]
n = 20 # data sizes

# Load the map
array = df_copy[(df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id )]['pO2Value'].tolist()[0]
X = df_copy[(df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id )]['pointsX'].tolist()[0]
Y = df_copy[(df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id )]['pointsY'].tolist()[0]
grid_shape = (n, n)
pO2_value = array.reshape(grid_shape, order='F') - np.min(array)

# Find circles
analyzer = Po2Analyzer(pO2_value)
analyzer.find_circles()

r_in = analyzer.rin
r_out = analyzer.rout
center = analyzer.center
mask_outer = analyzer.mask_outer
mask_inner = analyzer.mask_inner
mask_angle = analyzer.mask_angle
circumference_out = analyzer.circumference_out

print(f'r_in is: {r_in}')
print(f'r_out is: {r_out}')

# --------- Plot Original + Circles ---------
theta = np.linspace(0, 2 * np.pi, 100) # angles
rin = r_in / pixel_size
rout = r_out / pixel_size
# Outer circle
circle_outer_x = rout * np.cos(theta) + center[0]
circle_outer_y = rout * np.sin(theta) + center[1]
# Inner circle
circle_inner_x = rin * np.cos(theta) + center[0]
circle_inner_y = rin * np.sin(theta) + center[1]

for angle_deg in angle_ranges[0] + (angle_ranges[1] if angle_ranges[1] else ()):
  angle_rad = np.deg2rad(angle_deg)
  x_end = rout * np.cos(angle_rad) + center[0]
  y_end = rout * np.sin(angle_rad) + center[1]
  plt.plot([center[0], x_end], [center[1], y_end], 'k--', lw=1.5)

plt.pcolor(pO2_value, cmap='jet', shading='auto')
plt.plot(circle_outer_x, circle_outer_y, '--', linewidth=2, color='cyan', label=f'Outer | radius = {r_out} μm')
plt.plot(circle_inner_x, circle_inner_y, '-', linewidth=2, color='magenta', label=f'Inner | radius = {r_in} μm ')
plt.plot(center[0], center[1], 'x', color='black', label='Center')
plt.axis('equal')
plt.colorbar()
plt.title("Inner and Outer radius search")
plt.legend()
plt.show()

# # --------------------------------
# def get_cells_by_angle(grid_size, origin, angle_ranges, distance_range=None):
#     x0, y0 = origin
#     selected_cells = []

#     for y in range(grid_size):
#         for x in range(grid_size):
#             dx = x - x0
#             dy = y0 - y  # reverse y if needed (grid coordinates)
            
#             angle = math.degrees(math.atan2(dy, dx)) % 360
#             distance = math.hypot(dx, dy)

#             # Check angle ranges
#             in_angle = any(
#                 start <= angle <= end if start <= end else angle >= start or angle <= end
#                 for (start, end) in angle_ranges
#             )
            
#             # Check distance range if given
#             in_distance = True
#             if distance_range:
#                 min_d, max_d = distance_range
#                 in_distance = min_d <= distance <= max_d

#             if in_angle and in_distance:
#                 selected_cells.append((x, y))
    
#     return selected_cells

# grid_size = 20
# origin = (10, 10)

# # Modify distance range to target from a given radius (e.g., 8 units) to the edge of the map
# min_radius = 5
# max_radius = math.hypot(grid_size, grid_size)  # furthest possible distance in the grid

# # Get updated selected cells
# selected_cells_border = get_cells_by_angle(
#     grid_size,
#     origin,
#     angle_ranges,
#     distance_range=(min_radius, max_radius)
# )

# # Plotting
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.set_aspect('equal')
# ax.set_xlim(0, grid_size)
# ax.set_ylim(0, grid_size)

# # Draw grid
# for x in range(grid_size + 1):
#     ax.axvline(x, color='lightgray', linewidth=0.5)
# for y in range(grid_size + 1):
#     ax.axhline(y, color='lightgray', linewidth=0.5)

# # Highlight selected cells
# for x, y in selected_cells_border:
#     rect = plt.Rectangle((x, y), 1, 1, color='lightgreen')
#     ax.add_patch(rect)

# # Mark origin
# ox, oy = origin
# origin_rect = plt.Rectangle((ox, oy), 1, 1, color='red')
# ax.add_patch(origin_rect)

# # Axis labels
# ax.set_xticks(range(grid_size))
# ax.set_yticks(range(grid_size))
# ax.grid(False)

# for angle_deg in angle_ranges[1] + (angle_ranges[0] if angle_ranges[0] else ()):
#   angle_rad = np.deg2rad(angle_deg)
#   x_end = rout * np.cos(angle_rad) + center[0]
#   y_end = rout * np.sin(angle_rad) + center[1]
# #   plt.plot([center[0], x_end], [center[1], y_end], 'k--', lw=1.5)


# plt.title("Cells from Radius to Border in Given Angle Ranges")
# plt.gca().invert_yaxis()
# # plt.show()



# # Create a 400x400 matrix initialized to zeros
# matrix_size = 400
# matrix = np.zeros((matrix_size, matrix_size))

# # Set higher values for targeted cells and lower for non-targeted ones
# high_value = 15.
# low_value = 1.

# # Create a 400x400 matrix for diagonals representing each cell of the 20x20 grid
# matrix_diag = np.zeros((matrix_size, matrix_size))

# # Flatten the 20x20 grid into a 1D list of 400 positions corresponding to diagonals
# grid_cells = [(x, y) for y in range(grid_size) for x in range(grid_size)]

# # Assign higher or lower values to each diagonal cell based on whether it was targeted
# for i, (x, y) in enumerate(grid_cells):
#     value = high_value if (x, y) in selected_cells_border else low_value
#     matrix_diag[i, i] = value

# # Show the matrix
# plt.figure(figsize=(6, 6))
# plt.imshow(matrix_diag, cmap='viridis', origin='upper')
# plt.title("Diagonal Matrix: 2D Map Cell Encoding")
# plt.colorbar(label='Value')
# plt.show()


# ######################


# # Load and flatten
# df = pd.read_pickle(py_data_location + "/dataset.pkl")
# df_copy = df.copy()
# df_copy['pO2Value'] = df_copy['pO2Value'].apply(lambda x: np.asarray(x).ravel())

# # --------- Define exclusions here ---------
# # List of (arteriole_id, depth_id) pairs to skip

# # exclude_pairs = [
# #     (1, 4),
# #     (1, 5),
# #     (2, 1),
# #     (2, 5),
# #     (3, 1),
# #     (3, 4),
# #     (3, 5),
# #     (3, 6),
# #     (4, 1),
# #     (4, 2),
# #     (4, 3),
# #     (5, 1),
# #     (5, 5),
# #     (5, 6),
# #     (7, 4),
# #     (7, 5),
# #     (8, 3),
# #     (8, 4),
# #     (8, 5),
# #     (8, 6),
# #     (9, 5),
# #     (9, 6),
# #     (10, 3),
# #     (10, 4),
# #     (10, 5),
# #     (10, 6),
# #     (11, 1),
# #     (11, 3),
# #     (11, 4),
# #     (11, 5)
# #     # add more if needed
# # ]
# exclude_pairs = load_data(py_data_location + '/excluded_dataset.txt')

# # Apply exclusion filter
# mask_exclude = df_copy.apply(
#     lambda row: (row['arteriole_id'], row['depth_id']) not in exclude_pairs,
#     axis=1
# )
# df_copy = df_copy[mask_exclude].reset_index(drop=True)

# # --------- Overall stats across ALL remaining pixels ---------
# all_values = np.concatenate(df_copy['pO2Value'].values)

# stats = {
#     "min": float(np.min(all_values)),
#     "max": float(np.max(all_values)),
#     "mean": float(np.mean(all_values)),
#     "std": float(np.std(all_values)),
#     "median": float(np.median(all_values)),
#     "25th_percentile": float(np.percentile(all_values, 25)),
#     "75th_percentile": float(np.percentile(all_values, 75)),
#     "count": int(all_values.size),
# }
# print("OVERALL STATS (after exclusions):", stats)

# overall_mean = stats["mean"]

# # --------- Per-map summaries ---------
# df_copy['map_min']  = df_copy['pO2Value'].apply(np.min)
# df_copy['map_max']  = df_copy['pO2Value'].apply(np.max)
# df_copy['map_mean'] = df_copy['pO2Value'].apply(np.mean)
# df_copy['map_std']  = df_copy['pO2Value'].apply(np.std)

# # --------- Row containing the GLOBAL MAX pixel ---------
# ix_global_max_row = df_copy['map_max'].idxmax()
# row_gmax = df_copy.loc[ix_global_max_row]

# flat_max_idx = int(row_gmax['pO2Value'].argmax())

# print("\nROW WITH GLOBAL MAX PIXEL")
# print({
#     "arteriole_id": int(row_gmax['arteriole_id']),
#     "depth_id": int(row_gmax['depth_id']),
#     "global_max_value": float(row_gmax['map_max']),
#     "flat_pixel_index": flat_max_idx
# })

# # --------- Row with the HIGHEST MEAN ---------
# ix_highest_mean = df_copy['map_mean'].idxmax()
# row_hmean = df_copy.loc[ix_highest_mean]

# print("\nROW WITH HIGHEST MEAN")
# print({
#     "arteriole_id": int(row_hmean['arteriole_id']),
#     "depth_id": int(row_hmean['depth_id']),
#     "map_mean_value": float(row_hmean['map_mean']),
#     "map_std_value": float(row_hmean['map_std'])
# })

# # --------- Row whose mean is CLOSEST to the OVERALL MEAN ---------
# ix_closest_to_overall = (df_copy['map_mean'] - overall_mean).abs().idxmin()
# row_closest = df_copy.loc[ix_closest_to_overall]

# print("\nROW WITH MEAN CLOSEST TO OVERALL MEAN")
# print({
#     "arteriole_id": int(row_closest['arteriole_id']),
#     "depth_id": int(row_closest['depth_id']),
#     "map_mean_value": float(row_closest['map_mean']),
#     "delta_from_overall_mean": float(abs(row_closest['map_mean'] - overall_mean))
# })




# # Load and flatten each map
# df = pd.read_pickle(py_data_location + "/dataset.pkl")
# df_copy = df.copy()
# df_copy['pO2Value'] = df_copy['pO2Value'].apply(lambda x: np.asarray(x).ravel())

# # Compute per-map stats
# df_copy['map_min']  = df_copy['pO2Value'].apply(np.min)
# df_copy['map_max']  = df_copy['pO2Value'].apply(np.max)
# df_copy['map_mean'] = df_copy['pO2Value'].apply(np.mean)
# df_copy['map_std']  = df_copy['pO2Value'].apply(np.std)

# # Aggregate: mean and std for each quantity
# results = {
#     "minimums": {
#         "mean": float(df_copy['map_min'].mean()),
#         "std": float(df_copy['map_min'].std())
#     },
#     "maximums": {
#         "mean": float(df_copy['map_max'].mean()),
#         "std": float(df_copy['map_max'].std())
#     },
#     "means": {
#         "mean": float(df_copy['map_mean'].mean()),
#         "std": float(df_copy['map_mean'].std())
#     },
#     "std_devs": {
#         "mean": float(df_copy['map_std'].mean()),
#         "std": float(df_copy['map_std'].std())
#     }
# }

# # Pretty print results
# for k, v in results.items():
#     print(f"{k.capitalize()}: mean = {v['mean']:.4f}, std = {v['std']:.4f}")



# # ----------------------+ Plots the results +----------------------#
# observations_id = [f'obs_{i}' for i in range(1, 5)]

# # Simulated iteration steps
# x_obs = np.arange(1, len(observations_id) + 1)

# state_ensembles_1000 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/EnKF_real_data_iterative/state_ensembles_1000.npy")
# states_ensembles_100 = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/EnKF_plots/EnKF_real_data_iterative/state_ensembles_100.npy")

# # -----------------------
# # Ratio M Stats
# data_1000 = state_ensembles_1000.T * cmro2_by_M # shape: (n_ensembles, n_iterations)
# numBoxes = data_1000.shape[1]  # now robust

# names = [f'obs{i}' for i in range(1, numBoxes + 1)]

# P.figure()
# bp = P.boxplot(data_1000, labels=names)

# for i in range(numBoxes):
#     y = data_1000[:, i]
#     x = np.random.normal(1+i, 0.04, size=len(y))
#     P.plot(x, y, 'r.', alpha=0.2)
# P.xlabel('$PO_{2}$ Map ID')
# P.ylabel('State value CMRO2 (umol /cm^3 /min)')
# P.title('EnKF State Estimation with Uncertainty')
# P.grid(True)
# P.show()
# # P.savefig('enkf_state_estimation_test.png', dpi=300, bbox_inches='tight')

# data_1000_mean = np.mean(data_1000, axis=0) # shape: (n_ensembles, n_iterations)
# data_100_mean = np.mean(states_ensembles_100.T * cmro2_by_M, axis=0) # shape: (n_ensembles, n_iterations)
# data = np.vstack((data_100_mean, data_1000_mean)).T
# P.figure()
# bp = P.boxplot(data, labels=['Overall Stats 100', 'Overall Stats 1000'])

# for i in range(data.shape[1]):
#     y = data[:, i]
#     x = np.random.normal(1+i, 0.04, size=len(y))
#     P.plot(x, y, 'r.', alpha=0.2)
# P.xlabel('$PO_{2}$ Map ID') 
# P.ylabel('State value CMRO2 (umol /cm^3 /min)')
# P.title('EnKF State Estimation with Uncertainty - Overall')
# P.grid(True)
# P.show()    
# # P.savefig('enkf_state_estimation_overall_test.png', dpi=300, bbox_inches='tight')

