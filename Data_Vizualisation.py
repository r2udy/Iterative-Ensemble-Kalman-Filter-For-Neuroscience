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
from Po2Dataset import load_data, get_cells_by_angle
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
# Select your the data
art_id, dth_id = (2, 3)
# Angle ranges: from 0 to 90 degrees and from 270 to 360 degrees
angle_ranges = [(20, 60), (210, 250)]
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

# --------- Plotting the angle mask ---------
import math
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
def build_obs_covariance(
    grid_size=20,
    origin=(10,10),
    angle_ranges=[(20, 60), (210, 250)],
    min_radius=5,
    max_radius=None,
    high_var=15.0**2,
    sigma2=1.0**2,
    length_scale=3.0
):
    """
    Build a 2D covariance matrix for a grid_size x grid_size PO2 map.

    High-uncertainty cells (angle_ranges + min_radius) -> variance=high_var, uncorrelated.
    Low-uncertainty cells -> spatially correlated with Gaussian decay.

    Returns:
        C: covariance matrix (grid_size^2 x grid_size^2)
    """
    if max_radius is None:
        max_radius = math.hypot(grid_size, grid_size)

    # --- Function to get targeted cells ---
    def get_cells_by_angle(grid_size, origin, angle_ranges, distance_range=None):
        x0, y0 = origin
        selected_cells = []
        for y in range(grid_size):
            for x in range(grid_size):
                dx = x - x0
                dy = y0 - y
                angle = math.degrees(math.atan2(dy, dx)) % 360
                distance = math.hypot(dx, dy)
                in_angle = any(
                    start <= angle <= end if start <= end else angle >= start or angle <= end
                    for (start, end) in angle_ranges
                )
                in_distance = True
                if distance_range:
                    min_d, max_d = distance_range
                    in_distance = min_d <= distance <= max_d
                if in_angle and in_distance:
                    selected_cells.append((x, y))
        return selected_cells

    # --- Identify high-uncertainty cells ---
    selected_cells_border = get_cells_by_angle(
        grid_size, origin, angle_ranges, distance_range=(min_radius, max_radius)
    )

    # --- Create grid coordinates ---
    coords = np.array([(x, y) for y in range(grid_size) for x in range(grid_size)])
    matrix_size = grid_size * grid_size
    C = np.zeros((matrix_size, matrix_size))

    # --- Fill covariance matrix ---
    for i in range(matrix_size):
        xi, yi = coords[i]

        # High-uncertainty cell
        if (xi, yi) in selected_cells_border:
            C[i, i] = high_var
            continue

        # Low-uncertainty cells: compute correlations with other low-uncertainty cells
        for j in range(i, matrix_size):
            xj, yj = coords[j]

            if (xj, yj) in selected_cells_border:
                continue

            d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            cov_ij = sigma2 * np.exp(-d**2 / (2 * length_scale**2))
            C[i, j] = cov_ij
            C[j, i] = cov_ij

    return C

# --------------------------------
# Example usage
grid_size = 20
origin = (10, 10)
angle_ranges = [(20, 60), (210, 250)]
min_radius = 5
high_var = 3.0**2
sigma2 = 1.0**2
length_scale = 3.0

C = build_obs_covariance(
    grid_size=grid_size,
    origin=origin,
    angle_ranges=angle_ranges,
    min_radius=min_radius,
    high_var=high_var,
    sigma2=sigma2,
    length_scale=length_scale
)

# --- Quick visualization: diagonal (variance) map ---
uncertainty_map = np.diag(C).reshape((grid_size, grid_size))
plt.figure(figsize=(6,6))
plt.imshow(uncertainty_map, origin='upper', cmap='viridis')
plt.colorbar(label='Variance')
plt.title("Diagonal Variance Map of PO2 Covariance Matrix")
plt.show()

print("Covariance matrix shape:", C.shape)
plt.figure(figsize=(6,6))
plt.imshow(C, origin='upper', cmap='viridis')
plt.colorbar(label='Variance')
plt.title("Diagonal Variance Map of PO2 Covariance Matrix")
plt.show()


# ######################

# ---------- Target Cells -----------
# Adjust the observation covariance matrix to account very uncertain measurement
max_radius = math.hypot(n, n)  # furthest possible distance in the grid

# Targeted cells (by angle + from min_radius to border)
selected_cells_border = get_cells_by_angle(
    n,
    center,
    [angle_ranges[0], angle_ranges[1]],
    distance_range=(min_radius, max_radius)
)

# ---- Build diagonal R with per-cell variances ----
matrix_size = n * n
matrix      = np.zeros((matrix_size, matrix_size))

# Set higher values for targeted cells and lower for non-targeted ones
high_value  = 3.**2
low_value   = 1.**2

# Create a 400x400 matrix for diagonals representing each cell of the 20x20 grid
matrix_diag = np.zeros((matrix_size, matrix_size))

# Flatten the 20x20 grid into a 1D list of 400 positions corresponding to diagonals
grid_cells = [(x, y) for y in range(n) for x in range(n)]

# Assign higher or lower values to each diagonal cell based on whether it was targeted
for k, (x, y) in enumerate(grid_cells):
    matrix_diag[k, k] = high_value if (x, y) in selected_cells_border else low_value

plt.figure(figsize=(6,6))
plt.imshow(matrix_diag, origin='upper', cmap='viridis')
plt.colorbar(label='Variance')
plt.title("Diagonal Variance Map of PO2 Covariance Matrix")
plt.show()


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

