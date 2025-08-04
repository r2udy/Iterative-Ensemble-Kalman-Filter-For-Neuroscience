# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:35:42 2025

@author: ruudy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Import the data
data_two_holes_axis = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/FEM_code/SavedFiles/Results/square_two_holes_coordinates.npy")
data_two_holes = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/FEM_code/SavedFiles/Results/square_two_holes_solution.npy")

# data_one_hole_axis = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/FEM_code/SavedFiles/Results/square_one_hole_MM_coordinates.npy")
# data_one_hole = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/FEM_code/SavedFiles/Results/square_one_hole_MM_solution.npy")

data_one_hole_axis = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/FEM_dataset/square_one_hole_MM_coordinates.npy")
data_one_hole = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/FEM_dataset/square_one_hole_MM_solution.npy")

# Constant
R_ves = 20
R_t = 150
D = 4.0e3 #  m2 s−1
alpha = 1.39*1e-15 #  μM mmHg−1
CMRO2 =  1.80 / (60 * 1e12) # μmol μm−3 s−1
rO2 = CMRO2 / alpha
M = rO2 / D # mmHg μm−2
P_ves = 99.42 # mmHg


# Assign the data
z = data_two_holes
x = data_two_holes_axis[:, 0]
y = data_two_holes_axis[:, 1]

fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(x, y, c=z, cmap='viridis')
ax.set_xlabel('x - axis')
ax.set_ylabel('y - axis')
ax.set_title("PO2 fit as a function of the radial distance from the penetrating arteriole")
plt.colorbar(sc, ax=ax, label='Color scale: Po2 (mmHg)')
plt.tight_layout()
plt.show()

# 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(x, y, z, c=z, cmap='viridis')
ax.set_xlabel('x - axis')
ax.set_ylabel('y - axis')
ax.set_zlabel('z - axis')
ax.set_title("3D PO2 fit as a function of the radial distance from the penetrating arteriole")
plt.colorbar(sc, ax=ax, label='Color scale: Po2 (mmHg)')
plt.tight_layout()
plt.show()




# Assign the data
z = data_one_hole
x = data_one_hole_axis[:, 0]
y = data_one_hole_axis[:, 1]

fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(x, y, c=z, cmap='viridis')
ax.set_xlabel('x - axis')
ax.set_ylabel('y - axis')
ax.set_title("PO2 fit as a function of the radial distance from the penetrating arteriole")
plt.colorbar(sc, ax=ax, label='Color scale: Po2 (mmHg)')
plt.tight_layout()
plt.show()

# 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(x, y, z, c=z, cmap='viridis')
ax.set_xlabel('x - axis')
ax.set_ylabel('y - axis')
ax.set_zlabel('z - axis')
ax.set_title("3D PO2 fit as a function of the radial distance from the penetrating arteriole")
plt.colorbar(sc, ax=ax, label='Color scale: Po2 (mmHg)')
plt.tight_layout()
plt.show()

def find_closest_point(given_point, array_of_points):
    distances = np.abs(array_of_points - given_point)
    closest_index = np.argmin(distances)  # Index of the closest point
    return closest_index

def _interpolation_grid(grid_refined, grid_coarse):
    idx_list = []
    for point_coarse in grid_coarse:
        closest_idx = find_closest_point(point_coarse, grid_refined)
        idx_list.append(closest_idx)
    
    return np.array(idx_list)

x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

x_obs = np.linspace(x_min, x_max, 20)
y_obs = np.linspace(x_min, x_max, 20)
x_idx = _interpolation_grid(x, x_obs)
y_idx = _interpolation_grid(y, y_obs)

x_domain = x[x_idx]
y_domain = y[y_idx]
X_domain, Y_domain = np.meshgrid(x_domain, y_domain)
points = np.column_stack((X_domain.ravel(), Y_domain.ravel()))

# Interpolate z values at the grid points
z_grid = griddata((x, y), z, points, method='linear').reshape(20, 20)

plt.figure()
plt.scatter(X_domain, Y_domain, z_grid, c=z_grid)
plt.show()
