# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:35:42 2025

@author: ruudy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Import the data
data_neumann_40_axis = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/FEM_dataset/square_holes_neumann_R0_40_coordinates.npy")
data_neumann_40_hole = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/FEM_dataset/square_holes_neumann_R0_40_solution.npy")

data_neumann_120_axis = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/FEM_dataset/square_holes_neumann_R0_120_coordinates.npy")
data_neumann_120_hole = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/FEM_dataset/square_holes_neumann_R0_120_solution.npy")

"""+++++++++++++++++"""

data_no_neumann_40_axis = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/FEM_dataset/square_holes_no_neumann_R0_40_coordinates.npy")
data_no_neumann_40_hole = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/FEM_dataset/square_holes_no_neumann_R0_40_solution.npy")

data_no_neumann_120_axis = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/FEM_dataset/square_holes_no_neumann_R0_120_coordinates.npy")
data_no_neumann_120_hole = np.load("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/FEM_dataset/square_holes_no_neumann_R0_120_solution.npy")


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

# Assign the data
z_neumann = data_neumann_40_hole
x_neumann = data_neumann_40_axis[:, 0]
y_neumann = data_neumann_40_axis[:, 1]

z_noneumann = data_neumann_120_hole
x_noneumann = data_neumann_120_axis[:, 0]
y_noneumann = data_neumann_120_axis[:, 1]

x_axis = np.linspace(-190, 190, 20)
y_axis = np.linspace(-190, 190, 20)
X_axis, Y_axis = np.meshgrid(x_axis, y_axis)

x_idx = _interpolation_grid(x_neumann, x_axis)
y_idx = _interpolation_grid(y_neumann, y_axis)

x_domain = x_neumann[x_idx]
y_domain = y_neumann[y_idx]
X_domain, Y_domain = np.meshgrid(x_domain, y_domain)
points = np.column_stack((X_domain.ravel(), Y_domain.ravel()))

z_grid_neumann = griddata((x_neumann, y_neumann), z_neumann, points, method='linear').reshape(20, 20)
z_grid_noneumann = griddata((x_noneumann, y_noneumann), z_noneumann, points, method='linear').reshape(20, 20)

# 2D plot
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(x_neumann, y_neumann, c=z_neumann, cmap='viridis')
ax.set_xlabel('x - axis')
ax.set_ylabel('y - axis')
ax.set_title("PO2 fit")
plt.colorbar(sc, ax=ax, label='Color scale: Po2 (mmHg)')
plt.tight_layout()
plt.show()

# 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.plot_surface(X_axis, Y_axis, z_grid_neumann, cmap='viridis')
ax.plot_surface(X_axis, Y_axis, z_grid_noneumann, color='lightgrey', alpha=0.3)
ax.set_xlabel('x - axis')
ax.set_ylabel('y - axis')
ax.set_zlabel('z - axis')
# ax.set_title("3D $Po_2$ profile as a function of the radial distance from the penetrating arteriole")
plt.colorbar(sc, ax=ax, label='Color scale: Po2 (mmHg)')
plt.tight_layout()
plt.show()