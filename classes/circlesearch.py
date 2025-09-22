
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 09:30:06 2025

@author: ruudy
"""

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

class Po2Analyzer:
    def __init__(self,
                 pO2_array: np.ndarray,
                 X: np.ndarray,
                 Y: np.ndarray):
        
        # Constants
        self.SEC_MIN = 60
        self.CM3_M3 = 1e6
        self.UM3_M3 = 1e18
        self.D = 4.0e3
        self.alpha = 1.39e-15
        self.cmro2_by_M = self.SEC_MIN * self.UM3_M3 / self.CM3_M3 * self.D * self.alpha

        # Input fields
        self.pO2_value = pO2_array
        self.sigma = 2.0  # Gaussian smoothing
        self.smooth_data = gaussian_filter(self.pO2_value, sigma=self.sigma)

        self.rows, self.cols = self.pO2_value.shape
        self.X, self.Y = X, Y  # physical coordinates
        self.p_vessel = np.max(self.smooth_data)  # vessel pO2

        # Gradient magnitude
        self.Gx, self.Gy = np.gradient(self.pO2_value)
        self.Gmag = np.sqrt(self.Gx**2 + self.Gy**2)

    def find_circles(self):
        """
        In radial mode: 
        - Inner circle = circle with maximum average pO2 along circumference.
        - Outer circle = circle with minimum average pO2 along circumference.
        """
        # Center = maximum pO₂ location
        i_center, j_center = self.smooth_data.shape[0] // 2, self.smooth_data.shape[1] // 2
        win_size = 5  # half-width of search window

        # Define search window around center
        row_min, row_max = max(i_center - win_size, 0), min(i_center + win_size + 1, self.smooth_data.shape[0])
        col_min, col_max = max(j_center - win_size, 0), min(j_center + win_size + 1, self.smooth_data.shape[1])

        # Restrict to local window
        local_window = self.smooth_data[row_min:row_max, col_min:col_max]

        # local maximum inside the window
        local_imax, local_jmax = np.unravel_index(np.argmax(local_window), local_window.shape)
        imax = local_imax + row_min
        jmax = local_jmax + col_min

        if np.abs(imax - jmax) > 3:
            print("Warning: Center is not near the diagonal!")
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            c = ax.pcolormesh(self.X, self.Y, self.pO2_value, shading='auto', cmap='jet')
            fig.colorbar(c, label='pO₂')
            ax.set_title(f"Radial pO₂ Map of the ensemble member | Center at ({self.X[jmax, imax]:.1f}, {self.Y[jmax, imax]:.1f})")
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.axis('equal')
            plt.show()
        cx, cy = self.X[jmax, imax], self.Y[jmax, imax]
        self.center = (cx, cy)
        self.center_ij = (imax, jmax)

        # Radial distances from center
        self.R = np.sqrt((self.X - cx) ** 2 + (self.Y - cy) ** 2)
        r_vals = np.unique(self.R.round(decimals=2))  # test radii
        r_max = self.X[10, 19] - self.X[10, 10]
        r_vals = r_vals[r_vals < r_max]
        r_vals[0] = r_vals[1]/2

        circ_stats = []
        for r in r_vals:
            tol = 2 # thickness of ring
            mask_ring = (self.R >= r - tol) & (self.R <= r + tol)
            if np.any(mask_ring):
                avg_val = np.mean(self.smooth_data[mask_ring])
                avg_grad = np.mean(self.Gmag[mask_ring])
                circ_stats.append((r, avg_val, avg_grad))

        if not circ_stats:
            print("No circles found!")
            self.rin = self.rout = None
            return

        self.circ_stats = np.array(circ_stats)
        r_in = self.circ_stats[np.argmax(self.circ_stats[:, 1]), 0]   # max avg pO₂
        r_out = self.circ_stats[np.argmin(self.circ_stats[:, 2]), 0]  # min avg pO₂

        self.rin = r_in
        self.rout = r_out

        # Masks for visualization
        self.mask_inner = self.R <= r_in
        self.mask_outer = self.R <= r_out
