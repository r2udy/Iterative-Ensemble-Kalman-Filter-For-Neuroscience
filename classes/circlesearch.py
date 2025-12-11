
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

        self.rows, self.cols = self.pO2_value.shape
        self.X, self.Y = X, Y  # physical coordinates

        # Gradient magnitude
        self.Gx, self.Gy = np.gradient(self.pO2_value)
        self.Gmag = np.sqrt(self.Gx**2 + self.Gy**2)

    def find_circles(self):
        """
        In radial mode: 
        - Inner circle = circle with maximum average pO2 along circumference.
        - Outer circle = circle with minimum average pO2 along circumference.
        """
        imax, jmax = np.unravel_index(np.argmax(self.pO2_value.flatten()), self.pO2_value.shape)
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
                avg_val = np.mean(self.pO2_value[mask_ring])
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
