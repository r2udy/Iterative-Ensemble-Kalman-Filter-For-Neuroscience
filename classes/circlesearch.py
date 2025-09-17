
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 09:30:06 2025

@author: ruudy
"""

import numpy as np
from scipy.ndimage import gaussian_filter

class Po2Analyzer:
    def __init__(self,
                 pO2_array: np.ndarray,
                 X: np.ndarray,
                 Y: np.ndarray):
        
        # Constants conversion
        self.SEC_MIN = 60
        self.CM3_M3 = 1e6
        self.UM3_M3 = 1e18
        self.D = 4.0e3
        self.alpha = 1.39e-15
        self.cmro2_by_M = self.SEC_MIN * self.UM3_M3 / self.CM3_M3 * self.D * self.alpha
        self.pixel_size = 10  # um

        self.model = 'KE'
        self.pO2_value = pO2_array
        self.sigma = 2.0  # for Gaussian smoothing
        self.smooth_data = gaussian_filter(self.pO2_value, sigma=self.sigma)
        self.rows, self.cols = self.pO2_value.shape
        self.X, self.Y = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
        self.p_vessel = np.max(self.pO2_value)
        
        self.Gx, self.Gy = np.gradient(self.pO2_value)
        self.Gmag = np.sqrt(self.Gx**2 + self.Gy**2)
    
    def find_circles(self, min_r: int = 1, angle_range1_deg=(0, 0), angle_range2_deg=None, win_size: int = 3):
        max_row, max_col = np.unravel_index(np.argmax(self.pO2_value), self.pO2_value.shape)
        row_range = range(max(max_row - win_size, 0), min(max_row + win_size + 1, self.rows))
        col_range = range(max(max_col - win_size, 0), min(max_col + win_size + 1, self.cols))
        
        outer_circle_list = []
        inner_circle_list = []
        
        ## ------------------+ Find: Rves +------------------##
        for i_ in row_range:
            for j_ in col_range:
              dx_ = self.X - j_ + 1
              dy_ = self.Y - i_ + 1
              distance_squared_ = dx_**2 + dy_**2
              
              # Compute angles in degrees, between 0 and 360
              theta_ = (np.degrees(np.arctan2(dy_, dx_)) + 360) % 360
              
              t1_min, t1_max = angle_range1_deg
              mask1_angle = (theta_ >= t1_min) & (theta_ <= t1_max)
              
              max_r = min(i_, self.rows - i_, j_, self.cols - j_)
              for r in range(min_r, max_r):
                  circumference_in_full = (distance_squared_ == r**2)
                  mask_inner = (distance_squared_ <= r**2)
                  self.avgM_in = np.mean(self.pO2_value[mask_inner])
                  inner_circle_list.append({'center': (j_, i_), 'radius': r, 'avgM_in': self.avgM_in,
                                            'circumference_in':circumference_in_full * mask1_angle, 'mask_inner':mask_inner})
        
        # Sort and select top candidate
        self.inner_circle = sorted(inner_circle_list, key=lambda x: -x['avgM_in'])[0]
        self.rin = self.pixel_size * self.inner_circle['radius']
        self.center = self.inner_circle['center']
        self.circumference_in = self.inner_circle['circumference_in']
        self.mask_inner = self.inner_circle['mask_inner']
        
        
        # Center the outer circle in the middle of the pixel cells
        dx = self.X - self.center[0] + 1
        dy = self.Y - self.center[1] + 1
        
        # Compute angles in degrees, between 0 and 360
        theta = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
        
        
        ## ------------------+ Find: Ro +------------------##
        max_r = min(self.center[0], self.center[1], self.cols - self.center[0], self.rows - self.center[1])
        rin_idx = int(self.rin / self.pixel_size)
        for r in range(rin_idx, max_r):
            distance_squared = dx**2 + dy**2
            mask_outer = (distance_squared <= r**2)
            
            tolerance = 0.5
            circumference_out_full = (distance_squared >= (r - tolerance)**2) & (distance_squared <= (r + tolerance)**2)
            
            t1_min, t1_max = angle_range1_deg
            mask1_angle = (theta >= t1_min) & (theta <= t1_max)
        
            if angle_range2_deg:
                t2_min, t2_max = angle_range2_deg
                mask2_angle = (theta >= t2_min) & (theta <= t2_max)
                self.mask_angle = self.mask_inner | ~(mask1_angle | mask2_angle)
            else:
                self.mask_angle = self.mask_inner | ~mask1_angle
            
            # search over different circumferences
            circumference_out = self.mask_angle & circumference_out_full
            if np.any(circumference_out):
                # self.avgM_out = np.mean(self.Gmag[circumference_out])
                self.avgM_out = np.min(self.pO2_value[circumference_out])
                outer_circle_list.append({'center': (j_, i_), 'radius': r, 'avgM_out': self.avgM_out, 
                                          'circumference_out':circumference_out, 'mask_outer':mask_outer})
        
        self.outer_circle = None
        if outer_circle_list:
            # Sort and select top candidate
            self.outer_circle = sorted(outer_circle_list, key=lambda x: x['avgM_out'])[0]
            self.rout = self.pixel_size * self.outer_circle['radius']
            self.circumference_out = self.outer_circle['circumference_out']
            self.mask_outer = self.outer_circle['mask_outer']
        else:
            print("Warning: No valid circular segment found in the given angle range.")
            self.outer_circle = self.inner_circle = self.rin = self.rout = self.center = None
        
    def find_circles_coordinates(self):
        """
        In radial mode: 
        - Inner circle = circle with maximum average pO2 along circumference.
        - Outer circle = circle with minimum average pO2 along circumference.
        """
        # Center = maximum pO₂ location
        imax, jmax = np.unravel_index(np.argmax(self.smooth_data), self.smooth_data.shape)
        print(f"imax: {imax} and jmax: {jmax}")
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
                avg_grad = np.mean(self.pO2_value[mask_ring])
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
