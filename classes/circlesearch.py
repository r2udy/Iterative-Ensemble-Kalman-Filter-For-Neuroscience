
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
                 r0: float = 0.0,
                 pixel_size: float = 10.0,
                 ):
        
        # Constants conversion
        self.SEC_MIN = 60
        self.CM3_M3 = 1e6
        self.UM3_M3 = 1e18
        self.D = 4.0e3
        self.alpha = 1.39e-15
        self.cmro2_by_M = self.SEC_MIN * self.UM3_M3 / self.CM3_M3 * self.D * self.alpha
        
        self.model = 'KE'
        self.pO2_value = pO2_array
        self.sigma = 2.0  # for Gaussian smoothing
        self.smooth_data = gaussian_filter(self.pO2_value, sigma=self.sigma)
        self.pixel_size = pixel_size
        self.rows, self.cols = self.pO2_value.shape
        self.X, self.Y = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
        self.r0 = r0
        self.p_vessel = np.max(self.pO2_value)
        
        self.Gx, self.Gy = np.gradient(self.pO2_value)
        self.Gmag = np.sqrt(self.Gx**2 + self.Gy**2)
        self.distance_map = self._compute_distance_map()
        
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
        
        # self.rout = self._estimate_rout()

    def _compute_distance_map(self):
        self.idx_min = np.argmin(self.smooth_data.flatten())

        # Find the indices of the 5 minimum values in the flattened smooth_data array
        flat_smooth_data = self.pO2_value.flatten()
        idx_min_flat = np.argpartition(flat_smooth_data, 5)[:5]
        idx_min = np.unravel_index(idx_min_flat, self.smooth_data.shape)

        # Find the index of the maximum value in pO2_value
        imax, jmax = np.unravel_index(np.argmax(self.pO2_value), self.pO2_value.shape)

        distance_map = np.zeros_like(self.pO2_value)
        for row in range(self.pO2_value.shape[0]):
            for col in range(self.pO2_value.shape[1]):
                r_ = np.sqrt((row - imax)**2 + (col - jmax)**2)
                distance_map[row, col] = self.pixel_size * r_
        return distance_map

    def _estimate_rout(self):
        # return self.distance_map.flatten()[self.idx_min]
        # Calculate the average of the 5 distances
        distances = self._compute_distance_map()
        return np.mean(distances)