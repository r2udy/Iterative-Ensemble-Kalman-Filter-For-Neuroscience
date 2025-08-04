# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 16:41:00 2025

@author: ruudy
"""
import sys
import os

py_file_location = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/classes/"
sys.path.append(os.path.abspath(py_file_location))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares


class Po2Fitter:
    def __init__(self, pO2_array: np.ndarray, Rves, pixel_size: float = 10):

        # Constants conversion
        self.SEC_MIN = 60
        self.CM3_M3 = 1e6
        self.UM3_M3 = 1e18
        self.D = 4.0e3
        self.alpha = 1.39e-15
        self.cmro2_by_M = self.SEC_MIN * self.UM3_M3 / self.CM3_M3 * self.D * self.alpha
        
        self.raw_data = pO2_array
        self.pixel_size = pixel_size
        self.sigma = 2.0  # for Gaussian smoothing
        self.smooth_data = gaussian_filter(self.raw_data, sigma=self.sigma)
        self.pvessel = np.max(self.raw_data)

        self.distance_map = self._compute_distance_map()
        self.rout = self._estimate_rout() # um
        self.rin = Rves # um

    def _compute_distance_map(self):
        self.idx_min = np.argmin(self.smooth_data.flatten())
        imax, jmax = np.unravel_index(np.argmax(self.raw_data), self.raw_data.shape)
        
        distance_map = np.zeros_like(self.raw_data)
        for row in range(self.raw_data.shape[0]):
            for col in range(self.raw_data.shape[1]):
                r_ = np.sqrt((row - imax)**2 + (col - jmax)**2)
                distance_map[row, col] = self.pixel_size * r_
        return distance_map
    
    def _estimate_rout(self):
        return self.distance_map.flatten()[self.idx_min]

    def _partial_pressure(self, r, M, pvessel, rin, rout):
        return pvessel + (M / 4) * (r**2 - rin**2) - (M * rout**2 / 2) * np.log(r / rin)
    
    def fit(self):
        self.mask = (self.distance_map > 1) & (self.distance_map < self.rout)
        xdata = self.distance_map[self.mask].flatten()
        target = self.smooth_data[self.mask].flatten()
        sorted_indices = np.argsort(xdata)
        xdata = xdata[sorted_indices]
        target = target[sorted_indices]

        def residual(params):
            M = params
            return self._partial_pressure(xdata, M, self.pvessel, self.rin, self.rout) - target
        
        CMRO2 = 2.0e-14
        M_initial = CMRO2 / (self.D * self.alpha)
        initial_params = [M_initial] # M
        bounds = ([1e-6], [1e3])
        # Perform the fitting
        print("\nLeast squares nonlinear fitting - 1 paramter (CMRO2)\n")
        result = least_squares(residual, x0=initial_params, bounds=bounds, verbose=1, max_nfev=10000)

        self.estimated_params = result.x
        # 1D
        self.xdata = xdata
        self.target = target
        self.fitted_target = self._partial_pressure(xdata, result.x[0], self.pvessel, self.rin, self.rout)
        # 2D
        self.fitted_po2_values = np.zeros_like(self.raw_data.flatten())
        for (pos, _) in enumerate(self.fitted_po2_values.flatten()):
            if self.mask.flatten()[pos]:
                self.fitted_po2_values[pos] = self._partial_pressure(self.distance_map.flatten()[pos], result.x[0], self.pvessel, self.rin, self.rout)
            elif pos == np.argmax(self.raw_data):
                self.fitted_po2_values[pos] = self.raw_data.max()
        
    def plot_1D_results(self):
        # 1D Fit result
        plt.figure(figsize=(7, 5))
        plt.plot(self.xdata[self.mask], self.target, '*', label="Observed Data")
        plt.plot(self.xdata, self.fitted_target, 'r-', linewidth=2, label="Fitted Curve")
        plt.xlabel("Radius (μm)")
        plt.ylabel("Partial Oxygen Pressure (mmHg)")
        plt.title("Nonlinear Fit of PO2 vs. Radius")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_2D_results(self):
        # 2D Fit result
        plt.figure(figsize=(7, 5))
        plt.pcolor(self.fitted_po2_values, shading='auto', cmap='jet')
        plt.ylabel("Partial Oxygen Pressure (mmHg)")
        plt.axis('equal')
        plt.colorbar()
        plt.title("Nonlinear Fit of PO2 vs. Radius")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def get_results(self):
        self.cmro2_est = self.estimated_params[0] * self.cmro2_by_M
        self.M_est = self.estimated_params[0]
        
        return self.cmro2_est, self.M_est
    
    
    def plot_estimated_parameters(self):
        # Print estimates
        M_est = self.estimated_params[0]
        # print(f"Estimated M       : {M_est:.3e}")
        print(f"Estimated CMRO2   : {M_est * (60 * self.D * self.alpha * 1e12):.3e} umol.μm-3.min-1")
        print(f"Estimated Pvessel : {self.pvessel:.3f} mmHg")
        print(f"Estimated Rin     : {self.rin:.3f} μm")
        print(f"Estimated Rout    : {self.rout:.3f} μm")

if __name__ == "__main__":
    # --------- Load data ---------
    df = pd.read_pickle("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/dataset.pkl")
    df_copy = df.copy()
    df_copy['pO2Value'] = df_copy['pO2Value'].apply(lambda x: x.flatten())
    
    # Select your the data
    art_id = 3
    dth_id = 2
    
    # Load the map
    array = df_copy[(df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id )]['pO2Value'].tolist()[0]
    X = df_copy[(df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id )]['pointsX'].tolist()[0]
    Y = df_copy[(df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id )]['pointsY'].tolist()[0]
    n = 20 # data size
    pO2_value = array.reshape((n, n), order='F')
    
    fitter = Po2Fitter(pO2_array=pO2_value, Rves=10.)
    fitter.fit()
    fitter.plot_estimated_parameters()
    # 2D Fit result
    plt.figure(figsize=(7, 5))
    plt.pcolor(fitter.fitted_po2_values.reshape((20, 20), order='F'), shading='auto', cmap='jet')
    plt.colorbar()
    plt.title("Nonlinear Fit of PO2 vs. Data")
    plt.grid(True)
    plt.tight_layout()
    plt.show()