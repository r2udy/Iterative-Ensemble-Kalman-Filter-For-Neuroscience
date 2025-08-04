# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 16:41:00 2025

@author: ruudy
"""
import sys
import os

py_file_location = "C:/Users/ruudy/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/"
sys.path.append(os.path.abspath(py_file_location))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from MapGenerator import MapGenerator

class Po2Fitter_3:
    def __init__(self, pO2_array: np.ndarray, Rves, Rt, pixel_size: float = 10.0):
        
        # Constants conversion
        self.SEC_MIN = 60
        self.CM3_M3 = 1e6
        self.UM3_M3 = 1e18
        self.D = 4.0e3
        self.alpha = 1.39e-15
        self.cmro2_by_M = self.SEC_MIN * self.UM3_M3 / self.CM3_M3 * self.D * self.alpha
        
        # Partial Oxygen Array
        self.model = 'KE'
        self.raw_data = pO2_array
        self.pixel_size = pixel_size
        self.distance_map = self._compute_distance_map()
        
        # Parameters
        self.pvessel = np.max(self.raw_data)
        self.rin = Rves
        self.rt = Rt

    def _compute_distance_map(self):
        self.idx_min = np.argmin(self.raw_data.flatten())
        self.imax, self.jmax = np.unravel_index(np.argmax(self.raw_data), self.raw_data.shape)
        
        distance_map = np.zeros_like(self.raw_data)
        for row in range(self.raw_data.shape[0]):
            for col in range(self.raw_data.shape[1]):
                r_ = np.sqrt((row - self.imax)**2 + (col - self.jmax)**2)
                distance_map[row, col] = self.pixel_size * r_
        return distance_map
    
    def _partial_pressure(self, r, M, pvessel, Rves, R0, Rt):
        generator = MapGenerator(
            cmro2= M * self.cmro2_by_M,
            pvessel=pvessel,
            Rves=Rves,
            R0=R0,
            Rt=Rt,
            model=self.model,
            center=(self.imax, self.jmax)
        )
        return generator._partial_pressure(r)

    def fit(self):
        self.mask = (self.distance_map > self.rin)
        xdata = self.distance_map[self.mask].flatten()
        target = self.raw_data[self.mask].flatten()
        sorted_indices = np.argsort(xdata)
        xdata = xdata[sorted_indices]
        target = target[sorted_indices]
        
        def residual(params):
            M, pvessel, r0 = params
            
            generator = MapGenerator(
                cmro2=M * self.cmro2_by_M,
                pvessel=pvessel,
                Rves=self.rin,
                R0=r0,
                Rt=self.rt,
                model=self.model,
                center=(self.imax, self.jmax)
            )
            # return self._partial_pressure_KE(xdata, M, pvessel, self.rin, r0, self.rt) - target
            return generator._partial_pressure(xdata) - target
        
        cmro2 = np.random.uniform(2.5, 3.5)
        M_initial = cmro2 / self.cmro2_by_M
        r0_initial = 80.
        pvessel_initial = self.pvessel
        initial_params = [M_initial, pvessel_initial, r0_initial] # M, rin, rout
        bounds = ([1e-6, 10, 20], [1, 100, 90])
        print("\nLeast squares nonlinear fitting - 3 paramters (CMRO2, P_{vessel wall} and R_t)\n")
        result = least_squares(residual, x0=initial_params, bounds=bounds, verbose=1, max_nfev=10000)
        
        # Estimated parameters
        self.estimated_params = result.x
        # 1D
        self.xdata = xdata
        self.target = target
        
        # Initialize the po2 map generator object
        generator = MapGenerator(
            cmro2=result.x[0] * self.cmro2_by_M,
            pvessel=result.x[1],
            Rves=self.rin,
            R0=result.x[2],
            Rt=self.rt,
            model=self.model,
            center=(self.imax, self.jmax)
        )
        self.fitted_target = generator._partial_pressure_KE(self.xdata)
    
    def plot_1D_results(self):
        # 1D Fit result
        plt.figure(figsize=(7, 5))
        plt.plot(self.xdata, self.target, '*', label="Observed Data")
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
        plt.pcolor(self.fitted_po2_value, shading='auto', cmap='jet')
        plt.ylabel("Partial Oxygen Pressure (mmHg)")
        plt.title("Nonlinear Fit of PO2 vs. Radius")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_estimated_parameters(self):
        # Print estimates
        M_est, pvessel_est, r0 = self.estimated_params
        print("\n Non Linear Least Squares fitting paramaters estimation:")
        print("-"*65)
        print(f"Estimated CMRO2     : {M_est * self.cmro2_by_M:.3e} umol.μm-3.min-1")
        print(f"Estimated M         : {M_est:.3e}")
        print(f"Estimated Pvessel   : {pvessel_est:.3f} mmHg")
        print(f"Estimated R0        : {r0:.3f} μm")
        print(f"Rves                : {self.rin:.3f} μm")
        print(f"Rt                  : {self.rt:.3f} μm")
        print("-"*65)
    
    def get_results(self):
        self.cmro2_est = self.estimated_params[0] * self.cmro2_by_M
        self.M_est = self.estimated_params[0]
        self.pvessel_est = self.estimated_params[1]
        self.r0_est = self.estimated_params[2]
        
        return self.cmro2_est, self.M_est, self.pvessel_est, self.r0_est
        
if __name__ == "__main__":
    
    seed = np.random.seed(3)
    # Initialize Map generator
    # True observation
    cmro2_true = 2.0
    pvessel = 60.0
    Rves = 10.0
    R0 = 80.0
    Rt = 80.0
    model='KE'
    
    generator = MapGenerator(
        cmro2=cmro2_true,
        pvessel=pvessel,
        Rves=Rves,
        R0=R0,
        Rt=Rt,
        model=model
    )
    generator.plot_3d()
    
    # Initialize the Non Linear least squares fitter
    sigma = 2.0
    true_obs = generator.pO2_array
    obs_ = np.random.normal(true_obs.flatten(), scale=sigma)
    perturbated_obs = obs_.reshape((20, 20), order='F')
    generator.plot_3d(title=f'3D Oxygen Partial Pressure Distribution Perturbated, sigma: {sigma} mmHg', pO2_array=perturbated_obs)
    
    ##
    ## CMRO2 + pvessel + r0 fitting ##
    lsqnonlin_fitter = Po2Fitter_3(pO2_array=perturbated_obs, Rves=Rves, Rt=Rt)
    lsqnonlin_fitter.fit()
    lsqnonlin_fitter.plot_estimated_parameters()
    cmro2_est, _, pvessel_est, r0_est = lsqnonlin_fitter.get_results()
    
    # 3D Plot the results the results
    generator = MapGenerator(
        cmro2=cmro2_est,
        pvessel=pvessel_est,
        Rves=Rves,
        R0=r0_est,
        Rt=Rt,
        model=model
    )
    
    generator.plot_3d(title='3D Po2 Map estimation')
    print(f'Percentage Error: {np.abs(1- cmro2_est/cmro2_true) * 100}%')