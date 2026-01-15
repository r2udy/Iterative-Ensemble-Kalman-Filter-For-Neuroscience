# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 16:41:00 2025

@author: ruudy
"""
import sys
import os

ROOT = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/"
CLASSES = os.path.join(ROOT, "classes")
IMPL = os.path.join(ROOT, "Implementation")
CLUSTERING = os.path.join(IMPL, "Clustering")

# Add paths to Pyhton import system
print(">>> Added to sys.path:")
sys.path.append(ROOT)
sys.path.append(CLASSES)
sys.path.append(IMPL)
sys.path.append(CLUSTERING)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from EnKF_FEM_3 import EnKF
from FEM_code.generateMesh_Solver_multiple_holes import (
    DiffusionSolver, SolverParameters, HoleGeometry
)
from MapGenerator import MapGenerator
from synthetic_data_generation import load_synthetic_data

class Po2Fitter_3:
    def __init__(self, pO2_array: np.ndarray, Rves, X_axis: np.ndarray, Y_axis: np.ndarray):
        
        # Constants conversion
        self.SEC_MIN = 60
        self.CM3_M3 = 1e6
        self.UM3_M3 = 1e18
        self.D = 4.0e3
        self.alpha = 1.39e-15
        self.cmro2_by_M = self.SEC_MIN * self.UM3_M3 / self.CM3_M3 * self.D * self.alpha
        
        # Partial Oxygen Array
        self.raw_data = pO2_array
        self.X_axis = X_axis
        self.Y_axis = Y_axis
        self.distance_map = self._compute_distance_map()
        
        # Parameters
        self.pvessel = np.max(self.raw_data)
        self.rin = Rves

    def _compute_distance_map(self):
        self.idx_min = np.argmin(self.raw_data.flatten())
        imax, jmax = np.unravel_index(np.argmax(self.raw_data), self.raw_data.shape)
        cx, cy = self.X_axis[jmax, imax], self.Y_axis[jmax, imax]
        R = np.sqrt((self.X_axis - cx) ** 2 + (self.Y_axis - cy) ** 2)
        return R
    
    def _partial_pressure(self, r, M, pvessel, rin, rout):
            out = pvessel + (M / 4) * (rout**2 - rin**2) - (M * rout**2 / 2) * np.log(rout / rin)
            return out
    

    def fit(self):
        self.mask = (self.distance_map > self.rin)
        r_axis = self.distance_map[self.mask].flatten()
        target = self.raw_data[self.mask].flatten()
        sorted_indices = np.argsort(r_axis)
        r_axis = r_axis[sorted_indices]
        target = target[sorted_indices]
        
        def residual(params):
            M, pvessel, r0 = params
            
            obs_estimation = np.array([self._partial_pressure(r, M, pvessel, self.rin, r0) for r in r_axis])
            return obs_estimation[sorted_indices] - target
        
        cmro2_inital = 2.
        M_initial = cmro2_inital / self.cmro2_by_M
        r0_initial = 80.
        pvessel_initial = self.pvessel
        initial_params = [M_initial, pvessel_initial, r0_initial] # M, rin, rout
        bounds = ([1e-4, 30, self.rin], [1e-1, 120, r_axis.max()])
        print("\nLeast squares nonlinear fitting - 3 paramters (CMRO2, P_{vessel wall} and R0)\n")
        result = least_squares(residual, x0=initial_params, bounds=bounds, verbose=1, max_nfev=10000)
        
        # Estimated parameters
        self.estimated_params = result.x

        # 1D
        self.r_axis = r_axis
        self.target = target
        
        obs_estimation = np.array([self._partial_pressure(r, self.estimated_params[0], self.estimated_params[1], self.rin, self.estimated_params[2]) for r in self.distance_map.flatten()])
        self.fitted_target_array = obs_estimation.reshape(self.distance_map.shape, order='F')
        self.fitted_target = obs_estimation[sorted_indices]
                
    # def fit(self):
    #     self.mask = (self.distance_map > self.rin)
    #     r_axis = self.distance_map[self.mask].flatten()
    #     target = self.raw_data[self.mask].flatten()
    #     sorted_indices = np.argsort(r_axis)
    #     r_axis = r_axis[sorted_indices]
    #     target = target[sorted_indices]
        
    #     def residual(params):
    #         M, pvessel, r0 = params
            
    #         # Hole 1:
    #         cmro2_1     = M * self.cmro2_by_M
    #         Pves_1      = pvessel
    #         Rves_1      = self.rin
    #         R0_1        = r0
    #         center_1    = (0., 0., 0.)

    #         # Create solver parameters
    #         params = SolverParameters(filename="square_holes")

    #         # Define holes
    #         holes1 = [
    #             HoleGeometry(center=center_1, cmro2=cmro2_1, Pves=Pves_1, radius_ves=Rves_1, radius_0=R0_1),
    #             ]
        
    #         generator = MapGenerator(
    #             holes=holes1,
    #             params=params,
    #             X=self.X_axis,
    #             Y=self.Y_axis)
    #         obs_estimation = generator.pO2_array.flatten()
    #         return obs_estimation[sorted_indices] - target
        
    #     cmro2_inital = 2.
    #     M_initial = cmro2_inital / self.cmro2_by_M
    #     r0_initial = 80.
    #     pvessel_initial = self.pvessel
    #     initial_params = [M_initial, pvessel_initial, r0_initial] # M, rin, rout
    #     bounds = ([1e-3, 30, self.rin], [1e-2, 100, r_axis.max()])
    #     print("\nLeast squares nonlinear fitting - 3 paramters (CMRO2, P_{vessel wall} and R0)\n")
    #     result = least_squares(residual, x0=initial_params, bounds=bounds, verbose=1, max_nfev=10000)
        
    #     # Estimated parameters
    #     self.estimated_params = result.x
    #     # 1D
    #     self.r_axis = r_axis
    #     self.target = target
        
    #     # Hole 1:
    #     cmro2_1     = result.x[0] * self.cmro2_by_M
    #     Pves_1      = result.x[1]
    #     Rves_1      = self.rin
    #     R0_1        = result.x[2]
    #     center_1    = (0., 0., 0.)

    #     # Create solver parameters
    #     params = SolverParameters(filename="square_holes")

    #     # Define holes
    #     holes1 = [
    #         HoleGeometry(center=center_1, cmro2=cmro2_1, Pves=Pves_1, radius_ves=Rves_1, radius_0=R0_1),
    #         ]
        
    #     # Initialize the po2 map generator object
    #     generator = MapGenerator(
    #         holes=holes1,
    #         params=params,
    #         X=self.X_axis,
    #         Y=self.Y_axis)
    #     obs_estimation = generator.pO2_array
    #     self.fitted_target_array = obs_estimation
    #     self.fitted_target = obs_estimation.flatten()[sorted_indices]
                
    def plot_1D_results(self):
        # 1D Fit result
        plt.figure(figsize=(7, 5))
        plt.plot(self.r_axis, self.target, '*', label="Observed Data")
        plt.plot(self.r_axis, np.sort(self.fitted_target)[::-1], 'r-', linewidth=2, label="Fitted Curve")
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
        plt.pcolor(self.X_axis, self.Y_axis, self.fitted_target_array, shading='auto', cmap='jet')
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
        print("-"*65)
    
    def get_results(self):
        self.cmro2_est = self.estimated_params[0] * self.cmro2_by_M
        self.M_est = self.estimated_params[0]
        self.pvessel_est = self.estimated_params[1]
        self.r0_est = self.estimated_params[2]
        
        return self.cmro2_est, self.M_est, self.pvessel_est, self.r0_est
        
if __name__ == "__main__":
    
    # --------- Load data ---------
    df = pd.read_pickle("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/dataset.pkl")
    df_copy = df.copy()
    df_copy['pO2Value'] = df_copy['pO2Value'].apply(lambda x: x.flatten())
    
    # Select your the data
    art_id = 3
    dth_id = 2
    grid_size = 20

    # Load the map
    mask = (df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id )
    array = df_copy[mask]['pO2Value'].tolist()[0]
    X_axis = df_copy[mask]['pointsX'].tolist()[0]
    Y_axis = df_copy[mask]['pointsY'].tolist()[0]
    pO2_value = array.reshape((grid_size, grid_size), order='F')

    ##
    ## CMRO2 + pvessel + r0 fitting ##
    Rves = 17.
    center = (0.0, 0.0)
    lsqnonlin_fitter = Po2Fitter_3(pO2_array=pO2_value, Rves=Rves, X_axis=X_axis, Y_axis=Y_axis)
    lsqnonlin_fitter.fit()
    lsqnonlin_fitter.plot_estimated_parameters()
    cmro2_lsq, _, pvessel_lsq, R0_lsq = lsqnonlin_fitter.get_results()

    lsqnonlin_fitter.plot_1D_results()
    lsqnonlin_fitter.plot_2D_results()

    # 3D Plot the results the results
    params = SolverParameters(filename="square_holes")
    hole_estimated = [HoleGeometry(center=(*center, 0.0), cmro2=cmro2_lsq, Pves=pvessel_lsq, radius_ves=Rves, radius_0=R0_lsq)]
    generator = MapGenerator(
                        holes=hole_estimated,
                        params=params,
                        X=X_axis,
                        Y=Y_axis)
    obs_estimation = generator.pO2_array
    
    # 2D Fit result
    plt.figure(figsize=(7, 5))
    plt.pcolor(X_axis, Y_axis, obs_estimation, shading='auto', cmap='jet')
    plt.colorbar()
    plt.title("Nonlinear Fit of PO2 vs. Data")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
