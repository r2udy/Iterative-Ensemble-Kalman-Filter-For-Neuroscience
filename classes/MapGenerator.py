# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 16:54:17 2025

@author: ruudy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import Tuple
from mpi4py import MPI
from FEM_code.generateMesh_Solver_one_hole import DiffusionSolver, SolverParameters, HoleGeometry



class MapGenerator:
    def __init__(self,
                 cmro2: float,
                 pvessel: float,
                 Rves: float,
                 R0: float,
                 Rt: float,
                 model: str = 'KE',
                 pixel_size: float = 10.0,
                 center: Tuple[int, int] = (10, 11),
                 grid_size: Tuple[int, int] = (20, 20)):
        
        """
        Initialize MapGenerator with vascular parameters and grid configuration.
        
        Args:
           cmro2: Cerebral metabolic rate of oxygen (µmol/min/100g)
           pvessel: Vessel partial pressure (mmHg)
           Rves: Vessel radius (µm)
           R0: Inner radius (µm)
           Rt: Tissue radius (µm)
           pixel_size: Size of each pixel in µm
           center: Center coordinates (x,y) in grid units
           grid_size: Dimensions of the grid (cols, rows)
        """
        
        # Constants conversion
        self.SEC_MIN = 60
        self.CM3_M3 = 1e6
        self.UM3_M3 = 1e18
        self.D = 4.0e3
        self.alpha = 1.39e-15
        self.cmro2_by_M = self.SEC_MIN * self.UM3_M3 / self.CM3_M3 * self.D * self.alpha
        
        # Vascular parameters
        self.cols, self.rows = 20, 20
        self.X, self.Y = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
        self.cmro2 = cmro2
        self.M = self.cmro2 / self.cmro2_by_M
        self.pvessel = pvessel
        self.Rves = Rves
        self.R0 = R0
        self.Rt = Rt
        self.pixel_size = pixel_size
        
        # Grid configuration
        self.center = np.array(center)
        self.cols, self.rows = grid_size
        self.X, self.Y = np.meshgrid(np.arange(self.cols), np.arange(self.rows))
        
        # Generate pressure map
        self.model = model
        self.pO2_array = self.generate_map()
    
    def _partial_pressure_KE(self, r: np.ndarray) -> np.ndarray:
        """
        Calculate partial pressure from Krogh-Erlang model at given radial distances.
        
        Args:
            r: Array of radial distances from vessel center (µm)
            
        Returns:
            Array of partial pressures (mmHg)
        """
        r = np.asarray(r)
        if np.any(r <= 0):
            raise ValueError("All radius values must be positive and non-zero.")
        
        term_r_Rves     = self.pvessel
        term_Rves_r_Rt  = lambda x: self.pvessel + (self.M / 4) * (x**2 - self.Rves**2) - (self.M * self.R0**2 / 2) * np.log(x / self.Rves)
        
        # Initialize result array
        result = np.zeros_like(r)
    
        # Region masks
        inside_vessel = r < self.Rves
        vessel_wall = r >= self.Rves
    
        # Region 1: Inside vessel
        result[inside_vessel] = term_r_Rves
        
        # Region 2: Vessel wall
        if np.any(vessel_wall):
            r_masked = r[vessel_wall]
            result[vessel_wall] = term_Rves_r_Rt(r_masked) 

        return result
    
    def _partial_pressure(self, r: np.ndarray) -> np.ndarray:
        """
        Calculate partial pressure from ODACITI model at given radial distances.
        
        Args:
            r: Array of radial distances from vessel center (µm)
            
        Returns:
            Array of partial pressures (mmHg)
        """
        r = np.asarray(r)
        if np.any(r <= 0):
            raise ValueError("All radius values must be positive and non-zero.")
            
        beta            = (self.M / 2) * (self.Rves**2 - self.R0**2)
        term_r_Rves     = self.pvessel
        term_Rves_r_Rt  = lambda x: self.pvessel + (self.M / 4) * ((x**2 - self.Rves**2) - 2 * self.Rves**2 * np.log(x / self.Rves)) + beta * np.log(x / self.Rves)
        term_Rt_r       = lambda x: self.pvessel + (self.M / 4) * ((self.Rt**2 - self.Rves**2) - 2 * self.Rves**2 * np.log(x / self.Rves) + 2 * self.Rt**2 * np.log(x / self.Rt)) + beta * np.log(x / self.Rves)
        
        # Initialize result array
        result = np.zeros_like(r)
    
        # Region masks
        inside_vessel = r < self.Rves
        vessel_wall = (r >= self.Rves) & (r < self.Rt)
        tissue = r >= self.Rt
    
        # Region 1: Inside vessel
        result[inside_vessel] = term_r_Rves
    
        # Region 2: Vessel wall
        if np.any(vessel_wall):
            r_masked = r[vessel_wall]
            result[vessel_wall] = term_Rves_r_Rt(r_masked) 

        # Region 3: Tissue
        if np.any(tissue):
            r_masked = r[tissue]
            result[tissue] = term_Rt_r(r_masked)

        return result
    
    # def generate_map(self) -> np.ndarray:
    #     # Center the outer circle in the middle of the pixel cells
    #     dx = (self.X - (2*self.center[0] - 1) / 2) * self.pixel_size
    #     dy = (self.Y - (2*self.center[1] - 1) / 2) * self.pixel_size
    #     r = np.sqrt(dx**2 + dy**2)
    #     self.r_values = r
        
    #     if self.model == 'KE':
    #         return self._partial_pressure_KE(r)
    #     elif self.model == 'ODACITI':
    #         return self._partial_pressure(r)
        
    def generate_map(self) -> np.ndarray:

        # Generate mesh with dynamic radii
        # Initialize MPI
        comm = MPI.COMM_SELF

        # Create solver instance
        solver = DiffusionSolver(comm)

        # Create solver parameters
        params = SolverParameters(filename="square_one_hole", cmro2=self.cmro2, Pves=self.pvessel, Rves=self.Rves, R0=self.R0)
        
        # Define holes
        holes = [
            HoleGeometry(center=(0, 0, 0), radius_ves=self.Rves, radius_0=self.R0, marker=3),
            ]
        
        # Generate mesh
        solver.generate_mesh(holes)
        
        # Set up and solve problem
        solver.setup_problem(params, holes)
        solver.solve()

        uh = solver.uh.x.array
        domain_coordinate = solver.domain.geometry.x
        x = np.array(domain_coordinate[:, 0])
        y = np.array(domain_coordinate[:, 1])
        
        # -------------------------
        # Interpolate to observation grid
        # -------------------------
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        # Create observation grid points
        x_obs = np.linspace(x_min, x_max, self.cols)
        y_obs = np.linspace(y_min, y_max, self.cols)
        
        # Create simulation grid
        x_idx_domain = solver.interpolation_grid(x, x_obs)
        y_idx_domain = solver.interpolation_grid(y, y_obs)
        x_domain = x[x_idx_domain]
        y_domain = y[y_idx_domain]
        X_domain, Y_domain = np.meshgrid(x_domain, y_domain)
        points = np.column_stack((X_domain.ravel(), Y_domain.ravel()))

        # Evaluate FEM solution at observation points
        obs_model = griddata((x, y), uh, points, method='linear').reshape((self.cols, self.rows), order = 'F') # Interpolate z values at the grid points

        return obs_model
    
    def plot_2d(self, title: str = 'Partial Pressure Map'):
        """Visualize the pressure map."""
        plt.figure(figsize=(8, 6))
        plt.imshow(self.pO2_array, cmap='viridis', origin='lower')
        plt.colorbar(label='Partial Pressure (mmHg)')
        plt.title(title)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.show()
    
    def plot_3d(self, title: str = '3D Oxygen Partial Pressure Distribution', array = np.zeros((20, 20))):
        """Create a 3D surface plot of the pressure map"""
        
        if np.any(array) == 0:
            array = self.pO2_array
            
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create coordinate grids in physical units (microns)
        X = self.X * self.pixel_size
        Y = self.Y * self.pixel_size
    
        # Plot surface
        surf = ax.plot_surface(X, Y, array, 
                               cmap='viridis',
                               rstride=1, cstride=1,
                               linewidth=0, 
                               antialiased=True)
    
        # Add labels and colorbar
        ax.set_xlabel('X (µm)')
        ax.set_ylabel('Y (µm)')
        ax.set_zlabel('Partial Pressure (mmHg)')
        ax.set_title(title, fontsize=18)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        plt.tight_layout()
        plt.show()