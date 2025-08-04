# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 11:25:24 2025

@author: ruudy
"""

import numpy as np
import matplotlib.pyplot as plt
import ufl
import gmsh
import basix
import dolfinx.plot
import pyvista
from Core.Mesh import plot_mesh_with_physical_groups, create_mesh
from dolfinx import fem, io, mesh, log
from dolfinx.io import XDMFFile, gmshio
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from typing import Callable, Optional
from circlesearch import Po2Analyzer
from multiprocessing import Pool
from scipy.interpolate import griddata
from FEM_code.generateMesh_Solver_one_hole import HoleGeometry, DiffusionSolver, SolverParameters

class EnKF:
    def __init__(self,
                 state_dim: int,
                 obs_dim: int,
                 n_ensembles:int,
                 dynamics_model: Callable,
                 rng: Optional[np.random.Generator] = None,
                 seed: Optional[int] = 0):
        
        # Constants conversion
        self.SEC_MIN = 60
        self.CM3_M3 = 1e6
        self.UM3_M3 = 1e18
        self.D = 4.0e3
        self.alpha = 1.39e-15
        self.cmro2_by_M = self.SEC_MIN * self.UM3_M3 / self.CM3_M3 * self.D * self.alpha
    
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_ensembles = n_ensembles
        self.nu = 1 / 10 

        self.path = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/FEM_code/SavedFiles/Results/"
        
        self.dynamics_model = dynamics_model
        self.seed = seed
        if rng is None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = rng
        
        # Initialize ensemble
        self.ensemble = np.zeros((state_dim, n_ensembles))
        
        # Covariance matrices
        self.Q = np.eye(state_dim) # Background noise covariance
        self.R = np.eye(obs_dim) # Observation noise covariancemu
        
        
    def initialize_ensemble(self, a: np.ndarray, b: np.ndarray):
        """
        Initialize the ensemble with samples from a Gaussian distribution
        
        Parameters:
        -----------
        a : np.ndarray, shape (state_dim,)
            Mean of the initial state distribution (or lower bound for uniform distrib)
        b: np.ndarray, shape (state_dim,)
            Covariance of the initial state distribution (or upper bound for uniform distrib)
        """
        # ratio M
        self.ensemble[0] = self.rng.uniform(
            a[0], b[0], size=(1, self.n_ensembles)
            )  # Shape: (state_dim, n_ensembles)

    def set_process_noise(self, Q: np.ndarray):
        """Set the background noise covariance matrix"""
        self.Q = Q
    
    def set_observation_noise(self, R: np.ndarray):
        """Set the observation noise covariance matrix"""
        self.R = R

    def observation_operator(self, observation: np.ndarray, state: np.ndarray):
        """

        Parameters
        ----------
        observation: np.ndarray, shape (obs_dim,)
            The observed measurment 
        state: np.ndarray, shape (obs_dim,)
            The ratio of the oxygen consumption and permeability of the tissue M
        Returns
        -------
        analytic_map: np.ndarray, shape (obs_dim,)
            Anylitical Map of partial oxygen pressure
        annnular_idx: np.ndarray, shape (obs_dim,)
            Index of the 
            
        """
        
        assert observation.shape == (self.obs_dim,)
        assert state.shape == (self.state_dim,)
        
        # Initialize Circle Search
        n = 20 # observation dimension
        observation.mean()
        observation_array = np.reshape(observation, (n, n)) # ensemble member observation (n by n)
        analyzer = Po2Analyzer(observation_array)
        analyzer.find_circles()
        
        # Extract parameters from state
        cmro2   = state * self.cmro2_by_M
        Pves    = np.max(observation)
        Rves    = analyzer.rin
        R0      = analyzer.rout
        marker = 3

        # Generate mesh with dynamic radii
        # Initialize MPI
        comm = MPI.COMM_SELF

        # Create solver instance
        solver = DiffusionSolver(comm)

        # Create solver parameters
        params = SolverParameters(filename="square_one_hole", cmro2=cmro2, Pves=Pves, Rves=Rves, R0=R0)
        
        # Define holes
        holes = [
            HoleGeometry(center=(0, 0, 0), radius_ves=params.Rves, radius_0=params.R0, marker=3),
            ]
        
        # Generate mesh
        solver.generate_mesh(holes)
        
        # Set up and solve problem
        solver.setup_problem(params, holes)
        solver.solve()

        # Save results
        solver.save_results(params.filename)

        # -------------------------
        # Interpolate to observation grid
        uh = solver.uh.x.array
        domain_coordinate = solver.domain.geometry.x
        x = np.array(domain_coordinate[:, 0])
        y = np.array(domain_coordinate[:, 1])
        
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        # Create observation grid points
        x_obs = np.linspace(x_min, x_max, n)
        y_obs = np.linspace(y_min, y_max, n)
        
        # Create simulation grid
        x_idx_domain = solver.interpolation_grid(x, x_obs)
        y_idx_domain = solver.interpolation_grid(y, y_obs)
        x_domain = x[x_idx_domain]
        y_domain = y[y_idx_domain]
        X_domain, Y_domain = np.meshgrid(x_domain, y_domain)
        points = np.column_stack((X_domain.ravel(), Y_domain.ravel()))

        # Evaluate FEM solution at observation points
        obs_model = griddata((x, y), uh, points, method='linear') # Interpolate z values at the grid points
        
        return obs_model
        
        
    def predict(self, observation: np.ndarray):
        """
        Predictions step: propagate each ensemble member through the dynamics model
        and add background noise. Here there is no dynamics model.
        """

        for i in range(self.n_ensembles):
            # Propagate state through dynamics model
            self.ensemble[:, i] = self.dynamics_model(self.ensemble[:, i])
            
            # Add background noise
            self.ensemble[:, i] += self.rng.multivariate_normal(np.zeros(self.state_dim), self.Q)

        # Add Measurment noise
        noise = self.rng.multivariate_normal(np.zeros(self.obs_dim), self.nu * self.R)

        self.observation = observation + noise

    
    def update(self):
        
        """
        Update step: adjust the ensemble based on observations
        
        Parameters:
        -----------
        observation: np.ndarray, shape (obs_dim,)
            The observed measurement
        angle_range1_deg: First angle range in degrees (min, max)
        angle_range2_deg: Second angle range in degrees (min, max)
        """
        # Generate pertubated observations according to a Gaussian distributions
        obs_ensemble = np.zeros((self.obs_dim, self.n_ensembles))
        obs_model_ensembles = np.zeros_like(obs_ensemble)
        
        # Generate perturbed observations
        for i in range(self.n_ensembles):
            obs_ensemble[:, i] = self.observation + self.rng.multivariate_normal(
                np.zeros(self.obs_dim), self.R
            )
        
        # Filter out observation parameter outside the annular 
        for i in range(self.n_ensembles):
            # ensemble member state parameter
            state = self.ensemble[:, i]
            
            obs_model = self.observation_operator(obs_ensemble[:, i], state)
            obs_model_ensembles[:, i] = obs_model

        
        # 1. Compute ensemble means and deviations
        state_mean = np.mean(self.ensemble, axis=1)
        obs_mean = np.mean(obs_model_ensembles, axis=1)
        
        state_deviation = self.ensemble - state_mean[:, np.newaxis]
        obs_deviation = obs_model_ensembles - obs_mean[:, np.newaxis]
        
        # 2. Compute Kalman Gain
        A_BHT = (state_deviation @ obs_deviation.T) / (self.n_ensembles - 1)
        A_HBHT = (obs_deviation @ obs_deviation.T) / (self.n_ensembles - 1)
        self.K = A_BHT @ np.linalg.inv(A_HBHT + self.R)
        
        # 3. Update ensemble: innovation = observation - obs_model(ensemble)
        innovation = obs_ensemble - obs_model_ensembles
        self.ensemble += self.K @ innovation
        
    def get_state_estimate(self):
        """
        Get the current state estimate (mean and covariance)
        
        Returns:
        --------
        mean : np.ndarray, shape (state_dim,)
            Mean of the state estimate
        cov : np.ndarray, shape (state_dim, state_dim)
            Covariance of the state estimate
        """
        mean = np.mean(self.ensemble, axis=1)
        cov = np.cov(self.ensemble)
        return mean, cov
    
    def get_ensemble(self) -> np.ndarray:
        """
        Get the current ensemble
        
        Returns:
        --------
        ensemble : np.ndarray, shape (state_dim, n_ensembles)
            The ensemble members
        """
        return self.ensemble
    
    def get_Kalman_gain(self):
        """
        Get the current Kalman Gain
        
        Returns:
        --------
        ensemble : np.ndarray, shape (state_dim, n_ensembles)
            The ensemble members
        """
        return self.K