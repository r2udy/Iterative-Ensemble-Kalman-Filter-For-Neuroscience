# -*- coding: utf-8 -*-
"""
Created on Thurs Jul 24 09:39:00 2025

@author: ruudy
"""

import sys
import os
import math
import numpy as np
from scipy.stats import truncnorm
import concurrent.futures
from mpi4py import MPI
from typing import Callable, Optional
from circlesearch import Po2Analyzer
from scipy.interpolate import griddata
from Po2Dataset import get_cells_by_angle

py_file_location = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/"
sys.path.append(os.path.abspath(py_file_location))
from FEM_code.generateMesh_Solver_multiple_holes import HoleGeometry, DiffusionSolver, SolverParameters

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
        self.B = np.eye(state_dim) # Background noise covariance
        self.R = np.eye(obs_dim) # Observation noise covariance

        self.length_scale = 1.0 # Length scale for the update step
        self.grid_size = 20 # observation dimension
        
        
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
        for k in range(self.state_dim):
            self.ensemble[k, :] = self.rng.normal(
                a[k], b[k], size=(1, self.n_ensembles)
                )  # Shape: (state_dim, n_ensembles)
        
        self.R0_prior = a[1]
        self.sigma_R0 = b[1]

    def set_process_noise(self, B: np.ndarray):
        """Set the background noise covariance matrix"""
        self.B = B
    
    def set_observation_noise(self, R: np.ndarray):
        """Set the observation noise covariance matrix"""
        self.R = R

    def observation_operator(self, state: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
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
        
        assert state.shape == (self.state_dim,)
                
        # Extract parameters from state
        cmro2   = state[0] * self.cmro2_by_M
        Pves    = state[2]
        Rves    = 17.5#np.diff(X[0])[0]
        R0      = state[1]
        center  = (0.0, 0.0)
        marker = 3

        # Generate mesh with dynamic radii
        # Initialize MPI
        comm = MPI.COMM_SELF

        # Create solver instance
        solver = DiffusionSolver(comm)

        # Create solver parameters
        params = SolverParameters(filename="square_one_hole")

        # Define holes
        holes = [HoleGeometry(center=(*center, 0), 
                              cmro2=cmro2, Pves=Pves, 
                              radius_ves=Rves, 
                              radius_0=R0, marker=3),
        ]

        # Generate mesh
        solver.generate_mesh(holes)
        
        # Set up and solve problem
        solver.setup_problem(params, holes)
        solver.solve()

        # -------------------------
        # Interpolate to observation grid
        uh = solver.uh.x.array
        domain_coordinate = solver.domain.geometry.x
        x = np.array(domain_coordinate[:, 0])
        y = np.array(domain_coordinate[:, 1])
        
        # Create observation grid points
        x_obs = X[0] - X[0].mean()
        y_obs = Y[:,0] - Y[:,0].mean()
        
        # Create simulation grid
        x_idx_domain = solver.interpolation_grid(x, x_obs)
        y_idx_domain = solver.interpolation_grid(y, y_obs)
        x_domain = x[x_idx_domain]
        y_domain = y[y_idx_domain]
        X_domain, Y_domain = np.meshgrid(x_domain, y_domain)
        points = np.column_stack((X_domain.ravel(), Y_domain.ravel()))

        # Evaluate FEM solution at observation points
        obs_model = griddata((x, y), uh, points, method='linear').reshape((self.grid_size, self.grid_size), order='F').T # Interpolate z values at the grid points
        
        return obs_model.flatten()
        
        
    def predict(self):
        """
        Predictions step: propagate each ensemble member through the dynamics model
        and add background noise. Here there is no dynamics model.
        """

        for i in range(self.n_ensembles):
            # Propagate state through dynamics model
            self.ensemble[:, i] = self.dynamics_model(self.ensemble[:, i])
            
            # Add background noise using truncated normal distribution
            self.ensemble[:, i] += self.rng.multivariate_normal(np.zeros(self.state_dim), self.B)

    def update(self, observation: np.ndarray, X: np.ndarray, Y: np.ndarray):
        """
        Update step: adjust the ensemble based on observations
        
        Parameters:
        -----------
        observation: np.ndarray, shape (obs_dim,)
            The observed measurement
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    
        # Generate pertubated observations according to a Gaussian distributions
        obs_ensemble = np.zeros((self.obs_dim, self.n_ensembles))
        obs_model_ensembles = np.zeros_like(obs_ensemble)
        
        # Generate perturbed observations
        assert self.R.shape == (self.obs_dim, self.obs_dim)
        obs_perturbation = self.rng.multivariate_normal(np.zeros(self.obs_dim), self.R, size=self.n_ensembles).T
        obs_ensemble = observation[:, np.newaxis] + obs_perturbation
        
        # Split ensembles across ranks
        local_indices = np.array_split(np.arange(self.n_ensembles), size)[rank]
        local_results = []

        for i in local_indices:
            state = self.ensemble[:, i]
            result = self.observation_operator(state, X, Y)
            local_results.append(result)

        # Gather results from all ranks
        gathered_results = comm.allgather(local_results)
        gathered_results = [item for sublist in gathered_results for item in sublist]
        obs_model_ensembles = np.array(gathered_results).T  # shape (obs_dim, n_ensembles)

        # 1. Compute ensemble means and deviations
        state_mean  = np.mean(self.ensemble, axis=1)
        obs_mean    = np.mean(obs_model_ensembles, axis=1)
        state_deviation     = self.ensemble - state_mean[:, np.newaxis]
        obs_deviation       = obs_model_ensembles - obs_mean[:, np.newaxis]

        # Augmented observed ensemble
        R0_obs_pert = self.rng.normal(
            loc=self.R0_prior,
            scale=self.sigma_R0,
            size=self.n_ensembles
        )
        obs_ensemble_aug = np.vstack([
            obs_ensemble, 
            R0_obs_pert[np.newaxis, :]
        ])

        R0_model = self.ensemble[1, :]

        obs_model_ensembles_aug = np.vstack([
            obs_model_ensembles,
            R0_model[np.newaxis, :]
        ])

        obs_mean_aug = np.mean(obs_model_ensembles_aug, axis=1)
        obs_deviation_aug = obs_model_ensembles_aug - obs_mean_aug[:, np.newaxis]

        Q = np.block([
            [self.R,                        np.zeros((self.obs_dim, 1))],
            [np.zeros((1, self.obs_dim)),   np.array([[self.sigma_R0**2]])]
        ])
        # 2. Compute Kalman Gain
        # A_B = (state_deviation @ state_deviation.T) / (self.n_ensembles - 1)
        self.A_BHT = (state_deviation @ obs_deviation_aug.T) / (self.n_ensembles - 1)
        self.A_HBHT = (obs_deviation_aug @ obs_deviation_aug.T) / (self.n_ensembles - 1)
        self.K = self.A_BHT @ np.linalg.inv(self.A_HBHT + Q)
        
            # 3. Update ensemble: innovation = (perturbed) observation - model prediction (both augmented)
        self.innovation_aug = obs_ensemble_aug - obs_model_ensembles_aug
        # keep legacy attribute name for external code compatibility
        self.innovation = self.innovation_aug
        self.ensemble += self.length_scale * self.K @ self.innovation_aug

        

        # Compute NIS on the original (unaugmented) observation space
        A_HBHT_unaug = (obs_deviation @ obs_deviation.T) / (self.n_ensembles - 1)
        NIS = (observation - obs_mean).T @ np.linalg.inv(A_HBHT_unaug + self.R) @ (observation - obs_mean)
        self.NIS = NIS    
            
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
    
    def get_innovation(self):
        """
        Get the current innovation
        
        Returns:
        --------
        innovation : np.ndarray, shape (obs_dim, n_ensembles)
            The innovation for each ensemble member
        """
        return self.innovation
        