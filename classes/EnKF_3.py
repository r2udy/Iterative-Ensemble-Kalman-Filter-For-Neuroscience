# -*- coding: utf-8 -*-
"""
Created on Fri May  9 10:38:08 2025

@author: ruudy
"""

import numpy as np
from typing import Callable, Optional
from circlesearch import Po2Analyzer
from MapGenerator import MapGenerator

class EnKF:
    def __init__(self,
                 state_dim: int,
                 obs_dim: int,
                 n_ensembles:int,
                 dynamics_model: Callable,
                 rng: Optional[np.random.Generator] = None,
                 seed: Optional[int] = None):
        
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
        
        self.dynamics_model = dynamics_model
        
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
        
        # Pressure at vessel wall
        p_mean, p_std = a[1], b[1]
        self.ensemble[1] = self.rng.normal(
            p_mean, p_std, size=(1, self.n_ensembles)
            )
        
        r0_mean, r0_std = a[2], b[2]
        self.ensemble[2] = self.rng.normal(
            r0_mean, r0_std, size=(1, self.n_ensembles)
            )

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
        # Initialize Circle Search
        n = 20 # observation dimension
        observation_array = np.reshape(observation, (n, n)) # ensemble member observation (n by n)
        analyzer = Po2Analyzer(observation_array, M=state[0], r0=state[2])
        analyzer.find_circles()
        
        # Initialize Map generator
        cmro2   = state[0] * self.cmro2_by_M
        pvessel = state[1]
        Rves    = analyzer.rin
        R0      = state[2]
        Rt      = analyzer.rout
        
        generator = MapGenerator(
            cmro2=cmro2,
            pvessel=pvessel,
            Rves=Rves,
            R0=R0,
            Rt=Rt,
            model='KE'
        )
        obs_model = generator.pO2_array.flatten()
        
        mask_outer = analyzer.mask_outer
        mask_inner = analyzer.mask_inner
        mask_angle = analyzer.mask_angle
        annular_idx = mask_angle * (1 - mask_inner - ~mask_outer)
        
        return obs_model, annular_idx
        
    
    def predict(self):
        """
        Predictions step: propagate each ensemble member through the dynamics model
        and add background noise. Here there is no dynamics model.
        """

        for i in range(self.n_ensembles):
            # Propagate state through dynamics model
            self.ensemble[:, i] = self.dynamics_model(self.ensemble[:, i])
            
            # Add background noise
            self.ensemble[:, i] += self.rng.multivariate_normal(np.zeros(self.state_dim), self.Q)
    
    def update(self, observation: np.ndarray, angle_range1_deg: tuple = (0, 0), angle_range2_deg: tuple = (0, 0)):
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
            obs_ensemble[:, i] = observation + self.rng.multivariate_normal(
                np.zeros(self.obs_dim), self.R)
        
        # Filter out observation parameter outside the annular 
        for i in range(self.n_ensembles):
            # ensemble member state parameter
            state = self.ensemble[:, i]
            
            obs_model, annular_idx = self.observation_operator(obs_ensemble[:, i], state)
            obs_model_ensembles[:, i] = obs_model

        
        # 1. Compute ensemble means and deviations
        state_mean = np.mean(self.ensemble, axis=1)
        obs_mean = np.mean(obs_model_ensembles, axis=1)
        
        state_deviation = self.ensemble - state_mean[:, np.newaxis]
        obs_deviation = obs_model_ensembles - obs_mean[:, np.newaxis]
        
        # 2. Compute Kalman Gain
        A_BHT = (state_deviation @ obs_deviation.T) / (self.n_ensembles - 1)
        A_HBHT = (obs_deviation @ obs_deviation.T) / (self.n_ensembles - 1)
        K = A_BHT @ np.linalg.inv(A_HBHT + self.R)
        
        # 3. Update ensemble: innovation = observation - obs_model(ensemble)
        innovation = obs_ensemble - obs_model_ensembles
        self.ensemble += K @ innovation
        
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
        