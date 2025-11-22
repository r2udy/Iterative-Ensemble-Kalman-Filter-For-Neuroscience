"""
Created on Sat Nov 8 2:57:15 2025

@author: ruudybayonne
"""

import sys
import os

py_data_location = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/Synthetic Dataset/"
py_file_location = "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/classes/"
sys.path.append(os.path.abspath(py_file_location))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import EnKF_FEM
from FEM_code.generateMesh_Solver_multiple_holes import DiffusionSolver, SolverParameters, HoleGeometry
from MapGenerator import MapGenerator


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

from sklearn.mixture import GaussianMixture



def generate_synthetic_po2_maps():
    # ---------------------------
    # Parameters
    # ---------------------------
    grid_size = 20
    n_cells = grid_size * grid_size
    n_per_cmro2 = 15              # number of samples per cmro2 value (150 * 3 = 75 total)
    bank_R0 = np.array([80., 100., 120.])
    X_axis, Y_axis = np.meshgrid(np.linspace(-190, 190, 20), np.linspace(-190, 190, 20))
    sigma = 2.0

    # container for maps and metadata
    maps = []           # will hold (n_maps, grid_size, grid_size)
    meta = []           # will hold list of dicts with metadata

    # ---------------------------
    # Build synthetic bank
    # ---------------------------
    for i, R0_value in enumerate(bank_R0):
        # First vessel
        pvessel_true = 80.0
        pvessel_uncertainty = 2.0

        Rves_true = 10.
        Rves_uncertainty = 1.

        cmro2_true = 1.5
        cmro2_uncertainty = .2

        # draw perturbed parameter lists (n_per_cmro2 samples each)
        bank_pvessel_pertubed   = pvessel_true + np.random.normal(0, pvessel_uncertainty, size=n_per_cmro2)
        bank_Rves_pertubated    = Rves_true + np.random.normal(0, Rves_uncertainty, size=n_per_cmro2)
        # bank_R0_pertubed        = R0_true + np.random.normal(0, R0_uncertainty, size=n_per_cmro2)
        bank_cmro2_pertubed        = cmro2_true + np.random.normal(0, cmro2_uncertainty, size=n_per_cmro2)

        for j, (pvessel_sampled, Rves_sampled, cmro2_sampled) in enumerate(zip(bank_pvessel_pertubed, bank_Rves_pertubated, bank_cmro2_pertubed)):
            print(f"Maps {j+1 + i*n_per_cmro2} / {n_per_cmro2 * len(bank_R0)}")
            # Hole 1:
            cmro2_1     = cmro2_sampled
            Pves_1      = pvessel_sampled
            Rves_1      = Rves_sampled
            R0_1        = R0_value
            center_1    = (0., 0., 0.)

            # Hole 2:
            cmro2_2     = .5
            Pves_2      = pvessel_sampled * 0.8
            Rves_2      = 10.
            R0_2        = 120.
            position_x  = np.random.uniform(-170, -130)
            position_y  = np.random.uniform(-170, -130)
            center_2    = (position_x, position_y, 0.)

            # Hole 3:
            cmro2_3     = .5
            Pves_3      = pvessel_sampled * 0.8
            Rves_3      = 10.
            R0_3        = 120.
            position_x  = np.random.uniform(170, 130)
            position_y  = np.random.uniform(-170, 170)
            center_3    = (position_x, position_x, 0.)

            # Create solver parameters
            params = SolverParameters(filename="square_holes")

            # Define holes
            holes1 = [
                HoleGeometry(center=center_1, cmro2=cmro2_1, Pves=Pves_1, radius_ves=Rves_1, radius_0=R0_1, marker=params.marker),
                ]
            
            holes2 = [
                HoleGeometry(center=center_1, cmro2=cmro2_1, Pves=Pves_1, radius_ves=Rves_1, radius_0=R0_1, marker=params.marker),
                HoleGeometry(center=center_2, cmro2=cmro2_2, Pves=Pves_2, radius_ves=Rves_2, radius_0=R0_2, marker=params.marker + 1),
                ]

            holes3 = [
                HoleGeometry(center=center_1, cmro2=cmro2_1, Pves=Pves_1, radius_ves=Rves_1, radius_0=R0_1, marker=params.marker),
                HoleGeometry(center=center_2, cmro2=cmro2_2, Pves=Pves_2, radius_ves=Rves_2, radius_0=R0_2, marker=params.marker + 1),
                HoleGeometry(center=center_3, cmro2=cmro2_3, Pves=Pves_3, radius_ves=Rves_3, radius_0=R0_3, marker=params.marker + 2)
                ]

            # Combine profiles and add noise
            if j<n_per_cmro2//3:
                generator = MapGenerator(
                holes=holes1,
                params=params,
                X=X_axis,
                Y=Y_axis)
                profile = generator.pO2_array
                profile_perturbed = profile.flatten() + np.random.normal(np.zeros(n_cells), scale=sigma)

                maps.append(profile_perturbed.copy())
                meta.append({
                    "maps": profile_perturbed,
                    "cmro2": cmro2_sampled,
                    "pvessel_sampled": pvessel_sampled,
                    "Rves_sampled": Rves_sampled,
                    "R0_sampled": R0_value,
                    "capilary_1": None,
                    "capilary_2": None
                })
            
            elif j>=n_per_cmro2//2 and j<2*n_per_cmro2//3:
                generator = MapGenerator(
                holes=holes2,
                params=params,
                X=X_axis,
                Y=Y_axis)
                profile = generator.pO2_array
                profile_perturbed = profile.flatten() + np.random.normal(np.zeros(n_cells), scale=sigma)

                maps.append(profile_perturbed.copy())
                meta.append({
                    "maps": profile_perturbed,
                    "cmro2": cmro2_sampled,
                    "pvessel_sampled": pvessel_sampled,
                    "Rves_sampled": Rves_sampled,
                    "R0_sampled": R0_value,
                    "capilary_1": center_2,
                    "capilary_2": None
                })
            
            else:
                generator = MapGenerator(
                holes=holes3,
                params=params,
                X=X_axis,
                Y=Y_axis)
                profile = generator.pO2_array
                profile_perturbed = profile.flatten() + np.random.normal(np.zeros(n_cells), scale=sigma)

                maps.append(profile_perturbed.copy())
                meta.append({
                    "maps": profile_perturbed,
                    "cmro2": cmro2_sampled,
                    "pvessel_sampled": pvessel_sampled,
                    "Rves_sampled": Rves_sampled,
                    "R0_sampled": R0_value,
                    "capilary_1": center_2,
                    "capilary_2": center_3
                })
            
            # fig = plt.figure(figsize=(12, 8))
            # ax = fig.add_subplot(projection='3d')
            # obs_perturbated_array = profile_perturbed.reshape((grid_size, grid_size), order='F')
            # sc = ax.plot_surface(X_axis, Y_axis, obs_perturbated_array, cmap='viridis', edgecolor='none')
            # ax.set_xlabel('X (nm)')
            # ax.set_ylabel('Y (nm)')
            # ax.set_zlabel('pO2 (mmHg)')
            # plt.colorbar(sc, ax=ax, shrink=0.3, aspect=10, label='pO2 (mmHg)')
            # ax.set_title(f'Synthetic pO2 Data with Noise | cmro2={cmro2_value}')
            # plt.show()

    maps = np.array(maps)  # shape (n_maps, 20, 20)
    df_meta = pd.DataFrame(meta)

    return maps, df_meta

def save_synthetic_data(maps, df_meta, filename):
    np.savez(py_data_location + "/synthetic_po2_maps_" + filename + ".npz", maps=maps)
    df_meta.to_pickle(py_data_location + "/df_meta_" + filename + ".pkl")

def load_synthetic_data(filename):
    df_meta = pd.read_pickle(py_data_location + "/df_meta_" + filename + ".pkl")
    maps = np.load(py_data_location + "/synthetic_po2_maps_" + filename + ".npz")['maps']
    return maps, df_meta

if __name__ == "__main__":

    # ---------------------------
    # Generate synthetic data
    maps, df_meta = generate_synthetic_po2_maps()

    # Save generated data
    filename = "mulitple_sources_R0"
    save_synthetic_data(maps, df_meta, filename)

    # ---------------------------
    # Load generated data
    maps, df_meta = load_synthetic_data(filename)