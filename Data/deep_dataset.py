import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from Po2Dataset import load_data

# --------- Load data --------- #
py_data_location = os.getcwd()
df = pd.read_pickle(py_data_location + "/Data/dataset_radial.pkl")
df_copy = df.copy()
radial_dataset = load_data(py_data_location + '/Data/radial_dataset.txt')

# --------- Preprocess data --------- #
df_copy['pO2Value'] = df_copy['pO2Value'].apply(lambda x: x.flatten())
df_copy['pointsX'] = df_copy['pointsX'].apply(lambda x: x.flatten())
df_copy['pointsY'] = df_copy['pointsY'].apply(lambda x: x.flatten())
df_copy['arteriole_id'] = df_copy['arteriole_id'].astype(int)
df_copy.drop('tNew', axis=1, inplace=True)
df_copy.keys()

# --------- Constants --------- #
D = 4.0e3
alpha = 1.39e-15
cmro2_low, cmro2_high = 1, 3 # umol/cm3/min
cmro2_by_M = (60 * D * alpha * 1e12)
pixel_size = 10

for _, row in df_copy.iterrows():

    # Get arteriole ID, depth ID, and reshape the array
    art_id = row['arteriole_id']
    dth_id = row['depth_id']
    mask = (df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id)
    # Radial coords
    X = row['pointsX'].flatten()
    Y = row['pointsY'].flatten()
    Z = row['pO2Value'] - row['pO2Value'].mean(axis=0).flatten()

    # --- Create triangulation for irregular plotting ---
    triang = tri.Triangulation(X, Y)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.tripcolor(triang, Z, shading='flat', cmap='jet')
    plt.colorbar(label='pO_2')
    plt.axis('equal')
    plt.legend()
    plt.title(f"Radial pO₂ Map | Arteriole {art_id} | Depth {dth_id}")
    plt.show()

for i, entry in enumerate(radial_dataset):
    art_id = entry[0][0]
    dth_id = entry[0][1]

    angles_1 = entry[1]
    angles_2 = entry[2]

    min_radius = entry[3][0]

    # Observations
    obs = df_copy[(df_copy["arteriole_id"] == art_id) & (df_copy['depth_id'] == dth_id)]['pO2Value'].tolist()[0]
    pO2_array = obs.reshape((20, 20), order='F')
    
    plt.figure(figsize=(6, 6))
    plt.pcolor(pO2_array, cmap='jet', shading='auto')
    plt.colorbar(label='pO₂')
    plt.title(f"Radial pO₂ Map | Arteriole {art_id} | Depth {dth_id}")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')
    plt.show()

