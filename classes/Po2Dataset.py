# -*- coding: utf-8 -*-
"""
Created on Fri May  2 14:15:55 2025

@author: ruudy
"""

import math
import numpy as np
import pandas as pd
import scipy.io
import os
import re
from typing import List, Dict

# --------- Load data ---------
def load_data(filepath):
    with open(filepath, 'r') as f:
        # Read lines, remove comments, and skip empty lines
        lines = [
            line.split('#', 1)[0].strip()  # take text before '#' and strip spaces
            for line in f
            if line.strip() and not line.strip().startswith('#')
        ]
        # Remove lines that became empty after stripping comments
        lines = [line for line in lines if line]

        # Process each line into tuples
        data = [
            tuple(
                tuple(map(int, filter(None, pair.strip('()').split(','))))
                for pair in line.split('), (')
            )
            for line in lines
        ]
    return data

# ---------- Target Cells ----------- #
def get_cells_by_angle(grid_size, origin, angle_ranges, distance_range=None):
    x0, y0 = origin
    selected_cells = []

    for y in range(grid_size):
        for x in range(grid_size):
            dx = x - x0
            dy = y0 - y
            
            angle = math.degrees(math.atan2(dy, dx)) % 360
            distance = math.hypot(dx, dy)

            # Check angle ranges
            in_angle = any(
                start <= angle <= end if start <= end else angle >= start or angle <= end
                for (start, end) in angle_ranges
            )
            
            # Check distance range if given
            in_distance = True
            if distance_range:
                min_d, max_d = distance_range
                in_distance = min_d <= distance <= max_d

            if in_angle and in_distance:
                selected_cells.append((x, y))
    
    return selected_cells

# --------- Dataset creation ---------
class Po2Dataset:
    def __init__(self, base_dir: str, art_ids: List[int]):
        self.base_dir = base_dir
        self.art_ids = art_ids
        self.file_index = self._index_files()
        

    def _index_files(self) -> List[Dict]:
        index = []
        depth_map = {}
        metadata = scipy.io.loadmat(self.base_dir + 'database.mat')["main"]
        for art_id in self.art_ids:
            dir_path = self.base_dir + f"{art_id:02d}/" + "po2/"
            if not os.path.exists(dir_path):
                print(f"Warning: {dir_path} not found.")
                continue
            for file in os.listdir(dir_path):
                # if file.endswith(".mat") and not file.endswith("-r.mat"):
                if file.endswith(".mat"):
                    path = dir_path + file
                    mat_data = scipy.io.loadmat(path)
                    match = re.search(r'run(\d+)\.mat', path)
                    if match:
                        depth_id = int(match.group(1))
                        depth_val = metadata[art_id-1][0][17][0].tolist()[depth_id-1]
                        depth_map[depth_id] = depth_val
                    else:
                        depth_id = -1
                        depth_val = depth_map.get(depth_id, -1)
                    n_xy = int(np.sqrt(len(mat_data["pO2"]["pO2Value"].squeeze().tolist())))
                    n_time = len(mat_data["pO2"]["tNew"][0][0].squeeze())
                    index.append({
                        "arteriole_id": art_id,
                        "depth_id": depth_id,
                        "file_path": os.path.join(f"{art_id:02d}/" + "po2/", file),
                        "meta_sex": metadata[art_id-1][0][14].tolist()[0],
                        "depth": depth_val,
                        "pointsX": np.reshape(mat_data["pO2"]["pointsX"][0][0], (n_xy, n_xy)),
                        "pointsY": np.reshape(mat_data["pO2"]["pointsY"][0][0], (n_xy, n_xy)),
                        "data": np.reshape(mat_data["pO2"]["data"][0][0], (n_time, n_xy, n_xy)),
                        "tNew": mat_data["pO2"]["tNew"][0][0].squeeze(),
                        "pO2Value": np.array(mat_data["pO2"]["pO2Value"][0][0]),
                        "SR101": np.array(mat_data["pO2"]["references"].tolist()[0][0][0][0][1].tolist()),
                        "FITC": np.array(mat_data["pO2"]["references"].tolist()[0][0][0][1][1].tolist()),
                        "OxyPhor-2P": np.array(mat_data["pO2"]["references"].tolist()[0][0][0][2][1].tolist())
                    })
        return index

    def __len__(self):
        return len(self.file_index)

    def __getitem__(self, idx: int) -> Dict:
        file_entry = self.file_index[idx]
        mat_data = scipy.io.loadmat(file_entry["file_path"])
        return {
            "art_id": file_entry["art_id"],
            "depth_id": file_entry["depth_id"],
            "file_path": file_entry["file_path"],
            "data": mat_data
        }

# --- New radial dataset class ---
class Po2RadialDataset(Po2Dataset):
    def _index_files(self) -> List[Dict]:
        index = []
        depth_map = {}
        metadata = scipy.io.loadmat(self.base_dir + 'database.mat')["main"]

        # Iterate through arteriole IDs
        for art_id in self.art_ids:
            dir_path = self.base_dir + f"{art_id:02d}/" + "po2/"
            if not os.path.exists(dir_path):
                print(f"Warning: {dir_path} not found.")
                continue

            for file in os.listdir(dir_path):
                # Only keep radial files
                if file.endswith("-r.mat"):
                    path = dir_path + file
                    mat_data = scipy.io.loadmat(path)

                    # Find depth_id from filename if possible
                    match = re.search(r'run(\d+)(?:-r)?\.mat', path)
                    if match:
                        depth_id = int(match.group(1))
                    else:
                        print(f"Warning: Could not determine depth_id for {file}, skipping.")
                        continue

                    # Always use depth_val from metadata
                    depth_val = metadata[art_id-1][0][17][0].tolist()[depth_id-1]

                    n_xy = int(np.sqrt(len(mat_data["pO2"]["pO2Value"].squeeze().tolist())))
                    n_time = len(mat_data["pO2"]["tNew"][0][0].squeeze())

                    index.append({
                        "arteriole_id": art_id,
                        "depth_id": depth_id,
                        "file_path": os.path.join(f"{art_id:02d}/" + "po2/", file),
                        "meta_sex": metadata[art_id-1][0][14].tolist()[0],
                        "depth": depth_val,
                        "pointsX": np.reshape(mat_data["pO2"]["pointsX"][0][0], (n_xy, n_xy)),
                        "pointsY": np.reshape(mat_data["pO2"]["pointsY"][0][0], (n_xy, n_xy)),
                        "data": np.reshape(mat_data["pO2"]["data"][0][0], (n_time, n_xy, n_xy)),
                        "tNew": mat_data["pO2"]["tNew"][0][0].squeeze(),
                        "pO2Value": np.array(mat_data["pO2"]["pO2Value"][0][0]),
                        "SR101": np.array(mat_data["pO2"]["references"].tolist()[0][0][0][0][1].tolist()),
                        "FITC": np.array(mat_data["pO2"]["references"].tolist()[0][0][0][1][1].tolist()),
                        "OxyPhor-2P": np.array(mat_data["pO2"]["references"].tolist()[0][0][0][2][1].tolist())
                    })
        return index


if __name__ == "__main__":
    
    # # Original full dataset
    # dataset = Po2Dataset(
    #     base_dir="/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/TODEsource/dbase/", 
    #     art_ids=np.arange(1, 12)
    # )

    # # Save the file index to a pickle file
    # df = pd.DataFrame(dataset.file_index).to_pickle(
    #     "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/dataset.pkl"
    # )

    # New radial dataset
    radial_dataset = Po2RadialDataset(
        base_dir="/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/TODEsource/dbase/",
        art_ids=np.arange(1, 12)
    )
    pd.DataFrame(radial_dataset.file_index).to_pickle(
        "/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/dataset_radial.pkl"
    )
