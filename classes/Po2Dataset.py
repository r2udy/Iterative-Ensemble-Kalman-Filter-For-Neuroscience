# -*- coding: utf-8 -*-
"""
Created on Fri May  2 14:15:55 2025

@author: ruudy
"""


import numpy as np
import pandas as pd
import scipy.io
import os
import re
from typing import List, Dict

# --------- Load data ---------
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


if __name__ == "__main__":
    
    # create the dataset object
    dataset = Po2Dataset(base_dir="/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/TODEsource/dbase/", art_ids=np.arange(1, 12))
    # turn it into pandas dataframe
    df = pd.DataFrame(dataset.file_index)
    # Save it 
    df.to_pickle("/Users/ruudybayonne/Desktop/Stanford_Biology/PROJECT_OxyDiff/Python_code/Data/dataset.pkl")