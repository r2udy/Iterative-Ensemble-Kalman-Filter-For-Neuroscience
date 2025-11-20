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
from synthetic_data_generation import load_synthetic_data

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

from sklearn.mixture import GaussianMixture

# ---------------------------
# Save generated data
filename = "mulitple_sources"

# ---------------------------
# Load generated data
maps, df_meta = load_synthetic_data(filename)

# ---------------------------
# Prepare data for clustering
# ---------------------------
# number of maps
n_maps = maps.shape[0]

# Flatten maps to vectors
X = maps.reshape((n_maps, -1))  # shape (n_maps, 400)

# It's often useful to standardize each feature (pixel) across maps before kmeans
scaler = StandardScaler(with_mean=True, with_std=True)  # center and scale features
X_scaled = scaler.fit_transform(X)

# Optionally reduce dimension with PCA if desired (not required for small n_maps)
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)
# Use X_pca for clustering if used

labels_true = np.abs(df_meta['cmro2'].values-3).astype(int)
# ---------------------------
# Run KMeans
# ---------------------------
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
labels = kmeans.fit_predict(X_scaled)

# Score (quality metric): How well-separated are the clusters
if n_maps > n_clusters:
    ground_truth_score = metrics.rand_score(labels_true, labels)  # within-c
    sil = silhouette_score(X_scaled, labels) # silhouette score: How well-separated are the clusters (higher is better)
    dbi = davies_bouldin_score(X_scaled, labels) # Davies-Bouldin Index: measures cluster separation (lower is better)
    ch = calinski_harabasz_score(X_scaled, labels) # Calinski-Harabasz Score: measures cluster tightness and separation (higher values indicate better-defined clusters)
else:
    sil = np.nan

print("Running KMeans clustering...")
print(15*"-")
print("Rand Index (against true cmro2 clusters):", ground_truth_score)
print("Silhouette score:", sil)
print("Davies-Bouldin Index:", dbi)
print("Calinski-Harabasz Score:", ch)


df_meta['cluster'] = labels

# ---------------------------
# Visualize some results
# ---------------------------
# 1) show a few example maps
grid_size = 20
maps_examples_indexes = np.random.randint(0, n_maps, size=15)
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
axes = axes.flatten()
for i0, i in enumerate(maps_examples_indexes):
    ax = axes[i0]
    ax.imshow(maps[i].reshape(grid_size, grid_size), cmap='viridis', origin='lower')
    ax.set_title(f"idx={i}, c={labels[i]}, cmro2={df_meta.loc[i,'cmro2']}")
    ax.axis('off')
plt.suptitle("15 synthetic maps with cluster labels")
plt.tight_layout()
plt.show()

# 2) visualize cluster membership on index axis
plt.figure(figsize=(10, 4))
plt.scatter(np.arange(n_maps), labels, c=labels, cmap='tab10')
plt.xlabel("Map index")
plt.ylabel("Cluster label")
plt.title("Cluster assignment per map")
plt.tight_layout()
plt.show()

# 3) mean (centroid) maps per cluster
fig, axes = plt.subplots(1, n_clusters, figsize=(4*n_clusters, 4))
for c in range(n_clusters):
    idxs = np.where(labels == c)[0]
    mean_map = maps[idxs].mean(axis=0)

    im = axes[c].imshow(mean_map.reshape(grid_size, grid_size), cmap='viridis', origin='lower')
    axes[c].set_title(f"Cluster {c} (n={len(idxs)})")
    axes[c].axis('off')

    # Add a colorbar for each subplot
    cbar = plt.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04)
    cbar.set_label('pO₂', rotation=270, labelpad=12)
plt.suptitle("Cluster mean maps", fontsize=14)
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, n_clusters, figsize=(4*n_clusters, 4))
var_images = []
for c in range(n_clusters):
    idxs = np.where(labels == c)[0]
    var_map = maps[idxs].var(axis=0)

    im = axes[c].imshow(var_map.reshape(grid_size, grid_size), cmap='viridis', origin='lower')
    var_images.append(im)
    axes[c].set_title(f"Cluster {c} Variance (n={len(idxs)})")
    axes[c].axis('off')

    # Add a colorbar for each subplot
    cbar = plt.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04)
    cbar.set_label('pO₂', rotation=270, labelpad=12)
plt.suptitle("Cluster variance maps", fontsize=14)
plt.tight_layout()
plt.show()

# 4) show a few examples per cluster
for c in range(n_clusters):
    idxs = np.where(labels == c)[0][:6]
    fig, axes = plt.subplots(1, len(idxs), figsize=(3*len(idxs), 3))
    for k, idx in enumerate(idxs):
        ax = axes[k]
        ax.imshow(maps[idx].reshape(grid_size, grid_size), cmap='viridis', origin='lower')
        ax.set_title(f"idx {idx}")
        ax.axis('off')
    plt.suptitle(f"Examples from cluster {c}")
    plt.tight_layout()
    plt.show()

# 5) PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

eigenvalues = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)
print("PCA Eigenvalues:", eigenvalues)
print("Explained Variance Ratio:", explained_variance_ratio)
print("Cumulative Variance Explained:", cumulative_variance)
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(eigenvalues) + 1), eigenvalues, color='skyblue')
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue")
plt.title("PCA Eigenvalues")
plt.xticks(range(1, len(eigenvalues) + 1))
plt.tight_layout()
plt.show()

# ------------------------------------
# PCA scatter plot colored by cluster
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA of Synthetic pO2 Maps Colored by Cluster")
plt.tight_layout()
plt.show()

# Cluster Stability (Bootsrapping)
def bootstrap_cluster_stability_kmeans(X, n_clusters=n_clusters, n_bootstraps=100):
    n= X.shape[0]
    base_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    base_labels = base_kmeans.fit_predict(X)

    ari_scores = []

    for _ in range(n_bootstraps):
        # 1. Sample indicess with replacement
        idx = np.random.choice(n, size=n, replace=True)
        X_resampled = X[idx]

        # 2. Fit KMeans on resampled data
        kmeans_resampled = KMeans(n_clusters=n_clusters, random_state=None, n_init=20)
        labels_resampled = kmeans_resampled.fit_predict(X_resampled)

        ari = adjusted_rand_score(base_labels[idx], labels_resampled)
        ari_scores.append(ari)

    return np.mean(ari_scores), np.std(ari_scores)
mean_ari, std_ari = bootstrap_cluster_stability_kmeans(X_scaled, n_clusters=n_clusters, n_bootstraps=100)
print(f"Bootstrap Cluster Stability ARI (Adjusted Rand Index): {mean_ari:.4f} +/- {std_ari:.4f}")


# ------------------------------------
# Hierarchical Clustering and Dendrogram
# ------------------------------------

# Compute linkage
# 'ward' method minimizes variance within clusters
linked = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data Point Index')
plt.ylabel('Distance')
plt.show()

# Perform clustering and assign labels (e.g., 3 clusters)
cluster = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hierarchical = cluster.fit_predict(X_scaled)

# ------------------------------------
# PCA scatter plot colored by cluster
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_hierarchical, cmap='tab10', alpha=0.7)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA of Synthetic pO2 Maps Colored by Cluster")
plt.tight_layout()
plt.show()

print("Running Hierarchical clustering...")
print(15*"-")
# Score (quality metric): How well-separated are the clusters
ground_truth_score = metrics.rand_score(labels_true, labels_hierarchical)  # within-c
sil = silhouette_score(X_scaled, labels_hierarchical) # silhouette score: How well-separated are the clusters (higher is better)
dbi = davies_bouldin_score(X_scaled, labels_hierarchical) # Davies-Bouldin Index: measures cluster separation (lower is better)
ch = calinski_harabasz_score(X_scaled, labels_hierarchical) # Calinski-Harabasz Score: measures cluster tightness and separation (higher values indicate better-defined clusters)
print("Rand Index (against true cmro2 clusters):", ground_truth_score)
print("Silhouette score:", sil)
print("Davies-Bouldin Index:", dbi)
print("Calinski-Harabasz Score:", ch)



# ------------------------------------
# Gaussian Mixture Model Clustering
# ------------------------------------

# Data
X = X_scaled 

# Fit GMM
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)

# Predict cluster labels
labels_gmm = gmm.predict(X)

probabilities = gmm.predict_proba(X)
print("Probabilities of belonging to each component:\n", probabilities)

# Plot GMM clustering results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_gmm, cmap='tab10', alpha=0.7)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA of Synthetic pO2 Maps Colored by GMM Clusters")
plt.tight_layout()
plt.show()

print("Running GMM clustering...")
print(15*"-")
# Score (quality metric): How well-separated are the clusters
ground_truth_score = metrics.rand_score(labels_true, labels_gmm)  # within-c
sil = silhouette_score(X_scaled, labels_gmm) # silhouette score: How well-separated are the clusters (higher is better)
dbi = davies_bouldin_score(X_scaled, labels_gmm) # Davies-Bouldin Index: measures cluster separation (lower is better)
ch = calinski_harabasz_score(X_scaled, labels_gmm) # Calinski-Harabasz Score: measures cluster tightness and separation (higher values indicate better-defined clusters)
print("Rand Index (against true cmro2 clusters):", ground_truth_score)
print("Silhouette score:", sil)
print("Davies-Bouldin Index:", dbi)
print("Calinski-Harabasz Score:", ch)

# Cluster Stability (Bootsrapping)
def bootstrap_cluster_stability_gmm(X, n_clusters=n_clusters, n_bootstraps=100):
    n = X.shape[0]
    base_gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=20)
    base_labels = base_gmm.fit_predict(X)

    ari_scores = []

    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        X_resampled = X[idx]
        gmm_resampled = GaussianMixture(n_components=n_clusters, random_state=42, n_init=20)
        resampled_labels = gmm_resampled.fit_predict(X_resampled)

        ari = adjusted_rand_score(base_labels[idx], resampled_labels)
        ari_scores.append(ari)

    return np.mean(ari_scores), np.std(ari_scores)
# mean_ari, std_ari = bootstrap_cluster_stability_gmm(X, n_clusters=n_clusters, n_bootstraps=100)
# print(f"Bootstrap Cluster Stability ARI (Adjusted Rand Index): {mean_ari:.4f} +/- {std_ari:.4f}")

def SelBest(arr:list, X:int)->list:
    '''
    returns the set of X configurations with shorter distance
    '''
    dx=np.argsort(arr)[:X]
    return arr[dx]

n_clusters=np.arange(2, 8)
sils=[]
sils_err=[]
iterations=50
for n in n_clusters:
    tmp_sil=[]
    for _ in range(iterations):
        gmm=GaussianMixture(n, n_init=2).fit(X_pca)
        labels=gmm.predict(X_pca)
        sil=metrics.silhouette_score(X_pca, labels, metric='euclidean')
        tmp_sil.append(sil)
    val=np.mean(SelBest(np.array(tmp_sil), int(iterations/5)))
    err=np.std(tmp_sil)
    sils.append(val)
    sils_err.append(err)

plt.errorbar(n_clusters, sils, yerr=sils_err)
plt.title("Silhouette Scores for GMM", fontsize=20)
plt.xticks(n_clusters)
plt.xlabel("N. of clusters")
plt.ylabel("Score")

# # 5) Save clustered dataset if desired
# np.savez(py_data_location + "synthetic_po2_clustered_" + filename + ".npz", maps=maps, labels=labels_hierarchical, meta=df_meta.to_dict(orient='list'))
