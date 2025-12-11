import numpy as np
import matplotlib.pyplot as plt

class EnKFClusterRunner:
    def __init__(self, data_path, EnKFClass):
        """
        Parameters
        ----------
        data_path : str
            Path to your NPZ file containing maps, labels, and meta.
        EnKFClass : class
            The EnKF class to instantiate (e.g. EnKF from EnKF_FEM_3).
        """
        self.data = np.load(data_path, allow_pickle=True)
        self.maps = self.data["maps"]          # shape (N, H, W)
        self.labels = self.data["labels"]      # shape (N,)
        self.meta = self.data["meta"].item()   # dict of metadata
        self.EnKFClass = EnKFClass

        self.N, self.gridxgrid = self.maps.shape
        self.H, self.W = self.gridxgrid // 20, self.gridxgrid // 20

    # ------------------------------------------------------------------
    # Helper: extract maps in a cluster
    # ------------------------------------------------------------------
    def get_cluster(self, cluster_id):
        idxs = np.where(self.labels == cluster_id)[0]
        maps = self.maps[idxs]
        meta = {}
        for k, v in self.meta.items():
            v = list(v)  # ensure list indexing works

            try:
                # Try numpy indexing first
                arr = np.array(v)
                meta[k] = arr[idxs]
            except:
                # If numpy can't handle it, fall back to Python list indexing
                meta[k] = [v[i] for i in idxs]
        return maps, meta, idxs

    # ------------------------------------------------------------------
    # Strategy A: run EnKF for each map separately
    # ------------------------------------------------------------------
    def run_individual(self, cluster_id, X, Y, enkf_config, prior_bounds):
        """
        Run EnKF individually for each map.

        Parameters
        ----------
        cluster_id : int
        X, Y : 2D arrays
            Observation grid (your FEM grid used in observation_operator)
        enkf_config : dict
            kwargs for your EnKF class (state_dim, obs_dim, n_ensembles...)
        prior_bounds : tuple (a, b)
            Lower and upper bounds for uniform ensemble init
        """
        maps, meta, idxs = self.get_cluster(cluster_id)

        posteriors = []
        estimates = []
        estimates_cov = []

        print(f"Running INDIVIDUAL EnKF on cluster {cluster_id}...")
        print(f"Cluster size: {len(maps)}")

        a, b = prior_bounds
        
        # Extract constructor-only arguments
        enkf_ctor = {
                k: enkf_config[k] 
                for k in ["state_dim","obs_dim","n_ensembles","dynamics_model"]
            }
            
        R = enkf_config["R"]
        B = enkf_config["B"]

        for i, obs_map in enumerate(maps):
            print(f"[Cluster {cluster_id}] Map {i+1}/{len(maps)}")

            y = obs_map.reshape(-1)

            # -----------------------------
            # Initialize ensemble
            
            # Create EnKF
            enkf = self.EnKFClass(**enkf_ctor)

            # Set noise matrices AFTER construction
            enkf.set_observation_noise(R)
            enkf.set_process_noise(B)

            enkf.initialize_ensemble(a, b)

            # Update with the actual map
            enkf.predict()
            enkf.update(y, X, Y)

            # Store results
            mean, cov = enkf.get_state_estimate()

            posteriors.append(enkf.get_ensemble().copy())
            estimates.append(mean.copy())
            estimates_cov.append(cov.copy())

        return {
            "cluster_id": cluster_id,
            "indices": idxs,
            "posteriors": posteriors,   # ensemble for each map
            "estimates": np.array(estimates),
            "estimates_cov": np.array(estimates_cov),
            "meta": meta
        }

    # ------------------------------------------------------------------
    # Strategy B: run EnKF on the cluster mean
    # ------------------------------------------------------------------
    def run_on_centroid(self, cluster_id, X, Y, enkf_config, prior_bounds):
        maps, meta, idxs = self.get_cluster(cluster_id)
        mean_map = maps.mean(axis=0)
        y = mean_map.reshape(-1)

        a, b = prior_bounds

        # Extract constructor-only arguments
        enkf_ctor = {
                k: enkf_config[k] 
                for k in ["state_dim","obs_dim","n_ensembles","dynamics_model"]
            }

        R = enkf_config["R"]
        B = enkf_config["B"]

        enkf = self.EnKFClass(**enkf_ctor)

        # Initialize ensemble
        enkf.initialize_ensemble(a, b)
        enkf.set_observation_noise(R)
        enkf.set_process_noise(B)

        # Run EnKF on the MEAN
        enkf.predict()
        enkf.update(y, X, Y)

        mean, cov = enkf.get_state_estimate()

        return {
            "cluster_id": cluster_id,
            "indices": idxs,
            "posterior": enkf.get_ensemble(),
            "estimate": mean,
            "estimate_cov": cov,
            "meta": meta,
            "mean_map": mean_map
        }

    # ------------------------------------------------------------------
    # Strategy C: joint EnKF for all maps (advanced)
    # ------------------------------------------------------------------
    def run_joint(self, cluster_id, enkf_config):
        """
        Stack all maps in the cluster and update ensemble using multiple observations.

        Requires your EnKF class to support multi-observation updates.
        """
        maps, meta, idxs = self.get_cluster(cluster_id)
        Y = maps.reshape(len(maps), -1)   # shape (Nc, H*W)

        enkf = self.EnKFClass(**enkf_config)
        posterior = enkf.run(Y)           # requires your EnKF to accept 2D obs

        return {
            "cluster_id": cluster_id,
            "indices": idxs,
            "posterior": posterior,
            "meta": meta,
            "Y": Y
        }

    # ------------------------------------------------------------------
    # Utility: plot cluster centroid
    # ------------------------------------------------------------------
    def plot_cluster_centroid(self, cluster_id):
        maps, _, _ = self.get_cluster(cluster_id)
        mean_map = maps.mean(axis=0)

        plt.figure(figsize=(5,5))
        plt.imshow(mean_map, cmap="viridis", origin="lower")
        plt.colorbar(label="pO₂")
        plt.title(f"Cluster {cluster_id} Centroid Map")
        plt.show()

    # ------------------------------------------------------------------
    # Utility: plot cluster members
    # ------------------------------------------------------------------
    def plot_cluster_examples(self, cluster_id, n=6):
        maps, _, _ = self.get_cluster(cluster_id)
        n = min(n, len(maps))
        plt.figure(figsize=(3*n, 3))
        for i in range(n):
            plt.subplot(1, n, i+1)
            plt.imshow(maps[i], cmap="viridis", origin="lower")
            plt.axis("off")
        plt.suptitle(f"Cluster {cluster_id} Examples")
        plt.show()
    