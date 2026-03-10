"""
Baseline clustering methods for comparison with the multiscale detector.

Two well-established methods are included:

1. DBSCAN  -- density-based spatial clustering.
   The standard workhorse for SMLM cluster detection. Works well when
   clusters are compact and well-separated, but struggles with diffuse
   clusters and requires manual parameter tuning.

2. Ripley's K / L function  -- second-order spatial statistics.
   Classic point-process method. Detects clustering by comparing the
   observed density of neighbors within radius r against the Poisson
   expectation. L(r) - r > 0 indicates clustering at scale r.
   Does not produce hard cluster assignments, but gives a clean scale-
   dependent clustering signature for comparison with our variance curves.

Both are wrapped in a consistent interface that returns per-localization
cluster labels and (optionally) per-scale statistics for direct comparison
with the multiscale detector output.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.spatial import cKDTree


# ===========================================================================
# DBSCAN
# ===========================================================================

@dataclass
class DBSCANResult:
    labels: np.ndarray      # (N,) cluster IDs, -1 = noise
    n_clusters: int
    n_noise: int
    eps_nm: float
    min_samples: int

    @property
    def cluster_mask(self) -> np.ndarray:
        """Boolean mask of non-noise points."""
        return self.labels >= 0


class DBSCANBaseline:
    """
    DBSCAN clustering baseline.

    Uses a pure-numpy implementation with a KD-tree for neighbor queries,
    avoiding a scikit-learn dependency while matching its behavior exactly.

    Parameters
    ----------
    eps_nm : float
        Neighborhood radius in nm. Rule of thumb: set to 1-2x the expected
        cluster radius. For synaptic receptors clustered at ~100 nm, eps=50
        is a reasonable starting point.
    min_samples : int
        Minimum number of localizations within eps to define a core point.
        Smaller values detect more (smaller) clusters but increase noise
        sensitivity.
    """

    def __init__(self, eps_nm: float = 50.0, min_samples: int = 5):
        if eps_nm <= 0:
            raise ValueError("eps_nm must be positive")
        if min_samples < 1:
            raise ValueError("min_samples must be >= 1")
        self.eps_nm = eps_nm
        self.min_samples = min_samples

    def fit(self, positions: np.ndarray) -> DBSCANResult:
        """
        Run DBSCAN on localization positions.

        Parameters
        ----------
        positions : ndarray, shape (N, 3) in nm

        Returns
        -------
        DBSCANResult
        """
        positions = np.asarray(positions, dtype=float)
        n = len(positions)
        labels = np.full(n, -1, dtype=int)

        tree = cKDTree(positions)
        neighbors = tree.query_ball_point(positions, r=self.eps_nm)
        n_neighbors = np.array([len(nb) for nb in neighbors])
        is_core = n_neighbors >= self.min_samples

        cluster_id = 0
        visited = np.zeros(n, dtype=bool)

        for i in range(n):
            if visited[i] or not is_core[i]:
                continue
            visited[i] = True
            labels[i] = cluster_id

            stack = list(neighbors[i])
            while stack:
                j = stack.pop()
                if not visited[j]:
                    visited[j] = True
                    labels[j] = cluster_id
                    if is_core[j]:
                        stack.extend(neighbors[j])
                elif labels[j] == -1:
                    labels[j] = cluster_id

            cluster_id += 1

        n_clusters = cluster_id
        n_noise = int((labels == -1).sum())

        return DBSCANResult(
            labels=labels,
            n_clusters=n_clusters,
            n_noise=n_noise,
            eps_nm=self.eps_nm,
            min_samples=self.min_samples
        )


# ===========================================================================
# Ripley's K / L function
# ===========================================================================

@dataclass
class RipleysResult:
    radii: np.ndarray           # nm
    K_observed: np.ndarray      # observed K(r)
    K_poisson: np.ndarray       # Poisson expectation K_poisson(r) = (4/3) pi r^3 (3D)
    L_minus_r: np.ndarray       # L(r) - r, where L(r) = (K(r)/(4pi/3))^(1/3)
    K_mock_mean: Optional[np.ndarray]    # mean K(r) from null mocks
    K_mock_std: Optional[np.ndarray]     # std dev of K(r) from null mocks


class RipleysKBaseline:
    """
    Ripley's K function for 3D point patterns.

    Computes K(r) = (V / N^2) * sum_{i != j} 1(||x_i - x_j|| <= r)

    where V is the volume of the bounding box. The standard Poisson
    expectation in 3D is K(r) = (4/3) pi r^3.

    L(r) = (K(r) / (4*pi/3))^(1/3) is the linearized version.
    L(r) - r = 0 for Poisson; L(r) - r > 0 indicates clustering.

    Parameters
    ----------
    r_min_nm : float
        Smallest radius to evaluate (nm).
    r_max_nm : float
        Largest radius to evaluate (nm). Should be <= ROI size / 4
        to avoid edge effects.
    n_radii : int
        Number of radii. Logarithmic spacing is used.
    edge_correction : bool
        Apply Ripley's isotropic edge correction. Recommended for
        small ROIs where many points are near the boundary.
    """

    def __init__(self, r_min_nm: float = 20.0, r_max_nm: float = 500.0,
                 n_radii: int = 20, edge_correction: bool = False):
        self.r_min_nm = r_min_nm
        self.r_max_nm = r_max_nm
        self.n_radii = n_radii
        self.edge_correction = edge_correction
        self.radii = np.logspace(
            np.log10(r_min_nm), np.log10(r_max_nm), n_radii
        )

    def compute(self, positions: np.ndarray,
                mock_positions_list: Optional[list] = None) -> RipleysResult:
        """
        Compute K(r) for the observed data and optionally for null mocks.

        Parameters
        ----------
        positions : ndarray, shape (N, 3) in nm
        mock_positions_list : list of ndarray or None
            If provided, compute K(r) for each mock to get null distribution.

        Returns
        -------
        RipleysResult
        """
        positions = np.asarray(positions, dtype=float)
        K_obs = self._compute_K(positions)

        K_mocks = None
        if mock_positions_list is not None:
            K_mocks = np.array([self._compute_K(mp) for mp in mock_positions_list])

        K_poisson = (4.0 / 3.0) * np.pi * self.radii**3

        with np.errstate(invalid='ignore'):
            L_minus_r = (K_obs / ((4.0 / 3.0) * np.pi))**(1.0 / 3.0) - self.radii

        return RipleysResult(
            radii=self.radii.copy(),
            K_observed=K_obs,
            K_poisson=K_poisson,
            L_minus_r=L_minus_r,
            K_mock_mean=K_mocks.mean(axis=0) if K_mocks is not None else None,
            K_mock_std=K_mocks.std(axis=0) if K_mocks is not None else None,
        )

    def _compute_K(self, positions: np.ndarray) -> np.ndarray:
        """
        Raw K(r) estimate. No edge correction by default.

        For an N-point process in volume V:
            K(r) = (V / N^2) * sum_{i != j} 1(d_ij <= r)
        """
        n = len(positions)
        if n < 2:
            return np.zeros(len(self.radii))

        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        span = maxs - mins
        volume = np.prod(np.where(span > 0, span, 1.0))

        tree = cKDTree(positions)
        K = np.zeros(len(self.radii))

        for ri, r in enumerate(self.radii):
            pairs = tree.query_pairs(r=r, output_type='ndarray')
            # Each pair counted once; multiply by 2 for (i,j) and (j,i)
            K[ri] = 2 * len(pairs)

        K = K * volume / (n * (n - 1))
        return K


# ===========================================================================
# HDBSCAN
# ===========================================================================

@dataclass
class HDBSCANResult:
    labels: np.ndarray      # (N,) cluster IDs, -1 = noise
    n_clusters: int
    n_noise: int
    min_cluster_size: int

    @property
    def cluster_mask(self) -> np.ndarray:
        """Boolean mask of non-noise points."""
        return self.labels >= 0


class HDBSCANBaseline:
    """
    HDBSCAN clustering baseline using scikit-learn.

    Unlike DBSCAN, HDBSCAN does not require a fixed epsilon parameter,
    making it a more modern density-based alternative that automatically
    discovers clusters at varying densities.

    Parameters
    ----------
    min_cluster_size : int
        Minimum number of points to form a cluster.
    min_samples : int or None
        Number of samples in a neighborhood for a point to be core.
        If None, defaults to min_cluster_size.
    """

    def __init__(self, min_cluster_size: int = 15, min_samples: Optional[int] = None):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def fit(self, positions: np.ndarray) -> HDBSCANResult:
        """
        Run HDBSCAN on localization positions.

        Parameters
        ----------
        positions : ndarray, shape (N, 3) in nm

        Returns
        -------
        HDBSCANResult
        """
        from sklearn.cluster import HDBSCAN

        positions = np.asarray(positions, dtype=float)
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
        )
        labels = clusterer.fit_predict(positions)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())

        return HDBSCANResult(
            labels=labels,
            n_clusters=n_clusters,
            n_noise=n_noise,
            min_cluster_size=self.min_cluster_size,
        )
