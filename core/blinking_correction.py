"""
Blinking correction for SMLM localizations.

Fluorescent molecules in STORM/PALM cycle between bright and dark states.
A single molecule may generate dozens of separate localizations over time,
each appearing within a few nanometers of each other across non-consecutive
frames (the dark period). Without correction, one molecule gets counted
many times, inflating apparent cluster densities.

Algorithm: connected-component merging over a spatio-temporal graph.
- Two localizations are connected if they are within (r_lateral, r_axial)
  and separated by at most max_dark_frames frames.
- Connected components correspond to the same physical molecule.
- Each component is collapsed to a photon-weighted centroid.

The anisotropy parameter accounts for the fact that axial (z) localization
precision in astigmatic 3D-STORM is typically 2-3x worse than lateral.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BlinkerResult:
    """Output from blinking correction."""
    positions: np.ndarray        # (M, 3) merged positions in nm [x, y, z]
    photons: np.ndarray          # (M,) total photon count per merged event
    n_blinks: np.ndarray         # (M,) number of raw localizations merged
    original_indices: list       # list of arrays: original indices per cluster
    n_raw: int                   # total input localizations
    n_merged: int                # total output localizations

    @property
    def reduction_factor(self):
        return self.n_raw / self.n_merged if self.n_merged > 0 else 1.0


class UnionFind:
    """
    Union-Find (disjoint-set) data structure for connected-component merging.
    Path compression + union by rank for near-linear performance.
    """

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def components(self, n):
        """Return dict mapping root -> list of members."""
        groups = {}
        for i in range(n):
            root = self.find(i)
            groups.setdefault(root, []).append(i)
        return groups


class BlinkingCorrector:
    """
    Merge repeated localizations from the same fluorophore.

    Parameters
    ----------
    r_lateral : float
        Maximum lateral (xy) distance in nm to connect two localizations.
        Typical value: 30 nm (roughly 1-2x localization precision).
    r_axial : float
        Maximum axial (z) distance in nm to connect two localizations.
        Typical value: 60 nm (axial precision is ~2x worse than lateral).
    max_dark_frames : int
        Maximum number of consecutive dark frames allowed between two
        blinks from the same molecule. Typical value: 5 frames.
        Set to 1 to require strict consecutive-frame adjacency.

    Notes
    -----
    The spatial thresholds should be set conservatively (larger than the
    expected localization precision). Being too tight risks splitting one
    molecule into multiple events; being too loose merges distinct molecules.
    At typical STORM densities (< 1 molecule per 100x100 nm^2), r_lateral=30
    nm gives a < 1% false-merge rate.
    """

    def __init__(self, r_lateral: float = 30.0, r_axial: float = 60.0,
                 max_dark_frames: int = 5):
        if r_lateral <= 0 or r_axial <= 0:
            raise ValueError("Spatial radii must be positive")
        if max_dark_frames < 1:
            raise ValueError("max_dark_frames must be >= 1")

        self.r_lateral = r_lateral
        self.r_axial = r_axial
        self.max_dark_frames = max_dark_frames

    def correct(self, positions: np.ndarray, frames: np.ndarray,
                photons: Optional[np.ndarray] = None) -> BlinkerResult:
        """
        Merge localizations from the same fluorophore.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Localization positions in nm [x, y, z].
        frames : ndarray, shape (N,)
            Frame number for each localization (integer, 1-indexed or 0-indexed).
        photons : ndarray, shape (N,) or None
            Photon count per localization. Used for weighted centroid.
            If None, equal weighting is used.

        Returns
        -------
        BlinkerResult
            Merged localizations with metadata.
        """
        positions = np.asarray(positions, dtype=float)
        frames = np.asarray(frames, dtype=int)

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must be shape (N, 3)")
        if frames.shape[0] != positions.shape[0]:
            raise ValueError("frames and positions must have same length")

        n = len(positions)

        if photons is None:
            photons = np.ones(n, dtype=float)
        else:
            photons = np.asarray(photons, dtype=float)
            if photons.shape[0] != n:
                raise ValueError("photons and positions must have same length")

        # Sort by frame for efficient temporal search
        sort_order = np.argsort(frames, kind='stable')
        positions_s = positions[sort_order]
        frames_s = frames[sort_order]
        photons_s = photons[sort_order]

        uf = UnionFind(n)
        self._build_graph(positions_s, frames_s, uf)

        components = uf.components(n)
        merged_pos, merged_ph, merged_blinks, orig_idx = self._merge_components(
            components, positions_s, photons_s, sort_order
        )

        return BlinkerResult(
            positions=merged_pos,
            photons=merged_ph,
            n_blinks=merged_blinks,
            original_indices=orig_idx,
            n_raw=n,
            n_merged=len(merged_pos)
        )

    def _build_graph(self, positions_s: np.ndarray, frames_s: np.ndarray,
                     uf: UnionFind) -> None:
        """
        Connect localizations within the spatio-temporal threshold.

        For each localization i, look ahead to localizations j where
        frames_s[j] - frames_s[i] <= max_dark_frames, and connect those
        within the distance threshold.

        The anisotropic distance check is:
            (dx/r_lateral)^2 + (dy/r_lateral)^2 + (dz/r_axial)^2 <= 1
        """
        n = len(positions_s)
        # Normalize z by anisotropy ratio so we can use a single radius=1
        scale = np.array([1.0 / self.r_lateral,
                          1.0 / self.r_lateral,
                          1.0 / self.r_axial])
        scaled = positions_s * scale  # (N, 3)

        for i in range(n):
            frame_i = frames_s[i]
            # Look at subsequent localizations within the dark-frame window
            j = i + 1
            while j < n and frames_s[j] - frame_i <= self.max_dark_frames:
                diff = scaled[j] - scaled[i]
                dist_sq = diff[0]**2 + diff[1]**2 + diff[2]**2
                if dist_sq <= 1.0:
                    uf.union(i, j)
                j += 1

    def _merge_components(self, components: dict, positions_s: np.ndarray,
                          photons_s: np.ndarray,
                          sort_order: np.ndarray) -> tuple:
        """
        Collapse each connected component to a photon-weighted centroid.
        """
        merged_pos = []
        merged_ph = []
        merged_blinks = []
        orig_idx = []

        for members in components.values():
            idx = np.array(members)
            pos = positions_s[idx]
            ph = photons_s[idx]

            total_photons = ph.sum()
            if total_photons > 0:
                centroid = (pos * ph[:, None]).sum(axis=0) / total_photons
            else:
                centroid = pos.mean(axis=0)

            merged_pos.append(centroid)
            merged_ph.append(total_photons)
            merged_blinks.append(len(members))
            orig_idx.append(sort_order[idx])

        return (
            np.array(merged_pos),
            np.array(merged_ph),
            np.array(merged_blinks, dtype=int),
            orig_idx
        )
