"""
Biological null models for SMLM cluster analysis.

The standard cosmic null model (Poisson process in a rectangular box) is
inappropriate for cells. Biological membranes and organelles have irregular
geometry, and placing random points in a box would include extracellular
space and misrepresent the available volume.

The biologically correct null is Complete Spatial Randomness (CSR) constrained
to the cell's actual geometry: points drawn uniformly at random within the
convex hull (or alpha-shape) of the observed localization cloud.

Two null models are provided:

1. ConvexHullNull  -- CSR within the convex hull of the input data.
   Appropriate when the cell boundary is well-sampled by localizations.

2. BoundingBoxNull -- CSR within an axis-aligned bounding box.
   Useful for control validation and for near-rectangular ROIs.

Both models match the number of points in the observed data, so the null
distributions are directly comparable.
"""

import numpy as np
from scipy.spatial import ConvexHull, Delaunay


class BoundingBoxNull:
    """
    CSR within the axis-aligned bounding box of the data.

    The simplest null — useful for validation and debugging because the
    expected statistics are analytically known (Poisson process in a box).
    """

    def __init__(self, seed: int = None):
        self.rng = np.random.default_rng(seed)
        self.bounds = None

    def fit(self, positions: np.ndarray) -> 'BoundingBoxNull':
        """Compute bounding box from observed positions."""
        positions = np.asarray(positions)
        self.bounds = np.stack([positions.min(axis=0), positions.max(axis=0)])
        return self

    def sample(self, n: int) -> np.ndarray:
        """Draw n points uniformly from the bounding box."""
        if self.bounds is None:
            raise RuntimeError("Call fit() before sample()")
        lo, hi = self.bounds
        return self.rng.uniform(lo, hi, size=(n, 3))

    def generate_mocks(self, n_points: int, n_mocks: int) -> list:
        """Generate multiple mock datasets."""
        if self.bounds is None:
            raise RuntimeError("Call fit() before generate_mocks()")
        return [self.sample(n_points) for _ in range(n_mocks)]


class BiologicalNullModel:
    """
    CSR within the convex hull of the observed localizations.

    Uses rejection sampling: draw candidate points from the bounding box,
    keep only those inside the convex hull. The hull is computed once from
    the data and reused for all mock datasets.

    For highly non-convex cells (e.g., neurons with long processes), the
    convex hull overestimates the available volume, making the null slightly
    conservative. For roughly convex ROIs (e.g., a single spine or small
    organelle), this is exact.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility.
    batch_size : int
        Number of candidates to draw per rejection-sampling batch.
        Larger values are faster but use more memory.
    """

    def __init__(self, seed: int = None, batch_size: int = 10000):
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.hull = None
        self.delaunay = None
        self.bounds = None
        self._efficiency = None  # hull volume / box volume

    def fit(self, positions: np.ndarray) -> 'BiologicalNullModel':
        """
        Compute convex hull from observed positions.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
            Observed localizations in nm.
        """
        positions = np.asarray(positions, dtype=float)

        if len(positions) < 4:
            raise ValueError("Need at least 4 points to define a 3D convex hull")

        self.hull = ConvexHull(positions)
        self.delaunay = Delaunay(positions[self.hull.vertices])
        self.bounds = np.stack([positions.min(axis=0), positions.max(axis=0)])

        # Estimate acceptance fraction for efficiency
        box_vol = np.prod(self.bounds[1] - self.bounds[0])
        hull_vol = self.hull.volume
        self._efficiency = hull_vol / box_vol if box_vol > 0 else 1.0

        return self

    def _inside_hull(self, candidates: np.ndarray) -> np.ndarray:
        """
        Test which candidate points are inside the convex hull.

        Uses the Delaunay triangulation of hull vertices, which correctly
        handles the full interior including faces (not just vertices).
        """
        return self.delaunay.find_simplex(candidates) >= 0

    def sample(self, n: int, max_attempts: int = 50) -> np.ndarray:
        """
        Draw n points uniformly from the convex hull interior.

        Parameters
        ----------
        n : int
            Number of points to generate.
        max_attempts : int
            Maximum rejection-sampling rounds before raising an error.
            At typical hull efficiencies (> 20%) this should never trigger.

        Returns
        -------
        ndarray, shape (n, 3)
        """
        if self.hull is None:
            raise RuntimeError("Call fit() before sample()")

        lo, hi = self.bounds
        accepted = []
        attempts = 0
        needed = n

        while needed > 0 and attempts < max_attempts:
            # Oversample by ~1/efficiency to hit needed count quickly
            n_draw = max(int(needed / max(self._efficiency, 0.01)) + 100,
                         self.batch_size)
            candidates = self.rng.uniform(lo, hi, size=(n_draw, 3))
            inside = self._inside_hull(candidates)
            accepted.append(candidates[inside])
            needed -= inside.sum()
            attempts += 1

        if needed > 0:
            raise RuntimeError(
                f"Rejection sampling failed after {max_attempts} attempts. "
                f"Hull efficiency: {self._efficiency:.3f}. "
                "The convex hull may be degenerate."
            )

        result = np.vstack(accepted)[:n]
        return result

    def generate_mocks(self, n_points: int, n_mocks: int) -> list:
        """
        Generate multiple mock CSR datasets.

        Parameters
        ----------
        n_points : int
            Number of localizations per mock (should match real data).
        n_mocks : int
            Number of mock datasets.

        Returns
        -------
        list of ndarray, each shape (n_points, 3)
        """
        if self.hull is None:
            raise RuntimeError("Call fit() before generate_mocks()")
        return [self.sample(n_points) for _ in range(n_mocks)]

    @property
    def hull_volume(self) -> float:
        """Convex hull volume in nm^3."""
        if self.hull is None:
            return 0.0
        return self.hull.volume

    @property
    def acceptance_rate(self) -> float:
        """Fraction of rejection-sampled candidates that fall inside the hull."""
        return self._efficiency if self._efficiency is not None else 0.0
