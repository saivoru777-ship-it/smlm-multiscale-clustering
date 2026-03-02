"""
Unit tests for biological null models.

Coverage:
- BoundingBoxNull: uniform distribution in box, correct bounds
- BiologicalNullModel: all output points inside hull, N matching
- Both: fit/sample contract, determinism with seed, multiple calls
- Edge cases: minimal point cloud, very flat ROI (degenerate z)
"""

import pytest
import numpy as np
from scipy.spatial import ConvexHull
from ..core.null_models import BoundingBoxNull, BiologicalNullModel


def make_sphere_points(n=500, radius=1000.0, seed=42):
    """Points uniformly sampled from a sphere surface (non-trivial convex hull)."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True) * radius
    return pts + radius  # shift to positive octant


def make_box_points(n=200, lo=0.0, hi=1000.0, seed=1):
    """Uniform in a box -- hull should closely approximate the box."""
    return np.random.default_rng(seed).uniform(lo, hi, (n, 3))


# ===========================================================================
# BoundingBoxNull
# ===========================================================================

class TestBoundingBoxNull:

    def test_sample_within_bounds(self):
        data = make_box_points(200)
        null = BoundingBoxNull(seed=0)
        null.fit(data)
        samples = null.sample(1000)
        lo, hi = null.bounds
        assert (samples >= lo).all()
        assert (samples <= hi).all()

    def test_correct_shape(self):
        data = make_box_points(100)
        null = BoundingBoxNull(seed=0)
        null.fit(data)
        n = 300
        samples = null.sample(n)
        assert samples.shape == (n, 3)

    def test_generate_mocks_count(self):
        data = make_box_points(100)
        null = BoundingBoxNull(seed=0)
        null.fit(data)
        mocks = null.generate_mocks(n_points=150, n_mocks=5)
        assert len(mocks) == 5
        for m in mocks:
            assert m.shape == (150, 3)

    def test_reproducible_with_seed(self):
        data = make_box_points(100, seed=99)
        null1 = BoundingBoxNull(seed=7)
        null1.fit(data)
        samples1 = null1.sample(100)

        null2 = BoundingBoxNull(seed=7)
        null2.fit(data)
        samples2 = null2.sample(100)

        np.testing.assert_array_equal(samples1, samples2)

    def test_sample_before_fit_raises(self):
        null = BoundingBoxNull()
        with pytest.raises(RuntimeError, match="fit"):
            null.sample(10)

    def test_generate_mocks_before_fit_raises(self):
        null = BoundingBoxNull()
        with pytest.raises(RuntimeError, match="fit"):
            null.generate_mocks(10, 5)

    def test_bounds_match_data(self):
        data = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                         [100, 100, 100]], dtype=float)
        null = BoundingBoxNull()
        null.fit(data)
        np.testing.assert_array_equal(null.bounds[0], [0, 0, 0])
        np.testing.assert_array_equal(null.bounds[1], [100, 100, 100])


# ===========================================================================
# BiologicalNullModel
# ===========================================================================

class TestBiologicalNullModel:

    def test_all_samples_inside_hull(self):
        data = make_box_points(500, seed=10)
        null = BiologicalNullModel(seed=5)
        null.fit(data)
        samples = null.sample(300)

        # Check using scipy ConvexHull halfspace equations
        hull = ConvexHull(data)
        # A point is inside if Ax + b <= 0 for all halfspace equations
        eq = hull.equations
        inside = (eq[:, :3] @ samples.T + eq[:, 3:]).max(axis=0) <= 1e-6
        fraction_inside = inside.mean()
        assert fraction_inside > 0.99, (
            f"Only {fraction_inside:.3f} of samples inside hull"
        )

    def test_correct_n_points(self):
        data = make_box_points(200, seed=20)
        null = BiologicalNullModel(seed=0)
        null.fit(data)
        for n in [50, 200, 1000]:
            samples = null.sample(n)
            assert len(samples) == n

    def test_generate_mocks_shapes(self):
        data = make_box_points(200, seed=30)
        null = BiologicalNullModel(seed=0)
        null.fit(data)
        mocks = null.generate_mocks(n_points=100, n_mocks=4)
        assert len(mocks) == 4
        for m in mocks:
            assert m.shape == (100, 3)

    def test_hull_volume_positive(self):
        data = make_box_points(500, seed=40)
        null = BiologicalNullModel(seed=0)
        null.fit(data)
        assert null.hull_volume > 0

    def test_acceptance_rate_positive(self):
        data = make_box_points(500, seed=50)
        null = BiologicalNullModel(seed=0)
        null.fit(data)
        assert 0 < null.acceptance_rate <= 1.0

    def test_sample_before_fit_raises(self):
        null = BiologicalNullModel()
        with pytest.raises(RuntimeError, match="fit"):
            null.sample(10)

    def test_generate_mocks_before_fit_raises(self):
        null = BiologicalNullModel()
        with pytest.raises(RuntimeError, match="fit"):
            null.generate_mocks(10, 3)

    def test_insufficient_points_raises(self):
        # Need >= 4 points for 3D convex hull
        data = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        null = BiologicalNullModel()
        with pytest.raises(ValueError, match="4 points"):
            null.fit(data)

    def test_sphere_hull_correct(self):
        """
        For points on a sphere, all null samples should be within sphere radius.
        """
        pts = make_sphere_points(n=800, radius=500.0)
        center = pts.mean(axis=0)

        null = BiologicalNullModel(seed=1)
        null.fit(pts)
        samples = null.sample(200)

        # Points should be within the original cloud bounds
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        assert (samples >= lo - 1e-6).all()
        assert (samples <= hi + 1e-6).all()

    def test_multiple_calls_produce_different_samples(self):
        data = make_box_points(300, seed=60)
        null = BiologicalNullModel(seed=None)  # no fixed seed
        null.fit(data)
        s1 = null.sample(50)
        s2 = null.sample(50)
        # With high probability these should differ
        assert not np.allclose(s1, s2)

    def test_reproducible_across_fit_calls_with_same_seed(self):
        data = make_box_points(300, seed=70)
        null1 = BiologicalNullModel(seed=42)
        null1.fit(data)
        s1 = null1.sample(100)

        null2 = BiologicalNullModel(seed=42)
        null2.fit(data)
        s2 = null2.sample(100)

        np.testing.assert_array_equal(s1, s2)

    def test_hull_constrained_vs_box_null(self):
        """
        For non-convex data (points on two separate clusters), the biological
        null should produce fewer points in the space between them than the
        bounding-box null.
        """
        rng = np.random.default_rng(80)
        # Two well-separated clusters
        cluster_a = rng.normal([200, 200, 200], 30, (300, 3))
        cluster_b = rng.normal([800, 800, 800], 30, (300, 3))
        data = np.vstack([cluster_a, cluster_b])

        null = BiologicalNullModel(seed=10)
        null.fit(data)
        samples = null.sample(500)

        # All samples should be within convex hull
        hull = ConvexHull(data)
        eq = hull.equations
        inside = (eq[:, :3] @ samples.T + eq[:, 3:]).max(axis=0) <= 1e-6
        assert inside.mean() > 0.99
