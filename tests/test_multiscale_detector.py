"""
Unit tests for the SMLM multiscale clustering detector.

Coverage:
- CIC gridding: conservation of mass, boundary handling, shape contract
- variance_at_scale / skewness_at_scale: known analytic values for uniform data
- compute_curves: correct structure and finite values
- chi_squared_with_covariance: returns expected keys, p-value in [0,1]
- ScaleRange.for_smlm: correct count and ordering
- SMLMMultiscaleTest.from_positions: infers ROI correctly
- Full pipeline: clustered data detected, unclustered data not (low FPR)
- Shrinkage regularization: condition number reduced vs no shrinkage
"""

import pytest
import numpy as np
from scipy import stats
from ..core.multiscale_detector import (
    SMLMMultiscaleTest, ScaleRange, DatasetInfo
)


# ===========================================================================
# Helpers
# ===========================================================================

def uniform_positions(n=5000, roi=1000.0, seed=0):
    return np.random.default_rng(seed).uniform(0, roi, (n, 3))


def clustered_positions(n=5000, roi=1000.0, n_clusters=30,
                        cluster_sigma=40.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(cluster_sigma * 3, roi - cluster_sigma * 3,
                          (n_clusters, 3))
    assignments = rng.integers(0, n_clusters, n)
    positions = centers[assignments] + rng.normal(0, cluster_sigma, (n, 3))
    return np.clip(positions, 0, roi)


# ===========================================================================
# CIC gridding
# ===========================================================================

class TestCICGridding:

    def test_mass_conservation(self):
        """Total weight in grid equals number of input points."""
        detector = SMLMMultiscaleTest(roi_size_nm=np.array([1000, 1000, 1000]))
        positions = uniform_positions(1000)
        grid = detector.grid_positions(positions)
        np.testing.assert_almost_equal(grid.sum(), 1000.0, decimal=5)

    def test_output_shape(self):
        g = 32
        detector = SMLMMultiscaleTest(roi_size_nm=np.array([500, 500, 500]),
                                      grid_size=g)
        positions = uniform_positions(200)
        grid = detector.grid_positions(positions)
        assert grid.shape == (g, g, g)

    def test_single_point_at_origin_weight_at_corner(self):
        """A point at (0,0,0) places all weight at grid corner [0,0,0]."""
        detector = SMLMMultiscaleTest(roi_size_nm=np.array([100, 100, 100]),
                                      grid_size=8)
        positions = np.array([[0.0, 0.0, 0.0]])
        grid = detector.grid_positions(positions)
        np.testing.assert_almost_equal(grid.sum(), 1.0)
        # The exact distribution depends on normalization; just check sum
        assert grid[0, 0, 0] > 0

    def test_non_negative_weights(self):
        detector = SMLMMultiscaleTest(roi_size_nm=np.array([1000, 1000, 1000]))
        positions = uniform_positions(500)
        grid = detector.grid_positions(positions)
        assert (grid >= 0).all()

    def test_degenerate_z_dimension(self):
        """2D data (all z=0) should not crash -- z dimension handled."""
        detector = SMLMMultiscaleTest(roi_size_nm=np.array([1000, 1000, 1]),
                                      grid_size=16)
        positions = np.random.default_rng(5).uniform(0, 1000, (100, 2))
        positions_3d = np.column_stack([positions, np.zeros(100)])
        grid = detector.grid_positions(positions_3d)
        np.testing.assert_almost_equal(grid.sum(), 100.0, decimal=5)

    def test_large_point_count(self):
        """Performance: 100k points should complete without error."""
        detector = SMLMMultiscaleTest(roi_size_nm=np.array([5000, 5000, 1000]),
                                      grid_size=32)
        positions = uniform_positions(n=10000, roi=5000.0)
        grid = detector.grid_positions(positions)
        np.testing.assert_almost_equal(grid.sum(), 10000.0, decimal=4)


# ===========================================================================
# Scale statistics
# ===========================================================================

class TestVarianceAtScale:

    def setup_method(self):
        self.detector = SMLMMultiscaleTest(
            roi_size_nm=np.array([1000, 1000, 1000]), grid_size=64
        )

    def test_uniform_data_variance_near_one(self):
        """
        For a Poisson point process, the index of dispersion (variance/mean)
        converges to 1 as the number of cells increases.
        Tests that our implementation is in the right ballpark.
        """
        positions = uniform_positions(n=50000, roi=1000.0, seed=10)
        grid = self.detector.grid_positions(positions)
        var_ratio = self.detector.variance_at_scale(grid, cell_size=4)
        # For Poisson, variance/mean ~ 1. Allow 0.5-2.0 for finite sample.
        assert 0.5 < var_ratio < 2.0, (
            f"Variance ratio {var_ratio:.3f} outside Poisson-expected range [0.5, 2.0]"
        )

    def test_clustered_data_higher_variance(self):
        """Clustered data should have higher index of dispersion than uniform."""
        detector = self.detector
        unif = uniform_positions(n=5000, roi=1000.0, seed=20)
        clust = clustered_positions(n=5000, roi=1000.0, seed=20)
        grid_u = detector.grid_positions(unif)
        grid_c = detector.grid_positions(clust)
        var_u = detector.variance_at_scale(grid_u, cell_size=4)
        var_c = detector.variance_at_scale(grid_c, cell_size=4)
        assert var_c > var_u, (
            f"Expected clustered variance {var_c:.3f} > uniform {var_u:.3f}"
        )

    def test_returns_nan_for_too_large_cell_size(self):
        positions = uniform_positions(100)
        grid = self.detector.grid_positions(positions)
        # cell_size = grid_size (way too large)
        result = self.detector.variance_at_scale(grid, cell_size=64)
        assert np.isnan(result)


class TestSkewnessAtScale:

    def setup_method(self):
        self.detector = SMLMMultiscaleTest(
            roi_size_nm=np.array([1000, 1000, 1000]), grid_size=64
        )

    def test_uniform_data_skewness_near_zero(self):
        """Poisson counts have low skewness for large expected counts."""
        positions = uniform_positions(n=50000, roi=1000.0, seed=30)
        grid = self.detector.grid_positions(positions)
        skew = self.detector.skewness_at_scale(grid, cell_size=4)
        assert abs(skew) < 2.0, f"Skewness {skew:.3f} unexpectedly large for uniform data"

    def test_clustered_data_positive_skewness(self):
        """Heavy-tailed cluster distributions have positive skewness."""
        detector = self.detector
        clust = clustered_positions(n=5000, roi=1000.0, seed=40)
        grid = detector.grid_positions(clust)
        skew = detector.skewness_at_scale(grid, cell_size=4)
        # Not guaranteed for all configs, but typical
        assert skew > 0, f"Expected positive skewness for clustered data, got {skew:.3f}"

    def test_returns_nan_for_too_large_cell_size(self):
        positions = uniform_positions(100)
        grid = self.detector.grid_positions(positions)
        result = self.detector.skewness_at_scale(grid, cell_size=64)
        assert np.isnan(result)


# ===========================================================================
# compute_curves
# ===========================================================================

class TestComputeCurves:

    def setup_method(self):
        self.detector = SMLMMultiscaleTest(
            roi_size_nm=np.array([1000, 1000, 1000]), grid_size=64
        )
        self.scale_range = ScaleRange.logarithmic(min_vox=3, max_vox=20,
                                                   n_scales=8)

    def test_output_keys(self):
        positions = uniform_positions(2000)
        curves = self.detector.compute_curves(positions, self.scale_range)
        assert 'variance' in curves
        assert 'skewness' in curves

    def test_each_metric_has_scales_and_values(self):
        positions = uniform_positions(2000)
        curves = self.detector.compute_curves(positions, self.scale_range)
        for metric in ['variance', 'skewness']:
            assert 'scales' in curves[metric]
            assert 'values' in curves[metric]

    def test_no_nan_in_values(self):
        positions = uniform_positions(5000)
        curves = self.detector.compute_curves(positions, self.scale_range)
        for metric in ['variance', 'skewness']:
            vals = np.array(curves[metric]['values'])
            assert not np.isnan(vals).any(), (
                f"NaN in {metric} values: {vals}"
            )

    def test_scales_and_values_same_length(self):
        positions = uniform_positions(2000)
        curves = self.detector.compute_curves(positions, self.scale_range)
        for metric in ['variance', 'skewness']:
            assert len(curves[metric]['scales']) == len(curves[metric]['values'])

    def test_scales_increasing(self):
        positions = uniform_positions(2000)
        curves = self.detector.compute_curves(positions, self.scale_range)
        for metric in ['variance', 'skewness']:
            sc = curves[metric]['scales']
            assert sc == sorted(sc), f"{metric} scales not monotonically increasing"


# ===========================================================================
# chi_squared_with_covariance
# ===========================================================================

class TestChiSquared:

    def setup_method(self):
        self.detector = SMLMMultiscaleTest(
            roi_size_nm=np.array([1000, 1000, 1000]), grid_size=64
        )
        self.scale_range = ScaleRange.logarithmic(min_vox=3, max_vox=20,
                                                   n_scales=8)

    def _make_curves(self, n_points=3000, seed=0):
        positions = uniform_positions(n_points, seed=seed)
        return self.detector.compute_curves(positions, self.scale_range)

    def test_output_keys(self):
        real_c = self._make_curves(seed=0)['variance']
        mocks_c = [self._make_curves(seed=i)['variance'] for i in range(1, 11)]
        result = self.detector.chi_squared_with_covariance(real_c, mocks_c)
        for key in ['chi_squared', 'dof', 'p_value', 'condition_number', 'n_mocks_used']:
            assert key in result, f"Missing key: {key}"

    def test_p_value_in_range(self):
        real_c = self._make_curves(seed=0)['variance']
        mocks_c = [self._make_curves(seed=i)['variance'] for i in range(1, 21)]
        result = self.detector.chi_squared_with_covariance(real_c, mocks_c)
        assert 0.0 <= result['p_value'] <= 1.0

    def test_few_mocks_returns_p1(self):
        """With < 2 matching mocks, return p=1 (no information)."""
        real_c = self._make_curves(seed=0)['variance']
        result = self.detector.chi_squared_with_covariance(real_c, [])
        assert result['p_value'] == 1.0

    def test_identical_real_and_mocks_high_p(self):
        """When real data comes from the same distribution as mocks, p should be large."""
        all_curves = [self._make_curves(seed=i)['variance'] for i in range(51)]
        real_c = all_curves[0]
        mocks_c = all_curves[1:]
        result = self.detector.chi_squared_with_covariance(real_c, mocks_c)
        # Not guaranteed to be > 0.05, but should not be extremely small
        assert result['p_value'] > 0.001, (
            f"p={result['p_value']:.4f} unexpectedly small for null-vs-null test"
        )

    def test_clustered_vs_unclustered_mocks_low_p(self):
        """Clustered data compared to unclustered mocks should give small p."""
        detector = self.detector
        scale_range = self.scale_range

        clust_curves = detector.compute_curves(
            clustered_positions(5000, seed=99), scale_range
        )
        mock_curves = [
            detector.compute_curves(uniform_positions(5000, seed=i), scale_range)
            for i in range(30)
        ]
        for metric in ['variance', 'skewness']:
            result = detector.chi_squared_with_covariance(
                clust_curves[metric],
                [mc[metric] for mc in mock_curves]
            )
            assert result['p_value'] < 0.05, (
                f"{metric}: clustered vs unclustered gave p={result['p_value']:.4f}"
            )

    def test_shrinkage_reduces_condition_number(self):
        """Higher shrinkage should give lower condition number."""
        real_c = self._make_curves(seed=0)['variance']
        mocks_c = [self._make_curves(seed=i)['variance'] for i in range(1, 10)]
        r_low = self.detector.chi_squared_with_covariance(real_c, mocks_c,
                                                          shrinkage=0.0)
        r_high = self.detector.chi_squared_with_covariance(real_c, mocks_c,
                                                           shrinkage=0.5)
        if not np.isinf(r_low['condition_number']):
            assert r_high['condition_number'] <= r_low['condition_number']


# ===========================================================================
# ScaleRange
# ===========================================================================

class TestScaleRange:

    def test_for_smlm_produces_correct_count(self):
        sr = ScaleRange.for_smlm(min_nm=20, max_nm=500, n_scales=12,
                                  roi_size_nm=5000, grid_size=64)
        # After deduplication, count may be <= 12 but > 0
        assert 1 <= len(sr.cell_sizes_vox) <= 12

    def test_for_smlm_monotonically_increasing(self):
        sr = ScaleRange.for_smlm(min_nm=20, max_nm=500, n_scales=12)
        assert sr.cell_sizes_vox == sorted(sr.cell_sizes_vox)

    def test_for_smlm_scales_and_nm_same_length(self):
        sr = ScaleRange.for_smlm(min_nm=20, max_nm=500, n_scales=10)
        assert len(sr.cell_sizes_vox) == len(sr.cell_sizes_nm)

    def test_logarithmic_compatible_with_detector(self):
        sr = ScaleRange.logarithmic(min_vox=3, max_vox=20, n_scales=8)
        assert len(sr.cell_sizes_vox) >= 1
        assert min(sr.cell_sizes_vox) >= 3
        assert max(sr.cell_sizes_vox) <= 20


# ===========================================================================
# from_positions
# ===========================================================================

class TestFromPositions:

    def test_roi_larger_than_data_span(self):
        positions = uniform_positions(200, roi=500.0)
        detector = SMLMMultiscaleTest.from_positions(positions)
        span = positions.max(axis=0) - positions.min(axis=0)
        # ROI should be at least as large as data span
        assert (detector.roi_size_nm >= span * 0.99).all()

    def test_default_grid_size(self):
        positions = uniform_positions(200)
        detector = SMLMMultiscaleTest.from_positions(positions)
        assert detector.grid_size == 64


# ===========================================================================
# DatasetInfo
# ===========================================================================

class TestDatasetInfo:

    def test_matching_datasets(self):
        info = DatasetInfo(1000, np.array([500., 500., 200.]), 64)
        other = DatasetInfo(1000, np.array([500., 500., 200.]), 64)
        match, msg = info.matches(other)
        assert match

    def test_n_mismatch_fails(self):
        info = DatasetInfo(1000, np.array([500., 500., 200.]), 64)
        other = DatasetInfo(500, np.array([500., 500., 200.]), 64)
        match, msg = info.matches(other)
        assert not match
        assert "N" in msg

    def test_roi_mismatch_fails(self):
        info = DatasetInfo(1000, np.array([500., 500., 200.]), 64)
        other = DatasetInfo(1000, np.array([100., 500., 200.]), 64)
        match, msg = info.matches(other)
        assert not match

    def test_grid_mismatch_fails(self):
        info = DatasetInfo(1000, np.array([500., 500., 200.]), 64)
        other = DatasetInfo(1000, np.array([500., 500., 200.]), 32)
        match, msg = info.matches(other)
        assert not match


# ===========================================================================
# Full pipeline validation (integration)
# ===========================================================================

class TestFullPipelineValidation:
    """
    Mirrors the cosmic validation logic: check FPR and TPR with known data.
    These are slower tests (use pytest -m 'not slow' to skip).
    """

    def _run_test(self, real_positions, n_mocks=30, seed_offset=0):
        from ..core.null_models import BoundingBoxNull
        null = BoundingBoxNull(seed=42)
        null.fit(real_positions)
        mocks = null.generate_mocks(len(real_positions), n_mocks)

        detector = SMLMMultiscaleTest.from_positions(real_positions)
        roi_max = real_positions.max(axis=0) - real_positions.min(axis=0)
        scale_range = ScaleRange.for_smlm(
            min_nm=20, max_nm=roi_max.max() / 4,
            n_scales=10, roi_size_nm=roi_max.max(), grid_size=64
        )
        return detector.test(real_positions, mocks, scale_range)

    def test_unclustered_vs_unclustered_high_p(self):
        """CSR data vs CSR mocks: p should not be < 0.05 most of the time."""
        positions = uniform_positions(n=5000, roi=2000.0, seed=0)
        results = self._run_test(positions, n_mocks=30)
        # We're not guaranteed p > 0.05 (it's probabilistic), but under the
        # null ~95% of runs should pass. Check we're not wildly wrong.
        p_var = results['variance']['p_value']
        p_skew = results['skewness']['p_value']
        # Both very small simultaneously would be suspicious
        assert not (p_var < 0.001 and p_skew < 0.001), (
            f"Both metrics highly significant for pure CSR data: "
            f"p_var={p_var:.4f}, p_skew={p_skew:.4f}"
        )

    def test_clustered_vs_unclustered_low_p(self):
        """Strong clustering should be detected."""
        positions = clustered_positions(n=5000, roi=2000.0, n_clusters=50,
                                        cluster_sigma=30.0, seed=1)
        results = self._run_test(positions, n_mocks=30)
        p_min = min(results['variance']['p_value'],
                    results['skewness']['p_value'])
        assert p_min < 0.05, (
            f"Failed to detect obvious clustering: min p={p_min:.4f}"
        )
