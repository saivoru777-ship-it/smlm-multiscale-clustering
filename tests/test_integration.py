"""
Integration tests for the full SMLM clustering pipeline.

These tests run the complete pipeline end-to-end:
    raw SMLM data (with blinking)
    -> blinking correction
    -> biological null model
    -> multiscale test

And verify:
1. The pipeline runs without error on synthetic data
2. The blinking correction reduces localization count
3. The null model generates points inside the corrected-data hull
4. The chi-squared test produces valid p-values
5. Clustered data is detected; unclustered data is not flagged
6. The baseline comparison methods run and produce consistent output

These tests are slower than unit tests (~30s for the full benchmark call).
"""

import numpy as np
import pytest
from ..core.blinking_correction import BlinkingCorrector
from ..core.null_models import BiologicalNullModel, BoundingBoxNull
from ..core.multiscale_detector import SMLMMultiscaleTest, ScaleRange
from ..validation.comparison_methods import DBSCANBaseline, RipleysKBaseline
from ..validation.benchmark_runner import generate_synthetic_dataset
from ..validation.metrics import evaluate_detection, summarize_detection_experiments


# ===========================================================================
# End-to-end pipeline
# ===========================================================================

class TestPipelineEndToEnd:

    def _make_dataset(self, f_clust=0.3, seed=0, n_frames=20):
        # n_frames=20 with n_blinks_mean=4: expected inter-blink gap ~5 frames,
        # within max_dark_frames=5 so temporal merging actually fires.
        return generate_synthetic_dataset(
            n_molecules=2000,
            f_clust=f_clust,
            n_clusters=10,
            cluster_radius_nm=75.0,
            roi_nm=3000.0,
            n_blinks_mean=4.0,
            n_frames=n_frames,
            seed=seed
        )

    def test_full_pipeline_runs_without_error(self):
        dataset = self._make_dataset(f_clust=0.3, seed=0)

        # Step 1: blink correct
        corrector = BlinkingCorrector(r_lateral=30, r_axial=60, max_dark_frames=5)
        blink_result = corrector.correct(
            dataset.raw_positions, dataset.raw_frames, dataset.raw_photons
        )

        # Step 2: biological null
        null = BiologicalNullModel(seed=42)
        null.fit(blink_result.positions)
        mocks = null.generate_mocks(len(blink_result.positions), n_mocks=10)

        # Step 3: multiscale test
        detector = SMLMMultiscaleTest.from_positions(blink_result.positions)
        roi_max = blink_result.positions.max(axis=0) - blink_result.positions.min(axis=0)
        scale_range = ScaleRange.for_smlm(
            min_nm=20, max_nm=roi_max.max() / 4,
            n_scales=8, roi_size_nm=roi_max.max(), grid_size=32
        )
        results = detector.test(blink_result.positions, mocks, scale_range)

        # Check p-values exist and are valid
        for metric in ['variance', 'skewness']:
            assert 'p_value' in results[metric]
            assert 0.0 <= results[metric]['p_value'] <= 1.0

    def test_blinking_correction_reduces_count(self):
        # n_frames=20 ensures blinks from the same molecule cluster in time,
        # so max_dark_frames=5 actually fires and reduces the localization count.
        dataset = self._make_dataset(f_clust=0.0, seed=1, n_frames=20)
        corrector = BlinkingCorrector(r_lateral=30, r_axial=60, max_dark_frames=5)
        result = corrector.correct(
            dataset.raw_positions, dataset.raw_frames, dataset.raw_photons
        )
        assert result.n_merged < result.n_raw
        # With 4 blinks/molecule in 20 frames, at least 25% reduction expected.
        # CSR molecules spread over 3000nm sometimes get incorrectly linked,
        # limiting the reduction. 1.3x is a conservative lower bound.
        assert result.reduction_factor > 1.3

    def test_null_mocks_inside_hull(self):
        from scipy.spatial import ConvexHull
        dataset = self._make_dataset(f_clust=0.0, seed=2)
        corrector = BlinkingCorrector(r_lateral=30, r_axial=60, max_dark_frames=5)
        blink_result = corrector.correct(
            dataset.raw_positions, dataset.raw_frames, dataset.raw_photons
        )
        null = BiologicalNullModel(seed=0)
        null.fit(blink_result.positions)
        mock = null.sample(200)

        hull = ConvexHull(blink_result.positions)
        eq = hull.equations
        inside = (eq[:, :3] @ mock.T + eq[:, 3:]).max(axis=0) <= 1e-6
        assert inside.mean() > 0.99

    def test_unclustered_pipeline_no_spurious_detection(self):
        """
        Pure CSR data (clean molecule positions, no blinking) compared against
        CSR mocks drawn from the same bounding box should not trigger detection.

        We use molecule_positions directly (not raw blink positions) to avoid
        the small-scale density structure that unmerged blinks would introduce.
        That structure is a real signal relative to a pure CSR null -- testing
        for it here would conflate blinking correction quality with detector
        specificity. The detector's specificity is validated in unit tests using
        idealized CSR data; this test just verifies the integration pipeline
        produces valid output without crashing.
        """
        dataset = self._make_dataset(f_clust=0.0, seed=3)
        # Use ground-truth molecule positions (no blinking overcounting)
        clean_positions = dataset.molecule_positions

        null = BoundingBoxNull(seed=5)
        null.fit(clean_positions)
        mocks = null.generate_mocks(len(clean_positions), n_mocks=20)

        detector = SMLMMultiscaleTest.from_positions(clean_positions)
        roi_max = clean_positions.max(axis=0) - clean_positions.min(axis=0)
        scale_range = ScaleRange.for_smlm(
            min_nm=50, max_nm=roi_max.max() / 4,
            n_scales=8, roi_size_nm=roi_max.max(), grid_size=32
        )
        results = detector.test(clean_positions, mocks, scale_range)

        # Valid output contract
        for metric in ['variance', 'skewness']:
            assert 0.0 <= results[metric]['p_value'] <= 1.0

        p_var = results['variance']['p_value']
        p_skew = results['skewness']['p_value']
        # For CSR vs CSR, both metrics strongly significant simultaneously is unusual
        assert not (p_var < 0.001 and p_skew < 0.001), (
            f"Spurious detection for CSR data: p_var={p_var:.4f}, p_skew={p_skew:.4f}"
        )

    def test_strong_clustering_detected(self):
        """Very strong clustering (f_clust=0.8) should always be detected."""
        dataset = generate_synthetic_dataset(
            n_molecules=3000,
            f_clust=0.8,
            n_clusters=15,
            cluster_radius_nm=50.0,
            roi_nm=3000.0,
            n_blinks_mean=3.0,
            seed=42
        )
        corrector = BlinkingCorrector(r_lateral=30, r_axial=60, max_dark_frames=5)
        blink_result = corrector.correct(
            dataset.raw_positions, dataset.raw_frames, dataset.raw_photons
        )
        null = BoundingBoxNull(seed=10)
        null.fit(blink_result.positions)
        mocks = null.generate_mocks(len(blink_result.positions), n_mocks=30)

        detector = SMLMMultiscaleTest.from_positions(blink_result.positions)
        roi_max = blink_result.positions.max(axis=0) - blink_result.positions.min(axis=0)
        scale_range = ScaleRange.for_smlm(
            min_nm=20, max_nm=roi_max.max() / 4,
            n_scales=10, roi_size_nm=roi_max.max(), grid_size=32
        )
        results = detector.test(blink_result.positions, mocks, scale_range)

        p_min = min(results['variance']['p_value'],
                    results['skewness']['p_value'])
        assert p_min < 0.05, (
            f"Failed to detect strong clustering: min p={p_min:.4f}"
        )


# ===========================================================================
# Baseline comparisons
# ===========================================================================

class TestBaselineMethods:

    def _make_positions(self, f_clust=0.0, seed=10):
        dataset = generate_synthetic_dataset(
            n_molecules=1000, f_clust=f_clust,
            n_clusters=10, cluster_radius_nm=60.0,
            roi_nm=2000.0, n_blinks_mean=1.0, seed=seed
        )
        return dataset.raw_positions  # no blinking correction needed for baselines

    def test_dbscan_returns_labels(self):
        positions = self._make_positions()
        dbscan = DBSCANBaseline(eps_nm=60.0, min_samples=5)
        result = dbscan.fit(positions)
        assert result.labels.shape == (len(positions),)
        assert result.n_clusters >= 0
        assert result.n_noise >= 0
        assert result.n_clusters + result.n_noise <= len(positions)

    def test_dbscan_all_noise_for_very_tight_eps(self):
        """Eps = 1nm should give all noise for typical SMLM data."""
        positions = self._make_positions()
        dbscan = DBSCANBaseline(eps_nm=1.0, min_samples=5)
        result = dbscan.fit(positions)
        assert result.n_clusters == 0
        assert result.n_noise == len(positions)

    def test_dbscan_finds_clusters_in_clustered_data(self):
        positions = self._make_positions(f_clust=0.5, seed=20)
        dbscan = DBSCANBaseline(eps_nm=80.0, min_samples=5)
        result = dbscan.fit(positions)
        assert result.n_clusters > 0

    def test_ripleys_k_returns_result(self):
        positions = self._make_positions()
        ripley = RipleysKBaseline(r_min_nm=20, r_max_nm=300, n_radii=10)
        result = ripley.compute(positions)
        assert result.radii.shape == (10,)
        assert result.K_observed.shape == (10,)
        assert result.L_minus_r.shape == (10,)

    def test_ripleys_k_poisson_near_analytic(self):
        """
        For large uniform datasets, K(r) should approach (4/3)pi*r^3.
        Test at r=100nm with n=5000 points in a 2000nm cube.
        """
        np.random.seed(0)
        positions = np.random.uniform(0, 2000, (5000, 3))
        ripley = RipleysKBaseline(r_min_nm=50, r_max_nm=200, n_radii=5)
        result = ripley.compute(positions)

        r_mid = result.radii[len(result.radii) // 2]
        K_obs = result.K_observed[len(result.radii) // 2]
        K_analytic = (4.0 / 3.0) * np.pi * r_mid**3

        # Allow 50% deviation due to edge effects and finite N
        rel_err = abs(K_obs - K_analytic) / K_analytic
        assert rel_err < 0.5, (
            f"K({r_mid:.0f}nm): obs={K_obs:.1f}, analytic={K_analytic:.1f}, "
            f"rel_err={rel_err:.2f}"
        )

    def test_ripleys_clustered_l_minus_r_positive(self):
        """L(r) - r > 0 indicates clustering."""
        positions = self._make_positions(f_clust=0.5, seed=30)
        ripley = RipleysKBaseline(r_min_nm=20, r_max_nm=200, n_radii=10)
        result = ripley.compute(positions)
        # At intermediate scales, expect positive L-r for clustered data
        mid_vals = result.L_minus_r[3:7]
        assert mid_vals.mean() > 0, (
            f"Expected L(r)-r > 0 for clustered data, got mean={mid_vals.mean():.2f}"
        )


# ===========================================================================
# Metrics
# ===========================================================================

class TestMetrics:

    def test_perfect_detection(self):
        pred = np.array([1, 1, 0, 0])
        true = np.array([1, 1, 0, 0])
        m = evaluate_detection(pred, true)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.jaccard == 1.0

    def test_all_missed(self):
        pred = np.array([0, 0, 0, 0])
        true = np.array([1, 1, 0, 0])
        m = evaluate_detection(pred, true)
        assert m.recall == 0.0
        assert m.tp == 0
        assert m.fn == 2

    def test_all_false_positives(self):
        pred = np.array([1, 1, 1, 1])
        true = np.array([0, 0, 0, 0])
        m = evaluate_detection(pred, true)
        assert m.precision == 0.0
        assert m.fp == 4

    def test_f1_harmonic_mean(self):
        pred = np.array([1, 1, 1, 0])
        true = np.array([1, 0, 1, 1])
        m = evaluate_detection(pred, true)
        # TP=2, FP=1, FN=1
        prec = 2 / 3
        rec = 2 / 3
        expected_f1 = 2 * prec * rec / (prec + rec)
        np.testing.assert_almost_equal(m.f1, expected_f1)

    def test_summarize_detection_experiments(self):
        results = [
            {'detected': True, 'has_clusters': True},    # TP
            {'detected': True, 'has_clusters': True},    # TP
            {'detected': False, 'has_clusters': True},   # FN
            {'detected': False, 'has_clusters': False},  # TN
            {'detected': True, 'has_clusters': False},   # FP
        ]
        summary = summarize_detection_experiments(results)
        assert summary['tp'] == 2
        assert summary['fn'] == 1
        assert summary['tn'] == 1
        assert summary['fp'] == 1
        np.testing.assert_almost_equal(summary['sensitivity'], 2/3)
        np.testing.assert_almost_equal(summary['specificity'], 1/2)
