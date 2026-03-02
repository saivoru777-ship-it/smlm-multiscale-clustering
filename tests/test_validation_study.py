"""
Tests for the comprehensive validation study modules.

Tests:
    - Data loaders (ThunderSTORM, SMLM Challenge)
    - Synthetic scenarios (extended generators, biological presets)
    - Comprehensive study pipeline helper
    - Figure generator (smoke tests)
"""

import json
import tempfile
import numpy as np
import pytest
from pathlib import Path

from smlm_clustering.data.loaders import (
    load_thunderstorm_csv,
    load_smlm_challenge_csv,
)
from smlm_clustering.validation.synthetic_scenarios import (
    generate_extended_dataset,
    generate_from_preset,
    generate_cluster_points,
    BIOLOGICAL_PRESETS,
    BiologicalPreset,
)
from smlm_clustering.validation.comprehensive_study import (
    _run_pipeline,
    _corrected_chi2_pvalue,
    NumpyEncoder,
    ComprehensiveStudy,
)


# ============================================================================
# Data loaders
# ============================================================================

class TestThunderSTORMLoader:

    def _write_csv(self, tmp_path, content):
        fp = tmp_path / 'test.csv'
        fp.write_text(content)
        return fp

    def test_basic_3d(self, tmp_path):
        csv_content = (
            '"id","frame","x [nm]","y [nm]","z [nm]","intensity [photon]"\n'
            '1,1,100.5,200.3,50.0,1500.0\n'
            '2,1,300.0,400.0,75.0,2000.0\n'
            '3,2,150.0,250.0,60.0,1800.0\n'
        )
        fp = self._write_csv(tmp_path, csv_content)
        result = load_thunderstorm_csv(fp)

        assert result['positions'].shape == (3, 3)
        assert result['frames'].shape == (3,)
        assert result['photons'].shape == (3,)
        np.testing.assert_allclose(result['positions'][0], [100.5, 200.3, 50.0])
        assert result['frames'][0] == 1
        assert result['metadata']['has_z'] is True
        assert result['metadata']['n_localizations'] == 3

    def test_2d_no_z(self, tmp_path):
        csv_content = (
            '"id","frame","x [nm]","y [nm]","intensity [photon]"\n'
            '1,1,100.0,200.0,1500.0\n'
        )
        fp = self._write_csv(tmp_path, csv_content)
        result = load_thunderstorm_csv(fp)

        assert result['positions'].shape == (1, 3)
        assert result['positions'][0, 2] == 0.0  # default z
        assert result['metadata']['has_z'] is False

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_thunderstorm_csv('/nonexistent/file.csv')

    def test_empty_file(self, tmp_path):
        fp = self._write_csv(tmp_path, '"x [nm]","y [nm]"\n')
        with pytest.raises(ValueError, match="no data rows"):
            load_thunderstorm_csv(fp)


class TestSMLMChallengeLoader:

    def test_basic(self, tmp_path):
        csv_content = (
            'frame,x,y,z,photons\n'
            '1,100.0,200.0,50.0,1500\n'
            '2,300.0,400.0,75.0,2000\n'
        )
        fp = tmp_path / 'test.csv'
        fp.write_text(csv_content)
        result = load_smlm_challenge_csv(fp)

        assert result['positions'].shape == (2, 3)
        assert result['frames'].shape == (2,)
        assert result['metadata']['format'] == 'smlm_challenge'

    def test_tab_delimited(self, tmp_path):
        csv_content = (
            'frame\tx\ty\tz\tphotons\n'
            '1\t100.0\t200.0\t50.0\t1500\n'
        )
        fp = tmp_path / 'test.tsv'
        fp.write_text(csv_content)
        result = load_smlm_challenge_csv(fp)

        assert result['positions'].shape == (1, 3)


# ============================================================================
# Synthetic scenarios
# ============================================================================

class TestExtendedGenerator:

    def test_gaussian_shape(self):
        ds = generate_extended_dataset(
            n_molecules=100, f_clust=0.3, n_clusters=5,
            cluster_shape='gaussian', seed=42)
        assert ds.raw_positions.shape[1] == 3
        assert len(ds.molecule_labels) == 100
        assert ds.molecule_labels.sum() == 30  # 30% clustered

    def test_elongated_shape(self):
        ds = generate_extended_dataset(
            n_molecules=100, f_clust=0.3, n_clusters=5,
            cluster_shape='elongated', aspect_ratio=3.0, seed=42)
        assert ds.raw_positions.shape[1] == 3
        assert ds.f_clust == 0.3

    def test_ring_shape(self):
        ds = generate_extended_dataset(
            n_molecules=100, f_clust=0.3, n_clusters=5,
            cluster_shape='ring', ring_width_nm=10.0, seed=42)
        assert ds.raw_positions.shape[1] == 3

    def test_no_clusters(self):
        ds = generate_extended_dataset(
            n_molecules=100, f_clust=0.0, seed=42)
        assert ds.molecule_labels.sum() == 0
        assert ds.f_clust == 0.0

    def test_full_clustering(self):
        ds = generate_extended_dataset(
            n_molecules=100, f_clust=1.0, n_clusters=10, seed=42)
        assert ds.molecule_labels.sum() == 100

    def test_positions_in_roi(self):
        ds = generate_extended_dataset(
            n_molecules=200, f_clust=0.5, roi_nm=3000.0, seed=42)
        assert (ds.raw_positions >= 0).all()
        assert (ds.raw_positions <= 3000.0).all()

    def test_deterministic_seed(self):
        ds1 = generate_extended_dataset(n_molecules=50, seed=123)
        ds2 = generate_extended_dataset(n_molecules=50, seed=123)
        np.testing.assert_array_equal(
            ds1.molecule_positions, ds2.molecule_positions)


class TestClusterPointGenerator:

    def test_gaussian(self):
        rng = np.random.default_rng(42)
        pts = generate_cluster_points(
            center=np.array([0, 0, 0]), n_points=1000,
            radius_nm=50.0, shape='gaussian', rng=rng)
        assert pts.shape == (1000, 3)
        # Points should be centered near origin
        assert abs(pts.mean(axis=0)).max() < 10

    def test_elongated(self):
        rng = np.random.default_rng(42)
        pts = generate_cluster_points(
            center=np.array([0, 0, 0]), n_points=1000,
            radius_nm=50.0, shape='elongated', aspect_ratio=3.0, rng=rng)
        assert pts.shape == (1000, 3)

    def test_ring(self):
        rng = np.random.default_rng(42)
        pts = generate_cluster_points(
            center=np.array([0, 0, 0]), n_points=1000,
            radius_nm=50.0, shape='ring', ring_width_nm=5.0, rng=rng)
        assert pts.shape == (1000, 3)
        # Points should be roughly at radius distance from center
        distances_xy = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        # Most points should be near the ring radius (within 3 sigma)
        assert np.median(distances_xy) > 30  # not all at center

    def test_invalid_shape(self):
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="Unknown cluster shape"):
            generate_cluster_points(
                center=np.array([0, 0, 0]), n_points=10,
                radius_nm=50.0, shape='invalid', rng=rng)


class TestBiologicalPresets:

    def test_all_presets_exist(self):
        expected = ['synaptic_receptors', 'nuclear_pores',
                    'membrane_domains', 'negative_control']
        for name in expected:
            assert name in BIOLOGICAL_PRESETS

    def test_preset_types(self):
        for name, preset in BIOLOGICAL_PRESETS.items():
            assert isinstance(preset, BiologicalPreset)
            assert preset.n_molecules > 0
            assert 0 <= preset.f_clust <= 1
            assert preset.roi_nm > 0

    def test_generate_from_preset(self):
        for name in BIOLOGICAL_PRESETS:
            ds = generate_from_preset(name, seed=42)
            preset = BIOLOGICAL_PRESETS[name]
            assert len(ds.molecule_labels) == preset.n_molecules
            assert ds.roi_nm == preset.roi_nm

    def test_invalid_preset(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            generate_from_preset('nonexistent')

    def test_negative_control_no_clusters(self):
        ds = generate_from_preset('negative_control', seed=42)
        assert ds.f_clust == 0.0
        assert ds.molecule_labels.sum() == 0


# ============================================================================
# Pipeline and study
# ============================================================================

class TestCorrectedChi2:

    def test_null_returns_high_p(self):
        """Mock data tested against itself should give high p-values."""
        rng = np.random.default_rng(42)
        # Generate mock curves that are all from the same distribution
        n_mocks = 15
        n_scales = 5
        mock_curves = rng.normal(1.0, 0.1, (n_mocks, n_scales)).tolist()
        # "Real" data from the same distribution
        real_curve = rng.normal(1.0, 0.1, n_scales).tolist()

        p = _corrected_chi2_pvalue(real_curve, mock_curves)
        assert p > 0.01, f"p={p} should be high for null data"

    def test_signal_returns_low_p(self):
        """Real data far from mocks should give low p-value."""
        rng = np.random.default_rng(42)
        n_mocks = 15
        n_scales = 5
        mock_curves = rng.normal(1.0, 0.1, (n_mocks, n_scales)).tolist()
        # Real data with strong signal
        real_curve = (np.ones(n_scales) * 5.0).tolist()

        p = _corrected_chi2_pvalue(real_curve, mock_curves)
        assert p < 0.05, f"p={p} should be low for strong signal"

    def test_too_few_mocks(self):
        """Should return 1.0 with insufficient mocks."""
        p = _corrected_chi2_pvalue([1.0, 2.0], [[1.0, 2.0]])
        assert p == 1.0

    def test_empty_input(self):
        """Should return 1.0 with empty input."""
        p = _corrected_chi2_pvalue([], [])
        assert p == 1.0


class TestRunPipeline:

    def test_csr_pipeline(self):
        """Pipeline should run on CSR data without errors."""
        from smlm_clustering.validation.benchmark_runner import (
            generate_synthetic_dataset)
        dataset = generate_synthetic_dataset(
            n_molecules=200, f_clust=0.0, seed=42)
        result = _run_pipeline(dataset, n_mocks=5)

        assert 'ms_detected' in result
        assert 'ms_p_variance' in result
        assert 'ms_p_skewness' in result
        assert 'dbscan_detected' in result
        assert 'ripley_detected' in result
        assert 0 <= result['ms_p_variance'] <= 1
        assert 0 <= result['ms_p_skewness'] <= 1

    def test_clustered_pipeline(self):
        """Pipeline should detect strong clustering."""
        from smlm_clustering.validation.benchmark_runner import (
            generate_synthetic_dataset)
        dataset = generate_synthetic_dataset(
            n_molecules=500, f_clust=0.5, seed=42)
        result = _run_pipeline(dataset, n_mocks=10)

        assert 'ms_detected' in result
        assert result['has_clusters'] is True
        assert result['f_clust'] == 0.5

    def test_pipeline_with_custom_params(self):
        """Pipeline should accept custom parameters."""
        from smlm_clustering.validation.benchmark_runner import (
            generate_synthetic_dataset)
        dataset = generate_synthetic_dataset(
            n_molecules=200, f_clust=0.0, seed=42)
        result = _run_pipeline(
            dataset, n_mocks=5, grid_size=32,
            shrinkage=0.2, r_lateral=20.0, r_axial=40.0)

        assert 'ms_detected' in result


class TestNumpyEncoder:

    def test_ndarray(self):
        data = {'arr': np.array([1.0, 2.0, 3.0])}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed['arr'] == [1.0, 2.0, 3.0]

    def test_numpy_int(self):
        data = {'val': np.int64(42)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed['val'] == 42

    def test_numpy_float(self):
        data = {'val': np.float64(3.14)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert abs(parsed['val'] - 3.14) < 1e-10

    def test_numpy_nan(self):
        data = {'val': np.float64('nan')}
        result = json.dumps(data, cls=NumpyEncoder)
        assert 'nan' in result.lower()

    def test_numpy_bool(self):
        data = {'val': np.bool_(True)}
        result = json.dumps(data, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed['val'] is True


class TestComprehensiveStudy:

    def test_init(self, tmp_path):
        study = ComprehensiveStudy(
            output_dir=str(tmp_path), quick=True)
        assert study.quick is True
        assert study.output_dir == tmp_path

    def test_seed_determinism(self):
        study = ComprehensiveStudy(quick=True)
        s1 = study._seed(0, 0)
        s2 = study._seed(0, 0)
        s3 = study._seed(1, 0)
        assert s1 == s2
        assert s1 != s3

    def test_n_reps_quick(self):
        study = ComprehensiveStudy(quick=True)
        assert study._n_reps(100) == 10
        assert study._n_reps(20) == 3  # min of 3
        assert study._n_reps(10) == 3

    def test_n_reps_full(self):
        study = ComprehensiveStudy(quick=False)
        assert study._n_reps(100) == 100
        assert study._n_reps(20) == 20

    def test_n_mocks_quick(self):
        study = ComprehensiveStudy(quick=True)
        assert study._n_mocks() >= 15  # min for chi2 calibration
        assert study._n_mocks(50) >= 15
        assert study._n_mocks(min_mocks=30) >= 30  # custom minimum

    def test_n_mocks_full(self):
        study = ComprehensiveStudy(quick=False)
        assert study._n_mocks() == 50
        assert study._n_mocks(50) == 50


# ============================================================================
# Figure generator (smoke tests)
# ============================================================================

class TestFigureGenerator:

    def test_import(self):
        from smlm_clustering.validation.figure_generator import (
            generate_all_figures, COLORS, STYLE)
        assert 'multiscale' in COLORS
        assert 'dbscan' in COLORS
        assert 'ripley' in COLORS

    def test_missing_results_no_crash(self, tmp_path):
        """Figure generation should handle missing result files gracefully."""
        from smlm_clustering.validation.figure_generator import (
            generate_all_figures)
        # Should not raise — just print warnings
        generate_all_figures(
            results_dir=str(tmp_path / 'nonexistent'),
            figures_dir=str(tmp_path / 'figures'),
        )
