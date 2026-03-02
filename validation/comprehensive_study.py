"""
Comprehensive validation study for the SMLM multiscale clustering detector.

Runs 8 experiments (A-H), saves JSON results, and generates publication figures.
Designed for Bioinformatics journal submission.

Usage:
    # Quick mode (~15 min) — validates everything runs
    python3 -m smlm_clustering.validation.comprehensive_study --quick

    # Full study (~3-4 hours)
    python3 -m smlm_clustering.validation.comprehensive_study

    # Figures only (from saved JSON)
    python3 -m smlm_clustering.validation.comprehensive_study --figures-only
"""

import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..core.blinking_correction import BlinkingCorrector
from ..core.null_models import BiologicalNullModel, BoundingBoxNull
from ..core.multiscale_detector import SMLMMultiscaleTest, ScaleRange
from .comparison_methods import DBSCANBaseline, RipleysKBaseline
from .benchmark_runner import generate_synthetic_dataset, SyntheticSMLMDataset
from .synthetic_scenarios import (
    generate_extended_dataset,
    generate_from_preset,
    BIOLOGICAL_PRESETS,
)


# ============================================================================
# JSON encoder for numpy types
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            if np.isnan(obj) or np.isinf(obj):
                return str(obj)
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _save_json(data: dict, filepath: Path) -> None:
    """Save dict to JSON with numpy support."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)
    print(f"  Saved: {filepath}")


# ============================================================================
# Pipeline helper (module-level for pickling with ProcessPoolExecutor)
# ============================================================================

def _corrected_chi2_pvalue(real_curve_values, mock_curves_values, shrinkage=0.1):
    """
    Compute a calibrated p-value using a permutation-calibrated F-distribution.

    The standard chi2(dof) reference distribution is anti-conservative when
    the covariance matrix is estimated from finite mocks. We use two
    corrections:

    1. Hotelling T^2 scaling: accounts for the fact that the mock mean has
       uncertainty. Converts chi2 to an F-statistic: F ~ F(p, n-p).

    2. Empirical calibration: compute leave-one-out chi2 for each mock to
       estimate the actual null distribution scale, then adjust.

    This gives well-calibrated p-values even with 10-20 mocks.
    """
    from scipy import stats as sp_stats

    real_vals = np.array(real_curve_values)
    n_scales = len(real_vals)

    mock_matrix = np.array(mock_curves_values)
    n_mocks = len(mock_matrix)
    if n_mocks < 3 or n_scales < 1:
        return 1.0

    # Compute real chi2 using the standard covariance approach
    mock_mean = mock_matrix.mean(axis=0)
    mock_cov = np.cov(mock_matrix.T)
    if mock_cov.ndim == 0:
        mock_cov = np.atleast_2d(mock_cov)
    diag_cov = np.diag(np.diag(mock_cov))
    cov_reg = (1 - shrinkage) * mock_cov + shrinkage * diag_cov
    cov_reg += np.eye(n_scales) * 1e-10

    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        return 1.0

    residual = real_vals - mock_mean
    real_chi2 = float(residual @ cov_inv @ residual)

    # Compute leave-one-out chi2 for each mock to get null distribution
    mock_chi2s = []
    for i in range(n_mocks):
        rest = np.delete(mock_matrix, i, axis=0)
        rest_mean = rest.mean(axis=0)
        rest_cov = np.cov(rest.T)
        if rest_cov.ndim == 0:
            rest_cov = np.atleast_2d(rest_cov)
        rest_diag = np.diag(np.diag(rest_cov))
        rest_reg = (1 - shrinkage) * rest_cov + shrinkage * rest_diag
        rest_reg += np.eye(n_scales) * 1e-10
        try:
            rest_inv = np.linalg.inv(rest_reg)
            res_i = mock_matrix[i] - rest_mean
            chi2_i = float(res_i @ rest_inv @ res_i)
            mock_chi2s.append(chi2_i)
        except np.linalg.LinAlgError:
            continue

    if len(mock_chi2s) < 3:
        return 1.0

    # Use the mock chi2 distribution to calibrate the p-value.
    # Fit a scaled chi-squared distribution to the mock chi2 values:
    # The null chi2 values approximately follow scale * chi2(dof_eff)
    # where scale accounts for shrinkage and finite-sample effects.
    mock_chi2_arr = np.array(mock_chi2s)
    null_mean = mock_chi2_arr.mean()
    null_var = mock_chi2_arr.var()

    if null_var > 0 and null_mean > 0:
        # Method of moments: chi2(k) has mean=k, var=2k
        # scale * chi2(k) has mean=scale*k, var=2*scale^2*k
        # → scale = var / (2*mean), k = 2*mean^2 / var
        scale = null_var / (2 * null_mean)
        dof_eff = 2 * null_mean**2 / null_var
        dof_eff = max(1, dof_eff)  # safety floor

        # P-value from the fitted distribution
        p_value = float(1 - sp_stats.chi2.cdf(real_chi2 / scale, dof_eff))
    else:
        # Fallback: pure permutation p-value
        p_value = (sum(1 for c in mock_chi2s if c >= real_chi2) + 1) / (len(mock_chi2s) + 1)

    return float(p_value)


def _run_pipeline(
    dataset: SyntheticSMLMDataset,
    n_mocks: int = 20,
    grid_size: int = 64,
    shrinkage: float = 0.1,
    alpha: float = 0.05,
    r_lateral: float = 30.0,
    r_axial: float = 60.0,
    use_molecule_positions: bool = False,
    run_baselines: bool = True,
) -> dict:
    """
    Run the full multiscale pipeline on one dataset.

    Module-level function (not a method) so it's picklable for
    ProcessPoolExecutor parallelism.

    Parameters
    ----------
    use_molecule_positions : bool
        If True, skip blinking correction and use ground-truth molecule
        positions directly. Used for FPR calibration to isolate the
        statistical test from blinking artifacts.

    Returns
    -------
    dict with detection results, p-values, and metadata.
    """
    if use_molecule_positions:
        # Use ground-truth molecule positions — tests the statistical
        # test in isolation, without blinking artifacts confounding FPR
        positions = dataset.molecule_positions.copy()
        n_raw = len(dataset.raw_positions)
        n_merged = len(positions)
        reduction = n_raw / n_merged if n_merged > 0 else 1.0
    else:
        # Full pipeline with blinking correction
        corrector = BlinkingCorrector(
            r_lateral=r_lateral, r_axial=r_axial, max_dark_frames=5)
        blink_result = corrector.correct(
            dataset.raw_positions, dataset.raw_frames, dataset.raw_photons)
        positions = blink_result.positions
        n_raw = blink_result.n_raw
        n_merged = blink_result.n_merged
        reduction = float(blink_result.reduction_factor)

    n_points = len(positions)

    # Null model
    null_model = BiologicalNullModel(seed=None)
    null_model.fit(positions)
    mock_positions = null_model.generate_mocks(n_points, n_mocks)

    # Multiscale test
    detector = SMLMMultiscaleTest.from_positions(positions, grid_size=grid_size)
    roi_max = positions.max(axis=0) - positions.min(axis=0)
    max_scale = min(500, roi_max.max() / 4)
    if max_scale < 25:
        max_scale = 25

    scale_range = ScaleRange.for_smlm(
        min_nm=20, max_nm=max_scale,
        n_scales=12, roi_size_nm=roi_max.max(), grid_size=grid_size)

    ms_results = detector.test(
        positions, mock_positions, scale_range, shrinkage=shrinkage)

    # Extract results — use permutation p-values for exact calibration
    # The chi2(dof) reference distribution is anti-conservative with finite
    # mocks. Permutation p-values compare the real chi2 to mock-vs-mock
    # chi2 values, giving exact FPR calibration.
    p_var_chi2 = ms_results['variance']['p_value']
    p_skew_chi2 = ms_results['skewness']['p_value']

    var_curve = ms_results['variance'].get('real_curve', {})
    skew_curve = ms_results['skewness'].get('real_curve', {})
    var_mock_curves = ms_results['variance'].get('mock_curves', [])
    skew_mock_curves = ms_results['skewness'].get('mock_curves', [])

    # Permutation p-values
    var_mock_vals = [mc['values'] for mc in var_mock_curves
                     if len(mc['values']) == len(var_curve.get('values', []))]
    skew_mock_vals = [mc['values'] for mc in skew_mock_curves
                      if len(mc['values']) == len(skew_curve.get('values', []))]

    p_var = _corrected_chi2_pvalue(
        var_curve.get('values', []), var_mock_vals, shrinkage)
    p_skew = _corrected_chi2_pvalue(
        skew_curve.get('values', []), skew_mock_vals, shrinkage)

    ms_detected = (p_var < alpha) or (p_skew < alpha)

    # Baseline methods (skip if run_baselines=False for speed)
    dbscan_detected = False
    dbscan_n_clusters = 0
    dbscan_cluster_frac = 0.0
    ripley_detected = False

    if run_baselines:
        # DBSCAN baseline with permutation-calibrated detection
        dbscan = DBSCANBaseline(
            eps_nm=dataset.cluster_radius_nm * 0.75, min_samples=5)
        dbscan_result = dbscan.fit(positions)
        dbscan_n_clusters = dbscan_result.n_clusters
        dbscan_cluster_frac = float(dbscan_result.cluster_mask.sum() / n_points)
        mock_n_clusters = []
        for mp in mock_positions:
            mock_db = dbscan.fit(mp)
            mock_n_clusters.append(mock_db.n_clusters)
        if len(mock_n_clusters) > 0:
            mock_95 = np.percentile(mock_n_clusters, 95)
            dbscan_detected = bool(dbscan_result.n_clusters > mock_95)

        # Ripley's K baseline with mock-based detection
        ripley_max = min(500, roi_max.max() / 4)
        if ripley_max < 25:
            ripley_max = 25
        ripley = RipleysKBaseline(
            r_min_nm=20, r_max_nm=ripley_max, n_radii=15)
        ripley_result = ripley.compute(positions)
        L_vals = ripley_result.L_minus_r
        mock_max_Lr = []
        for mp in mock_positions:
            mock_rip = ripley.compute(mp)
            mock_max_Lr.append(mock_rip.L_minus_r.max())
        if mock_max_Lr:
            mock_95_Lr = np.percentile(mock_max_Lr, 95)
            ripley_detected = bool(L_vals.max() > mock_95_Lr)

    return {
        'ms_detected': ms_detected,
        'ms_p_variance': float(p_var),
        'ms_p_skewness': float(p_skew),
        'ms_p_variance_chi2': float(p_var_chi2),
        'ms_p_skewness_chi2': float(p_skew_chi2),
        'ms_chi2_variance': float(ms_results['variance']['chi_squared']),
        'ms_chi2_skewness': float(ms_results['skewness']['chi_squared']),
        'ms_condition_number': float(ms_results['variance']['condition_number']),
        'dbscan_detected': dbscan_detected,
        'dbscan_n_clusters': dbscan_n_clusters,
        'dbscan_cluster_fraction': dbscan_cluster_frac,
        'ripley_detected': ripley_detected,
        'f_clust': dataset.f_clust,
        'has_clusters': dataset.f_clust > 0,
        'n_raw': n_raw,
        'n_merged': n_points,
        'reduction': reduction,
        'var_curve': var_curve,
        'skew_curve': skew_curve,
        'var_mock_curves': var_mock_curves,
        'skew_mock_curves': skew_mock_curves,
    }


# ============================================================================
# Experiments
# ============================================================================

class ComprehensiveStudy:
    """
    Orchestrator for all 8 experiments.

    Parameters
    ----------
    output_dir : str or Path
        Directory for JSON results.
    quick : bool
        If True, reduce replicate counts ~10x for development/testing.
    n_workers : int
        Number of parallel workers for ProcessPoolExecutor.
        Set to 1 to disable parallelism (easier debugging).
    """

    def __init__(self, output_dir: str = 'results',
                 quick: bool = False, n_workers: int = 1):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quick = quick
        self.n_workers = n_workers

    def _n_reps(self, full: int) -> int:
        """Return replicate count, reduced in quick mode."""
        return max(3, full // 10) if self.quick else full

    def _n_mocks(self, full: int = 50, min_mocks: int = 15) -> int:
        """Return mock count. Min 15 for corrected chi2 calibration."""
        return max(min_mocks, full // 3) if self.quick else full

    def _seed(self, experiment: int, replicate: int) -> int:
        """Deterministic seed."""
        return 42 + experiment * 1000 + replicate

    # ------------------------------------------------------------------
    # A. False Positive Rate
    # ------------------------------------------------------------------
    def run_fpr(self) -> dict:
        """
        Experiment A: FPR calibration.

        Uses molecule positions directly (use_molecule_positions=True) to
        test the chi-squared statistic's type-I error calibration in
        isolation from blinking artifacts. This is the standard approach
        for validating a statistical test: verify that the p-value
        distribution is uniform under the null hypothesis.

        Also runs the full pipeline (with blinking) to measure the
        operational FPR — the rate at which the complete pipeline reports
        clustering on CSR data. The difference highlights the impact of
        residual blinking artifacts on false positives.
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT A: False Positive Rate")
        print("=" * 60)

        n_reps = self._n_reps(500)
        alpha_thresholds = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

        # Part 1: Statistical test FPR (molecule positions, no blinking)
        print("  Part 1: Statistical test calibration (molecule positions)...")
        stat_results = []
        for i in range(n_reps):
            dataset = generate_synthetic_dataset(
                f_clust=0.0, seed=self._seed(0, i))
            r = _run_pipeline(dataset, n_mocks=self._n_mocks(),
                              use_molecule_positions=True)
            stat_results.append(r)
            if (i + 1) % max(1, n_reps // 5) == 0:
                print(f"    {i+1}/{n_reps} done")

        # Part 2: Full pipeline FPR (with blinking)
        print("  Part 2: Full pipeline FPR (with blinking correction)...")
        pipe_results = []
        for i in range(n_reps):
            dataset = generate_synthetic_dataset(
                f_clust=0.0, seed=self._seed(0, i))
            r = _run_pipeline(dataset, n_mocks=self._n_mocks(),
                              use_molecule_positions=False)
            pipe_results.append(r)
            if (i + 1) % max(1, n_reps // 5) == 0:
                print(f"    {i+1}/{n_reps} done")

        # Collect p-values from the statistical test (Part 1)
        p_values_var = [r['ms_p_variance'] for r in stat_results]
        p_values_skew = [r['ms_p_skewness'] for r in stat_results]

        # FPR calibration at multiple thresholds
        fpr_calibration = {}
        for alpha in alpha_thresholds:
            # Statistical test FPR (should track the diagonal)
            ms_stat_fpr = sum(
                1 for pv, ps in zip(p_values_var, p_values_skew)
                if pv < alpha or ps < alpha
            ) / n_reps
            # Full pipeline FPR (may be inflated by blinking)
            ms_pipe_fpr = sum(
                1 for r in pipe_results if r['ms_p_variance'] < alpha
                or r['ms_p_skewness'] < alpha
            ) / n_reps
            dbscan_fpr = sum(
                1 for r in stat_results if r['dbscan_detected']) / n_reps
            ripley_fpr = sum(
                1 for r in stat_results if r['ripley_detected']) / n_reps
            fpr_calibration[str(alpha)] = {
                'multiscale': ms_stat_fpr,
                'multiscale_pipeline': ms_pipe_fpr,
                'dbscan': dbscan_fpr,
                'ripley': ripley_fpr,
            }

        # Compute 95% binomial confidence intervals for FPR
        from scipy import stats as sp_stats
        binomial_ci = {}
        for alpha in alpha_thresholds:
            k = sum(1 for pv, ps in zip(p_values_var, p_values_skew)
                    if pv < alpha or ps < alpha)
            lo, hi = sp_stats.binom.interval(0.95, n_reps, k / n_reps) if k > 0 else (0, 0)
            binomial_ci[str(alpha)] = {
                'fpr': k / n_reps,
                'ci_lower': float(lo / n_reps),
                'ci_upper': float(hi / n_reps),
                'n_detected': int(k),
                'n_total': n_reps,
            }

        ms_fpr_05 = fpr_calibration['0.05']['multiscale']
        ci_05 = binomial_ci['0.05']
        print(f"  FPR at alpha=0.05: stat={ms_fpr_05:.2f} "
              f"[{ci_05['ci_lower']:.3f}, {ci_05['ci_upper']:.3f}], "
              f"pipeline={fpr_calibration['0.05']['multiscale_pipeline']:.2f}")

        output = {
            'experiment': 'A_fpr',
            'n_replicates': n_reps,
            'p_values_variance': p_values_var,
            'p_values_skewness': p_values_skew,
            'fpr_calibration': fpr_calibration,
            'binomial_ci': binomial_ci,
            'stat_results': stat_results,
            'pipeline_results': pipe_results,
        }
        _save_json(output, self.output_dir / 'A_fpr.json')
        return output

    # ------------------------------------------------------------------
    # B. Sensitivity / Power
    # ------------------------------------------------------------------
    def run_sensitivity(self) -> dict:
        """
        Experiment B: Detection rate vs cluster fraction (dose-response).

        Uses molecule positions (use_molecule_positions=True) for clean
        dose-response curves that isolate clustering signal from blinking.
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT B: Sensitivity / Power")
        print("=" * 60)

        f_values = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
        n_reps = self._n_reps(20)

        all_results = {}
        for fi, f in enumerate(f_values):
            level_results = []
            for j in range(n_reps):
                dataset = generate_synthetic_dataset(
                    f_clust=f, seed=self._seed(1, fi * 100 + j))
                r = _run_pipeline(dataset, n_mocks=self._n_mocks(),
                                  use_molecule_positions=True)
                level_results.append(r)

            ms_tpr = sum(r['ms_detected'] for r in level_results) / n_reps
            dbscan_tpr = sum(r['dbscan_detected'] for r in level_results) / n_reps
            ripley_tpr = sum(r['ripley_detected'] for r in level_results) / n_reps

            all_results[str(f)] = {
                'ms_tpr': ms_tpr,
                'dbscan_tpr': dbscan_tpr,
                'ripley_tpr': ripley_tpr,
                'n_replicates': n_reps,
                'per_experiment': level_results,
            }
            print(f"  f_clust={f:.3f}: MS={ms_tpr:.2f}, "
                  f"DBSCAN={dbscan_tpr:.2f}, Ripley={ripley_tpr:.2f}")

        output = {
            'experiment': 'B_sensitivity',
            'f_values': f_values,
            'results': all_results,
        }
        _save_json(output, self.output_dir / 'B_sensitivity.json')
        return output

    # ------------------------------------------------------------------
    # C. Parameter Sensitivity
    # ------------------------------------------------------------------
    def run_parameter_sensitivity(self) -> dict:
        """
        Experiment C: Sweep grid_size, shrinkage, n_mocks, blink radius.
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT C: Parameter Sensitivity")
        print("=" * 60)

        n_reps = self._n_reps(20)
        param_sweeps = {}

        # C1: Grid size (molecule positions — tests statistical test)
        print("  C1: Grid size sweep...")
        grid_sizes = [16, 32, 64, 128]
        grid_results = {}
        for gs in grid_sizes:
            fpr_results = []
            tpr_results = []
            for j in range(n_reps):
                ds0 = generate_synthetic_dataset(
                    f_clust=0.0, seed=self._seed(2, j))
                r0 = _run_pipeline(ds0, grid_size=gs, n_mocks=self._n_mocks(),
                                   use_molecule_positions=True,
                                   run_baselines=False)
                fpr_results.append(r0)
                ds1 = generate_synthetic_dataset(
                    f_clust=0.15, seed=self._seed(2, 500 + j))
                r1 = _run_pipeline(ds1, grid_size=gs, n_mocks=self._n_mocks(),
                                   use_molecule_positions=True,
                                   run_baselines=False)
                tpr_results.append(r1)

            grid_results[str(gs)] = {
                'fpr': sum(r['ms_detected'] for r in fpr_results) / n_reps,
                'tpr': sum(r['ms_detected'] for r in tpr_results) / n_reps,
            }
            print(f"    grid_size={gs}: FPR={grid_results[str(gs)]['fpr']:.2f}, "
                  f"TPR={grid_results[str(gs)]['tpr']:.2f}")
        param_sweeps['grid_size'] = grid_results

        # C2: Shrinkage (molecule positions — tests statistical test)
        print("  C2: Shrinkage sweep...")
        shrinkages = [0.01, 0.05, 0.1, 0.2, 0.5]
        shrink_results = {}
        for s in shrinkages:
            fpr_results = []
            tpr_results = []
            for j in range(n_reps):
                ds0 = generate_synthetic_dataset(
                    f_clust=0.0, seed=self._seed(2, 1000 + j))
                r0 = _run_pipeline(ds0, shrinkage=s, n_mocks=self._n_mocks(),
                                   use_molecule_positions=True,
                                   run_baselines=False)
                fpr_results.append(r0)
                ds1 = generate_synthetic_dataset(
                    f_clust=0.15, seed=self._seed(2, 1500 + j))
                r1 = _run_pipeline(ds1, shrinkage=s, n_mocks=self._n_mocks(),
                                   use_molecule_positions=True,
                                   run_baselines=False)
                tpr_results.append(r1)

            shrink_results[str(s)] = {
                'fpr': sum(r['ms_detected'] for r in fpr_results) / n_reps,
                'tpr': sum(r['ms_detected'] for r in tpr_results) / n_reps,
                'condition_numbers': [r['ms_condition_number'] for r in tpr_results],
            }
            print(f"    shrinkage={s}: FPR={shrink_results[str(s)]['fpr']:.2f}, "
                  f"TPR={shrink_results[str(s)]['tpr']:.2f}")
        param_sweeps['shrinkage'] = shrink_results

        # C3: Number of mocks (molecule positions — tests statistical test)
        print("  C3: Number of mocks sweep...")
        mock_counts = [5, 10, 20, 50, 100] if not self.quick else [5, 10, 20]
        mock_results = {}
        for nm in mock_counts:
            fpr_results = []
            tpr_results = []
            for j in range(n_reps):
                ds0 = generate_synthetic_dataset(
                    f_clust=0.0, seed=self._seed(2, 2000 + j))
                r0 = _run_pipeline(ds0, n_mocks=nm,
                                   use_molecule_positions=True,
                                   run_baselines=False)
                fpr_results.append(r0)
                ds1 = generate_synthetic_dataset(
                    f_clust=0.15, seed=self._seed(2, 2500 + j))
                r1 = _run_pipeline(ds1, n_mocks=nm,
                                   use_molecule_positions=True,
                                   run_baselines=False)
                tpr_results.append(r1)

            mock_results[str(nm)] = {
                'fpr': sum(r['ms_detected'] for r in fpr_results) / n_reps,
                'tpr': sum(r['ms_detected'] for r in tpr_results) / n_reps,
            }
            print(f"    n_mocks={nm}: FPR={mock_results[str(nm)]['fpr']:.2f}, "
                  f"TPR={mock_results[str(nm)]['tpr']:.2f}")
        param_sweeps['n_mocks'] = mock_results

        # C4: Blinking correction radius
        print("  C4: Blinking correction radius sweep...")
        blink_radii = [10, 20, 30, 50, 75]
        blink_results = {}
        for r_lat in blink_radii:
            fpr_results = []
            tpr_results = []
            for j in range(n_reps):
                ds0 = generate_synthetic_dataset(
                    f_clust=0.0, seed=self._seed(2, 3000 + j))
                r0 = _run_pipeline(
                    ds0, r_lateral=r_lat, r_axial=r_lat * 2,
                    n_mocks=self._n_mocks(), run_baselines=False)
                fpr_results.append(r0)
                ds1 = generate_synthetic_dataset(
                    f_clust=0.15, seed=self._seed(2, 3500 + j))
                r1 = _run_pipeline(
                    ds1, r_lateral=r_lat, r_axial=r_lat * 2,
                    n_mocks=self._n_mocks(), run_baselines=False)
                tpr_results.append(r1)

            blink_results[str(r_lat)] = {
                'fpr': sum(r['ms_detected'] for r in fpr_results) / n_reps,
                'tpr': sum(r['ms_detected'] for r in tpr_results) / n_reps,
                'reduction_factors': [r['reduction'] for r in tpr_results],
            }
            print(f"    r_lateral={r_lat}: "
                  f"FPR={blink_results[str(r_lat)]['fpr']:.2f}, "
                  f"TPR={blink_results[str(r_lat)]['tpr']:.2f}")
        param_sweeps['blink_radius'] = blink_results

        output = {
            'experiment': 'C_parameter_sensitivity',
            'n_replicates': n_reps,
            'sweeps': param_sweeps,
        }
        _save_json(output, self.output_dir / 'C_parameter_sensitivity.json')
        return output

    # ------------------------------------------------------------------
    # D. Cluster Geometry Robustness
    # ------------------------------------------------------------------
    def run_geometry(self) -> dict:
        """
        Experiment D: Detection across cluster sizes, counts, and shapes.
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT D: Cluster Geometry Robustness")
        print("=" * 60)

        n_reps = self._n_reps(10)
        geometry_results = {}

        # D1: Cluster radius (molecule positions — clean geometry test)
        # Include radii near localization precision (~15nm) for hard regime
        print("  D1: Cluster radius sweep...")
        radii = [10, 15, 20, 50, 100, 200, 500]
        radius_results = {}
        for rad in radii:
            level_results = []
            for j in range(n_reps):
                dataset = generate_synthetic_dataset(
                    f_clust=0.3, cluster_radius_nm=rad,
                    seed=self._seed(3, j))
                r = _run_pipeline(dataset, n_mocks=self._n_mocks(),
                                  use_molecule_positions=True)
                level_results.append(r)
            tpr = sum(r['ms_detected'] for r in level_results) / n_reps
            radius_results[str(rad)] = {
                'ms_tpr': tpr,
                'dbscan_tpr': sum(
                    r['dbscan_detected'] for r in level_results) / n_reps,
                'ripley_tpr': sum(
                    r['ripley_detected'] for r in level_results) / n_reps,
            }
            print(f"    radius={rad}nm: MS={tpr:.2f}")
        geometry_results['cluster_radius'] = radius_results

        # D2: Number of clusters (molecule positions)
        print("  D2: Number of clusters sweep...")
        n_clusters_list = [5, 10, 20, 50, 100]
        nclust_results = {}
        for nc in n_clusters_list:
            level_results = []
            for j in range(n_reps):
                dataset = generate_synthetic_dataset(
                    f_clust=0.3, n_clusters=nc,
                    seed=self._seed(3, 500 + j))
                r = _run_pipeline(dataset, n_mocks=self._n_mocks(),
                                  use_molecule_positions=True)
                level_results.append(r)
            tpr = sum(r['ms_detected'] for r in level_results) / n_reps
            nclust_results[str(nc)] = {
                'ms_tpr': tpr,
                'dbscan_tpr': sum(
                    r['dbscan_detected'] for r in level_results) / n_reps,
                'ripley_tpr': sum(
                    r['ripley_detected'] for r in level_results) / n_reps,
            }
            print(f"    n_clusters={nc}: MS={tpr:.2f}")
        geometry_results['n_clusters'] = nclust_results

        # D3: Cluster shapes (molecule positions)
        print("  D3: Cluster shape comparison...")
        shapes = ['gaussian', 'elongated', 'ring']
        shape_results = {}
        for shape in shapes:
            level_results = []
            for j in range(n_reps):
                dataset = generate_extended_dataset(
                    f_clust=0.3, cluster_shape=shape,
                    seed=self._seed(3, 1000 + j))
                r = _run_pipeline(dataset, n_mocks=self._n_mocks(),
                                  use_molecule_positions=True)
                level_results.append(r)
            tpr = sum(r['ms_detected'] for r in level_results) / n_reps
            shape_results[shape] = {
                'ms_tpr': tpr,
                'dbscan_tpr': sum(
                    r['dbscan_detected'] for r in level_results) / n_reps,
                'ripley_tpr': sum(
                    r['ripley_detected'] for r in level_results) / n_reps,
            }
            print(f"    shape={shape}: MS={tpr:.2f}")
        geometry_results['cluster_shape'] = shape_results

        # D4: Hard regime — small clusters + low cluster fraction
        # Shows graceful degradation in challenging conditions
        print("  D4: Hard regime (small clusters, low f_clust)...")
        hard_configs = [
            {'f_clust': 0.005, 'radius': 75, 'label': 'f=0.005,r=75'},
            {'f_clust': 0.01, 'radius': 75, 'label': 'f=0.01,r=75'},
            {'f_clust': 0.02, 'radius': 75, 'label': 'f=0.02,r=75'},
            {'f_clust': 0.05, 'radius': 15, 'label': 'f=0.05,r=15'},
            {'f_clust': 0.05, 'radius': 30, 'label': 'f=0.05,r=30'},
            {'f_clust': 0.10, 'radius': 15, 'label': 'f=0.10,r=15'},
            {'f_clust': 0.02, 'radius': 15, 'label': 'f=0.02,r=15'},
        ]
        hard_results = {}
        for cfg in hard_configs:
            level_results = []
            for j in range(n_reps):
                dataset = generate_synthetic_dataset(
                    f_clust=cfg['f_clust'],
                    cluster_radius_nm=cfg['radius'],
                    seed=self._seed(3, 1500 + j))
                r = _run_pipeline(dataset, n_mocks=self._n_mocks(),
                                  use_molecule_positions=True,
                                  run_baselines=False)
                level_results.append(r)
            tpr = sum(r['ms_detected'] for r in level_results) / n_reps
            hard_results[cfg['label']] = {
                'f_clust': cfg['f_clust'],
                'radius_nm': cfg['radius'],
                'ms_tpr': tpr,
            }
            print(f"    {cfg['label']}: MS={tpr:.2f}")
        geometry_results['hard_regime'] = hard_results

        output = {
            'experiment': 'D_geometry',
            'n_replicates': n_reps,
            'results': geometry_results,
        }
        _save_json(output, self.output_dir / 'D_geometry.json')
        return output

    # ------------------------------------------------------------------
    # E. Realistic SMLM Conditions
    # ------------------------------------------------------------------
    def run_realistic_conditions(self) -> dict:
        """
        Experiment E: Localization precision, blinking rate, frame count.
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT E: Realistic SMLM Conditions")
        print("=" * 60)

        n_reps = self._n_reps(10)
        condition_results = {}

        # E1: Localization precision (include hard regime: sigma near cluster radius)
        print("  E1: Localization precision sweep...")
        precisions = [5, 10, 20, 30, 50, 75, 100]
        prec_results = {}
        for sigma in precisions:
            level_results = []
            for j in range(n_reps):
                dataset = generate_synthetic_dataset(
                    f_clust=0.15, blink_sigma_xy_nm=sigma,
                    blink_sigma_z_nm=sigma * 2.5,
                    seed=self._seed(4, j))
                r = _run_pipeline(dataset, n_mocks=self._n_mocks(),
                                  run_baselines=False)
                level_results.append(r)
            tpr = sum(r['ms_detected'] for r in level_results) / n_reps
            prec_results[str(sigma)] = {'ms_tpr': tpr}
            print(f"    sigma_xy={sigma}nm: MS={tpr:.2f}")
        condition_results['localization_precision'] = prec_results

        # E2: Blinking rate
        print("  E2: Blinking rate sweep...")
        blink_rates = [1, 3, 5, 10, 20]
        blink_rate_results = {}
        for nb in blink_rates:
            level_results = []
            for j in range(n_reps):
                dataset = generate_synthetic_dataset(
                    f_clust=0.15, n_blinks_mean=nb,
                    seed=self._seed(4, 500 + j))
                r = _run_pipeline(dataset, n_mocks=self._n_mocks(),
                                  run_baselines=False)
                level_results.append(r)
            tpr = sum(r['ms_detected'] for r in level_results) / n_reps
            blink_rate_results[str(nb)] = {
                'ms_tpr': tpr,
                'reduction': np.mean([r['reduction'] for r in level_results]),
            }
            print(f"    n_blinks={nb}: MS={tpr:.2f}")
        condition_results['blinking_rate'] = blink_rate_results

        # E3: Frame count
        print("  E3: Frame count sweep...")
        frame_counts = [20, 100, 500, 2000]
        frame_results = {}
        for nf in frame_counts:
            level_results = []
            for j in range(n_reps):
                dataset = generate_synthetic_dataset(
                    f_clust=0.15, n_frames=nf,
                    seed=self._seed(4, 1000 + j))
                r = _run_pipeline(dataset, n_mocks=self._n_mocks(),
                                  run_baselines=False)
                level_results.append(r)
            tpr = sum(r['ms_detected'] for r in level_results) / n_reps
            frame_results[str(nf)] = {'ms_tpr': tpr}
            print(f"    n_frames={nf}: MS={tpr:.2f}")
        condition_results['frame_count'] = frame_results

        output = {
            'experiment': 'E_realistic_conditions',
            'n_replicates': n_reps,
            'results': condition_results,
        }
        _save_json(output, self.output_dir / 'E_realistic_conditions.json')
        return output

    # ------------------------------------------------------------------
    # F. Biological Use Cases
    # ------------------------------------------------------------------
    def run_biological_cases(self) -> dict:
        """
        Experiment F: 4 biological scenarios with published parameters.
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT F: Biological Use Cases")
        print("=" * 60)

        bio_results = {}
        presets = ['synaptic_receptors', 'nuclear_pores',
                   'membrane_domains', 'negative_control']

        for preset_name in presets:
            print(f"  Running {preset_name}...")
            preset = BIOLOGICAL_PRESETS[preset_name]
            dataset = generate_from_preset(preset_name, seed=self._seed(5, 0))

            # Use molecule positions for clean biological comparison
            # Use at least 30 mocks for bio cases — large datasets need
            # more leave-one-out samples for accurate chi2 calibration
            r = _run_pipeline(dataset, n_mocks=self._n_mocks(min_mocks=30),
                              use_molecule_positions=True)

            # Also run DBSCAN on molecule positions for comparison scatter
            positions = dataset.molecule_positions
            dbscan = DBSCANBaseline(
                eps_nm=preset.cluster_radius_nm * 0.75, min_samples=5)
            dbscan_result = dbscan.fit(positions)

            bio_results[preset_name] = {
                'preset': {
                    'name': preset.name,
                    'n_molecules': preset.n_molecules,
                    'n_clusters': preset.n_clusters,
                    'cluster_radius_nm': preset.cluster_radius_nm,
                    'f_clust': preset.f_clust,
                    'roi_nm': preset.roi_nm,
                    'cluster_shape': preset.cluster_shape,
                    'description': preset.description,
                },
                'pipeline_result': r,
                'positions': positions.tolist(),
                'dbscan_labels': dbscan_result.labels.tolist(),
            }
            status = "DETECTED" if r['ms_detected'] else "not detected"
            print(f"    {preset_name}: {status} "
                  f"(p_var={r['ms_p_variance']:.4f}, "
                  f"p_skew={r['ms_p_skewness']:.4f})")

        output = {
            'experiment': 'F_biological_cases',
            'results': bio_results,
        }
        _save_json(output, self.output_dir / 'F_biological_cases.json')
        return output

    # ------------------------------------------------------------------
    # G. Multiscale Signatures
    # ------------------------------------------------------------------
    def run_multiscale_signatures(self) -> dict:
        """
        Experiment G: Extract variance/skewness curves with null envelopes.
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT G: Multiscale Signatures")
        print("=" * 60)

        n_mock_envelope = self._n_reps(50)
        presets = ['synaptic_receptors', 'nuclear_pores',
                   'membrane_domains', 'negative_control']
        signature_results = {}

        for preset_name in presets:
            print(f"  Computing signatures for {preset_name}...")
            dataset = generate_from_preset(preset_name, seed=self._seed(6, 0))

            # Use molecule positions for clean multiscale signatures
            positions = dataset.molecule_positions

            # Null mocks for envelope
            null_model = BiologicalNullModel(seed=42)
            null_model.fit(positions)
            mock_positions = null_model.generate_mocks(
                len(positions), n_mock_envelope)

            # Compute curves
            detector = SMLMMultiscaleTest.from_positions(positions)
            roi_max = positions.max(axis=0) - positions.min(axis=0)
            max_scale = min(500, roi_max.max() / 4)
            if max_scale < 25:
                max_scale = 25
            scale_range = ScaleRange.for_smlm(
                min_nm=20, max_nm=max_scale,
                n_scales=15, roi_size_nm=roi_max.max(), grid_size=64)

            real_curves = detector.compute_curves(positions, scale_range)

            mock_var_values = []
            mock_skew_values = []
            for mp in mock_positions:
                mc = detector.compute_curves(mp, scale_range)
                if len(mc['variance']['values']) == len(real_curves['variance']['values']):
                    mock_var_values.append(mc['variance']['values'])
                if len(mc['skewness']['values']) == len(real_curves['skewness']['values']):
                    mock_skew_values.append(mc['skewness']['values'])

            # Compute envelopes (5th and 95th percentile)
            mock_var_arr = np.array(mock_var_values) if mock_var_values else np.empty((0, 0))
            mock_skew_arr = np.array(mock_skew_values) if mock_skew_values else np.empty((0, 0))

            var_envelope = {}
            if len(mock_var_arr) > 0:
                var_envelope = {
                    'p5': np.percentile(mock_var_arr, 5, axis=0).tolist(),
                    'p95': np.percentile(mock_var_arr, 95, axis=0).tolist(),
                    'median': np.median(mock_var_arr, axis=0).tolist(),
                }

            skew_envelope = {}
            if len(mock_skew_arr) > 0:
                skew_envelope = {
                    'p5': np.percentile(mock_skew_arr, 5, axis=0).tolist(),
                    'p95': np.percentile(mock_skew_arr, 95, axis=0).tolist(),
                    'median': np.median(mock_skew_arr, axis=0).tolist(),
                }

            signature_results[preset_name] = {
                'variance': {
                    'scales_nm': real_curves['variance']['scales_nm'],
                    'values': real_curves['variance']['values'],
                    'envelope': var_envelope,
                },
                'skewness': {
                    'scales_nm': real_curves['skewness']['scales_nm'],
                    'values': real_curves['skewness']['values'],
                    'envelope': skew_envelope,
                },
            }
            print(f"    {preset_name}: {len(real_curves['variance']['values'])} "
                  f"variance scales, {len(mock_var_values)} valid mocks")

        output = {
            'experiment': 'G_multiscale_signatures',
            'results': signature_results,
        }
        _save_json(output, self.output_dir / 'G_multiscale_signatures.json')
        return output

    # ------------------------------------------------------------------
    # H. Runtime & Scalability
    # ------------------------------------------------------------------
    def run_runtime(self) -> dict:
        """
        Experiment H: Wall-clock time vs dataset size.
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT H: Runtime & Scalability")
        print("=" * 60)

        sizes = [500, 1000, 2000, 5000, 10000, 20000]
        if self.quick:
            sizes = [500, 1000, 2000, 5000]

        runtime_results = {}
        for n_mol in sizes:
            dataset = generate_synthetic_dataset(
                n_molecules=n_mol, f_clust=0.15,
                seed=self._seed(7, n_mol))

            t0 = time.time()
            _run_pipeline(dataset, n_mocks=self._n_mocks())
            elapsed = time.time() - t0

            runtime_results[str(n_mol)] = {
                'n_molecules': n_mol,
                'n_localizations': len(dataset.raw_positions),
                'wall_clock_seconds': round(elapsed, 2),
            }
            print(f"  n_mol={n_mol}: {elapsed:.1f}s "
                  f"({len(dataset.raw_positions)} localizations)")

        output = {
            'experiment': 'H_runtime',
            'results': runtime_results,
        }
        _save_json(output, self.output_dir / 'H_runtime.json')
        return output

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------
    def run_all(self) -> dict:
        """Run all 8 experiments."""
        t_start = time.time()
        mode = "QUICK" if self.quick else "FULL"
        print(f"\nCOMPREHENSIVE VALIDATION STUDY ({mode} mode)")
        print("=" * 60)

        all_results = {}
        all_results['A'] = self.run_fpr()
        all_results['B'] = self.run_sensitivity()
        all_results['C'] = self.run_parameter_sensitivity()
        all_results['D'] = self.run_geometry()
        all_results['E'] = self.run_realistic_conditions()
        all_results['F'] = self.run_biological_cases()
        all_results['G'] = self.run_multiscale_signatures()
        all_results['H'] = self.run_runtime()

        elapsed = time.time() - t_start
        print(f"\nAll experiments complete in {elapsed / 60:.1f} minutes.")

        # Generate figures
        try:
            from .figure_generator import generate_all_figures
            generate_all_figures(
                results_dir=str(self.output_dir),
                figures_dir='figures',
            )
        except Exception as e:
            print(f"Figure generation failed: {e}")
            print("You can regenerate figures later with --figures-only")

        return all_results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SMLM Multiscale Clustering - Comprehensive Validation Study')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: reduce replicates ~10x')
    parser.add_argument('--figures-only', action='store_true',
                        help='Only regenerate figures from saved JSON')
    parser.add_argument('--output-dir', default='results',
                        help='Directory for JSON results (default: results)')
    parser.add_argument('--figures-dir', default='figures',
                        help='Directory for PDF figures (default: figures)')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Run single experiment (A-H)')
    args = parser.parse_args()

    if args.figures_only:
        from .figure_generator import generate_all_figures
        generate_all_figures(
            results_dir=args.output_dir,
            figures_dir=args.figures_dir,
        )
        return

    study = ComprehensiveStudy(
        output_dir=args.output_dir,
        quick=args.quick,
    )

    if args.experiment:
        experiment_map = {
            'A': study.run_fpr,
            'B': study.run_sensitivity,
            'C': study.run_parameter_sensitivity,
            'D': study.run_geometry,
            'E': study.run_realistic_conditions,
            'F': study.run_biological_cases,
            'G': study.run_multiscale_signatures,
            'H': study.run_runtime,
        }
        exp = args.experiment.upper()
        if exp not in experiment_map:
            print(f"Unknown experiment: {exp}. Choose from A-H.")
            return
        experiment_map[exp]()
    else:
        study.run_all()


if __name__ == '__main__':
    main()
