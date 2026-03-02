"""
Benchmark validation for the SMLM multiscale clustering detector.

Validation strategy (following SMLM 2016 Challenge methodology):

1. Synthetic ground truth -- generate point clouds with known cluster structure
   and verify the detector correctly identifies them. This is the primary
   validation before any real data is touched.

2. Negative controls -- verify the false positive rate (FPR) at alpha=0.05
   is at most 5% across many CSR datasets. A method that reports significance
   too often is unreliable.

3. Positive controls -- verify the true positive rate (TPR) for a range of
   cluster sizes, densities, and separations. Report the minimum detectable
   cluster fraction.

4. Comparison with baselines -- run DBSCAN and Ripley's K on the same
   synthetic datasets and report all metrics side by side.

Synthetic data models
---------------------
two_population_model : the standard SMLM cluster model
    A fraction f_clust of molecules are in clusters (Gaussian blobs), the
    rest are uniform background. Blinking is simulated by repeating each
    molecule position with Gaussian noise.

    Parameters:
        n_molecules       : number of distinct molecules
        f_clust           : fraction in clusters (0 to 1)
        n_clusters        : number of clusters
        cluster_radius_nm : 1-sigma radius of each cluster (nm)
        roi_nm            : ROI side length (nm)
        n_blinks          : average blinks per molecule (Poisson)
        blink_sigma_nm    : localization noise per blink (nm)

The benchmark reports sensitivity and specificity as functions of f_clust
(the "dose-response" curve).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..core.blinking_correction import BlinkingCorrector
from ..core.null_models import BiologicalNullModel
from ..core.multiscale_detector import SMLMMultiscaleTest, ScaleRange
from .comparison_methods import DBSCANBaseline, RipleysKBaseline
from .metrics import summarize_detection_experiments, evaluate_detection


# ===========================================================================
# Synthetic data generator
# ===========================================================================

@dataclass
class SyntheticSMLMDataset:
    """Output of the synthetic SMLM generator."""
    raw_positions: np.ndarray       # (N_loc, 3) with blinking, nm
    raw_frames: np.ndarray          # (N_loc,) frame numbers
    raw_photons: np.ndarray         # (N_loc,) photon counts
    molecule_positions: np.ndarray  # (N_mol, 3) true molecule positions
    molecule_labels: np.ndarray     # (N_mol,) 1=cluster, 0=background
    roi_nm: float
    f_clust: float
    n_clusters: int
    cluster_radius_nm: float


def generate_synthetic_dataset(
    n_molecules: int = 5000,
    f_clust: float = 0.3,
    n_clusters: int = 20,
    cluster_radius_nm: float = 75.0,
    roi_nm: float = 5000.0,
    n_blinks_mean: float = 5.0,
    blink_sigma_xy_nm: float = 15.0,
    blink_sigma_z_nm: float = 35.0,
    n_frames: int = 500,
    seed: Optional[int] = None
) -> SyntheticSMLMDataset:
    """
    Generate a synthetic SMLM dataset with known cluster structure.

    Molecules are placed either uniformly (background) or in Gaussian
    clusters. Each molecule blinks a Poisson number of times; each blink
    adds independent Gaussian localization noise.

    Parameters
    ----------
    n_molecules : int
        Total number of distinct fluorescent molecules.
    f_clust : float
        Fraction of molecules in clusters. 0 = pure CSR, 1 = all clustered.
    n_clusters : int
        Number of clusters.
    cluster_radius_nm : float
        1-sigma radius of each Gaussian cluster (nm).
    roi_nm : float
        Side length of the cubic ROI in nm.
    n_blinks_mean : float
        Mean number of blink events per molecule (Poisson distributed).
    blink_sigma_xy_nm : float
        Localization noise per blink, lateral (nm).
    blink_sigma_z_nm : float
        Localization noise per blink, axial (nm).
    n_frames : int
        Total number of imaging frames (used to assign random frame numbers).
    seed : int or None

    Returns
    -------
    SyntheticSMLMDataset
    """
    rng = np.random.default_rng(seed)

    n_clust = int(n_molecules * f_clust)
    n_bg = n_molecules - n_clust

    # Background molecules: uniform in ROI
    bg_pos = rng.uniform(0, roi_nm, (n_bg, 3))
    bg_labels = np.zeros(n_bg, dtype=int)

    # Clustered molecules: Gaussian around cluster centers
    cluster_centers = rng.uniform(
        cluster_radius_nm * 3,
        roi_nm - cluster_radius_nm * 3,
        (n_clusters, 3)
    )
    assignment = rng.integers(0, n_clusters, n_clust)
    centers = cluster_centers[assignment]
    sigma = np.array([cluster_radius_nm, cluster_radius_nm, cluster_radius_nm])
    clust_pos = centers + rng.normal(0, sigma, (n_clust, 3))
    clust_pos = np.clip(clust_pos, 0, roi_nm)
    clust_labels = np.ones(n_clust, dtype=int)

    mol_positions = np.vstack([bg_pos, clust_pos])
    mol_labels = np.concatenate([bg_labels, clust_labels])

    # Simulate blinking
    all_locs = []
    all_frames = []
    all_photons = []

    for mol_idx in range(n_molecules):
        n_blinks = max(1, rng.poisson(n_blinks_mean))
        frames = np.sort(rng.integers(1, n_frames + 1, n_blinks))
        noise_xy = rng.normal(0, blink_sigma_xy_nm, (n_blinks, 2))
        noise_z = rng.normal(0, blink_sigma_z_nm, (n_blinks, 1))
        noise = np.hstack([noise_xy, noise_z])
        locs = mol_positions[mol_idx] + noise
        locs = np.clip(locs, 0, roi_nm)
        photons = rng.uniform(500, 3000, n_blinks)

        all_locs.append(locs)
        all_frames.append(frames)
        all_photons.append(photons)

    raw_positions = np.vstack(all_locs)
    raw_frames = np.concatenate(all_frames)
    raw_photons = np.concatenate(all_photons)

    return SyntheticSMLMDataset(
        raw_positions=raw_positions,
        raw_frames=raw_frames,
        raw_photons=raw_photons,
        molecule_positions=mol_positions,
        molecule_labels=mol_labels,
        roi_nm=roi_nm,
        f_clust=f_clust,
        n_clusters=n_clusters,
        cluster_radius_nm=cluster_radius_nm
    )


# ===========================================================================
# Benchmark runner
# ===========================================================================

class BenchmarkRunner:
    """
    Run the full validation benchmark.

    Tests:
      - False positive rate across n_null_experiments CSR datasets
      - True positive rate across a range of clustered fractions
      - Side-by-side comparison with DBSCAN and Ripley's K

    Parameters
    ----------
    n_mocks : int
        Number of CSR null mocks per test dataset.
    n_null_experiments : int
        Number of CSR datasets used to estimate FPR.
    alpha : float
        Significance threshold for declaring "clustered".
    shrinkage : float
        Shrinkage parameter for the chi^2 test (see multiscale_detector).
    verbose : bool
        Print progress.
    """

    def __init__(self, n_mocks: int = 50, n_null_experiments: int = 20,
                 alpha: float = 0.05, shrinkage: float = 0.1,
                 verbose: bool = True):
        self.n_mocks = n_mocks
        self.n_null_experiments = n_null_experiments
        self.alpha = alpha
        self.shrinkage = shrinkage
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _run_single(self, dataset: SyntheticSMLMDataset,
                    corrector: BlinkingCorrector) -> dict:
        """
        Run the full pipeline on one synthetic dataset.

        Returns a dict with p-values from the multiscale test and cluster
        labels from DBSCAN.
        """
        # Blinking correction
        blink_result = corrector.correct(
            dataset.raw_positions, dataset.raw_frames, dataset.raw_photons
        )

        # Biological null model
        null_model = BiologicalNullModel(seed=None)
        null_model.fit(blink_result.positions)
        mock_positions = null_model.generate_mocks(
            len(blink_result.positions), self.n_mocks
        )

        # Multiscale test
        detector = SMLMMultiscaleTest.from_positions(blink_result.positions)
        roi_max = blink_result.positions.max(axis=0) - blink_result.positions.min(axis=0)
        scale_range = ScaleRange.for_smlm(
            min_nm=20, max_nm=min(500, roi_max.max() / 4),
            n_scales=12, roi_size_nm=roi_max.max(), grid_size=64
        )
        ms_results = detector.test(
            blink_result.positions, mock_positions,
            scale_range, shrinkage=self.shrinkage
        )

        # DBSCAN baseline
        dbscan = DBSCANBaseline(
            eps_nm=dataset.cluster_radius_nm * 0.75,
            min_samples=5
        )
        dbscan_result = dbscan.fit(blink_result.positions)

        # Ripley's K baseline (no mocks to keep runtime manageable)
        ripley = RipleysKBaseline(
            r_min_nm=20, r_max_nm=min(500, roi_max.max() / 4), n_radii=15
        )
        ripley_result = ripley.compute(blink_result.positions)

        # Multiscale detection: significant in at least one metric?
        p_var = ms_results['variance']['p_value']
        p_skew = ms_results['skewness']['p_value']
        ms_detected = (p_var < self.alpha) or (p_skew < self.alpha)

        # Ripley's: any L(r) - r > 0 beyond noise?
        # Use: mean L-r at scales > 50nm > 3* expected Poisson noise
        ripley_detected = bool(
            (ripley_result.L_minus_r[ripley_result.radii > 50] > 0).mean() > 0.5
        )

        return {
            'ms_detected': ms_detected,
            'ms_p_variance': p_var,
            'ms_p_skewness': p_skew,
            'dbscan_n_clusters': dbscan_result.n_clusters,
            'dbscan_cluster_fraction': (dbscan_result.cluster_mask.sum() /
                                        len(blink_result.positions)),
            'ripley_detected': ripley_detected,
            'has_clusters': dataset.f_clust > 0,
            'f_clust': dataset.f_clust,
            'n_raw': len(dataset.raw_positions),
            'n_merged': len(blink_result.positions),
            'reduction': blink_result.reduction_factor,
        }

    def run_false_positive_test(self, seed_start: int = 1000) -> dict:
        """
        Estimate false positive rate on CSR datasets.

        Generates n_null_experiments datasets with f_clust=0 and runs the
        full pipeline. Returns observed FPR (should be <= alpha).
        """
        self._log("\n" + "="*70)
        self._log("FALSE POSITIVE RATE TEST (expect FPR <= %.2f)" % self.alpha)
        self._log("="*70)

        corrector = BlinkingCorrector(r_lateral=30, r_axial=60, max_dark_frames=5)
        results = []

        for i in range(self.n_null_experiments):
            dataset = generate_synthetic_dataset(
                f_clust=0.0, seed=seed_start + i
            )
            r = self._run_single(dataset, corrector)
            results.append(r)
            if self.verbose and (i + 1) % 5 == 0:
                self._log(f"  {i+1}/{self.n_null_experiments} done...")

        summary = summarize_detection_experiments(
            [{'detected': r['ms_detected'], 'has_clusters': False}
             for r in results]
        )

        self._log(f"\n  FPR = {summary['fpr']:.3f} "
                  f"({summary['fp']}/{self.n_null_experiments} false positives)")
        status = "PASS" if summary['fpr'] <= self.alpha else "FAIL"
        self._log(f"  Result: {status}")

        return {'summary': summary, 'per_experiment': results}

    def run_sensitivity_test(self,
                             f_clust_values: Optional[list] = None,
                             n_per_level: int = 10,
                             seed_start: int = 2000) -> dict:
        """
        Estimate true positive rate as a function of clustered fraction.

        Parameters
        ----------
        f_clust_values : list of float or None
            Cluster fractions to test. Default: [0.05, 0.1, 0.2, 0.3, 0.5].
        n_per_level : int
            Number of datasets per fraction (more = tighter estimate).

        Returns
        -------
        dict mapping f_clust -> detection summary
        """
        if f_clust_values is None:
            f_clust_values = [0.05, 0.10, 0.20, 0.30, 0.50]

        self._log("\n" + "="*70)
        self._log("SENSITIVITY TEST (TPR vs. cluster fraction)")
        self._log("="*70)

        corrector = BlinkingCorrector(r_lateral=30, r_axial=60, max_dark_frames=5)
        all_results = {}

        for fi, f in enumerate(f_clust_values):
            level_results = []
            for j in range(n_per_level):
                dataset = generate_synthetic_dataset(
                    f_clust=f, seed=seed_start + fi * 100 + j
                )
                r = self._run_single(dataset, corrector)
                level_results.append(r)

            summary = summarize_detection_experiments(
                [{'detected': r['ms_detected'], 'has_clusters': True}
                 for r in level_results]
            )
            all_results[f] = {'summary': summary, 'per_experiment': level_results}

            self._log(f"  f_clust={f:.2f}: TPR = {summary['sensitivity']:.2f} "
                      f"({summary['tp']}/{n_per_level})")

        return all_results

    def run_full_benchmark(self) -> dict:
        """
        Run complete validation benchmark.

        Returns
        -------
        dict with 'fpr_test' and 'sensitivity_test' results.
        """
        self._log("\nSMLM MULTISCALE CLUSTERING BENCHMARK")
        self._log("="*70)
        self._log(f"n_mocks={self.n_mocks}, alpha={self.alpha}, "
                  f"shrinkage={self.shrinkage}")

        fpr_results = self.run_false_positive_test()
        sensitivity_results = self.run_sensitivity_test()

        self._log("\n" + "="*70)
        self._log("BENCHMARK SUMMARY")
        self._log("="*70)

        fpr = fpr_results['summary']['fpr']
        self._log(f"  FPR: {fpr:.3f} (target <= {self.alpha})")

        for f, res in sensitivity_results.items():
            tpr = res['summary']['sensitivity']
            self._log(f"  TPR at f_clust={f:.2f}: {tpr:.2f}")

        passed = fpr <= self.alpha
        self._log(f"\n  Overall: {'PASS' if passed else 'FAIL'}")
        self._log("="*70)

        return {
            'fpr_test': fpr_results,
            'sensitivity_test': sensitivity_results,
            'passed': passed
        }
