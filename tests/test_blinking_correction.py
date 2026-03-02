"""
Unit tests for blinking correction.

Coverage:
- Identical position, same frames -> single cluster
- Identical positions, adjacent frames -> single cluster
- Identical positions, dark gap within threshold -> merged
- Identical positions, dark gap exceeding threshold -> separate
- Spatial separation exceeding threshold -> separate
- Anisotropic threshold respected (lateral vs axial)
- Photon-weighted centroid correctness
- Output dtype and shape contracts
- Edge cases: single point, all same molecule, no blinking
"""

import pytest
import numpy as np
from ..core.blinking_correction import BlinkingCorrector, BlinkerResult


class TestBlinkingCorrectorBasic:

    def setup_method(self):
        self.corrector = BlinkingCorrector(
            r_lateral=30.0, r_axial=60.0, max_dark_frames=5
        )

    def test_single_localization_returns_one_molecule(self):
        positions = np.array([[100.0, 200.0, 50.0]])
        frames = np.array([1])
        result = self.corrector.correct(positions, frames)
        assert result.n_merged == 1
        assert result.n_raw == 1
        np.testing.assert_array_almost_equal(result.positions[0], [100.0, 200.0, 50.0])

    def test_two_same_position_same_frame_counted_once_if_adjacent(self):
        # Two points at same location on consecutive frames: one molecule
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        frames = np.array([1, 2])
        result = self.corrector.correct(positions, frames)
        assert result.n_merged == 1
        assert result.n_raw == 2

    def test_same_position_within_dark_gap_merged(self):
        # Same molecule, dark for 3 frames (within threshold of 5)
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        frames = np.array([1, 4])   # gap = 3 frames
        result = self.corrector.correct(positions, frames)
        assert result.n_merged == 1

    def test_same_position_exceeding_dark_gap_separate(self):
        # Same molecule, dark for 10 frames (exceeds threshold of 5)
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        frames = np.array([1, 12])  # gap = 11 frames
        result = self.corrector.correct(positions, frames)
        assert result.n_merged == 2

    def test_spatial_separation_lateral_exceeds_threshold(self):
        # Two nearby molecules in the same frame: should be separate
        positions = np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
        frames = np.array([1, 2])
        corrector = BlinkingCorrector(r_lateral=30.0, r_axial=60.0)
        result = corrector.correct(positions, frames)
        assert result.n_merged == 2

    def test_spatial_separation_lateral_within_threshold(self):
        positions = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        frames = np.array([1, 2])
        corrector = BlinkingCorrector(r_lateral=30.0, r_axial=60.0)
        result = corrector.correct(positions, frames)
        assert result.n_merged == 1

    def test_anisotropic_axial_within_threshold(self):
        # z separation = 50nm, axial threshold = 60nm -> same molecule
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 50.0]])
        frames = np.array([1, 2])
        corrector = BlinkingCorrector(r_lateral=30.0, r_axial=60.0)
        result = corrector.correct(positions, frames)
        assert result.n_merged == 1

    def test_anisotropic_axial_exceeds_threshold(self):
        # z separation = 70nm, axial threshold = 60nm -> separate molecules
        positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 70.0]])
        frames = np.array([1, 2])
        corrector = BlinkingCorrector(r_lateral=30.0, r_axial=60.0)
        result = corrector.correct(positions, frames)
        assert result.n_merged == 2

    def test_anisotropic_combined_distance_within_ellipsoid(self):
        # x=15/30=0.5, z=30/60=0.5 -> scaled dist_sq = 0.25+0.25 = 0.5 < 1 -> merge
        positions = np.array([[0.0, 0.0, 0.0], [15.0, 0.0, 30.0]])
        frames = np.array([1, 2])
        corrector = BlinkingCorrector(r_lateral=30.0, r_axial=60.0)
        result = corrector.correct(positions, frames)
        assert result.n_merged == 1

    def test_anisotropic_combined_distance_outside_ellipsoid(self):
        # x=20/30=0.667, z=50/60=0.833 -> scaled dist_sq = 0.444+0.694 = 1.138 > 1 -> separate
        # Each dimension is within its own threshold, but the combined ellipsoidal
        # distance exceeds 1.0, so they should not merge.
        positions = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 50.0]])
        frames = np.array([1, 2])
        corrector = BlinkingCorrector(r_lateral=30.0, r_axial=60.0)
        result = corrector.correct(positions, frames)
        assert result.n_merged == 2


class TestBlinkingCorrectorPhotonWeighting:

    def test_equal_photons_gives_arithmetic_mean(self):
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        frames = np.array([1, 2])
        photons = np.array([1.0, 1.0])
        corrector = BlinkingCorrector(r_lateral=30.0, r_axial=60.0)
        result = corrector.correct(positions, frames, photons)
        assert result.n_merged == 1
        np.testing.assert_almost_equal(result.positions[0, 0], 5.0, decimal=10)

    def test_unequal_photons_weights_toward_brighter(self):
        # Point at x=0 has 3x photons of point at x=10
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        frames = np.array([1, 2])
        photons = np.array([3.0, 1.0])
        corrector = BlinkingCorrector(r_lateral=30.0, r_axial=60.0)
        result = corrector.correct(positions, frames, photons)
        assert result.n_merged == 1
        # Expected centroid: (3*0 + 1*10) / 4 = 2.5
        np.testing.assert_almost_equal(result.positions[0, 0], 2.5, decimal=10)

    def test_total_photons_summed_correctly(self):
        positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        frames = np.array([1, 2])
        photons = np.array([100.0, 200.0])
        corrector = BlinkingCorrector(r_lateral=30.0, r_axial=60.0)
        result = corrector.correct(positions, frames, photons)
        assert result.n_merged == 1
        np.testing.assert_almost_equal(result.photons[0], 300.0)

    def test_no_photons_argument_uses_uniform_weighting(self):
        positions = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        frames = np.array([1, 2])
        corrector = BlinkingCorrector(r_lateral=30.0, r_axial=60.0)
        result = corrector.correct(positions, frames, photons=None)
        assert result.n_merged == 1
        np.testing.assert_almost_equal(result.positions[0, 0], 5.0, decimal=10)


class TestBlinkingCorrectorOutputContracts:

    def setup_method(self):
        self.corrector = BlinkingCorrector(r_lateral=30.0, r_axial=60.0)

    def test_output_is_blinker_result(self):
        positions = np.random.uniform(0, 1000, (50, 3))
        frames = np.random.randint(1, 100, 50)
        result = self.corrector.correct(positions, frames)
        assert isinstance(result, BlinkerResult)

    def test_positions_shape(self):
        positions = np.random.uniform(0, 1000, (100, 3))
        frames = np.random.randint(1, 200, 100)
        result = self.corrector.correct(positions, frames)
        assert result.positions.ndim == 2
        assert result.positions.shape[1] == 3

    def test_photons_and_positions_same_length(self):
        positions = np.random.uniform(0, 1000, (100, 3))
        frames = np.random.randint(1, 200, 100)
        result = self.corrector.correct(positions, frames)
        assert len(result.photons) == len(result.positions)
        assert len(result.n_blinks) == len(result.positions)

    def test_total_blinks_equals_n_raw(self):
        positions = np.random.uniform(0, 1000, (100, 3))
        frames = np.random.randint(1, 200, 100)
        result = self.corrector.correct(positions, frames)
        assert result.n_blinks.sum() == result.n_raw

    def test_merged_never_exceeds_raw(self):
        positions = np.random.uniform(0, 1000, (200, 3))
        frames = np.random.randint(1, 100, 200)
        result = self.corrector.correct(positions, frames)
        assert result.n_merged <= result.n_raw

    def test_n_raw_stored_correctly(self):
        positions = np.random.uniform(0, 1000, (77, 3))
        frames = np.random.randint(1, 50, 77)
        result = self.corrector.correct(positions, frames)
        assert result.n_raw == 77

    def test_reduction_factor_geq_one(self):
        positions = np.random.uniform(0, 500, (100, 3))
        frames = np.random.randint(1, 10, 100)  # dense in time -> lots of merging
        result = self.corrector.correct(positions, frames)
        assert result.reduction_factor >= 1.0

    def test_well_separated_molecules_not_merged(self):
        # 10 molecules far apart, 1 blink each
        centers = np.array([[i * 500.0, 0.0, 0.0] for i in range(10)])
        frames = np.arange(1, 11)
        result = self.corrector.correct(centers, frames)
        assert result.n_merged == 10


class TestBlinkingCorrectorInputValidation:

    def test_wrong_positions_shape_raises(self):
        corrector = BlinkingCorrector()
        with pytest.raises(ValueError, match="shape"):
            corrector.correct(np.ones((10, 2)), np.ones(10, dtype=int))

    def test_mismatched_frames_length_raises(self):
        corrector = BlinkingCorrector()
        with pytest.raises(ValueError, match="same length"):
            corrector.correct(np.ones((10, 3)), np.ones(5, dtype=int))

    def test_mismatched_photons_length_raises(self):
        corrector = BlinkingCorrector()
        with pytest.raises(ValueError, match="same length"):
            corrector.correct(np.ones((10, 3)), np.ones(10, dtype=int),
                              photons=np.ones(5))

    def test_negative_radius_raises(self):
        with pytest.raises(ValueError, match="positive"):
            BlinkingCorrector(r_lateral=-1.0)

    def test_zero_dark_frames_raises(self):
        with pytest.raises(ValueError, match="max_dark_frames"):
            BlinkingCorrector(max_dark_frames=0)


class TestBlinkingCorrectorChain:
    """Integration-style tests: simulate realistic blinking patterns."""

    def test_five_blinks_same_molecule_merged(self):
        """One molecule blinks 5 times over 10 frames."""
        base = np.array([100.0, 200.0, 50.0])
        positions = base + np.random.default_rng(0).normal(0, 5, (5, 3))
        frames = np.array([1, 3, 5, 7, 9])  # on every other frame
        corrector = BlinkingCorrector(r_lateral=30.0, r_axial=60.0,
                                      max_dark_frames=3)
        result = corrector.correct(positions, frames)
        assert result.n_merged == 1
        assert result.n_blinks[0] == 5

    def test_two_molecules_interleaved_frames(self):
        """Two molecules in adjacent frames, spatially separated."""
        rng = np.random.default_rng(1)
        mol1 = np.array([0.0, 0.0, 0.0]) + rng.normal(0, 5, (4, 3))
        mol2 = np.array([200.0, 0.0, 0.0]) + rng.normal(0, 5, (4, 3))
        positions = np.vstack([mol1, mol2])
        frames = np.array([1, 3, 5, 7, 2, 4, 6, 8])  # interleaved
        corrector = BlinkingCorrector(r_lateral=30.0, r_axial=60.0,
                                      max_dark_frames=3)
        result = corrector.correct(positions, frames)
        assert result.n_merged == 2

    def test_centroid_near_ground_truth(self):
        """Centroid of merged cluster should be close to the true position."""
        true_pos = np.array([500.0, 500.0, 500.0])
        rng = np.random.default_rng(42)
        n_blinks = 20
        positions = true_pos + rng.normal(0, 10, (n_blinks, 3))
        frames = np.arange(1, n_blinks + 1)
        corrector = BlinkingCorrector(r_lateral=40.0, r_axial=80.0,
                                      max_dark_frames=2)
        result = corrector.correct(positions, frames)
        assert result.n_merged == 1
        err = np.linalg.norm(result.positions[0] - true_pos)
        assert err < 10.0, f"Centroid error {err:.1f} nm > 10 nm"
