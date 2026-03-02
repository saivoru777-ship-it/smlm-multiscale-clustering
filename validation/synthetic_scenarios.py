"""
Extended synthetic data generators for comprehensive validation.

Adds:
    - Elongated (elliptical) clusters
    - Ring-shaped clusters (nuclear pore model)
    - Non-cubic ROI support
    - Biological preset configurations (synaptic, nuclear pore, membrane)
    - Parametric cluster generation for geometry robustness studies

All generators return SyntheticSMLMDataset for compatibility with the
existing benchmark pipeline.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List

from .benchmark_runner import SyntheticSMLMDataset


# ============================================================================
# Biological presets
# ============================================================================

@dataclass
class BiologicalPreset:
    """Configuration for a biologically-motivated synthetic scenario."""
    name: str
    n_molecules: int
    n_clusters: int
    cluster_radius_nm: float
    f_clust: float
    roi_nm: float
    cluster_shape: str       # 'gaussian', 'elongated', 'ring'
    n_blinks_mean: float = 5.0
    blink_sigma_xy_nm: float = 15.0
    blink_sigma_z_nm: float = 35.0
    n_frames: int = 500
    aspect_ratio: float = 1.0   # for elongated clusters
    ring_width_nm: float = 10.0  # for ring clusters
    description: str = ''


BIOLOGICAL_PRESETS = {
    'synaptic_receptors': BiologicalPreset(
        name='synaptic_receptors',
        n_molecules=2000,
        n_clusters=30,
        cluster_radius_nm=35.0,
        f_clust=0.7,
        roi_nm=2000.0,
        cluster_shape='gaussian',
        n_blinks_mean=5.0,
        blink_sigma_xy_nm=10.0,
        description='Synaptic receptor nanodomains (~35nm, high f_clust)',
    ),
    'nuclear_pores': BiologicalPreset(
        name='nuclear_pores',
        n_molecules=5000,
        n_clusters=200,
        cluster_radius_nm=50.0,
        f_clust=0.4,
        roi_nm=10000.0,
        cluster_shape='ring',
        ring_width_nm=10.0,
        n_blinks_mean=3.0,
        blink_sigma_xy_nm=15.0,
        description='Nuclear pore complexes (ring geometry, ~50nm radius)',
    ),
    'membrane_domains': BiologicalPreset(
        name='membrane_domains',
        n_molecules=10000,
        n_clusters=50,
        cluster_radius_nm=150.0,
        f_clust=0.3,
        roi_nm=5000.0,
        cluster_shape='gaussian',
        n_blinks_mean=5.0,
        blink_sigma_xy_nm=20.0,
        description='Membrane protein domains (~150nm, moderate f_clust)',
    ),
    'negative_control': BiologicalPreset(
        name='negative_control',
        n_molecules=5000,
        n_clusters=0,
        cluster_radius_nm=75.0,
        f_clust=0.0,
        roi_nm=5000.0,
        cluster_shape='gaussian',
        description='Pure CSR negative control',
    ),
}


# ============================================================================
# Extended cluster generators
# ============================================================================

def _generate_gaussian_cluster(
    center: np.ndarray,
    n_points: int,
    radius_nm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate isotropic Gaussian cluster points."""
    return center + rng.normal(0, radius_nm, (n_points, 3))


def _generate_elongated_cluster(
    center: np.ndarray,
    n_points: int,
    radius_nm: float,
    aspect_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate an elongated (ellipsoidal) cluster.

    The major axis is along a random direction with length aspect_ratio * radius,
    minor axes have the base radius.
    """
    # Generate in local coordinates
    sigma = np.array([radius_nm * aspect_ratio, radius_nm, radius_nm])
    local_points = rng.normal(0, sigma, (n_points, 3))

    # Random rotation via QR decomposition of random matrix
    random_matrix = rng.standard_normal((3, 3))
    q, _ = np.linalg.qr(random_matrix)
    rotated = local_points @ q.T

    return center + rotated


def _generate_ring_cluster(
    center: np.ndarray,
    n_points: int,
    radius_nm: float,
    ring_width_nm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a ring-shaped cluster (nuclear pore model).

    Points are distributed on a ring of given radius in the xy-plane
    with Gaussian scatter of ring_width_nm.
    """
    # Random angles around the ring
    theta = rng.uniform(0, 2 * np.pi, n_points)

    # Points on the ring + radial noise
    radial_noise = rng.normal(0, ring_width_nm, n_points)
    r = radius_nm + radial_noise

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = rng.normal(0, ring_width_nm, n_points)

    local_points = np.column_stack([x, y, z])

    # Random 3D orientation
    random_matrix = rng.standard_normal((3, 3))
    q, _ = np.linalg.qr(random_matrix)
    rotated = local_points @ q.T

    return center + rotated


def generate_cluster_points(
    center: np.ndarray,
    n_points: int,
    radius_nm: float,
    shape: str = 'gaussian',
    aspect_ratio: float = 3.0,
    ring_width_nm: float = 10.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate points for a single cluster with the specified geometry.

    Parameters
    ----------
    center : ndarray (3,)
    n_points : int
    radius_nm : float
    shape : str
        'gaussian', 'elongated', or 'ring'
    aspect_ratio : float
        Major/minor axis ratio for elongated clusters.
    ring_width_nm : float
        Width of ring scatter for ring clusters.
    rng : Generator

    Returns
    -------
    ndarray (n_points, 3)
    """
    if rng is None:
        rng = np.random.default_rng()

    if shape == 'gaussian':
        return _generate_gaussian_cluster(center, n_points, radius_nm, rng)
    elif shape == 'elongated':
        return _generate_elongated_cluster(
            center, n_points, radius_nm, aspect_ratio, rng)
    elif shape == 'ring':
        return _generate_ring_cluster(
            center, n_points, radius_nm, ring_width_nm, rng)
    else:
        raise ValueError(f"Unknown cluster shape: {shape}")


# ============================================================================
# Main generator
# ============================================================================

def generate_extended_dataset(
    n_molecules: int = 5000,
    f_clust: float = 0.3,
    n_clusters: int = 20,
    cluster_radius_nm: float = 75.0,
    roi_nm: float = 5000.0,
    cluster_shape: str = 'gaussian',
    aspect_ratio: float = 3.0,
    ring_width_nm: float = 10.0,
    n_blinks_mean: float = 5.0,
    blink_sigma_xy_nm: float = 15.0,
    blink_sigma_z_nm: float = 35.0,
    n_frames: int = 500,
    seed: Optional[int] = None,
) -> SyntheticSMLMDataset:
    """
    Generate a synthetic SMLM dataset with extended cluster geometries.

    Extends generate_synthetic_dataset() with support for elongated
    and ring-shaped clusters.

    Parameters
    ----------
    n_molecules : int
        Total distinct molecules.
    f_clust : float
        Fraction of molecules in clusters.
    n_clusters : int
        Number of clusters.
    cluster_radius_nm : float
        Characteristic radius of each cluster (nm).
    roi_nm : float
        Side length of the cubic ROI (nm).
    cluster_shape : str
        'gaussian', 'elongated', or 'ring'.
    aspect_ratio : float
        For elongated clusters: major/minor axis ratio.
    ring_width_nm : float
        For ring clusters: Gaussian scatter width (nm).
    n_blinks_mean : float
        Mean blinks per molecule (Poisson).
    blink_sigma_xy_nm : float
        Lateral localization noise per blink (nm).
    blink_sigma_z_nm : float
        Axial localization noise per blink (nm).
    n_frames : int
        Total imaging frames.
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

    # Clustered molecules
    if n_clust > 0 and n_clusters > 0:
        margin = max(cluster_radius_nm * 3, roi_nm * 0.05)
        lo = min(margin, roi_nm * 0.1)
        hi = max(roi_nm - margin, roi_nm * 0.9)
        cluster_centers = rng.uniform(lo, hi, (n_clusters, 3))

        assignment = rng.integers(0, n_clusters, n_clust)
        clust_pos_list = []
        for ci in range(n_clusters):
            mask = assignment == ci
            nc = mask.sum()
            if nc > 0:
                pts = generate_cluster_points(
                    center=cluster_centers[ci],
                    n_points=nc,
                    radius_nm=cluster_radius_nm,
                    shape=cluster_shape,
                    aspect_ratio=aspect_ratio,
                    ring_width_nm=ring_width_nm,
                    rng=rng,
                )
                clust_pos_list.append(pts)

        clust_pos = np.vstack(clust_pos_list) if clust_pos_list else np.empty((0, 3))
        clust_pos = np.clip(clust_pos, 0, roi_nm)
        clust_labels = np.ones(n_clust, dtype=int)
    else:
        clust_pos = np.empty((0, 3))
        clust_labels = np.empty(0, dtype=int)

    mol_positions = np.vstack([bg_pos, clust_pos]) if n_clust > 0 else bg_pos
    mol_labels = np.concatenate([bg_labels, clust_labels]) if n_clust > 0 else bg_labels

    # Simulate blinking
    all_locs = []
    all_frames = []
    all_photons = []

    for mol_idx in range(len(mol_positions)):
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
        cluster_radius_nm=cluster_radius_nm,
    )


def generate_from_preset(
    preset_name: str,
    seed: Optional[int] = None,
) -> SyntheticSMLMDataset:
    """
    Generate a dataset from a biological preset.

    Parameters
    ----------
    preset_name : str
        One of: 'synaptic_receptors', 'nuclear_pores',
        'membrane_domains', 'negative_control'
    seed : int or None

    Returns
    -------
    SyntheticSMLMDataset
    """
    if preset_name not in BIOLOGICAL_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available: {list(BIOLOGICAL_PRESETS.keys())}"
        )

    preset = BIOLOGICAL_PRESETS[preset_name]

    return generate_extended_dataset(
        n_molecules=preset.n_molecules,
        f_clust=preset.f_clust,
        n_clusters=preset.n_clusters,
        cluster_radius_nm=preset.cluster_radius_nm,
        roi_nm=preset.roi_nm,
        cluster_shape=preset.cluster_shape,
        aspect_ratio=preset.aspect_ratio,
        ring_width_nm=preset.ring_width_nm,
        n_blinks_mean=preset.n_blinks_mean,
        blink_sigma_xy_nm=preset.blink_sigma_xy_nm,
        blink_sigma_z_nm=preset.blink_sigma_z_nm,
        n_frames=preset.n_frames,
        seed=seed,
    )
