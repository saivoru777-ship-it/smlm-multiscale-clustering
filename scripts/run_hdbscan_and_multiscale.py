#!/usr/bin/env python3
"""
Run HDBSCAN on experimental data + mixed-scale clustering simulation.
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/Users/prethamsai')
import numpy as np, json, time
from scipy.io import loadmat
from smlm_clustering.core.null_models import BiologicalNullModel
from smlm_clustering.core.multiscale_detector import SMLMMultiscaleTest, ScaleRange
from smlm_clustering.validation.comprehensive_study import _corrected_chi2_pvalue
from smlm_clustering.validation.comparison_methods import (
    DBSCANBaseline, RipleysKBaseline, HDBSCANBaseline
)

roi_nm = 5000.0

# ============================================================
# PART 1: HDBSCAN on experimental data
# ============================================================
print("=" * 60)
print("PART 1: HDBSCAN on experimental data")
print("=" * 60, flush=True)

data_dir = '/Users/prethamsai/smlm_clustering/data/experimental'

def load_mat(path, n_particles, pixel_nm=100.0):
    mat = loadmat(path, squeeze_me=True)
    particles = mat['particles']
    rng = np.random.default_rng(42)
    pts_list = []
    for i in range(min(n_particles, len(particles))):
        pts = particles[i]['points'] * pixel_nm
        pts = pts - pts.mean(axis=0)
        offset = rng.uniform(500, roi_nm-500, 3)
        offset[2] = rng.uniform(50, 450)
        pts = np.clip(pts + offset, 0, [roi_nm, roi_nm, 500])
        pts_list.append(pts)
    combined = np.vstack(pts_list)
    bg = np.column_stack([rng.uniform(0, roi_nm, (2000, 2)), rng.uniform(0, 500, 2000)])
    return np.vstack([combined, bg])

datasets = [
    ('NPC STORM', load_mat(f'{data_dir}/NUP107_STORM.mat', 30)),
    ('NPC PAINT', load_mat(f'{data_dir}/NUP107_PAINT.mat', 25)),
    ('DNA-origami tetra', load_mat(f'{data_dir}/Tetra_PAINT.mat', 15)),
]

rng = np.random.default_rng(42)
storm_n = len(datasets[0][1])
csr = np.column_stack([rng.uniform(0, roi_nm, (storm_n, 2)), rng.uniform(0, 500, storm_n)])
datasets.append(('CSR control', csr))

rng2 = np.random.default_rng(123)
perm = np.column_stack([rng2.uniform(0, roi_nm, (storm_n, 2)), rng2.uniform(0, 500, storm_n)])
datasets.append(('Permutation control', perm))

hdbscan_results = {}
for name, pts in datasets:
    print(f"\n{name} (N={len(pts)})...", flush=True)
    null = BiologicalNullModel(seed=42)
    null.fit(pts)
    mocks = null.generate_mocks(len(pts), 15)

    hdb = HDBSCANBaseline(min_cluster_size=15)
    hr = hdb.fit(pts)
    mock_clusters = [hdb.fit(m).n_clusters for m in mocks]
    m95 = np.percentile(mock_clusters, 95)
    det = bool(hr.n_clusters > m95)
    print(f"  HDBSCAN: {hr.n_clusters} clusters (95th mock={m95:.0f}) det={det}", flush=True)
    print(f"  noise points: {hr.n_noise}/{len(pts)}", flush=True)
    hdbscan_results[name] = {
        'n_clusters': int(hr.n_clusters),
        'n_noise': int(hr.n_noise),
        'mock_95': float(m95),
        'detected': det,
    }

print("\n\nHDBSCAN Summary:")
for name, r in hdbscan_results.items():
    print(f"  {name}: {r['n_clusters']} clusters (95th={r['mock_95']:.0f}) det={r['detected']}")

# ============================================================
# PART 2: Mixed-scale clustering simulation
# ============================================================
print("\n" + "=" * 60)
print("PART 2: Mixed-scale (hierarchical) clustering simulation")
print("=" * 60, flush=True)

def generate_mixed_scale(n_molecules=5000, seed=42):
    """Generate data with clusters at TWO different scales simultaneously."""
    rng = np.random.default_rng(seed)
    positions = []

    # Large domains (r=200nm): 10% of molecules in 5 domains
    n_large = int(n_molecules * 0.10)
    n_per_large = n_large // 5
    for _ in range(5):
        center = rng.uniform(500, roi_nm-500, 3)
        center[2] = rng.uniform(50, 450)
        pts = center + rng.normal(0, 200, (n_per_large, 3))
        positions.append(pts)

    # Small nanoclusters (r=30nm): 15% of molecules in 50 nanoclusters
    n_small = int(n_molecules * 0.15)
    n_per_small = n_small // 50
    for _ in range(50):
        center = rng.uniform(500, roi_nm-500, 3)
        center[2] = rng.uniform(50, 450)
        pts = center + rng.normal(0, 30, (n_per_small, 3))
        positions.append(pts)

    # Background CSR: remaining molecules
    n_bg = n_molecules - n_large - n_small
    bg = np.column_stack([
        rng.uniform(0, roi_nm, (n_bg, 2)),
        rng.uniform(0, 500, n_bg)
    ])
    positions.append(bg)

    all_pts = np.vstack(positions)
    all_pts = np.clip(all_pts, 0, [roi_nm, roi_nm, 500])
    return all_pts

def run_ms_test(positions, n_mocks=15, min_scale_nm=20):
    null = BiologicalNullModel(seed=None)
    null.fit(positions)
    mock_positions = null.generate_mocks(len(positions), n_mocks)

    detector = SMLMMultiscaleTest.from_positions(positions, grid_size=64)
    roi_max = positions.max(axis=0) - positions.min(axis=0)
    max_scale = min(500, roi_max.max() / 4)
    if max_scale < 25:
        max_scale = 25

    scale_range = ScaleRange.for_smlm(
        min_nm=min_scale_nm, max_nm=max_scale,
        n_scales=12, roi_size_nm=roi_max.max(), grid_size=64)

    ms = detector.test(positions, mock_positions, scale_range, shrinkage=0.1)

    var_curve = ms['variance'].get('real_curve', {})
    skew_curve = ms['skewness'].get('real_curve', {})
    var_mocks = [mc['values'] for mc in ms['variance'].get('mock_curves', [])
                 if len(mc['values']) == len(var_curve.get('values', []))]
    skew_mocks = [mc['values'] for mc in ms['skewness'].get('mock_curves', [])
                  if len(mc['values']) == len(skew_curve.get('values', []))]

    pv = _corrected_chi2_pvalue(var_curve.get('values', []), var_mocks, 0.1)
    ps = _corrected_chi2_pvalue(skew_curve.get('values', []), skew_mocks, 0.1)

    # Also return the variance curve for fingerprint analysis
    return pv, ps, var_curve, skew_curve

# Run mixed-scale test
mixed_results = {}
for trial in range(3):
    seed = 800 + trial
    t0 = time.time()
    pts = generate_mixed_scale(n_molecules=5000, seed=seed)
    print(f"\nMixed-scale trial {trial+1}/3 (N={len(pts)})...", flush=True)

    # Multiscale
    pv, ps, var_c, skew_c = run_ms_test(pts, n_mocks=15)
    ms_det = pv < 0.05 or ps < 0.05
    print(f"  MS: pv={pv:.4f} ps={ps:.4f} det={ms_det}", flush=True)

    # DBSCAN at small scale (r=30)
    null = BiologicalNullModel(seed=42)
    null.fit(pts)
    mocks = null.generate_mocks(len(pts), 15)

    db_small = DBSCANBaseline(eps_nm=22.5, min_samples=5)
    dr_small = db_small.fit(pts)
    mc_small = [db_small.fit(m).n_clusters for m in mocks]
    m95_small = np.percentile(mc_small, 95)
    db_small_det = bool(dr_small.n_clusters > m95_small)
    print(f"  DBSCAN(eps=22.5nm): {dr_small.n_clusters} clusters (95th={m95_small:.0f}) det={db_small_det}", flush=True)

    # DBSCAN at large scale (r=200)
    db_large = DBSCANBaseline(eps_nm=150, min_samples=5)
    dr_large = db_large.fit(pts)
    mc_large = [db_large.fit(m).n_clusters for m in mocks]
    m95_large = np.percentile(mc_large, 95)
    db_large_det = bool(dr_large.n_clusters > m95_large)
    print(f"  DBSCAN(eps=150nm): {dr_large.n_clusters} clusters (95th={m95_large:.0f}) det={db_large_det}", flush=True)

    # HDBSCAN
    hdb = HDBSCANBaseline(min_cluster_size=15)
    hr = hdb.fit(pts)
    mc_hdb = [hdb.fit(m).n_clusters for m in mocks]
    m95_hdb = np.percentile(mc_hdb, 95)
    hdb_det = bool(hr.n_clusters > m95_hdb)
    print(f"  HDBSCAN: {hr.n_clusters} clusters (95th={m95_hdb:.0f}) det={hdb_det}", flush=True)

    # Ripley
    rip = RipleysKBaseline(r_min_nm=20, r_max_nm=500, n_radii=15)
    rr = rip.compute(pts)
    Lmax = float(rr.L_minus_r.max())
    mL = [float(rip.compute(m).L_minus_r.max()) for m in mocks]
    m95L = np.percentile(mL, 95)
    rip_det = bool(Lmax > m95L)
    print(f"  Ripley: Lmax={Lmax:.1f} (95th={m95L:.1f}) det={rip_det}", flush=True)

    # Variance curve info for fingerprint
    if var_c.get('scales'):
        print(f"  Variance curve scales: {[f'{s:.0f}' for s in var_c['scales']]}nm", flush=True)
        print(f"  Variance curve values: {[f'{v:.2f}' for v in var_c['values']]}", flush=True)

    elapsed = time.time() - t0
    mixed_results[f'trial_{trial+1}'] = {
        'ms_det': bool(ms_det), 'ms_pv': float(pv), 'ms_ps': float(ps),
        'dbscan_small_det': bool(db_small_det), 'dbscan_small_clusters': int(dr_small.n_clusters),
        'dbscan_large_det': bool(db_large_det), 'dbscan_large_clusters': int(dr_large.n_clusters),
        'hdbscan_det': bool(hdb_det), 'hdbscan_clusters': int(hr.n_clusters),
        'ripley_det': bool(rip_det), 'ripley_Lmax': float(Lmax),
        'variance_scales': var_c.get('scales', []),
        'variance_values': var_c.get('values', []),
    }
    print(f"  ({elapsed:.0f}s)", flush=True)

# Save all results
output = {
    'hdbscan_experimental': hdbscan_results,
    'mixed_scale': mixed_results,
}

with open('/Users/prethamsai/results/hdbscan_and_multiscale.json', 'w') as f:
    json.dump(output, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

print("\n\nAll results saved to results/hdbscan_and_multiscale.json")
