#!/usr/bin/env python3
"""
Two experiments:
1. Test pipeline FPR fixes (min_scale approach)
2. Run baselines on real experimental data + permutation control

Writes results to /Users/prethamsai/results/fix_study_results.json
"""
import sys, json, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/Users/prethamsai')

import numpy as np
from scipy.io import loadmat
from smlm_clustering.validation.benchmark_runner import generate_synthetic_dataset
from smlm_clustering.core.blinking_correction import BlinkingCorrector
from smlm_clustering.core.multiscale_detector import SMLMMultiscaleTest, ScaleRange
from smlm_clustering.core.null_models import BiologicalNullModel
from smlm_clustering.validation.comparison_methods import DBSCANBaseline, RipleysKBaseline

roi_nm = 5000.0
all_results = {}

# ============================================================
# PART 1: Pipeline FPR fixes (3 trials)
# ============================================================
print("=" * 60)
print("PART 1: Pipeline FPR fix approaches")
print("=" * 60)

fpr_results = {k: [] for k in ['mol_pos', 'std_corr', 'min50nm', 'min80nm', 'min100nm']}

for trial in range(3):
    seed = 300 + trial
    t0 = time.time()
    print(f"\nTrial {trial+1}/3 ...", flush=True)

    ds = generate_synthetic_dataset(
        n_molecules=5000, f_clust=0.0, roi_nm=5000.0,
        n_blinks_mean=5, blink_sigma_xy_nm=15.0, blink_sigma_z_nm=35.0,
        n_frames=500, seed=seed
    )

    corrector = BlinkingCorrector(r_lateral=30, r_axial=60, max_dark_frames=5)
    r = corrector.correct(ds.raw_positions, ds.raw_frames, ds.raw_photons)

    for label, positions, min_nm in [
        ('mol_pos', ds.molecule_positions, 20),
        ('std_corr', r.positions, 20),
        ('min50nm', r.positions, 50),
        ('min80nm', r.positions, 80),
        ('min100nm', r.positions, 100),
    ]:
        null = BiologicalNullModel(seed=seed + 1000)
        null.fit(positions)
        mocks = null.generate_mocks(10, len(positions))
        roi_max = np.array([roi_nm, roi_nm, roi_nm * 0.1])
        sr = ScaleRange.for_smlm(min_nm=min_nm, max_nm=min(500, roi_nm / 4),
                                  n_scales=12, roi_size_nm=roi_nm, grid_size=64)
        det = SMLMMultiscaleTest(roi_size_nm=roi_max, grid_size=64)
        res = det.test(positions, mocks, sr, shrinkage=0.1)
        detected = res['p_variance'] < 0.05 or res['p_skewness'] < 0.05
        fpr_results[label].append(detected)
        print(f"  {label}: det={detected} (pv={res['p_variance']:.3f} ps={res['p_skewness']:.3f})", flush=True)

    print(f"  Trial took {time.time()-t0:.0f}s", flush=True)

print("\nFPR Summary:")
for k, v in fpr_results.items():
    print(f"  {k}: {sum(v)}/{len(v)} = {100*sum(v)/len(v):.0f}%")
all_results['fpr_fixes'] = {k: {'detected': sum(v), 'total': len(v)} for k, v in fpr_results.items()}

# ============================================================
# PART 2: Baselines on experimental data
# ============================================================
print("\n" + "=" * 60)
print("PART 2: Baselines on experimental data")
print("=" * 60)

data_dir = '/Users/prethamsai/smlm_clustering/data/experimental'

def load_mat_particles(path, n_particles, pixel_nm=100.0):
    mat = loadmat(path, squeeze_me=True)
    particles = mat['particles']
    rng = np.random.default_rng(42)
    all_points = []
    for i in range(min(n_particles, len(particles))):
        p = particles[i]
        pts = p['points'] * pixel_nm
        pts = pts - pts.mean(axis=0)
        offset = rng.uniform(500, roi_nm - 500, 3)
        offset[2] = rng.uniform(50, 450)
        pts = pts + offset
        pts = np.clip(pts, 0, [roi_nm, roi_nm, 500])
        all_points.append(pts)
    particle_pts = np.vstack(all_points)
    n_bg = 2000
    bg = np.column_stack([rng.uniform(0, roi_nm, (n_bg, 2)), rng.uniform(0, 500, n_bg)])
    return np.vstack([particle_pts, bg])

def run_all_methods(positions, label, est_r=50.0, n_mocks=15):
    t0 = time.time()
    print(f"\n{label} (N={len(positions)})...", flush=True)
    roi_max = np.array([roi_nm, roi_nm, 500.0])

    # Mocks
    null = BiologicalNullModel(seed=42)
    null.fit(positions)
    mocks = null.generate_mocks(n_mocks, len(positions))

    # Multiscale
    sr = ScaleRange.for_smlm(min_nm=20, max_nm=1250, n_scales=12, roi_size_nm=roi_nm, grid_size=64)
    detector = SMLMMultiscaleTest(roi_size_nm=roi_max, grid_size=64)
    ms = detector.test(positions, mocks, sr, shrinkage=0.1)
    ms_det = ms['p_variance'] < 0.05 or ms['p_skewness'] < 0.05
    print(f"  MS: pv={ms['p_variance']:.4f} ps={ms['p_skewness']:.4f} det={ms_det}", flush=True)

    # DBSCAN
    db = DBSCANBaseline(eps_nm=est_r * 0.75, min_samples=5)
    db_r = db.fit(positions)
    mc = [db.fit(m).n_clusters for m in mocks]
    m95 = np.percentile(mc, 95)
    db_det = bool(db_r.n_clusters > m95)
    print(f"  DBSCAN: {db_r.n_clusters} clusters (95th={m95:.0f}) det={db_det}", flush=True)

    # Ripley (compute once per dataset, not redundantly)
    rip = RipleysKBaseline(r_min_nm=0, r_max_nm=500, n_radii=50)
    rr = rip.compute(positions)
    Lmax = float((rr.L_values - rr.radii).max()) if len(rr.L_values) > 0 else 0.0
    mock_Lmaxes = []
    for m in mocks:
        mr = rip.compute(m)
        if len(mr.L_values) > 0:
            mock_Lmaxes.append(float((mr.L_values - mr.radii).max()))
    m95L = np.percentile(mock_Lmaxes, 95) if mock_Lmaxes else 0
    rip_det = bool(Lmax > m95L)
    print(f"  Ripley: Lmax={Lmax:.1f} (95th={m95L:.1f}) det={rip_det}", flush=True)
    print(f"  Took {time.time()-t0:.0f}s", flush=True)

    return {
        'label': label, 'n': len(positions),
        'ms_p_var': float(ms['p_variance']), 'ms_p_skew': float(ms['p_skewness']), 'ms_det': bool(ms_det),
        'dbscan_clusters': int(db_r.n_clusters), 'dbscan_det': bool(db_det),
        'ripley_Lmax': Lmax, 'ripley_det': bool(rip_det),
    }

exp_results = {}

pts_storm = load_mat_particles(f'{data_dir}/NUP107_STORM.mat', 30)
exp_results['npc_storm'] = run_all_methods(pts_storm, 'NPC STORM (30 particles)', est_r=50)

pts_paint = load_mat_particles(f'{data_dir}/NUP107_PAINT.mat', 25)
exp_results['npc_paint'] = run_all_methods(pts_paint, 'NPC PAINT (25 particles)', est_r=50)

pts_tetra = load_mat_particles(f'{data_dir}/Tetra_PAINT.mat', 15)
exp_results['tetra'] = run_all_methods(pts_tetra, 'DNA-origami tetra (15)', est_r=30)

# CSR control
rng = np.random.default_rng(42)
csr = np.column_stack([rng.uniform(0, roi_nm, (len(pts_storm), 2)), rng.uniform(0, 500, len(pts_storm))])
exp_results['csr'] = run_all_methods(csr, 'CSR control', est_r=50)

# Permutation control
rng2 = np.random.default_rng(123)
perm = np.column_stack([rng2.uniform(0, roi_nm, (len(pts_storm), 2)), rng2.uniform(0, 500, len(pts_storm))])
exp_results['permutation'] = run_all_methods(perm, 'Permutation control', est_r=50)

all_results['experimental'] = exp_results

# Save
with open('/Users/prethamsai/results/fix_study_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print("\n\nAll results saved to results/fix_study_results.json")
