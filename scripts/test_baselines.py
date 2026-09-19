#!/usr/bin/env python3
"""Run DBSCAN + Ripley baselines on experimental data."""
import sys; sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/Users/prethamsai')
import numpy as np, json, time
from scipy.io import loadmat
from smlm_clustering.core.multiscale_detector import SMLMMultiscaleTest, ScaleRange
from smlm_clustering.core.null_models import BiologicalNullModel
from smlm_clustering.validation.comparison_methods import DBSCANBaseline, RipleysKBaseline

roi_nm = 5000.0
N_MOCKS = 15

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

def analyze(positions, label, est_r=50.0):
    t0 = time.time()
    print(f"\n{label} (N={len(positions)})...", flush=True)
    roi_max = np.array([roi_nm, roi_nm, 500.0])
    null = BiologicalNullModel(seed=42)
    null.fit(positions)
    mocks = null.generate_mocks(N_MOCKS, len(positions))

    # Multiscale
    sr = ScaleRange.for_smlm(min_nm=20, max_nm=1250, n_scales=12, roi_size_nm=roi_nm, grid_size=64)
    ms = SMLMMultiscaleTest(roi_size_nm=roi_max, grid_size=64).test(positions, mocks, sr, shrinkage=0.1)
    pv = ms['variance']['p_value']
    ps = ms['skewness']['p_value']
    ms_det = pv < 0.05 or ps < 0.05
    print(f"  MS: pv={pv:.4f} ps={ps:.4f} det={ms_det}", flush=True)

    # DBSCAN
    db = DBSCANBaseline(eps_nm=est_r*0.75, min_samples=5)
    db_r = db.fit(positions)
    mc = [db.fit(m).n_clusters for m in mocks]
    m95 = np.percentile(mc, 95)
    db_det = bool(db_r.n_clusters > m95)
    print(f"  DBSCAN: {db_r.n_clusters} clusters (95th={m95:.0f}) det={db_det}", flush=True)

    # Ripley
    rip = RipleysKBaseline(r_min_nm=0, r_max_nm=500, n_radii=50)
    rr = rip.compute(positions)
    Lmax = float((rr.L_values - rr.radii).max()) if len(rr.L_values) > 0 else 0
    mL = []
    for m in mocks:
        mr = rip.compute(m)
        if len(mr.L_values) > 0:
            mL.append(float((mr.L_values - mr.radii).max()))
    m95L = np.percentile(mL, 95) if mL else 0
    rip_det = bool(Lmax > m95L)
    print(f"  Ripley: Lmax={Lmax:.1f} (95th={m95L:.1f}) det={rip_det}", flush=True)
    print(f"  Took {time.time()-t0:.0f}s", flush=True)

    return {'label': label, 'n': len(positions),
            'ms_p_var': float(pv), 'ms_p_skew': float(ps), 'ms_det': bool(ms_det),
            'dbscan_clusters': int(db_r.n_clusters), 'dbscan_det': bool(db_det),
            'ripley_Lmax': Lmax, 'ripley_det': bool(rip_det)}

data_dir = '/Users/prethamsai/smlm_clustering/data/experimental'
results = {}

pts_storm = load_mat(f'{data_dir}/NUP107_STORM.mat', 30)
results['npc_storm'] = analyze(pts_storm, 'NPC STORM')

pts_paint = load_mat(f'{data_dir}/NUP107_PAINT.mat', 25)
results['npc_paint'] = analyze(pts_paint, 'NPC PAINT')

pts_tetra = load_mat(f'{data_dir}/Tetra_PAINT.mat', 15)
results['tetra'] = analyze(pts_tetra, 'DNA-origami tetra', est_r=30)

rng = np.random.default_rng(42)
csr = np.column_stack([rng.uniform(0, roi_nm, (len(pts_storm), 2)), rng.uniform(0, 500, len(pts_storm))])
results['csr'] = analyze(csr, 'CSR control')

rng2 = np.random.default_rng(123)
perm = np.column_stack([rng2.uniform(0, roi_nm, (len(pts_storm), 2)), rng2.uniform(0, 500, len(pts_storm))])
results['permutation'] = analyze(perm, 'Permutation control')

with open('/Users/prethamsai/results/I_experimental_with_baselines.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved to results/I_experimental_with_baselines.json")
