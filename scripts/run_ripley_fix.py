#!/usr/bin/env python3
"""Just run Ripley on experimental data with r_min=20 (fixing the log(0) bug)."""
import sys; sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/Users/prethamsai')
import numpy as np, json
from scipy.io import loadmat
from smlm_clustering.core.null_models import BiologicalNullModel
from smlm_clustering.validation.comparison_methods import RipleysKBaseline

roi_nm = 5000.0

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

def run_ripley(positions, label, n_mocks=15):
    print(f"{label} (N={len(positions)})...", flush=True)
    null = BiologicalNullModel(seed=42)
    null.fit(positions)
    mocks = null.generate_mocks(len(positions), n_mocks)

    rip = RipleysKBaseline(r_min_nm=20, r_max_nm=500, n_radii=15)
    rr = rip.compute(positions)
    Lmax = float(rr.L_minus_r.max())
    mL = []
    for m in mocks:
        mr = rip.compute(m)
        mL.append(float(mr.L_minus_r.max()))
    m95 = np.percentile(mL, 95)
    det = bool(Lmax > m95)
    print(f"  Ripley: Lmax={Lmax:.1f} (95th={m95:.1f}) det={det}", flush=True)
    return det, Lmax, m95

data_dir = '/Users/prethamsai/smlm_clustering/data/experimental'

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

results = {}
for name, pts in datasets:
    det, Lmax, m95 = run_ripley(pts, name)
    results[name] = {'detected': det, 'Lmax': Lmax, 'mock_95': m95}

print("\nSummary:")
for name, r in results.items():
    print(f"  {name}: det={r['detected']} Lmax={r['Lmax']:.1f} 95th={r['mock_95']:.1f}")

with open('/Users/prethamsai/results/ripley_experimental.json', 'w') as f:
    json.dump(results, f, indent=2)
