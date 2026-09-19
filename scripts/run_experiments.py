#!/usr/bin/env python3
"""
Run two experiments:
1. FPR fix: test min_scale on pipeline-corrected CSR data (500 molecules, very fast)
2. Baselines on real data + permutation controls

Uses the comprehensive_study pipeline infrastructure.
"""
import sys; sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/Users/prethamsai')

import numpy as np, json, time
from scipy.io import loadmat
from smlm_clustering.validation.benchmark_runner import generate_synthetic_dataset
from smlm_clustering.core.blinking_correction import BlinkingCorrector
from smlm_clustering.core.multiscale_detector import SMLMMultiscaleTest, ScaleRange
from smlm_clustering.core.null_models import BiologicalNullModel
from smlm_clustering.validation.comprehensive_study import _corrected_chi2_pvalue
from smlm_clustering.validation.comparison_methods import DBSCANBaseline, RipleysKBaseline

roi_nm = 5000.0

def run_ms_test(positions, n_mocks=15, grid_size=64, shrinkage=0.1, min_scale_nm=20):
    """Run multiscale test, return corrected p-values."""
    null = BiologicalNullModel(seed=None)
    null.fit(positions)
    mock_positions = null.generate_mocks(len(positions), n_mocks)

    detector = SMLMMultiscaleTest.from_positions(positions, grid_size=grid_size)
    roi_max = positions.max(axis=0) - positions.min(axis=0)
    max_scale = min(500, roi_max.max() / 4)
    if max_scale < 25:
        max_scale = 25

    scale_range = ScaleRange.for_smlm(
        min_nm=min_scale_nm, max_nm=max_scale,
        n_scales=12, roi_size_nm=roi_max.max(), grid_size=grid_size)

    ms = detector.test(positions, mock_positions, scale_range, shrinkage=shrinkage)

    # Corrected chi2 p-values
    var_curve = ms['variance'].get('real_curve', {})
    skew_curve = ms['skewness'].get('real_curve', {})
    var_mocks = [mc['values'] for mc in ms['variance'].get('mock_curves', [])
                 if len(mc['values']) == len(var_curve.get('values', []))]
    skew_mocks = [mc['values'] for mc in ms['skewness'].get('mock_curves', [])
                  if len(mc['values']) == len(skew_curve.get('values', []))]

    pv = _corrected_chi2_pvalue(var_curve.get('values', []), var_mocks, shrinkage)
    ps = _corrected_chi2_pvalue(skew_curve.get('values', []), skew_mocks, shrinkage)
    return pv, ps

# ============================================================
# PART 1: Pipeline FPR test with min_scale fix
# ============================================================
print("=" * 60)
print("PART 1: Pipeline FPR — min_scale fix")
print("Using 500 molecules (fast mock generation)")
print("=" * 60, flush=True)

fpr = {k: [] for k in ['mol_pos_20nm', 'pipeline_20nm', 'pipeline_50nm', 'pipeline_80nm']}

for trial in range(10):
    seed = 500 + trial
    t0 = time.time()
    ds = generate_synthetic_dataset(
        n_molecules=500, f_clust=0.0, roi_nm=roi_nm,
        n_blinks_mean=5, blink_sigma_xy_nm=15.0, blink_sigma_z_nm=35.0,
        n_frames=500, seed=seed
    )
    corrector = BlinkingCorrector(r_lateral=30, r_axial=60, max_dark_frames=5)
    r = corrector.correct(ds.raw_positions, ds.raw_frames, ds.raw_photons)

    # Molecule positions, standard min_scale=20nm
    pv, ps = run_ms_test(ds.molecule_positions, n_mocks=15, min_scale_nm=20)
    det = pv < 0.05 or ps < 0.05
    fpr['mol_pos_20nm'].append(det)

    # Pipeline (corrected), min_scale=20nm
    pv, ps = run_ms_test(r.positions, n_mocks=15, min_scale_nm=20)
    det = pv < 0.05 or ps < 0.05
    fpr['pipeline_20nm'].append(det)

    # Pipeline, min_scale=50nm
    pv, ps = run_ms_test(r.positions, n_mocks=15, min_scale_nm=50)
    det = pv < 0.05 or ps < 0.05
    fpr['pipeline_50nm'].append(det)

    # Pipeline, min_scale=80nm
    pv, ps = run_ms_test(r.positions, n_mocks=15, min_scale_nm=80)
    det = pv < 0.05 or ps < 0.05
    fpr['pipeline_80nm'].append(det)

    elapsed = time.time() - t0
    print(f"Trial {trial+1}/10: mol={fpr['mol_pos_20nm'][-1]} pipe20={fpr['pipeline_20nm'][-1]} "
          f"pipe50={fpr['pipeline_50nm'][-1]} pipe80={fpr['pipeline_80nm'][-1]} ({elapsed:.0f}s)", flush=True)

print("\nFPR Summary (10 trials):")
for k, v in fpr.items():
    print(f"  {k}: {sum(v)}/10 = {100*sum(v)/10:.0f}%")

# ============================================================
# PART 2: Baselines on experimental data
# ============================================================
print("\n" + "=" * 60)
print("PART 2: Baselines on experimental data")
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

def full_analysis(positions, label, est_r=50.0, n_mocks=15):
    t0 = time.time()
    print(f"\n{label} (N={len(positions)})...", flush=True)

    # Multiscale
    pv, ps = run_ms_test(positions, n_mocks=n_mocks, min_scale_nm=20)
    ms_det = pv < 0.05 or ps < 0.05
    print(f"  MS: pv={pv:.4f} ps={ps:.4f} det={ms_det}", flush=True)

    # Generate mocks for baselines (reuse same null model)
    null = BiologicalNullModel(seed=42)
    null.fit(positions)
    mocks = null.generate_mocks(len(positions), n_mocks)

    # DBSCAN
    db = DBSCANBaseline(eps_nm=est_r*0.75, min_samples=5)
    db_r = db.fit(positions)
    mc = [db.fit(m).n_clusters for m in mocks]
    m95 = np.percentile(mc, 95)
    db_det = bool(db_r.n_clusters > m95)
    print(f"  DBSCAN: {db_r.n_clusters} clusters (95th={m95:.0f}) det={db_det}", flush=True)

    # Ripley
    rip = RipleysKBaseline(r_min_nm=20, r_max_nm=500, n_radii=15)
    rr = rip.compute(positions)
    Lmax = float(rr.L_minus_r.max()) if hasattr(rr, 'L_minus_r') and len(rr.L_minus_r) > 0 else 0
    mL = []
    for m in mocks:
        mr = rip.compute(m)
        if hasattr(mr, 'L_minus_r') and len(mr.L_minus_r) > 0:
            mL.append(float(mr.L_minus_r.max()))
    m95L = np.percentile(mL, 95) if mL else 0
    rip_det = bool(Lmax > m95L)
    print(f"  Ripley: Lmax={Lmax:.1f} (95th={m95L:.1f}) det={rip_det}", flush=True)
    print(f"  Took {time.time()-t0:.0f}s", flush=True)

    return {'label': label, 'n': len(positions),
            'ms_p_var': float(pv), 'ms_p_skew': float(ps), 'ms_det': bool(ms_det),
            'dbscan_clusters': int(db_r.n_clusters), 'dbscan_det': bool(db_det),
            'ripley_Lmax': Lmax, 'ripley_det': bool(rip_det)}

exp = {}
pts_storm = load_mat(f'{data_dir}/NUP107_STORM.mat', 30)
exp['npc_storm'] = full_analysis(pts_storm, 'NPC STORM', est_r=50)

pts_paint = load_mat(f'{data_dir}/NUP107_PAINT.mat', 25)
exp['npc_paint'] = full_analysis(pts_paint, 'NPC PAINT', est_r=50)

pts_tetra = load_mat(f'{data_dir}/Tetra_PAINT.mat', 15)
exp['tetra'] = full_analysis(pts_tetra, 'DNA-origami tetra', est_r=30)

# Controls
rng = np.random.default_rng(42)
csr = np.column_stack([rng.uniform(0, roi_nm, (len(pts_storm), 2)), rng.uniform(0, 500, len(pts_storm))])
exp['csr'] = full_analysis(csr, 'CSR control', est_r=50)

rng2 = np.random.default_rng(123)
perm = np.column_stack([rng2.uniform(0, roi_nm, (len(pts_storm), 2)), rng2.uniform(0, 500, len(pts_storm))])
exp['permutation'] = full_analysis(perm, 'Permutation control', est_r=50)

# Save everything
output = {'fpr_fixes': {k: {'detected': sum(v), 'total': len(v),
                             'fpr': sum(v)/len(v)} for k, v in fpr.items()},
          'experimental_baselines': exp}

with open('/Users/prethamsai/results/experiments_results.json', 'w') as f:
    json.dump(output, f, indent=2, default=str)

print("\n\nAll results saved to results/experiments_results.json")
