#!/usr/bin/env python3
"""Test whether raising min_scale fixes pipeline FPR.
Uses only 5 mocks and 2000 molecules to keep it fast."""
import sys; sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, '/Users/prethamsai')
import numpy as np
from smlm_clustering.validation.benchmark_runner import generate_synthetic_dataset
from smlm_clustering.core.blinking_correction import BlinkingCorrector
from smlm_clustering.core.multiscale_detector import SMLMMultiscaleTest, ScaleRange
from smlm_clustering.core.null_models import BiologicalNullModel
import time, json

roi_nm = 5000.0
results = {k: [] for k in ['mol_pos', 'std_corr', 'min50nm', 'min80nm', 'min100nm']}

for trial in range(5):
    seed = 400 + trial
    t0 = time.time()
    # Smaller dataset = faster mocks
    ds = generate_synthetic_dataset(
        n_molecules=2000, f_clust=0.0, roi_nm=roi_nm,
        n_blinks_mean=5, blink_sigma_xy_nm=15.0, blink_sigma_z_nm=35.0,
        n_frames=500, seed=seed
    )
    corrector = BlinkingCorrector(r_lateral=30, r_axial=60, max_dark_frames=5)
    r = corrector.correct(ds.raw_positions, ds.raw_frames, ds.raw_photons)
    print(f"Trial {trial+1}: {ds.molecule_positions.shape[0]} mol, {r.n_raw} raw → {r.n_merged} merged", flush=True)

    for label, pos, min_nm in [
        ('mol_pos', ds.molecule_positions, 20),
        ('std_corr', r.positions, 20),
        ('min50nm', r.positions, 50),
        ('min80nm', r.positions, 80),
        ('min100nm', r.positions, 100),
    ]:
        null = BiologicalNullModel(seed=seed+1000)
        null.fit(pos)
        mocks = null.generate_mocks(5, len(pos))  # only 5 mocks
        roi_max = np.array([roi_nm, roi_nm, roi_nm * 0.1])
        sr = ScaleRange.for_smlm(min_nm=min_nm, max_nm=min(500, roi_nm/4),
                                  n_scales=12, roi_size_nm=roi_nm, grid_size=64)
        det = SMLMMultiscaleTest(roi_size_nm=roi_max, grid_size=64)
        res = det.test(pos, mocks, sr, shrinkage=0.1)
        pv = res['variance']['p_value']
        ps = res['skewness']['p_value']
        detected = pv < 0.05 or ps < 0.05
        results[label].append(detected)
        print(f"  {label}: det={detected} pv={pv:.3f} ps={ps:.3f}", flush=True)
    print(f"  ({time.time()-t0:.0f}s)", flush=True)

print("\n=== FPR SUMMARY (5 trials) ===")
for k, v in results.items():
    print(f"  {k}: {sum(v)}/{len(v)} = {100*sum(v)/len(v):.0f}%")

with open('/Users/prethamsai/results/fpr_fix_results.json', 'w') as f:
    json.dump({k: {'detected': sum(v), 'total': len(v)} for k, v in results.items()}, f, indent=2)
