"""
Run experiments B-H at full replicates (skip A which we already have at 200 reps).
Expected runtime: ~3-4 hours.
"""
import sys, json, time
sys.path.insert(0, '/Users/prethamsai')

from smlm_clustering.validation.comprehensive_study import ComprehensiveStudy

start = time.time()

# Full mode but we'll only run B-H
study = ComprehensiveStudy(output_dir='/Users/prethamsai/results/full', quick=False)

print("Running experiments B-H at FULL replicates (skipping A)")
print("="*60)

# B: Sensitivity (20 reps per f_clust value)
print("\nStarting Experiment B...")
r_b = study.run_sensitivity()
elapsed = (time.time() - start) / 60
print(f"  B done in {elapsed:.1f} min")

# C: Parameter sensitivity (20 reps)
print("\nStarting Experiment C...")
r_c = study.run_parameter_sensitivity()
elapsed = (time.time() - start) / 60
print(f"  C done in {elapsed:.1f} min")

# D: Geometry (10 reps)
print("\nStarting Experiment D...")
r_d = study.run_geometry()
elapsed = (time.time() - start) / 60
print(f"  D done in {elapsed:.1f} min")

# E: Realistic conditions (10 reps)
print("\nStarting Experiment E...")
r_e = study.run_realistic_conditions()
elapsed = (time.time() - start) / 60
print(f"  E done in {elapsed:.1f} min")

# F: Biological cases
print("\nStarting Experiment F...")
r_f = study.run_biological_cases()
elapsed = (time.time() - start) / 60
print(f"  F done in {elapsed:.1f} min")

# G: Signatures (50 mocks)
print("\nStarting Experiment G...")
r_g = study.run_multiscale_signatures()
elapsed = (time.time() - start) / 60
print(f"  G done in {elapsed:.1f} min")

# H: Runtime
print("\nStarting Experiment H...")
r_h = study.run_runtime()
elapsed = (time.time() - start) / 60
print(f"  H done in {elapsed:.1f} min")

total = (time.time() - start) / 60
print(f"\nAll experiments B-H complete in {total:.1f} min")

# Copy the existing 200-rep A results into full/
import shutil, os
src = '/Users/prethamsai/results_200/A_fpr.json'
dst = '/Users/prethamsai/results/full/A_fpr.json'
if os.path.exists(src) and not os.path.exists(dst):
    shutil.copy2(src, dst)
    print(f"Copied 200-rep A_fpr.json to results/full/")
