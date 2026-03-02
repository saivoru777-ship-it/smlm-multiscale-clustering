# Multiscale Covariance-Aware Clustering Detection for SMLM

A statistical framework for detecting molecular clustering in single-molecule localization microscopy (SMLM) data using multiscale counts-in-cells analysis with covariance-aware chi-squared testing.

## Installation

```bash
pip install numpy scipy
pip install pytest  # for tests
```

## Usage

### Run the full validation study

```bash
cd /path/to/parent/  # directory containing smlm_clustering/
python3 -m smlm_clustering.validation.comprehensive_study --output-dir results/final
```

### Quick mode (reduced replicates, ~1-2 hours)

```bash
python3 -m smlm_clustering.validation.comprehensive_study --quick --output-dir results/final
```

### Regenerate figures only (from existing results)

```bash
python3 -m smlm_clustering.validation.comprehensive_study --figures-only --output-dir results/final --figures-dir figures
```

### Run tests

```bash
python3 -m pytest smlm_clustering/tests/ -v
```

## Directory Structure

```
smlm_clustering/
  core/
    multiscale_detector.py   # CIC gridding, chi-squared test
    blinking_correction.py   # UnionFind blink merging
    null_models.py           # Biological null (convex hull CSR)
    scale_range.py           # Logarithmic scale spacing
  data/
    loaders.py               # ThunderSTORM, SMLM Challenge CSV
  validation/
    comprehensive_study.py   # 8-experiment validation suite
    figure_generator.py      # Publication figure generation
    benchmark_runner.py      # Synthetic data + pipeline runner
    synthetic_scenarios.py   # Extended generators, biological presets
  tests/
    test_*.py                # Unit and integration tests
results/
  final/                     # Canonical results (200-rep FPR + quick B-H)
  quick/                     # Quick-mode results
figures/                     # Generated PDF figures
```
