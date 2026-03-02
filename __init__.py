"""
smlm_clustering
---------------
Multiscale clustering analysis for Single-Molecule Localization Microscopy data.

Adapted from a cosmic structure detection pipeline originally developed for
galaxy clustering analysis (cosmic_pattern_analysis).

Core pipeline:
    BlinkingCorrector       -- merge repeated blink events from the same molecule
    BiologicalNullModel     -- CSR null constrained to cell convex hull
    SMLMMultiscaleTest      -- covariance-aware chi-squared multiscale test

Validation:
    BenchmarkRunner         -- synthetic benchmark with sensitivity/FPR curves
    DBSCANBaseline          -- standard DBSCAN comparison
    RipleysKBaseline        -- Ripley's K/L function comparison
    ClusterMetrics          -- precision, recall, F1, Jaccard
"""

__version__ = "0.1.0"
