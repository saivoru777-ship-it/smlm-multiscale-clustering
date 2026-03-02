# Figure Captions

## Main Figures

### Figure 2 (`figures/fig2_fpr_sensitivity.pdf`)

**False positive rate calibration and sensitivity.**
**(A)** QQ plot of p-value distributions from 200 CSR datasets against the uniform distribution. Approximate linearity confirms calibration.
**(B)** Histogram of p-values (variance and skewness) showing approximate uniformity.
**(C)** Combined FPR (OR rule) vs. significance threshold alpha, with 95% binomial confidence intervals (shading). The diagonal represents perfect calibration. FPR inflation above the diagonal is expected from the OR combination of two tests.
**(D)** Detection power (TPR) vs. cluster fraction f for the multiscale test, DBSCAN (permutation-calibrated), and Ripley's L function (envelope test).

### Figure 3 (`figures/fig3_parameter_sensitivity.pdf`)

**Parameter sensitivity.**
FPR (circles) and TPR (squares) for sweeps of grid size, shrinkage parameter lambda, number of mocks, and blinking correction radius. The test is robust across all parameter ranges, with FPR <= 0.33 and TPR = 1.0 for the statistical test parameters (grid size, shrinkage, mocks). Pipeline FPR = 1.0 for all blinking radii reflects a blinking simulation artifact.

### Figure 4 (`figures/fig4_geometry.pdf`)

**Geometry robustness and hard regime.**
**(A)** TPR vs. cluster radius (10-500 nm).
**(B)** TPR vs. number of clusters (5-100).
**(C)** TPR across cluster shapes (Gaussian, elongated, ring-shaped).
**(D)** Hard regime: TPR for seven configurations with low cluster fractions and/or small radii, showing graceful degradation from TPR = 1.0 at f = 0.02 to TPR = 0.0 at f = 0.005.

### Figure 5 (`figures/fig5_biological.pdf`)

**Biological use cases.**
Point cloud visualizations and detection results for four biologically motivated scenarios. Synaptic receptors (small dense clusters), nuclear pores (ring-shaped), and membrane domains (large sparse clusters) are all correctly detected (p < 0.001). The negative control (pure CSR) is correctly not detected (p_var = 0.849, p_skew = 0.249).

### Figure 6 (`figures/fig6_signatures.pdf`)

**Multiscale variance signatures.**
Variance-to-mean ratio curves across spatial scales for the four biological scenarios. Different cluster morphologies produce distinct scale-dependent signatures. The mock envelope (gray shading) represents the 95% range from null datasets. The negative control remains within the envelope at all scales.

## Supplementary Figures

### Figure S1 (`figures/figS1_blinking.pdf`)

**Blinking correction performance.**
Reduction factors (raw localizations / merged molecules) as a function of blinking rate. Even at 20 blinks per molecule, the correction reduces the localization count by only ~10%, indicating conservative merging. Detection power (TPR) remains 1.0 across all blinking rates.

### Figure S2 (`figures/figS2_conditions.pdf`)

**Realistic SMLM conditions.**
**(A)** Detection power vs. localization precision (sigma_xy = 5-100 nm). TPR = 1.0 across all precisions.
**(B)** Detection power vs. number of imaging frames (20-2000). TPR = 1.0 regardless of frame count.

### Figure S3 (`figures/figS3_condition_number.pdf`)

**Covariance matrix condition numbers.**
Condition numbers of the regularized covariance matrix as a function of the shrinkage parameter lambda. With lambda = 0.1 (default), condition numbers remain below 10^3, ensuring stable matrix inversion.

### Figure S4 (`figures/figS4_runtime.pdf`)

**Runtime scaling.**
Wall-clock time for the full pipeline as a function of the number of molecules (500-5000). Scaling is near-linear, with 5000 molecules processed in approximately 21 seconds.
