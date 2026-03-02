"""
Publication figure generator for the SMLM multiscale clustering study.

Generates Figures 2-6 (main) and S1-S4 (supplementary) as PDF files
from saved JSON experiment results.

Style: Arial/Helvetica, 8pt, 300 DPI, colorblind-safe palette (Wong 2011).

Usage:
    from smlm_clustering.validation.figure_generator import generate_all_figures
    generate_all_figures(results_dir='results', figures_dir='figures')
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ============================================================================
# Style constants
# ============================================================================

# Wong 2011 colorblind-safe palette
COLORS = {
    'multiscale': '#0072B2',
    'dbscan': '#D55E00',
    'ripley': '#009E73',
    'neutral': '#999999',
    'highlight': '#E69F00',
    'fill': '#56B4E9',
}

# Bioinformatics style
STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.0,
    'lines.markersize': 4,
}

# Bioinformatics column widths (inches)
SINGLE_COL = 3.35
DOUBLE_COL = 7.0


def _apply_style():
    """Apply publication style globally."""
    plt.rcParams.update(STYLE)


def _load_json(filepath: Path) -> Optional[dict]:
    """Load JSON, returning None if file doesn't exist."""
    if not filepath.exists():
        print(f"  Warning: {filepath} not found, skipping figure")
        return None
    with open(filepath) as f:
        return json.load(f)


def _save_fig(fig, filepath: Path):
    """Save figure as PDF."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, format='pdf')
    plt.close(fig)
    print(f"  Saved: {filepath}")


# ============================================================================
# Figure 2: FPR and Sensitivity (A, B, C)
# ============================================================================

def figure_2(results_dir: Path, figures_dir: Path):
    """Fig 2: A) p-value QQ plot, B) p-value histogram, C) FPR calibration with CI, D) dose-response."""
    _apply_style()

    fpr_data = _load_json(results_dir / 'A_fpr.json')
    sens_data = _load_json(results_dir / 'B_sensitivity.json')
    if fpr_data is None or sens_data is None:
        return

    fig, axes = plt.subplots(1, 4, figsize=(DOUBLE_COL, 2.2))

    p_var = np.array(fpr_data['p_values_variance'])
    p_skew = np.array(fpr_data['p_values_skewness'])
    n_reps = fpr_data['n_replicates']

    # 2A: QQ plot — most rigorous uniformity check
    ax = axes[0]
    for p_vals, color, label in [
        (p_var, COLORS['multiscale'], 'Variance'),
        (p_skew, COLORS['fill'], 'Skewness'),
    ]:
        sorted_p = np.sort(p_vals)
        n = len(sorted_p)
        expected = (np.arange(1, n + 1) - 0.5) / n
        ax.scatter(expected, sorted_p, s=6, color=color, alpha=0.7, label=label)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, label='Uniform')
    ax.set_xlabel('Expected quantile')
    ax.set_ylabel('Observed p-value')
    ax.set_title('A', loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=5)
    ax.set_aspect('equal')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.annotate(f'n={n_reps}', xy=(0.95, 0.05), xycoords='axes fraction',
                ha='right', fontsize=5, color=COLORS['neutral'])

    # 2B: P-value histogram (should be uniform under null)
    ax = axes[1]
    ax.hist(p_var, bins=10, range=(0, 1), alpha=0.7,
            color=COLORS['multiscale'], label='Variance', density=True)
    ax.hist(p_skew, bins=10, range=(0, 1), alpha=0.5,
            color=COLORS['fill'], label='Skewness', density=True)
    ax.axhline(1.0, color='k', linestyle='--', linewidth=0.5, label='Uniform')
    ax.set_xlabel('p-value')
    ax.set_ylabel('Density')
    ax.set_title('B', loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=5)

    # 2C: FPR calibration plot with 95% binomial CI
    ax = axes[2]
    cal = fpr_data['fpr_calibration']
    alphas = sorted([float(a) for a in cal.keys()])
    ms_fpr = [cal[str(a)]['multiscale'] for a in alphas]
    db_fpr = [cal[str(a)]['dbscan'] for a in alphas]
    rip_fpr = [cal[str(a)]['ripley'] for a in alphas]

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5, label='Ideal')

    # Add binomial CI shading for multiscale if available
    ci_data = fpr_data.get('binomial_ci', {})
    if ci_data:
        ci_lo = [ci_data[str(a)]['ci_lower'] for a in alphas if str(a) in ci_data]
        ci_hi = [ci_data[str(a)]['ci_upper'] for a in alphas if str(a) in ci_data]
        ci_alphas = [a for a in alphas if str(a) in ci_data]
        ax.fill_between(ci_alphas, ci_lo, ci_hi,
                        color=COLORS['multiscale'], alpha=0.15, label='95% CI')

    ax.plot(alphas, ms_fpr, 'o-', color=COLORS['multiscale'],
            label='Multiscale', markersize=3)
    ax.plot(alphas, db_fpr, 's-', color=COLORS['dbscan'],
            label='DBSCAN', markersize=3)
    ax.plot(alphas, rip_fpr, '^-', color=COLORS['ripley'],
            label="Ripley's K", markersize=3)
    if 'multiscale_pipeline' in cal[str(alphas[0])]:
        ms_pipe_fpr = [cal[str(a)]['multiscale_pipeline'] for a in alphas]
        ax.plot(alphas, ms_pipe_fpr, 'x--', color=COLORS['highlight'],
                label='MS (pipeline)', markersize=3)
    ax.set_xlabel('Nominal $\\alpha$')
    ax.set_ylabel('Observed FPR')
    ax.set_title('C', loc='left', fontweight='bold')
    ax.legend(frameon=False, loc='upper left', fontsize=5)
    ax.set_xlim(-0.01, 0.22)
    ax.set_ylim(-0.01, max(0.25, max(ms_fpr) * 1.1 if ms_fpr else 0.25))

    # 2D: Dose-response (sensitivity)
    ax = axes[3]
    f_values = [float(f) for f in sens_data['f_values']]
    ms_tpr = [sens_data['results'][str(f)]['ms_tpr'] for f in f_values]
    db_tpr = [sens_data['results'][str(f)]['dbscan_tpr'] for f in f_values]
    rip_tpr = [sens_data['results'][str(f)]['ripley_tpr'] for f in f_values]

    ax.plot(f_values, ms_tpr, 'o-', color=COLORS['multiscale'],
            label='Multiscale', markersize=3)
    ax.plot(f_values, db_tpr, 's-', color=COLORS['dbscan'],
            label='DBSCAN', markersize=3)
    ax.plot(f_values, rip_tpr, '^-', color=COLORS['ripley'],
            label="Ripley's K", markersize=3)
    ax.set_xlabel('Cluster fraction $f_{clust}$')
    ax.set_ylabel('Detection rate (TPR)')
    ax.set_title('D', loc='left', fontweight='bold')
    ax.legend(frameon=False, fontsize=5)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    _save_fig(fig, figures_dir / 'fig2_fpr_sensitivity.pdf')


# ============================================================================
# Figure 3: Parameter Sensitivity (2x2)
# ============================================================================

def figure_3(results_dir: Path, figures_dir: Path):
    """Fig 3: Parameter sensitivity — grid, shrinkage, mocks, blink radius."""
    _apply_style()

    data = _load_json(results_dir / 'C_parameter_sensitivity.json')
    if data is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.0))
    sweeps = data['sweeps']

    # Helper for FPR/TPR twin plots
    def _plot_fpr_tpr(ax, param_dict, xlabel, title_letter):
        params = sorted([float(k) for k in param_dict.keys()])
        fprs = [param_dict[str(int(p) if p == int(p) else p)]['fpr']
                for p in params]
        tprs = [param_dict[str(int(p) if p == int(p) else p)]['tpr']
                for p in params]
        ax.plot(params, fprs, 'o--', color=COLORS['dbscan'],
                label='FPR (null)', markersize=3)
        ax.plot(params, tprs, 's-', color=COLORS['multiscale'],
                label='TPR ($f_{clust}$=0.15)', markersize=3)
        ax.axhline(0.05, color='k', linestyle=':', linewidth=0.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Rate')
        ax.set_title(title_letter, loc='left', fontweight='bold')
        ax.legend(frameon=False, fontsize=6)
        ax.set_ylim(-0.05, 1.05)

    # 3A: Grid size
    _plot_fpr_tpr(axes[0, 0], sweeps['grid_size'], 'Grid size', 'A')

    # 3B: Shrinkage
    _plot_fpr_tpr(axes[0, 1], sweeps['shrinkage'], 'Shrinkage $\\lambda$', 'B')
    axes[0, 1].set_xscale('log')

    # 3C: Number of mocks
    _plot_fpr_tpr(axes[1, 0], sweeps['n_mocks'], 'Number of mocks', 'C')

    # 3D: Blinking correction radius
    _plot_fpr_tpr(axes[1, 1], sweeps['blink_radius'],
                  'Blink radius $r_{lat}$ (nm)', 'D')
    axes[1, 1].annotate('with blinking\ncorrection',
                        xy=(0.5, 0.5), xycoords='axes fraction',
                        ha='center', fontsize=6, color=COLORS['neutral'])

    fig.tight_layout()
    _save_fig(fig, figures_dir / 'fig3_parameter_sensitivity.pdf')


# ============================================================================
# Figure 4: Cluster Geometry Robustness
# ============================================================================

def figure_4(results_dir: Path, figures_dir: Path):
    """Fig 4: A) cluster radius, B) n_clusters, C) cluster shape, D) hard regime."""
    _apply_style()

    data = _load_json(results_dir / 'D_geometry.json')
    if data is None:
        return

    results = data['results']
    has_hard = 'hard_regime' in results
    if has_hard:
        fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.0))
        ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.2))
        ax_a, ax_b, ax_c = axes[0], axes[1], axes[2]

    # 4A: Cluster radius
    radii = sorted([int(k) for k in results['cluster_radius'].keys()])
    for method, color, marker, label in [
        ('ms_tpr', COLORS['multiscale'], 'o', 'Multiscale'),
        ('dbscan_tpr', COLORS['dbscan'], 's', 'DBSCAN'),
        ('ripley_tpr', COLORS['ripley'], '^', "Ripley's K"),
    ]:
        vals = [results['cluster_radius'][str(r)][method] for r in radii]
        ax_a.plot(radii, vals, f'{marker}-', color=color, label=label, markersize=3)
    ax_a.set_xlabel('Cluster radius (nm)')
    ax_a.set_ylabel('Detection rate')
    ax_a.set_title('A', loc='left', fontweight='bold')
    ax_a.legend(frameon=False, fontsize=6)
    ax_a.set_xscale('log')
    ax_a.set_ylim(-0.05, 1.05)

    # 4B: Number of clusters
    nclusts = sorted([int(k) for k in results['n_clusters'].keys()])
    for method, color, marker, label in [
        ('ms_tpr', COLORS['multiscale'], 'o', 'Multiscale'),
        ('dbscan_tpr', COLORS['dbscan'], 's', 'DBSCAN'),
        ('ripley_tpr', COLORS['ripley'], '^', "Ripley's K"),
    ]:
        vals = [results['n_clusters'][str(nc)][method] for nc in nclusts]
        ax_b.plot(nclusts, vals, f'{marker}-', color=color, label=label, markersize=3)
    ax_b.set_xlabel('Number of clusters')
    ax_b.set_ylabel('Detection rate')
    ax_b.set_title('B', loc='left', fontweight='bold')
    ax_b.set_xscale('log')
    ax_b.set_ylim(-0.05, 1.05)

    # 4C: Cluster shape (bar chart)
    shapes = list(results['cluster_shape'].keys())
    x = np.arange(len(shapes))
    width = 0.25
    for i, (method, color, label) in enumerate([
        ('ms_tpr', COLORS['multiscale'], 'Multiscale'),
        ('dbscan_tpr', COLORS['dbscan'], 'DBSCAN'),
        ('ripley_tpr', COLORS['ripley'], "Ripley's K"),
    ]):
        vals = [results['cluster_shape'][s][method] for s in shapes]
        ax_c.bar(x + i * width, vals, width, color=color, label=label)
    ax_c.set_xticks(x + width)
    ax_c.set_xticklabels(shapes, fontsize=6)
    ax_c.set_ylabel('Detection rate')
    ax_c.set_title('C', loc='left', fontweight='bold')
    ax_c.legend(frameon=False, fontsize=6)
    ax_c.set_ylim(0, 1.15)

    # 4D: Hard regime — shows graceful degradation
    if has_hard:
        hard = results['hard_regime']
        labels = list(hard.keys())
        tprs = [hard[k]['ms_tpr'] for k in labels]
        x = np.arange(len(labels))
        colors = [COLORS['multiscale'] if t > 0.5 else
                  COLORS['highlight'] if t > 0 else
                  COLORS['neutral'] for t in tprs]
        ax_d.bar(x, tprs, color=colors)
        ax_d.set_xticks(x)
        ax_d.set_xticklabels(labels, fontsize=5, rotation=45, ha='right')
        ax_d.set_ylabel('Detection rate')
        ax_d.set_title('D  Hard regime', loc='left', fontweight='bold',
                        fontsize=8)
        ax_d.set_ylim(0, 1.15)
        ax_d.axhline(0.5, color='k', linestyle=':', linewidth=0.5)

    fig.tight_layout()
    _save_fig(fig, figures_dir / 'fig4_geometry.pdf')


# ============================================================================
# Figure 5: Biological Use Cases (4 rows x 3 cols)
# ============================================================================

def figure_5(results_dir: Path, figures_dir: Path):
    """Fig 5: Biological use cases — scatter, variance+envelope, DBSCAN."""
    _apply_style()

    data = _load_json(results_dir / 'F_biological_cases.json')
    sig_data = _load_json(results_dir / 'G_multiscale_signatures.json')
    if data is None:
        return

    presets = ['synaptic_receptors', 'nuclear_pores',
               'membrane_domains', 'negative_control']
    n_rows = len(presets)
    fig, axes = plt.subplots(n_rows, 3, figsize=(DOUBLE_COL, n_rows * 1.8))

    for row, preset_name in enumerate(presets):
        bio = data['results'][preset_name]
        positions = np.array(bio['positions'])
        dbscan_labels = np.array(bio['dbscan_labels'])
        pr = bio['pipeline_result']

        # Col 1: XY scatter (raw positions colored by molecule identity)
        ax = axes[row, 0]
        ax.scatter(positions[:, 0], positions[:, 1], s=0.3,
                   c=COLORS['neutral'], alpha=0.3, rasterized=True)
        ax.set_aspect('equal')
        ax.set_xlabel('x (nm)')
        ax.set_ylabel('y (nm)')
        label = preset_name.replace('_', ' ').title()
        ax.set_title(label, fontsize=7, fontweight='bold')

        # Col 2: Variance curve + null envelope
        ax = axes[row, 1]
        if sig_data and preset_name in sig_data.get('results', {}):
            sig = sig_data['results'][preset_name]['variance']
            scales = sig['scales_nm']
            values = sig['values']
            ax.plot(scales, values, '-', color=COLORS['multiscale'],
                    linewidth=1.2, label='Data')
            if sig.get('envelope'):
                env = sig['envelope']
                ax.fill_between(scales, env['p5'], env['p95'],
                                color=COLORS['fill'], alpha=0.3,
                                label='Null 5-95%')
                ax.plot(scales, env['median'], '--',
                        color=COLORS['neutral'], linewidth=0.5)
            ax.set_xlabel('Scale (nm)')
            ax.set_ylabel('Variance / mean')
            ax.legend(frameon=False, fontsize=5)
        else:
            # Fallback: use curves from pipeline result
            var_curve = pr.get('var_curve', {})
            if var_curve and var_curve.get('scales_nm'):
                ax.plot(var_curve['scales_nm'], var_curve['values'],
                        '-', color=COLORS['multiscale'])
            ax.set_xlabel('Scale (nm)')
            ax.set_ylabel('Variance / mean')

        # Col 3: DBSCAN scatter
        ax = axes[row, 2]
        cluster_mask = dbscan_labels >= 0
        noise_mask = ~cluster_mask
        if noise_mask.any():
            ax.scatter(positions[noise_mask, 0], positions[noise_mask, 1],
                       s=0.3, c=COLORS['neutral'], alpha=0.2, rasterized=True)
        if cluster_mask.any():
            ax.scatter(positions[cluster_mask, 0], positions[cluster_mask, 1],
                       s=0.3, c=COLORS['dbscan'], alpha=0.5, rasterized=True)
        ax.set_aspect('equal')
        ax.set_xlabel('x (nm)')
        n_clust = len(set(dbscan_labels[cluster_mask])) if cluster_mask.any() else 0
        ax.set_title(f'DBSCAN ({n_clust} clusters)', fontsize=6)

    fig.tight_layout()
    _save_fig(fig, figures_dir / 'fig5_biological.pdf')


# ============================================================================
# Figure 6: Multiscale Signatures
# ============================================================================

def figure_6(results_dir: Path, figures_dir: Path):
    """Fig 6: A) variance vs scale, B) skewness vs scale (all bio scenarios)."""
    _apply_style()

    data = _load_json(results_dir / 'G_multiscale_signatures.json')
    if data is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.5))

    preset_colors = {
        'synaptic_receptors': COLORS['multiscale'],
        'nuclear_pores': COLORS['dbscan'],
        'membrane_domains': COLORS['ripley'],
        'negative_control': COLORS['neutral'],
    }
    preset_labels = {
        'synaptic_receptors': 'Synaptic',
        'nuclear_pores': 'Nuclear pores',
        'membrane_domains': 'Membrane',
        'negative_control': 'Control (CSR)',
    }

    for metric, ax, ylabel, title_letter in [
        ('variance', axes[0], 'Variance / mean', 'A'),
        ('skewness', axes[1], 'Skewness', 'B'),
    ]:
        for preset_name, color in preset_colors.items():
            if preset_name not in data['results']:
                continue
            sig = data['results'][preset_name][metric]
            scales = sig['scales_nm']
            values = sig['values']
            label = preset_labels[preset_name]
            ax.plot(scales, values, '-', color=color, label=label,
                    linewidth=1.2)

            # Show envelope for negative control only
            if preset_name == 'negative_control' and sig.get('envelope'):
                env = sig['envelope']
                ax.fill_between(scales, env['p5'], env['p95'],
                                color=color, alpha=0.15)

        ax.set_xlabel('Scale (nm)')
        ax.set_ylabel(ylabel)
        ax.set_title(title_letter, loc='left', fontweight='bold')
        ax.legend(frameon=False, fontsize=6)

    fig.tight_layout()
    _save_fig(fig, figures_dir / 'fig6_signatures.pdf')


# ============================================================================
# Supplementary Figures
# ============================================================================

def figure_s1(results_dir: Path, figures_dir: Path):
    """Fig S1: Blinking correction efficiency vs parameters."""
    _apply_style()

    data = _load_json(results_dir / 'E_realistic_conditions.json')
    if data is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL * 2, 2.2))

    # S1A: TPR vs blinking rate
    ax = axes[0]
    blink_data = data['results'].get('blinking_rate', {})
    if blink_data:
        rates = sorted([int(k) for k in blink_data.keys()])
        tprs = [blink_data[str(r)]['ms_tpr'] for r in rates]
        reductions = [blink_data[str(r)].get('reduction', 1.0) for r in rates]
        ax.plot(rates, tprs, 'o-', color=COLORS['multiscale'], markersize=3)
        ax.set_xlabel('Mean blinks per molecule')
        ax.set_ylabel('Detection rate (TPR)')
        ax.set_title('A', loc='left', fontweight='bold')
        ax.set_ylim(-0.05, 1.05)

    # S1B: Reduction factor vs blinking rate
    ax = axes[1]
    if blink_data:
        ax.plot(rates, reductions, 's-', color=COLORS['highlight'], markersize=3)
        ax.set_xlabel('Mean blinks per molecule')
        ax.set_ylabel('Reduction factor (raw/merged)')
        ax.set_title('B', loc='left', fontweight='bold')

    fig.tight_layout()
    _save_fig(fig, figures_dir / 'figS1_blinking.pdf')


def figure_s2(results_dir: Path, figures_dir: Path):
    """Fig S2: Localization precision and frame count effects."""
    _apply_style()

    data = _load_json(results_dir / 'E_realistic_conditions.json')
    if data is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL * 2, 2.2))

    # S2A: Localization precision
    ax = axes[0]
    prec_data = data['results'].get('localization_precision', {})
    if prec_data:
        sigmas = sorted([int(k) for k in prec_data.keys()])
        tprs = [prec_data[str(s)]['ms_tpr'] for s in sigmas]
        ax.plot(sigmas, tprs, 'o-', color=COLORS['multiscale'], markersize=3)
        ax.set_xlabel('Localization precision $\\sigma_{xy}$ (nm)')
        ax.set_ylabel('Detection rate (TPR)')
        ax.set_title('A', loc='left', fontweight='bold')
        ax.set_ylim(-0.05, 1.05)

    # S2B: Frame count
    ax = axes[1]
    frame_data = data['results'].get('frame_count', {})
    if frame_data:
        frames = sorted([int(k) for k in frame_data.keys()])
        tprs = [frame_data[str(f)]['ms_tpr'] for f in frames]
        ax.plot(frames, tprs, 'o-', color=COLORS['multiscale'], markersize=3)
        ax.set_xlabel('Number of frames')
        ax.set_ylabel('Detection rate (TPR)')
        ax.set_title('B', loc='left', fontweight='bold')
        ax.set_xscale('log')
        ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    _save_fig(fig, figures_dir / 'figS2_conditions.pdf')


def figure_s3(results_dir: Path, figures_dir: Path):
    """Fig S3: Condition number vs shrinkage."""
    _apply_style()

    data = _load_json(results_dir / 'C_parameter_sensitivity.json')
    if data is None:
        return

    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL, 2.2))

    shrink_data = data['sweeps'].get('shrinkage', {})
    if shrink_data:
        lambdas = sorted([float(k) for k in shrink_data.keys()])
        for lam in lambdas:
            conds = shrink_data[str(lam)].get('condition_numbers', [])
            if conds:
                median_cond = np.median(conds)
                ax.scatter([lam] * len(conds), conds,
                           s=8, alpha=0.3, color=COLORS['multiscale'])
                ax.scatter(lam, median_cond, s=20, color=COLORS['dbscan'],
                           marker='_', linewidths=2, zorder=5)

        ax.set_xlabel('Shrinkage $\\lambda$')
        ax.set_ylabel('Condition number')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Covariance matrix conditioning', fontsize=8)

    fig.tight_layout()
    _save_fig(fig, figures_dir / 'figS3_condition_number.pdf')


def figure_s4(results_dir: Path, figures_dir: Path):
    """Fig S4: Runtime scaling vs dataset size."""
    _apply_style()

    data = _load_json(results_dir / 'H_runtime.json')
    if data is None:
        return

    fig, ax = plt.subplots(1, 1, figsize=(SINGLE_COL, 2.2))

    rt = data['results']
    sizes = sorted([int(k) for k in rt.keys()])
    n_locs = [rt[str(s)]['n_localizations'] for s in sizes]
    times = [rt[str(s)]['wall_clock_seconds'] for s in sizes]

    ax.plot(n_locs, times, 'o-', color=COLORS['multiscale'], markersize=4)
    ax.set_xlabel('Number of localizations')
    ax.set_ylabel('Wall-clock time (s)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Pipeline runtime scaling', fontsize=8)

    fig.tight_layout()
    _save_fig(fig, figures_dir / 'figS4_runtime.pdf')


# ============================================================================
# Generate all
# ============================================================================

def generate_all_figures(results_dir: str = 'results',
                         figures_dir: str = 'figures'):
    """Generate all publication figures from saved JSON results."""
    results_path = Path(results_dir)
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating publication figures...")

    figure_2(results_path, figures_path)
    figure_3(results_path, figures_path)
    figure_4(results_path, figures_path)
    figure_5(results_path, figures_path)
    figure_6(results_path, figures_path)
    figure_s1(results_path, figures_path)
    figure_s2(results_path, figures_path)
    figure_s3(results_path, figures_path)
    figure_s4(results_path, figures_path)

    print("Done.")
