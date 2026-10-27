# ============================================================================
# Reporting — printing and figures
# ============================================================================
# This module owns the rebalancing summary and the five accumulation
# matplotlib figures. It performs no simulation.
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from montecarlo.engine import time_weighted_cagr

sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)


def print_rebalancing_summary(rebalancing_stats):
    """Print compact nested rebalancing statistics"""

    print(f"    Partial rebalances: {rebalancing_stats['total_partial']:,} "
          f"({rebalancing_stats['partial_pct']:.1f}%)")
    print(f"    Full rebalances: {rebalancing_stats['total_full']:,} "
          f"({rebalancing_stats['full_pct']:.1f}%)")
    print(f"    Total rebalancing events: {rebalancing_stats['total_events']:,}")
    # TURNOVER, which nothing reported until 2026-09-14. It is a portfolio
    # CHARACTERISTIC, not a diagnostic: VT trades exactly 0.00% because one fund
    # cannot drift, while a 9-fund book runs ~3%/yr. Measured at 0.15%
    # round-trip that is half a basis point a year — 0.3% over 58 years — so it
    # is reported and NOT charged. The number that matters is the tax one: 3%/yr
    # of realised gains in a taxable sleeve is roughly 60x the commission.
    if rebalancing_stats.get('turnover_per_year') is not None:
        t = rebalancing_stats['turnover_per_year']
        print(f"    Turnover: {t:.2%}/yr of average balance "
              f"(~{t * 0.0015:.3%}/yr at a 15bp round trip, not charged)")
    print(f"    Events per year:")
    print(f"        Mean: {rebalancing_stats['avg_events_per_year']:.2f}")
    print(f"        Median: {rebalancing_stats['median_events_per_year']:.2f}")

    print(f"\n    Quarterly Rebalancing Frequency:")
    quarterly = rebalancing_stats['quarterly_totals']
    total_events = rebalancing_stats['total_events']
    for quarter in ['Q1', 'Q2', 'Q3', 'Q4']:
        count = quarterly[quarter]
        pct = (count / total_events * 100) if total_events > 0 else 0
        print(f"        {quarter}: {count:6,} ({pct:5.1f}%)")

    print(f"\n    Q1-Q3 Decision Breakdown (Diagnostics):")
    reasons = rebalancing_stats['q1q3_decision_reasons']
    q1q3_total = sum(reasons.values())
    print(f"        Total Q1-Q3 rebalances: {q1q3_total:,}")
    if q1q3_total > 0:
        print(f"        Multiple breaches (≥2): {reasons['multiple_breaches']:,} ({reasons['multiple_breaches']/q1q3_total*100:5.1f}%)")
        print(f"        Single breach + near bands: {reasons['near_bands']:,} ({reasons['near_bands']/q1q3_total*100:5.1f}%)")
        print(f"        Partial (single breach only): {reasons['partial']:,} ({reasons['partial']/q1q3_total*100:5.1f}%)")

def build_figures(results, portfolios, selection, config, max_drawdowns):
    """Build the five comparison figures. Returns them as a list.

    `results[label]` supplies REDUCED statistics, not the raw simulation output:

        final_values   per-path final wealth, in dollars
        sortino_stats  the full aggregate_sortino_stats dict
        twr_cagr       per-path time-weighted CAGR (decimal)
        annual_vol     annualised volatility, in PERCENT
        median_path    the median wealth path, one point per month

    It used to take the raw `returns` and `paths` matrices and reduce them here.
    Those are n_sims x ~700 floats EACH — 56MB per strategy at 5,000 paths — and
    holding them for every strategy so that figures could be drawn at the end put
    a 12-strategy run near 700MB for two scalars and a median. The caller
    reduces and drops them now; nothing else ever needed them.
    """
    # Generate colors dynamically using matplotlib colormap
    cmap = plt.cm.get_cmap('tab20')  # tab20 provides 20 distinct colors
    num_portfolios = len(selection)
    colors = {label: cmap(i / max(num_portfolios - 1, 1)) for i, label in enumerate(selection)}

    print("  Generating figures...")

    figures = []

    # ============================================================================
    # FIGURE 1: Risk-Return Profile and Drawdowns
    # ============================================================================

    fig1 = plt.figure(figsize=(16, 9))
    gs1 = fig1.add_gridspec(1, 2, hspace=0.3, wspace=0.3, bottom=0.15)

    # Panel 1: Risk-Return Scatter
    ax1 = fig1.add_subplot(gs1[0, 0])

    for label in selection:
        median_return = np.median(results[label]['twr_cagr']) * 100
        annual_vol = results[label]['annual_vol']

        ax1.scatter(annual_vol, median_return, s=200, color=colors[label], edgecolors='black', linewidth=2, alpha=0.8, zorder=3)
        # Offset, not centred on the marker: `run.py` can hand this 12+
        # strategies and the labels are glide specs, so centred text overprinted
        # both the marker and its neighbours.
        ax1.annotate(label, (annual_vol, median_return), fontsize=7,
                     fontweight='bold', textcoords='offset points',
                     xytext=(0, 9), ha='center')

    ax1.set_xlabel('Volatility (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time-Weighted Return (% / yr, real)', fontsize=12, fontweight='bold')
    ax1.set_title('Risk-Return Profile', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Maximum Drawdown Box Plot
    ax2 = fig1.add_subplot(gs1[0, 1])

    drawdown_data = [max_drawdowns[label] for label in selection]
    bp = ax2.boxplot(drawdown_data, labels=selection, patch_artist=True, widths=0.6, showfliers=True, flierprops=dict(marker='o', markersize=3, alpha=0.5))
    ax2.tick_params(axis='x', labelrotation=45, labelsize=7)
    for tick in ax2.get_xticklabels():
        tick.set_horizontalalignment('right')

    for patch, label in zip(bp['boxes'], selection):
        patch.set_facecolor(colors[label])
        patch.set_alpha(0.7)

    for i, label in enumerate(selection):
        median_dd = np.median(max_drawdowns[label])
        ax2.plot(i+1, median_dd, 'D', color='black', markersize=8, zorder=4)

    ax2.axhline(-30, color='orange', linestyle='--', linewidth=2, label='Target: -30%', alpha=0.7)
    ax2.set_xlabel('Portfolio', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Max Drawdown (%)', fontsize=12, fontweight='bold')
    # NOT a hardcoded 50: the horizon is derived from your plan-to age, and this
    # title claimed 50 years on a 58-year run.
    ax2.set_title(f"Maximum Drawdown ({config['time_horizon_years']} Years)",
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    fig1.suptitle('Figure 1 of 5: Risk-Return Profile and Drawdowns', fontsize=16, fontweight='bold', y=0.98)
    figures.append(fig1)

    # ============================================================================
    # FIGURE 2: Sortino Distribution Comparison
    # ============================================================================

    n_portfolios = len(selection)
    n_cols = min(3, n_portfolios)
    n_rows = int(np.ceil(n_portfolios / n_cols))

    fig2 = plt.figure(figsize=(16, 9))
    gs2 = fig2.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3, bottom=0.15, top=0.92)

    for idx, label in enumerate(selection):
        i, j = divmod(idx, n_cols)
        ax = fig2.add_subplot(gs2[i, j])

        portfolio = portfolios[label]
        cost_alloc = portfolio['allocation'].get('COST', 0.0)

        sortino_ratios = results[label]['sortino_stats']['all_sortino_ratios']
        ax.hist(sortino_ratios, bins=50, alpha=0.7, color=colors[label], edgecolor='black', linewidth=0.5)
        ax.axvline(results[label]['sortino_stats']['sortino_median'], color='red', linestyle='--', linewidth=2, label=f"Median: {results[label]['sortino_stats']['sortino_median']:.2f}")
        ax.axvline(results[label]['sortino_stats']['sortino_10th'], color='orange', linestyle=':', linewidth=1.5, label=f"10th: {results[label]['sortino_stats']['sortino_10th']:.2f}")
        ax.axvline(results[label]['sortino_stats']['sortino_90th'], color='green', linestyle=':', linewidth=1.5, label=f"90th: {results[label]['sortino_stats']['sortino_90th']:.2f}")
        ax.set_xlabel('Sortino Ratio', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(f'Portfolio {label} ({cost_alloc:.0%} COST)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig2.suptitle('Figure 2 of 5: Per-Path Sortino Ratio Distributions (MAR=4.0%)', fontsize=16, fontweight='bold')
    figures.append(fig2)

    # ============================================================================
    # FIGURE 3: Final Wealth Distributions
    # ============================================================================

    fig3 = plt.figure(figsize=(16, 9))
    gs3 = fig3.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3, bottom=0.15, top=0.92)

    for idx, label in enumerate(selection):
        i, j = divmod(idx, n_cols)
        ax = fig3.add_subplot(gs3[i, j])

        portfolio = portfolios[label]
        cost_alloc = portfolio['allocation'].get('COST', 0.0)

        final_vals = results[label]['final_values'] / 1e6
        median_val = np.median(final_vals)
        p10_val = np.percentile(final_vals, 10)
        p90_val = np.percentile(final_vals, 90)
        p95_val = np.percentile(final_vals, 95)

        ax.hist(final_vals, bins=100, alpha=0.8, color=colors[label], edgecolor='black', linewidth=0.5)
        ax.axvline(median_val, color='red', linestyle='--', linewidth=2, label=f'Median: ${median_val:.2f}M')
        ax.axvline(p10_val, color='orange', linestyle=':', linewidth=1.5, label=f'10th: ${p10_val:.2f}M')
        ax.axvline(p90_val, color='green', linestyle=':', linewidth=1.5, label=f'90th: ${p90_val:.2f}M')
        ax.set_xlim(0, p95_val * 1.1)
        ax.set_xlabel('Final Value ($M, real)', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.set_title(f'Portfolio {label} ({cost_alloc:.0%} COST)', fontsize=10, fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig3.suptitle(f"Figure 3 of 5: {config['time_horizon_years']}-Year Final Wealth Distributions (real $)",
                  fontsize=16, fontweight='bold')
    figures.append(fig3)

    # ============================================================================
    # FIGURE 4: Median Growth Paths
    # ============================================================================

    fig4 = plt.figure(figsize=(16, 9))
    gs4 = fig4.add_gridspec(1, 1, hspace=0.3, wspace=0.3, bottom=0.15)
    ax = fig4.add_subplot(gs4[0, 0])

    months = np.arange(config['time_horizon_years'] * 12 + 1)
    years = config['age'] + months / 12

    for label in selection:
        portfolio = portfolios[label]
        cost_alloc = portfolio['allocation'].get('COST', 0.0)

        median_path = results[label]['median_path']
        ax.plot(years, median_path / 1e6, linewidth=2.5, label=f'{label}: {portfolio["name"][:20]} ({cost_alloc:.0%} COST)', color=colors[label])

    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Portfolio Value ($M, real)', fontsize=12)
    ax.set_title('Median Growth Paths (Monthly Steps)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3)

    fig4.suptitle('Figure 4 of 5: Median Growth Paths', fontsize=16, fontweight='bold', y=0.98)
    figures.append(fig4)

    # ============================================================================
    # FIGURE 5: Sortino & Wealth Comparison
    # ============================================================================

    fig5 = plt.figure(figsize=(16, 9))
    gs5 = fig5.add_gridspec(1, 2, hspace=0.3, wspace=0.3, bottom=0.15)

    sortino_medians = [results[label]['sortino_stats']['sortino_median'] for label in selection]
    wealth_medians = [np.median(results[label]['final_values']) / 1e6 for label in selection]

    # Left panel: Sortino by portfolio
    ax1 = fig5.add_subplot(gs5[0, 0])
    x = np.arange(len(selection))
    ax1.bar(x, sortino_medians, color=[colors[label] for label in selection], alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Portfolio', fontsize=11)
    ax1.set_ylabel('Median Sortino Ratio', fontsize=11)
    ax1.set_title('Median Sortino by Portfolio', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{label}\n({portfolios[label]["allocation"].get("COST", 0.0):.0%} COST)' for label in selection],
                        rotation=45, ha='right', fontsize=7)
    ax1.grid(axis='y', alpha=0.3)

    # Right panel: Wealth by portfolio
    ax2 = fig5.add_subplot(gs5[0, 1])
    ax2.bar(x, wealth_medians, color=[colors[label] for label in selection], alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Portfolio', fontsize=11)
    ax2.set_ylabel('Median Wealth ($M, real)', fontsize=11)
    ax2.set_title('Median Wealth by Portfolio', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{label}\n({portfolios[label]["allocation"].get("COST", 0.0):.0%} COST)' for label in selection],
                        rotation=45, ha='right', fontsize=7)
    ax2.grid(axis='y', alpha=0.3)

    fig5.suptitle('Figure 5 of 5: Sortino & Wealth Comparison', fontsize=16, fontweight='bold', y=0.98)
    figures.append(fig5)

    return figures


def save_figures(figures, outdir, prefix='accumulation'):
    """Write the figures to PNG and return the paths.

    Saved rather than shown, matching report_retirement.figures: a run should
    not block on a window being closed, and the tables are the primary output.
    """
    from datetime import datetime
    from pathlib import Path

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d_%H%M')
    written = []
    for i, fig in enumerate(figures, 1):
        path = outdir / f"{prefix}_{stamp}_fig{i}.png"
        fig.savefig(path, dpi=110, bbox_inches='tight')
        plt.close(fig)
        written.append(path)
    return written
