#!/usr/bin/env python3
"""
06_state_its_analysis.py
Run interrupted time series analysis for each state.

Extends 03_its_analysis.py to run the same ITS model on state-level data,
producing forest plots and heterogeneity statistics.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
DATA_FILE = DATA_DIR / "state_overdose_monthly.csv"
RESULTS_FILE = DATA_DIR / "state_its_results.csv"


def run_state_its(df_state, outcome_col):
    """
    Run ITS regression for a single state.

    Returns dict with coefficients, SEs, p-values, or None if model fails.
    """
    # Drop missing values for this outcome
    df_clean = df_state.dropna(subset=[outcome_col]).copy()

    if len(df_clean) < 20:  # Minimum observations needed
        return None

    # Check we have data on both sides of intervention
    if df_clean['post_intervention'].sum() == 0 or df_clean['post_intervention'].sum() == len(df_clean):
        return None

    # Prepare variables
    y = df_clean[outcome_col].values
    time = df_clean['time'].values
    post = df_clean['post_intervention'].values
    time_post = time * post

    # Create design matrix
    X = np.column_stack([np.ones(len(y)), time, post, time_post])

    try:
        # Fit OLS with Newey-West HAC standard errors
        model = sm.OLS(y, X)
        # Use fewer lags for smaller samples
        max_lags = min(6, len(y) // 10)
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': max(1, max_lags)})

        return {
            'n_obs': len(df_clean),
            'beta0': results.params[0],  # Intercept
            'beta1': results.params[1],  # Pre-trend
            'beta2': results.params[2],  # Level change
            'beta2_se': results.bse[2],
            'beta2_pval': results.pvalues[2],
            'beta3': results.params[3],  # Slope change
            'beta3_se': results.bse[3],
            'beta3_pval': results.pvalues[3],
            'rsquared': results.rsquared,
        }
    except Exception as e:
        print(f"    Model failed: {e}")
        return None


def run_all_states(df, outcome_col, outcome_label):
    """
    Run ITS for all states for a given outcome.

    Returns DataFrame with results for each state.
    """
    print(f"\n{'='*70}")
    print(f"State-Level ITS: {outcome_label}")
    print(f"{'='*70}")

    states = df['state'].unique()
    results_list = []

    for state in sorted(states):
        df_state = df[df['state'] == state].copy()
        state_name = df_state['state_name'].iloc[0]

        result = run_state_its(df_state, outcome_col)

        if result is not None:
            result['state'] = state
            result['state_name'] = state_name
            result['outcome'] = outcome_col
            results_list.append(result)
            print(f"  {state} ({state_name}): n={result['n_obs']}, "
                  f"beta3={result['beta3']:+.1f} (p={result['beta3_pval']:.3f})")
        else:
            print(f"  {state} ({state_name}): SKIPPED (insufficient data)")

    if not results_list:
        return pd.DataFrame()

    results_df = pd.DataFrame(results_list)

    # Reorder columns
    cols = ['state', 'state_name', 'outcome', 'n_obs',
            'beta2', 'beta2_se', 'beta2_pval',
            'beta3', 'beta3_se', 'beta3_pval',
            'beta0', 'beta1', 'rsquared']
    results_df = results_df[cols]

    return results_df


def compute_summary_stats(results_df, national_beta3=None):
    """
    Compute summary statistics across states.

    Returns dict with heterogeneity metrics.
    """
    # Filter to valid results
    valid = results_df.dropna(subset=['beta3', 'beta3_se'])

    if len(valid) == 0:
        return {}

    # Count significant effects
    sig_neg = (valid['beta3_pval'] < 0.05) & (valid['beta3'] < 0)
    sig_pos = (valid['beta3_pval'] < 0.05) & (valid['beta3'] > 0)
    not_sig = valid['beta3_pval'] >= 0.05

    # Weighted mean (inverse variance weighting)
    weights = 1 / (valid['beta3_se'] ** 2)
    weighted_mean = (valid['beta3'] * weights).sum() / weights.sum()
    weighted_se = np.sqrt(1 / weights.sum())

    # Simple mean and SD
    simple_mean = valid['beta3'].mean()
    simple_sd = valid['beta3'].std()

    # I-squared heterogeneity statistic
    # Q = sum(w_i * (beta_i - weighted_mean)^2)
    Q = (weights * (valid['beta3'] - weighted_mean) ** 2).sum()
    df_q = len(valid) - 1
    I_squared = max(0, (Q - df_q) / Q * 100) if Q > 0 else 0

    summary = {
        'n_states': len(valid),
        'n_sig_negative': sig_neg.sum(),
        'n_sig_positive': sig_pos.sum(),
        'n_not_significant': not_sig.sum(),
        'weighted_mean_beta3': weighted_mean,
        'weighted_se_beta3': weighted_se,
        'simple_mean_beta3': simple_mean,
        'simple_sd_beta3': simple_sd,
        'I_squared': I_squared,
        'Q_statistic': Q,
        'Q_df': df_q,
    }

    if national_beta3 is not None:
        summary['national_beta3'] = national_beta3

    return summary


def print_summary(summary):
    """Print formatted summary statistics."""
    print(f"\n{'='*70}")
    print("SUMMARY: State-Level Heterogeneity")
    print(f"{'='*70}")

    print(f"\nStates analyzed: {summary['n_states']}")
    print(f"  Significant negative slope change: {summary['n_sig_negative']}")
    print(f"  Significant positive slope change: {summary['n_sig_positive']}")
    print(f"  Not significant: {summary['n_not_significant']}")

    print(f"\nPooled estimates (slope change, beta3):")
    print(f"  Weighted mean: {summary['weighted_mean_beta3']:+.2f} (SE: {summary['weighted_se_beta3']:.2f})")
    print(f"  Simple mean:   {summary['simple_mean_beta3']:+.2f} (SD: {summary['simple_sd_beta3']:.2f})")

    if 'national_beta3' in summary:
        print(f"  National estimate: {summary['national_beta3']:+.2f}")

    print(f"\nHeterogeneity:")
    print(f"  I-squared: {summary['I_squared']:.1f}%")
    print(f"  Q statistic: {summary['Q_statistic']:.1f} (df={summary['Q_df']})")

    if summary['I_squared'] > 75:
        print("  -> High heterogeneity across states")
    elif summary['I_squared'] > 50:
        print("  -> Moderate heterogeneity across states")
    else:
        print("  -> Low heterogeneity across states")


def plot_forest(results_df, outcome_label, national_beta3=None):
    """
    Create forest plot of state-level slope changes (beta3).
    """
    # Sort by beta3 for visual clarity
    df_plot = results_df.sort_values('beta3', ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, max(8, len(df_plot) * 0.35)))

    y_positions = range(len(df_plot))
    colors = []

    for _, row in df_plot.iterrows():
        if row['beta3_pval'] < 0.05:
            colors.append('darkblue' if row['beta3'] < 0 else 'darkred')
        else:
            colors.append('gray')

    # Plot point estimates with 95% CIs
    ci_95 = 1.96 * df_plot['beta3_se']

    ax.errorbar(
        df_plot['beta3'], y_positions,
        xerr=ci_95,
        fmt='none', ecolor='gray', elinewidth=1, capsize=3, alpha=0.7
    )

    ax.scatter(df_plot['beta3'], y_positions, c=colors, s=50, zorder=5)

    # Add reference line at 0
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    # Add national estimate line if provided
    if national_beta3 is not None:
        ax.axvline(x=national_beta3, color='red', linestyle='--', linewidth=1.5,
                   label=f'National estimate ({national_beta3:+.1f})')
        ax.legend(loc='lower right')

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{row['state']} ({row['state_name']})" for _, row in df_plot.iterrows()])
    ax.set_xlabel('Slope Change (beta3: deaths/month)', fontsize=11)
    ax.set_title(f'Forest Plot: State-Level Slope Changes\n{outcome_label}',
                 fontsize=12, fontweight='bold')

    # Add significance legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue',
               markersize=8, label='Sig. negative (p<0.05)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred',
               markersize=8, label='Sig. positive (p<0.05)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=8, label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    # Save
    filename = f"state_forest_plot_{results_df['outcome'].iloc[0]}"
    plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / f"{filename}.pdf", bbox_inches='tight')
    plt.close()

    print(f"\nSaved forest plot to {FIGURES_DIR / filename}.png")


def plot_histogram(results_df, outcome_label, national_beta3=None):
    """
    Create histogram of state-level slope changes.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(results_df['beta3'], bins=15, edgecolor='black', alpha=0.7, color='steelblue')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

    if national_beta3 is not None:
        ax.axvline(x=national_beta3, color='red', linestyle='--', linewidth=2,
                   label=f'National estimate ({national_beta3:+.1f})')
        ax.legend()

    ax.axvline(x=results_df['beta3'].mean(), color='green', linestyle=':',
               linewidth=2, label=f"State mean ({results_df['beta3'].mean():+.1f})")
    ax.legend()

    ax.set_xlabel('Slope Change (beta3: deaths/month)', fontsize=11)
    ax.set_ylabel('Number of States', fontsize=11)
    ax.set_title(f'Distribution of State-Level Slope Changes\n{outcome_label}',
                 fontsize=12, fontweight='bold')

    plt.tight_layout()

    filename = f"state_beta3_histogram_{results_df['outcome'].iloc[0]}"
    plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved histogram to {FIGURES_DIR / filename}.png")


# ============================================================================
# PLACEBO TESTS
# ============================================================================

TRUE_INTERVENTION = pd.Timestamp('2017-07-01')


def run_its_at_date(df_state, outcome_col, intervention_date):
    """Run ITS model with a specific intervention date for one state."""
    df_clean = df_state.dropna(subset=[outcome_col]).copy()

    if len(df_clean) < 20:
        return None

    # Redefine post variable for this intervention date
    df_clean['post'] = (df_clean['date'] >= intervention_date).astype(int)

    # Check we have data on both sides
    if df_clean['post'].sum() == 0 or df_clean['post'].sum() == len(df_clean):
        return None

    df_clean['time_post'] = df_clean['time'] * df_clean['post']

    y = df_clean[outcome_col].values
    X = np.column_stack([
        np.ones(len(y)),
        df_clean['time'].values,
        df_clean['post'].values,
        df_clean['time_post'].values
    ])

    try:
        model = sm.OLS(y, X)
        max_lags = min(6, len(y) // 10)
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': max(1, max_lags)})

        return {
            'beta3': results.params[3],
            'beta3_se': results.bse[3],
            'beta3_pval': results.pvalues[3],
        }
    except:
        return None


def run_state_placebo_tests(df, state, outcome_col):
    """
    Run placebo tests for a single state.

    Returns dict with:
    - results at each placebo date
    - whether July 2017 is an outlier (most extreme negative beta3)
    """
    df_state = df[df['state'] == state].copy()

    # Test dates: every 6 months from 2016-01 to 2019-01
    test_dates = pd.date_range('2016-01-01', '2019-01-01', freq='6MS')

    results = []
    for date in test_dates:
        res = run_its_at_date(df_state, outcome_col, date)
        if res is not None:
            res['date'] = date
            res['is_true'] = (date == TRUE_INTERVENTION)
            results.append(res)

    if not results:
        return None

    results_df = pd.DataFrame(results)

    # Check if true intervention date was tested
    true_result = results_df[results_df['is_true']]
    if len(true_result) == 0:
        return None

    true_beta3 = true_result['beta3'].iloc[0]

    # Is July 2017 the most extreme negative? (what we'd expect if effect is real)
    is_most_negative = true_beta3 == results_df['beta3'].min()

    # Is July 2017 significantly more negative than average placebo?
    placebo_results = results_df[~results_df['is_true']]
    if len(placebo_results) > 0:
        placebo_mean = placebo_results['beta3'].mean()
        placebo_sd = placebo_results['beta3'].std()
        # Z-score of true effect relative to placebo distribution
        if placebo_sd > 0:
            z_score = (true_beta3 - placebo_mean) / placebo_sd
        else:
            z_score = 0
    else:
        placebo_mean = np.nan
        z_score = np.nan

    return {
        'state': state,
        'true_beta3': true_beta3,
        'true_pval': true_result['beta3_pval'].iloc[0],
        'placebo_mean_beta3': placebo_mean,
        'is_most_negative': is_most_negative,
        'z_score': z_score,
        'n_dates_tested': len(results_df),
        'all_results': results_df,
    }


def run_all_placebo_tests(df, outcome_col, outcome_label):
    """
    Run placebo tests for all states and summarize.
    """
    print(f"\n{'='*70}")
    print(f"PLACEBO TESTS: {outcome_label}")
    print(f"{'='*70}")

    states = df['state'].unique()
    placebo_results = []

    for state in sorted(states):
        result = run_state_placebo_tests(df, state, outcome_col)
        if result is not None:
            placebo_results.append(result)

    if not placebo_results:
        print("No states with sufficient data for placebo tests")
        return None

    # Summary
    placebo_df = pd.DataFrame([{
        'state': r['state'],
        'true_beta3': r['true_beta3'],
        'true_pval': r['true_pval'],
        'placebo_mean': r['placebo_mean_beta3'],
        'is_most_negative': r['is_most_negative'],
        'z_score': r['z_score'],
    } for r in placebo_results])

    print(f"\n{'State':<6} {'True β3':>10} {'p-val':>8} {'Placebo Mean':>12} {'Most Neg?':>10} {'Z-score':>8}")
    print("-" * 60)
    for _, row in placebo_df.iterrows():
        most_neg = "YES" if row['is_most_negative'] else "no"
        print(f"{row['state']:<6} {row['true_beta3']:>+10.1f} {row['true_pval']:>8.3f} "
              f"{row['placebo_mean']:>+12.1f} {most_neg:>10} {row['z_score']:>+8.2f}")

    # Summary statistics
    n_most_negative = placebo_df['is_most_negative'].sum()
    n_sig_and_most_neg = ((placebo_df['true_pval'] < 0.05) & placebo_df['is_most_negative']).sum()

    print(f"\n--- Placebo Test Summary ---")
    print(f"States analyzed: {len(placebo_df)}")
    print(f"States where July 2017 is most negative date: {n_most_negative} ({100*n_most_negative/len(placebo_df):.0f}%)")
    print(f"States with sig. effect AND most negative: {n_sig_and_most_neg}")
    print(f"Mean Z-score (true vs placebos): {placebo_df['z_score'].mean():.2f}")

    if n_most_negative / len(placebo_df) < 0.3:
        print("\n⚠️  WARNING: July 2017 is NOT consistently the most extreme date.")
        print("   The state-level effects may reflect general trends, not DNM takedowns.")
    else:
        print("\n✓  July 2017 appears to be a uniquely important date in many states.")

    return placebo_df


def main():
    print("Loading state-level data...")
    df = pd.read_csv(DATA_FILE, parse_dates=['date'])

    print(f"Data shape: {df.shape}")
    print(f"States: {df['state'].nunique()}")

    # Note: National ITS uses 12-month rolling totals, state analysis uses monthly counts
    # These are different scales, so we don't plot the national estimate on the state forest plot

    # Run analysis for synthetic opioid deaths (primary outcome)
    results_synth = run_all_states(df, 'synthetic_opioid_deaths', 'Synthetic Opioid Deaths')

    if len(results_synth) > 0:
        # Summary statistics (no national comparison due to scale mismatch)
        summary = compute_summary_stats(results_synth)
        print_summary(summary)

        # Visualizations (no national line - different scale)
        plot_forest(results_synth, 'Synthetic Opioid Deaths')
        plot_histogram(results_synth, 'Synthetic Opioid Deaths')

        # Save results
        results_synth.to_csv(RESULTS_FILE, index=False)
        print(f"\nSaved state results to {RESULTS_FILE}")

    # Run for secondary outcomes
    secondary_outcomes = [
        ('total_overdose_deaths', 'Total Overdose Deaths'),
        ('heroin_deaths', 'Heroin Deaths'),
        ('cocaine_deaths', 'Cocaine Deaths'),
    ]

    all_results = [results_synth] if len(results_synth) > 0 else []

    for outcome_col, outcome_label in secondary_outcomes:
        results = run_all_states(df, outcome_col, outcome_label)
        if len(results) > 0:
            all_results.append(results)
            summary = compute_summary_stats(results)
            print_summary(summary)

    # Combine all results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(DATA_DIR / "state_its_results_all.csv", index=False)
        print(f"\nSaved all results to {DATA_DIR / 'state_its_results_all.csv'}")

    # Run placebo tests for primary outcome
    print("\n\n" + "#"*70)
    print("ROBUSTNESS: PLACEBO TESTS")
    print("#"*70)
    placebo_df = run_all_placebo_tests(df, 'synthetic_opioid_deaths', 'Synthetic Opioid Deaths')

    print("\n" + "="*70)
    print("State-level ITS analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
