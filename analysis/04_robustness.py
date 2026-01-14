#!/usr/bin/env python3
"""
04_robustness.py
Robustness checks for the ITS analysis.

1. Placebo tests: Fake intervention dates
2. Different time windows
3. Alternative model specifications
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
DATA_FILE = DATA_DIR / "cdc_overdose_monthly.csv"

# True intervention
TRUE_INTERVENTION = pd.Timestamp('2017-07-01')


def run_its_at_date(df, outcome_col, intervention_date):
    """Run ITS model with a specific intervention date."""
    df = df.copy()

    # Redefine post variable
    df['post'] = (df['date'] >= intervention_date).astype(int)
    df['time_post'] = df['time'] * df['post']

    y = df[outcome_col].values
    X = np.column_stack([
        np.ones(len(y)),
        df['time'].values,
        df['post'].values,
        df['time_post'].values
    ])

    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 6})

    return {
        'date': intervention_date,
        'beta2': results.params[2],
        'beta2_se': results.bse[2],
        'beta2_p': results.pvalues[2],
        'beta3': results.params[3],
        'beta3_se': results.bse[3],
        'beta3_p': results.pvalues[3],
    }


def placebo_tests(df, outcome_col, outcome_label):
    """
    Run placebo tests at fake intervention dates.
    If the true effect is real, we should NOT see significant effects at placebo dates.
    """
    print(f"\n{'='*80}")
    print(f"PLACEBO TESTS: {outcome_label}")
    print(f"{'='*80}")

    # Test dates: every 6 months, excluding true intervention
    test_dates = pd.date_range('2015-07-01', '2019-01-01', freq='6MS')
    test_dates = [d for d in test_dates if d != TRUE_INTERVENTION]

    results = []
    for date in test_dates:
        res = run_its_at_date(df, outcome_col, date)
        results.append(res)

    # Add true intervention
    true_res = run_its_at_date(df, outcome_col, TRUE_INTERVENTION)
    true_res['is_true'] = True
    results.append(true_res)

    results_df = pd.DataFrame(results)
    results_df['is_true'] = results_df['date'] == TRUE_INTERVENTION

    print(f"\n{'Intervention Date':<20} {'β2 (Level)':>12} {'p-value':>10} {'β3 (Slope)':>12} {'p-value':>10}")
    print("-"*70)
    for _, row in results_df.sort_values('date').iterrows():
        marker = " <-- TRUE" if row['is_true'] else ""
        sig2 = "*" if row['beta2_p'] < 0.05 else ""
        sig3 = "*" if row['beta3_p'] < 0.05 else ""
        print(f"{row['date'].strftime('%Y-%m-%d'):<20} {row['beta2']:>+10,.0f}{sig2:<2} {row['beta2_p']:>10.4f} "
              f"{row['beta3']:>+10,.1f}{sig3:<2} {row['beta3_p']:>10.4f}{marker}")

    # Plot placebo test results
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, coef, ylabel in [(axes[0], 'beta2', 'Level Change (β2)'),
                              (axes[1], 'beta3', 'Slope Change (β3)')]:
        colors = ['red' if is_true else 'blue' for is_true in results_df['is_true']]
        ax.bar(range(len(results_df)), results_df[coef], color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # Add significance markers
        for i, (_, row) in enumerate(results_df.iterrows()):
            p = row[f'{coef}_p']
            if p < 0.05:
                ax.annotate('*', xy=(i, row[coef]), ha='center',
                           va='bottom' if row[coef] > 0 else 'top', fontsize=14)

        ax.set_xticks(range(len(results_df)))
        ax.set_xticklabels([d.strftime('%Y-%m') for d in results_df['date']], rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Placebo Intervention Date')
        ax.set_title(f'{outcome_label}: {ylabel}')

    plt.suptitle('Placebo Tests (Red = True Intervention Date)', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'placebo_{outcome_col}.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / f'placebo_{outcome_col}.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nSaved placebo plot to {FIGURES_DIR / 'placebo_{outcome_col}.png'}")

    return results_df


def window_sensitivity(df, outcome_col, outcome_label):
    """
    Test sensitivity to different time windows around intervention.
    """
    print(f"\n{'='*80}")
    print(f"WINDOW SENSITIVITY: {outcome_label}")
    print(f"{'='*80}")

    windows = [
        ('Full (2015-2019)', None, None),
        ('±18 months', '2016-01-01', '2018-12-01'),
        ('±12 months', '2016-07-01', '2018-06-01'),
        ('±9 months', '2016-10-01', '2018-03-01'),
    ]

    results = []
    for name, start, end in windows:
        df_window = df.copy()
        if start:
            df_window = df_window[df_window['date'] >= start]
        if end:
            df_window = df_window[df_window['date'] <= end]

        # Re-index time
        df_window = df_window.reset_index(drop=True)
        df_window['time'] = range(1, len(df_window) + 1)

        res = run_its_at_date(df_window, outcome_col, TRUE_INTERVENTION)
        res['window'] = name
        res['n_obs'] = len(df_window)
        results.append(res)

    results_df = pd.DataFrame(results)

    print(f"\n{'Window':<25} {'N':>5} {'β2 (Level)':>12} {'p-value':>10} {'β3 (Slope)':>12} {'p-value':>10}")
    print("-"*80)
    for _, row in results_df.iterrows():
        sig2 = "*" if row['beta2_p'] < 0.05 else ""
        sig3 = "*" if row['beta3_p'] < 0.05 else ""
        print(f"{row['window']:<25} {row['n_obs']:>5} {row['beta2']:>+10,.0f}{sig2:<2} {row['beta2_p']:>10.4f} "
              f"{row['beta3']:>+10,.1f}{sig3:<2} {row['beta3_p']:>10.4f}")

    return results_df


def segmented_regression_with_seasonality(df, outcome_col, outcome_label):
    """
    ITS model with seasonal controls (month fixed effects).
    """
    print(f"\n{'='*80}")
    print(f"ITS WITH SEASONALITY CONTROLS: {outcome_label}")
    print(f"{'='*80}")

    # Create month dummies (January = reference)
    month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)

    y = df[outcome_col].values
    X_base = np.column_stack([
        np.ones(len(y)),
        df['time'].values,
        df['post_intervention'].values,
        df['time'].values * df['post_intervention'].values,
    ])
    X = np.hstack([X_base, month_dummies.values])

    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 6})

    var_names = ['Intercept', 'Time', 'Post', 'Time×Post'] + list(month_dummies.columns)

    print("\nKey coefficients (with seasonal controls):")
    print(f"{'Variable':<25} {'Coef':>12} {'Robust SE':>12} {'p-value':>10}")
    print("-"*60)
    for i, name in enumerate(var_names[:4]):  # Just show main vars
        sig = "*" if results.pvalues[i] < 0.05 else ""
        print(f"{name:<25} {results.params[i]:>+10,.1f}{sig:<2} {results.bse[i]:>12,.1f} {results.pvalues[i]:>10.4f}")

    print(f"\nR-squared (with seasonality): {results.rsquared:.4f}")

    return results


def main():
    print("Loading prepared data...")
    df = pd.read_csv(DATA_FILE, parse_dates=['date'])

    # Primary outcome: synthetic opioids
    outcome_col = 'synthetic_opioid_deaths'
    outcome_label = 'Synthetic Opioid Deaths'

    # Run robustness checks
    placebo_results = placebo_tests(df, outcome_col, outcome_label)
    window_results = window_sensitivity(df, outcome_col, outcome_label)
    seasonal_results = segmented_regression_with_seasonality(df, outcome_col, outcome_label)

    # Also run for total overdose deaths
    print("\n\n" + "#"*80)
    print("ROBUSTNESS CHECKS FOR TOTAL OVERDOSE DEATHS")
    print("#"*80)

    outcome_col2 = 'total_overdose_deaths'
    outcome_label2 = 'Total Overdose Deaths'

    placebo_tests(df, outcome_col2, outcome_label2)
    window_sensitivity(df, outcome_col2, outcome_label2)
    segmented_regression_with_seasonality(df, outcome_col2, outcome_label2)

    # Save combined robustness results
    all_robustness = {
        'placebo_synthetic': placebo_results.to_dict(),
        'window_synthetic': window_results.to_dict(),
    }

    print("\n" + "="*80)
    print("Robustness checks complete!")
    print("="*80)


if __name__ == "__main__":
    main()
