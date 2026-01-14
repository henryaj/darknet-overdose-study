#!/usr/bin/env python3
"""
03_its_analysis.py
Interrupted Time Series (ITS) analysis of overdose deaths.

Model specification:
    Y_t = β0 + β1*time + β2*post + β3*(time × post) + ε_t

Where:
    - Y_t = monthly overdose deaths
    - time = months since start (1, 2, 3, ...)
    - post = 1 if t ≥ July 2017, else 0
    - time × post = interaction term for slope change

Key coefficients:
    - β2 = immediate level change at intervention
    - β3 = change in trend post-intervention
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
DATA_FILE = DATA_DIR / "cdc_overdose_monthly.csv"


def run_its_model(df, outcome_col, outcome_label):
    """
    Run interrupted time series regression.

    Returns regression results and key statistics.
    """
    print(f"\n{'='*80}")
    print(f"ITS Analysis: {outcome_label}")
    print(f"{'='*80}")

    # Prepare variables
    y = df[outcome_col].values
    time = df['time'].values
    post = df['post_intervention'].values
    time_post = time * post  # Interaction term

    # Create design matrix
    X = np.column_stack([np.ones(len(y)), time, post, time_post])
    X = sm.add_constant(X, has_constant='skip')

    # Fit OLS model
    model = sm.OLS(y, X)
    results = model.fit()

    # Also fit with Newey-West HAC standard errors (robust to autocorrelation)
    results_robust = model.fit(cov_type='HAC', cov_kwds={'maxlags': 6})

    print("\n--- OLS Results (Standard Errors) ---")
    print(f"{'Variable':<25} {'Coef':>12} {'Std Err':>12} {'t':>10} {'P>|t|':>10}")
    print("-"*70)
    var_names = ['Intercept', 'Time (trend)', 'Post (level shift)', 'Time×Post (slope change)']
    for i, name in enumerate(var_names):
        print(f"{name:<25} {results.params[i]:>12,.2f} {results.bse[i]:>12,.2f} "
              f"{results.tvalues[i]:>10.3f} {results.pvalues[i]:>10.4f}")

    print(f"\nR-squared: {results.rsquared:.4f}")
    print(f"Adj. R-squared: {results.rsquared_adj:.4f}")

    print("\n--- Robust Results (Newey-West HAC SE, 6 lags) ---")
    print(f"{'Variable':<25} {'Coef':>12} {'Robust SE':>12} {'t':>10} {'P>|t|':>10}")
    print("-"*70)
    for i, name in enumerate(var_names):
        print(f"{name:<25} {results_robust.params[i]:>12,.2f} {results_robust.bse[i]:>12,.2f} "
              f"{results_robust.tvalues[i]:>10.3f} {results_robust.pvalues[i]:>10.4f}")

    # Check for autocorrelation in residuals
    lb_test = acorr_ljungbox(results.resid, lags=[6, 12], return_df=True)
    print("\n--- Ljung-Box Test for Autocorrelation ---")
    print(lb_test)

    # Interpretation
    print("\n--- Interpretation ---")
    b2 = results_robust.params[2]
    b3 = results_robust.params[3]
    p2 = results_robust.pvalues[2]
    p3 = results_robust.pvalues[3]

    print(f"Level change (β2): {b2:+,.1f} deaths")
    if p2 < 0.05:
        direction = "increase" if b2 > 0 else "decrease"
        print(f"  → Statistically significant immediate {direction} at intervention (p={p2:.4f})")
    else:
        print(f"  → Not statistically significant (p={p2:.4f})")

    print(f"Slope change (β3): {b3:+,.1f} deaths/month")
    if p3 < 0.05:
        direction = "acceleration" if b3 > 0 else "deceleration"
        print(f"  → Statistically significant {direction} of trend post-intervention (p={p3:.4f})")
    else:
        print(f"  → Not statistically significant (p={p3:.4f})")

    return results, results_robust


def plot_its_fitted(df, outcome_col, outcome_label, results):
    """Plot observed data with fitted ITS model."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot observed data
    ax.plot(df['date'], df[outcome_col], 'ko', markersize=4, alpha=0.6, label='Observed')

    # Generate fitted values
    time = df['time'].values
    post = df['post_intervention'].values
    time_post = time * post
    X = np.column_stack([np.ones(len(time)), time, post, time_post])
    fitted = X @ results.params

    # Plot fitted line (split by intervention)
    pre_mask = df['post_intervention'] == 0
    post_mask = df['post_intervention'] == 1

    ax.plot(df.loc[pre_mask, 'date'], fitted[pre_mask], 'b-', linewidth=2, label='Fitted (pre)')
    ax.plot(df.loc[post_mask, 'date'], fitted[post_mask], 'r-', linewidth=2, label='Fitted (post)')

    # Counterfactual: what would have happened without intervention
    time_cf = df['time'].values
    counterfactual = results.params[0] + results.params[1] * time_cf
    ax.plot(df['date'], counterfactual, 'b--', linewidth=1.5, alpha=0.5, label='Counterfactual (no intervention)')

    # Intervention line
    intervention_date = pd.Timestamp('2017-07-01')
    ax.axvline(x=intervention_date, color='gray', linestyle='--', linewidth=1.5)
    ax.annotate('AlphaBay/Hansa\nTakedown', xy=(intervention_date, ax.get_ylim()[1]*0.9),
                fontsize=10, ha='center')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{outcome_label}\n(12-month ending)', fontsize=11)
    ax.set_title(f'Interrupted Time Series Analysis: {outcome_label}', fontsize=13, fontweight='bold')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    plt.tight_layout()

    # Save
    filename = f"figure_its_{outcome_col}"
    plt.savefig(FIGURES_DIR / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / f"{filename}.pdf", bbox_inches='tight')
    plt.close()

    print(f"\nSaved ITS plot to {FIGURES_DIR / filename}.png")


def create_results_table(all_results):
    """Create formatted results table for all outcomes."""
    print("\n" + "="*100)
    print("TABLE 2: Interrupted Time Series Regression Results")
    print("="*100)
    print(f"\n{'Outcome':<35} {'β2 (Level)':>15} {'p-value':>10} {'β3 (Slope)':>15} {'p-value':>10}")
    print("-"*100)

    table_data = []
    for outcome, label, results in all_results:
        b2 = results.params[2]
        p2 = results.pvalues[2]
        b3 = results.params[3]
        p3 = results.pvalues[3]

        sig2 = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else ""
        sig3 = "***" if p3 < 0.001 else "**" if p3 < 0.01 else "*" if p3 < 0.05 else ""

        print(f"{label:<35} {b2:>+12,.1f}{sig2:<3} {p2:>10.4f} {b3:>+12,.1f}{sig3:<3} {p3:>10.4f}")

        table_data.append({
            'outcome': label,
            'beta2_level': b2,
            'beta2_pvalue': p2,
            'beta3_slope': b3,
            'beta3_pvalue': p3
        })

    print("-"*100)
    print("Notes: Newey-West HAC standard errors with 6 lags. * p<0.05, ** p<0.01, *** p<0.001")
    print("="*100)

    # Save
    results_df = pd.DataFrame(table_data)
    results_df.to_csv(DATA_DIR / 'its_results.csv', index=False)
    print(f"\nSaved results table to {DATA_DIR / 'its_results.csv'}")


def main():
    print("Loading prepared data...")
    df = pd.read_csv(DATA_FILE, parse_dates=['date'])

    # Define outcomes to analyze
    outcomes = [
        ('total_overdose_deaths', 'Total Drug Overdose Deaths'),
        ('synthetic_opioid_deaths', 'Synthetic Opioid Deaths'),
        ('heroin_deaths', 'Heroin Deaths'),
        ('cocaine_deaths', 'Cocaine Deaths'),
        ('natural_opioid_deaths', 'Natural/Semi-synthetic Opioid Deaths'),
        ('psychostimulant_deaths', 'Psychostimulant Deaths'),
    ]

    all_results = []

    for outcome_col, outcome_label in outcomes:
        results_ols, results_robust = run_its_model(df, outcome_col, outcome_label)
        all_results.append((outcome_col, outcome_label, results_robust))

        # Plot for key outcomes
        if outcome_col in ['synthetic_opioid_deaths', 'total_overdose_deaths']:
            plot_its_fitted(df, outcome_col, outcome_label, results_ols)

    # Create summary table
    create_results_table(all_results)

    print("\n" + "="*80)
    print("ITS analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
