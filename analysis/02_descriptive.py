#!/usr/bin/env python3
"""
02_descriptive.py
Descriptive statistics and time series visualization of overdose data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
DATA_FILE = DATA_DIR / "cdc_overdose_monthly.csv"

# Intervention date
INTERVENTION_DATE = pd.Timestamp('2017-07-01')


def create_summary_table(df):
    """Generate summary statistics table for the paper."""
    # Split by intervention
    pre = df[df['post_intervention'] == 0]
    post = df[df['post_intervention'] == 1]

    outcomes = [
        ('total_overdose_deaths', 'Total Drug Overdose Deaths'),
        ('synthetic_opioid_deaths', 'Synthetic Opioid Deaths (Fentanyl)'),
        ('heroin_deaths', 'Heroin Deaths'),
        ('cocaine_deaths', 'Cocaine Deaths'),
        ('natural_opioid_deaths', 'Natural/Semi-synthetic Opioid Deaths'),
        ('psychostimulant_deaths', 'Psychostimulant Deaths'),
    ]

    print("\n" + "="*80)
    print("TABLE 1: Summary Statistics (Monthly Death Counts)")
    print("="*80)
    print(f"\n{'Outcome':<45} {'Pre-Intervention':>16} {'Post-Intervention':>18}")
    print(f"{'':45} {'(Jan 2015-Jun 2017)':>16} {'(Jul 2017-Dec 2019)':>18}")
    print("-"*80)

    summary_data = []
    for col, label in outcomes:
        pre_mean = pre[col].mean()
        pre_sd = pre[col].std()
        post_mean = post[col].mean()
        post_sd = post[col].std()
        pct_change = ((post_mean - pre_mean) / pre_mean) * 100

        print(f"{label:<45} {pre_mean:>8,.0f} ({pre_sd:>5,.0f}) {post_mean:>9,.0f} ({post_sd:>5,.0f})")
        summary_data.append({
            'outcome': label,
            'pre_mean': pre_mean,
            'pre_sd': pre_sd,
            'post_mean': post_mean,
            'post_sd': post_sd,
            'pct_change': pct_change
        })

    print("-"*80)
    print(f"{'N (months)':<45} {len(pre):>16} {len(post):>18}")
    print("="*80)

    # Save as CSV for LaTeX
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(DATA_DIR / 'summary_statistics.csv', index=False)
    print(f"\nSaved summary statistics to {DATA_DIR / 'summary_statistics.csv'}")

    return summary_df


def plot_time_series(df):
    """Create main time series figure showing overdose deaths over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    outcomes = [
        ('total_overdose_deaths', 'Total Drug Overdose Deaths', axes[0, 0]),
        ('synthetic_opioid_deaths', 'Synthetic Opioid Deaths (incl. Fentanyl)', axes[0, 1]),
        ('heroin_deaths', 'Heroin Deaths', axes[1, 0]),
        ('cocaine_deaths', 'Cocaine Deaths', axes[1, 1]),
    ]

    for col, title, ax in outcomes:
        # Plot pre-intervention
        pre = df[df['date'] < INTERVENTION_DATE]
        post = df[df['date'] >= INTERVENTION_DATE]

        ax.plot(pre['date'], pre[col], 'b-', linewidth=1.5, label='Pre-intervention')
        ax.plot(post['date'], post[col], 'r-', linewidth=1.5, label='Post-intervention')

        # Add intervention line
        ax.axvline(x=INTERVENTION_DATE, color='black', linestyle='--', linewidth=2,
                   label='AlphaBay/Hansa Takedown (Jul 2017)')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Monthly Deaths (12-month ending)', fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.grid(True, alpha=0.3)

        # Format y-axis with commas
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Add legend to first plot only
    axes[0, 0].legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure1_time_series.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure1_time_series.pdf', bbox_inches='tight')
    plt.close()

    print(f"\nSaved time series plot to {FIGURES_DIR / 'figure1_time_series.png'}")


def plot_synthetic_opioids_focus(df):
    """Create focused plot on synthetic opioids (the key outcome)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot data
    ax.plot(df['date'], df['synthetic_opioid_deaths'], 'ko-', markersize=4, linewidth=1)

    # Add intervention line
    ax.axvline(x=INTERVENTION_DATE, color='red', linestyle='--', linewidth=2)

    # Add annotation
    ax.annotate('AlphaBay/Hansa\nTakedown', xy=(INTERVENTION_DATE, 28000),
                xytext=(INTERVENTION_DATE + pd.Timedelta(days=90), 20000),
                fontsize=10, ha='left',
                arrowprops=dict(arrowstyle='->', color='red'))

    # Shade pre/post periods
    ax.axvspan(df['date'].min(), INTERVENTION_DATE, alpha=0.1, color='blue', label='Pre-intervention')
    ax.axvspan(INTERVENTION_DATE, df['date'].max(), alpha=0.1, color='red', label='Post-intervention')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Monthly Synthetic Opioid Deaths\n(12-month ending period)', fontsize=12)
    ax.set_title('Synthetic Opioid Deaths in the United States, 2015-2019', fontsize=14, fontweight='bold')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'figure2_synthetic_opioids.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'figure2_synthetic_opioids.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved synthetic opioids plot to {FIGURES_DIR / 'figure2_synthetic_opioids.png'}")


def compute_growth_rates(df):
    """Compute and compare growth rates pre/post intervention."""
    pre = df[df['post_intervention'] == 0].copy()
    post = df[df['post_intervention'] == 1].copy()

    print("\n" + "="*80)
    print("GROWTH RATE ANALYSIS")
    print("="*80)

    outcomes = ['total_overdose_deaths', 'synthetic_opioid_deaths', 'heroin_deaths', 'cocaine_deaths']

    for col in outcomes:
        # Monthly growth rate = (end - start) / (n_months - 1)
        pre_growth = (pre[col].iloc[-1] - pre[col].iloc[0]) / (len(pre) - 1)
        post_growth = (post[col].iloc[-1] - post[col].iloc[0]) / (len(post) - 1)

        # Percentage of baseline
        pre_baseline = pre[col].iloc[0]
        pre_growth_pct = (pre_growth / pre_baseline) * 100
        post_growth_pct = (post_growth / post[col].iloc[0]) * 100

        print(f"\n{col}:")
        print(f"  Pre-intervention monthly growth:  {pre_growth:>8,.1f} deaths/month ({pre_growth_pct:.2f}% of baseline)")
        print(f"  Post-intervention monthly growth: {post_growth:>8,.1f} deaths/month ({post_growth_pct:.2f}% of baseline)")
        print(f"  Change in growth rate: {post_growth - pre_growth:>+,.1f} deaths/month")


def main():
    print("Loading prepared data...")
    df = pd.read_csv(DATA_FILE, parse_dates=['date'])

    print(f"Data range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total observations: {len(df)}")
    print(f"Intervention month: {INTERVENTION_DATE}")

    # Generate outputs
    create_summary_table(df)
    plot_time_series(df)
    plot_synthetic_opioids_focus(df)
    compute_growth_rates(df)

    print("\n" + "="*80)
    print("Descriptive analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()
