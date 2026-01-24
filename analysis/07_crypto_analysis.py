#!/usr/bin/env python3
"""
07_crypto_analysis.py
Analyze darknet market cryptocurrency volumes alongside overdose death trends.

Uses estimated transaction volumes based on Chainalysis Crypto Crime Reports
to show the timing of DNM activity drop and recovery after July 2017 takedowns.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
CRYPTO_FILE = DATA_DIR / "dnm_crypto_volumes.csv"
OVERDOSE_FILE = DATA_DIR / "cdc_overdose_monthly.csv"


def load_crypto_data():
    """Load DNM cryptocurrency volume data."""
    df = pd.read_csv(CRYPTO_FILE, comment='#')
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_overdose_data():
    """Load CDC overdose data."""
    df = pd.read_csv(OVERDOSE_FILE, parse_dates=['date'])
    return df


def plot_crypto_overdose_comparison(crypto_df, overdose_df):
    """
    Create dual-axis plot comparing DNM volumes to overdose trends.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Filter overdose data to crypto data range
    overdose_df = overdose_df[
        (overdose_df['date'] >= '2016-01-01') &
        (overdose_df['date'] <= '2019-01-01')
    ]

    # Left axis: DNM crypto volumes
    color1 = '#2E86AB'  # Blue
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('DNM Transaction Volume ($M/month)', color=color1, fontsize=11)
    ax1.plot(crypto_df['date'], crypto_df['dnm_volume_millions_usd'],
             'o-', color=color1, linewidth=2, markersize=6, label='DNM Bitcoin Volume')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 80)

    # Fill the drop period
    ax1.axvspan(pd.Timestamp('2017-07-01'), pd.Timestamp('2018-01-01'),
                alpha=0.15, color='red', label='Post-takedown period')

    # Right axis: Synthetic opioid deaths
    ax2 = ax1.twinx()
    color2 = '#E94F37'  # Red
    ax2.set_ylabel('Synthetic Opioid Deaths (12-mo rolling)', color=color2, fontsize=11)
    ax2.plot(overdose_df['date'], overdose_df['synthetic_opioid_deaths'],
             '-', color=color2, linewidth=2, alpha=0.8, label='Synthetic Opioid Deaths')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Format y-axis with commas
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Intervention line
    intervention_date = pd.Timestamp('2017-07-01')
    ax1.axvline(x=intervention_date, color='black', linestyle='--', linewidth=1.5,
                label='July 2017 takedowns')

    # Annotations
    ax1.annotate('AlphaBay/Hansa\nTakedown',
                 xy=(intervention_date, 70), xytext=(intervention_date - pd.Timedelta(days=60), 75),
                 fontsize=9, ha='right',
                 arrowprops=dict(arrowstyle='->', color='black', lw=1))

    ax1.annotate('60% drop in\nDNM activity',
                 xy=(pd.Timestamp('2017-08-01'), 28), xytext=(pd.Timestamp('2017-10-01'), 15),
                 fontsize=9, ha='left',
                 arrowprops=dict(arrowstyle='->', color=color1, lw=1))

    ax1.annotate('Recovery by\nearly 2018',
                 xy=(pd.Timestamp('2018-04-01'), 40), xytext=(pd.Timestamp('2018-06-01'), 20),
                 fontsize=9, ha='left',
                 arrowprops=dict(arrowstyle='->', color=color1, lw=1))

    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    # Title
    ax1.set_title('Darknet Market Activity vs. Synthetic Opioid Deaths\n'
                  'DNM volumes recovered within 6 months of takedowns',
                  fontsize=12, fontweight='bold')

    # Grid
    ax1.grid(True, alpha=0.3)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    plt.tight_layout()

    # Save
    plt.savefig(FIGURES_DIR / 'crypto_overdose_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'crypto_overdose_comparison.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved comparison plot to {FIGURES_DIR / 'crypto_overdose_comparison.png'}")


def print_summary(crypto_df):
    """Print summary of crypto volume changes."""
    print("\n" + "="*70)
    print("DNM CRYPTOCURRENCY VOLUME ANALYSIS")
    print("="*70)

    # Pre-takedown peak
    pre_peak = crypto_df[crypto_df['date'] < '2017-07-01']['dnm_volume_millions_usd'].max()
    post_drop = crypto_df[crypto_df['date'] == '2017-07-01']['dnm_volume_millions_usd'].iloc[0]
    recovery = crypto_df[crypto_df['date'] >= '2018-01-01']['dnm_volume_millions_usd'].iloc[0]
    late_2018 = crypto_df[crypto_df['date'] >= '2018-10-01']['dnm_volume_millions_usd'].iloc[0]

    print(f"\nKey data points (from Chainalysis reports):")
    print(f"  Pre-takedown peak (June 2017):  ${pre_peak}M/month")
    print(f"  Post-takedown (July 2017):      ${post_drop}M/month")
    print(f"  Drop magnitude:                 {100*(pre_peak-post_drop)/pre_peak:.0f}%")
    print(f"  Early 2018 (recovery start):    ${recovery}M/month")
    print(f"  Late 2018 (recovered):          ${late_2018}M/month")

    print(f"\n--- Interpretation ---")
    print("DNM transaction volumes dropped sharply (~60%) after the July 2017")
    print("AlphaBay/Hansa takedowns, but recovered within approximately 6 months.")
    print("Users rapidly migrated to alternative markets (Dream Market, Hydra)")
    print("rather than being permanently displaced to street drug markets.")
    print("\nThis undermines the 'forced to street dealers' hypothesis: the window")
    print("during which DNM users lacked online alternatives was brief (~6 months).")

    print("\n" + "="*70)
    print("Data sources:")
    print("  - Chainalysis Crypto Crime Report 2019")
    print("  - Chainalysis blog: 'Decoding Darknet Markets'")
    print("  - Note: Monthly values estimated from reported annual totals")
    print("="*70)


def main():
    print("Loading data...")
    crypto_df = load_crypto_data()
    overdose_df = load_overdose_data()

    print(f"Crypto data: {len(crypto_df)} observations")
    print(f"Overdose data: {len(overdose_df)} observations")

    # Print summary
    print_summary(crypto_df)

    # Create visualization
    plot_crypto_overdose_comparison(crypto_df, overdose_df)

    print("\nCrypto analysis complete!")


if __name__ == "__main__":
    main()
