#!/usr/bin/env python3
"""
05_state_data_prep.py
Prepare state-level CDC VSRR overdose data for interrupted time series analysis.

Extends 01_data_prep.py to process data for individual states rather than
national aggregate. Only includes states with sufficient data completeness.
"""

import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_FILE = DATA_DIR / "vsrr_overdose_raw.csv"
OUTPUT_FILE = DATA_DIR / "state_overdose_monthly.csv"

# Month mapping for sorting
MONTH_ORDER = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

# Indicators we care about (mapping to clean names)
INDICATORS = {
    'Number of Drug Overdose Deaths': 'total_overdose_deaths',
    'Synthetic opioids, excl. methadone (T40.4)': 'synthetic_opioid_deaths',
    'Heroin (T40.1)': 'heroin_deaths',
    'Cocaine (T40.5)': 'cocaine_deaths',
}

# Minimum observations required for a state to be included
MIN_OBS_REQUIRED = 50


def get_qualifying_states(df):
    """
    Identify states with sufficient synthetic opioid death data.

    Returns list of state codes that have at least MIN_OBS_REQUIRED
    non-null observations for synthetic opioid deaths.
    """
    # Filter to synthetic opioid indicator
    synth = df[df['Indicator'].str.contains('Synthetic', case=False, na=False)]

    # Count non-null Data Value by state
    state_counts = synth.groupby('State').apply(
        lambda x: x['Data Value'].notna().sum(),
        include_groups=False
    )

    # Filter to states meeting threshold (exclude US aggregate)
    qualifying = state_counts[
        (state_counts >= MIN_OBS_REQUIRED) &
        (state_counts.index != 'US')
    ]

    return list(qualifying.index)


def process_state_data(df, state_code):
    """
    Process data for a single state into monthly time series format.

    Returns DataFrame with columns:
    - state, state_name, date, year, month, time, post_intervention, time_since_intervention
    - total_overdose_deaths, synthetic_opioid_deaths, heroin_deaths, cocaine_deaths
    """
    # Filter to this state
    df_state = df[df['State'] == state_code].copy()

    # Get state name
    state_name = df_state['State Name'].iloc[0] if 'State Name' in df_state.columns else state_code

    # Filter to study period (2015-2019)
    df_state = df_state[(df_state['Year'] >= 2015) & (df_state['Year'] <= 2019)]

    # Map month names to numbers
    df_state['month_num'] = df_state['Month'].map(MONTH_ORDER)

    # Create pivot table: rows = year-month, columns = indicators
    pivot_data = []

    for (year, month), group in df_state.groupby(['Year', 'month_num']):
        row = {
            'state': state_code,
            'state_name': state_name,
            'year': int(year),
            'month': int(month)
        }

        for indicator, clean_name in INDICATORS.items():
            # Find this indicator (handle naming variations)
            matches = group[group['Indicator'].str.contains(
                indicator.split('(')[0].strip(), case=False, na=False
            )]
            if len(matches) > 0:
                val = matches['Data Value'].iloc[0]
                try:
                    row[clean_name] = int(float(val)) if pd.notna(val) and val != '' else None
                except (ValueError, TypeError):
                    row[clean_name] = None
            else:
                row[clean_name] = None

        pivot_data.append(row)

    if not pivot_data:
        return None

    # Create output dataframe
    df_out = pd.DataFrame(pivot_data)
    df_out = df_out.sort_values(['year', 'month']).reset_index(drop=True)

    # Create date column
    df_out['date'] = pd.to_datetime(
        df_out['year'].astype(str) + '-' + df_out['month'].astype(str) + '-01'
    )

    # Create intervention indicator (July 2017 = AlphaBay/Hansa takedown)
    intervention_date = pd.Timestamp('2017-07-01')
    df_out['post_intervention'] = (df_out['date'] >= intervention_date).astype(int)

    # Create time index (months since start)
    df_out['time'] = range(1, len(df_out) + 1)

    # Create time since intervention
    intervention_rows = df_out[df_out['date'] == intervention_date]
    if len(intervention_rows) > 0:
        intervention_time = intervention_rows['time'].iloc[0]
        df_out['time_since_intervention'] = df_out['time'] - intervention_time
        df_out.loc[df_out['time_since_intervention'] < 0, 'time_since_intervention'] = 0
    else:
        # Handle case where intervention date not in data
        df_out['time_since_intervention'] = 0

    return df_out


def main():
    print(f"Loading raw data from {RAW_FILE}...")
    df = pd.read_csv(RAW_FILE)

    print(f"Raw data shape: {df.shape}")
    print(f"Total states/territories: {df['State'].nunique()}")

    # Get qualifying states
    print(f"\nIdentifying states with ≥{MIN_OBS_REQUIRED} observations...")
    qualifying_states = get_qualifying_states(df)
    print(f"Found {len(qualifying_states)} qualifying states:")
    print(f"  {', '.join(sorted(qualifying_states))}")

    # Process each state
    all_state_data = []
    state_summary = []

    for state_code in sorted(qualifying_states):
        df_state = process_state_data(df, state_code)
        if df_state is not None:
            all_state_data.append(df_state)

            # Summary info
            n_obs = len(df_state)
            n_synth = df_state['synthetic_opioid_deaths'].notna().sum()
            state_summary.append({
                'state': state_code,
                'state_name': df_state['state_name'].iloc[0],
                'n_obs': n_obs,
                'n_synth_obs': n_synth
            })

    # Combine all states
    df_combined = pd.concat(all_state_data, ignore_index=True)

    # Reorder columns
    cols = ['state', 'state_name', 'date', 'year', 'month', 'time',
            'post_intervention', 'time_since_intervention',
            'total_overdose_deaths', 'synthetic_opioid_deaths',
            'heroin_deaths', 'cocaine_deaths']
    df_combined = df_combined[cols]

    # Print summary
    print(f"\nState-level data summary:")
    print("-" * 60)
    summary_df = pd.DataFrame(state_summary)
    print(summary_df.to_string(index=False))

    print(f"\nCombined output shape: {df_combined.shape}")
    print(f"Total states: {df_combined['state'].nunique()}")
    print(f"Date range: {df_combined['date'].min()} to {df_combined['date'].max()}")

    # Missing value summary
    print(f"\nMissing values by column:")
    for col in ['total_overdose_deaths', 'synthetic_opioid_deaths', 'heroin_deaths', 'cocaine_deaths']:
        missing = df_combined[col].isna().sum()
        pct = 100 * missing / len(df_combined)
        print(f"  {col}: {missing} ({pct:.1f}%)")

    # Save
    df_combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
