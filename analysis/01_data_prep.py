#!/usr/bin/env python3
"""
01_data_prep.py
Prepare CDC VSRR overdose data for interrupted time series analysis.

Data source: https://data.cdc.gov/NCHS/VSRR-Provisional-Drug-Overdose-Death-Counts/xkb8-kh2a
"""

import pandas as pd
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_FILE = DATA_DIR / "vsrr_overdose_raw.csv"
OUTPUT_FILE = DATA_DIR / "cdc_overdose_monthly.csv"

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
    'Natural & semi-synthetic opioids (T40.2)': 'natural_opioid_deaths',
    'Psychostimulants with abuse potential (T43.6)': 'psychostimulant_deaths',
}


def main():
    print(f"Loading raw data from {RAW_FILE}...")
    df = pd.read_csv(RAW_FILE)

    print(f"Raw data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nUnique States: {df['State'].nunique()}")
    print(f"Unique Years: {sorted(df['Year'].unique())}")

    # Filter to national US data only
    df_us = df[df['State'] == 'US'].copy()
    print(f"\nFiltered to US national: {len(df_us)} rows")

    # Filter to our study period (2015-2019)
    df_us = df_us[(df_us['Year'] >= 2015) & (df_us['Year'] <= 2019)]
    print(f"Filtered to 2015-2019: {len(df_us)} rows")

    # Check available indicators
    print(f"\nAvailable indicators in US data:")
    for ind in df_us['Indicator'].unique():
        count = len(df_us[df_us['Indicator'] == ind])
        print(f"  {ind}: {count} rows")

    # Map month names to numbers
    df_us['month_num'] = df_us['Month'].map(MONTH_ORDER)

    # Create pivot table: rows = year-month, columns = indicators
    pivot_data = []

    for (year, month), group in df_us.groupby(['Year', 'month_num']):
        row = {'year': int(year), 'month': int(month)}

        for indicator, clean_name in INDICATORS.items():
            # Try to find this indicator (handle slight naming variations)
            matches = group[group['Indicator'].str.contains(indicator.split('(')[0].strip(), case=False, na=False)]
            if len(matches) > 0:
                # Use 'Data Value' column, handle missing values
                val = matches['Data Value'].iloc[0]
                try:
                    row[clean_name] = int(float(val)) if pd.notna(val) and val != '' else None
                except (ValueError, TypeError):
                    row[clean_name] = None
            else:
                row[clean_name] = None

        pivot_data.append(row)

    # Create output dataframe
    df_out = pd.DataFrame(pivot_data)
    df_out = df_out.sort_values(['year', 'month']).reset_index(drop=True)

    # Create date column for easier plotting
    df_out['date'] = pd.to_datetime(df_out['year'].astype(str) + '-' + df_out['month'].astype(str) + '-01')

    # Create intervention indicator (July 2017 = AlphaBay/Hansa takedown)
    intervention_date = pd.Timestamp('2017-07-01')
    df_out['post_intervention'] = (df_out['date'] >= intervention_date).astype(int)

    # Create time index (months since start)
    df_out['time'] = range(1, len(df_out) + 1)

    # Create time since intervention (0 before, 1, 2, 3... after)
    df_out['time_since_intervention'] = df_out['time'] - df_out[df_out['date'] == intervention_date]['time'].iloc[0]
    df_out.loc[df_out['time_since_intervention'] < 0, 'time_since_intervention'] = 0

    # Reorder columns
    cols = ['date', 'year', 'month', 'time', 'post_intervention', 'time_since_intervention',
            'total_overdose_deaths', 'synthetic_opioid_deaths', 'heroin_deaths',
            'cocaine_deaths', 'natural_opioid_deaths', 'psychostimulant_deaths']
    df_out = df_out[cols]

    print(f"\nOutput data shape: {df_out.shape}")
    print(f"\nSample of output data:")
    print(df_out.head(10).to_string())
    print("\n...")
    print(df_out.tail(5).to_string())

    # Check for missing values
    print(f"\nMissing values per column:")
    print(df_out.isnull().sum())

    # Summary stats
    print(f"\nSummary statistics:")
    print(df_out.describe())

    # Save
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
