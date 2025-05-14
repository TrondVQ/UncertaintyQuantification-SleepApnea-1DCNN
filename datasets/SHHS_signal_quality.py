#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
from typing import Dict, List, Any, NoReturn, Optional # Import types used

def analyze_signal_quality(csv_file_path: str) -> NoReturn:
    """
    Analyzes signal quality variables from the SHHS2 dataset CSV for a defined cohort.

    The analysis cohort is defined by individuals with non-missing AHI ('ahi_a0h3a') values.
    Statistics (counts and percentages) for defined signal quality variables
    ('quoxim', 'quhr', 'quchest', 'quabdo') based on their category codes are printed.

    Args:
        csv_file_path: Path to the SHHS2 dataset CSV file (e.g., 'shhs2-dataset-0.21.0.csv').
                       This file is typically available from the NSRR website.
    """
    print(f"--- Analyzing SHHS2 Signal Quality from: {csv_file_path} ---")
    print("="*70)

    # --- Configuration: Define column names and signal quality variable details ---
    # Based on SHHS2 dataset documentation from NSRR
    TARGET_AHI_COL: str = 'ahi_a0h3a' # Column used to define the analysis cohort

    # Configuration for signal quality variables to analyze
    # Keys are column names, values are dictionaries with display name and category mappings
    SIGNAL_QUALITY_VARS_CONFIG: Dict[str, Dict[str, Any]] = {
        'quoxim': {'name': 'SaO2 Signal Quality (Oximeter)',
                   'categories': {
                       1: '<25% artifact-free', 2: '25-49% artifact-free',
                       3: '50-74% artifact-free', 4: '75-94% artifact-free',
                       5: '>=95% artifact-free'}},
        'quhr': {'name': 'Heart Rate Signal Quality (Pulse)',
                 'categories': {
                     1: '<25% artifact-free', 2: '25-49% artifact-free',
                     3: '50-74% artifact-free', 4: '75-94% artifact-free',
                     5: '>=95% artifact-free'}},
        'quchest': {'name': 'Thoracic Effort Signal Quality (Chest Inductance)',
                    'categories': {
                        1: '<25% artifact-free', 2: '25-49% artifact-free',
                        3: '50-74% artifact-free', 4: '75-94% artifact-free',
                        5: '>=95% artifact-free'}},
        'quabdo': {'name': 'Abdominal Effort Signal Quality (Abdominal Inductance)',
                   'categories': {
                       1: '<25% artifact-free', 2: '25-49% artifact-free',
                       3: '50-74% artifact-free', 4: '75-94% artifact-free',
                       5: '>=95% artifact-free'}}
        # Add other 'qu*' variables here if needed from the SHHS2 dataset, e.g., 'quairflow', 'qusleep'
    }

    try:
        # Load the CSV file into a pandas DataFrame
        # Using encoding='latin1' and low_memory=False as often required for large SHHS CSVs
        print("Loading dataset...")
        df_data = pd.read_csv(csv_file_path, encoding='latin1', low_memory=False)
        print(f"Successfully loaded dataset. Total records: {df_data.shape[0]}")

        # --- 1. Define analysis cohort based on non-missing target AHI ---
        if TARGET_AHI_COL not in df_data.columns:
            print(f"Error: Target AHI column '{TARGET_AHI_COL}' not found in the dataset!")
            print("Please verify the column name against your SHHS2 CSV file.")
            return # Exit if essential column is missing

        # Filter for records where the target AHI is not missing
        cohort_df = df_data[df_data[TARGET_AHI_COL].notna()].copy() # Use .copy() to avoid SettingWithCopyWarning
        num_cohort = len(cohort_df)

        if num_cohort == 0:
            print(f"Warning: No records found with non-missing '{TARGET_AHI_COL}'. Cannot perform analysis.")
            return # Exit if no valid cohort members

        print(f"\nAnalysis cohort defined by non-missing '{TARGET_AHI_COL}'. N = {num_cohort}")
        print("="*70)

        # --- 2. Analyze Signal Quality Variables for the Cohort ---
        print(f"\n--- Signal Quality Statistics for Cohort (N={num_cohort}) ---")

        # Identify which of the target quality columns exist in the DataFrame
        available_quality_vars = {
            col: config for col, config in SIGNAL_QUALITY_VARS_CONFIG.items()
            if col in cohort_df.columns
        }

        if not available_quality_vars:
            print("No specified signal quality variable columns found in the dataset.")
            print(f"Checked for columns: {list(SIGNAL_QUALITY_VARS_CONFIG.keys())}")
            print("Please verify column names against your SHHS2 CSV file.")
            return

        for var_col_name, info in available_quality_vars.items():
            print(f"\n--- Statistics for {info['name']} ({var_col_name}) ---")

            signal_series = cohort_df[var_col_name].copy()
            num_total_in_cohort = len(signal_series) # Total cohort size for context
            signal_series_dropna = signal_series.dropna()
            num_valid = len(signal_series_dropna)
            num_missing = num_total_in_cohort - num_valid

            print(f"N (total in cohort for this var): {num_total_in_cohort}")
            print(f"N (non-missing values): {num_valid}")
            # Calculate and format missing percentage carefully
            missing_percentage = (num_missing / num_total_in_cohort * 100) if num_total_in_cohort > 0 else 0
            print(f"N (missing values): {num_missing} ({missing_percentage:.1f}%)")


            if num_valid == 0:
                print(f"No valid (non-missing) data for '{var_col_name}' in the defined cohort.")
                continue

            # Ensure the data is treated numerically for mean/median/std, coercing errors
            signal_series_numeric = pd.to_numeric(signal_series_dropna, errors='coerce').dropna()

            if not signal_series_numeric.empty:
                # Descriptive statistics for the quality score (treating as ordinal/numeric)
                print(f"Mean score: {signal_series_numeric.mean():.2f}")
                print(f"Median score: {signal_series_numeric.median():.2f}")
                print(f"Std Dev of scores: {signal_series_numeric.std():.2f}")
                print(f"Min score: {signal_series_numeric.min():.0f}") # Use .0f for integer display
                print(f"Max score: {signal_series_numeric.max():.0f}")

                print("\nCounts per Category:")
                # Use value_counts on the numeric series for counts
                counts = signal_series_numeric.value_counts().sort_index()
                total_valid_numeric = counts.sum() # Sum of counts should equal num_valid_numeric

                for val_code, count_val in counts.items():
                    # Ensure val_code is int for dict lookup, handle potential floats from data
                    val_code_int: Optional[int] = int(round(val_code)) if pd.notna(val_code) else None
                    # Look up category label, providing a fallback if code is not in config
                    label = info['categories'].get(val_code_int, f"Unknown code: {val_code}")
                    print(f"  Category {val_code_int} ({label}): {count_val}")

                print("\nPercentages per Category:")
                # Use value_counts with normalize=True on the numeric series for percentages
                percentages = signal_series_numeric.value_counts(normalize=True).sort_index() * 100

                for val_code, perc in percentages.items():
                    val_code_int: Optional[int] = int(round(val_code)) if pd.notna(val_code) else None
                    label = info['categories'].get(val_code_int, f"Unknown code: {val_code}")
                    print(f"  Category {val_code_int} ({label}): {perc:.1f}%")

                # Check for any non-numeric values that were dropped by pd.to_numeric
                num_dropped_non_numeric = num_valid - len(signal_series_numeric)
                if num_dropped_non_numeric > 0:
                    print(f"\nWarning: {num_dropped_non_numeric} values were non-numeric or invalid and excluded from numeric stats and category counts.")


            else:
                print(f"No valid numeric data found for '{var_col_name}' after coercing errors.")


        print("\n" + "="*70)
        print("--- Signal Quality Analysis Finished ---")

    except FileNotFoundError:
        print(f"\nError: The file '{csv_file_path}' was not found.")
        print("Please ensure the SHHS2 dataset CSV is in the correct location or provide the full path using the --csv_file argument.")
    except KeyError as e:
        print(f"\nKeyError: A specified column name does not exist in the CSV: {e}")
        print(f"Please verify column names ({TARGET_AHI_COL} and signal quality variables) against your SHHS2 CSV file.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during analysis: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze SHHS2 signal quality variables from the main SHHS2 dataset CSV."
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default="shhs2-dataset-0.21.0.csv", # Default filename, user might need to change
        help="Path to the SHHS2 dataset CSV file (e.g., 'shhs2-dataset-0.21.0.csv'). "
             "Defaults to looking for 'shhs2-dataset-0.21.0.csv' in the current directory."
    )
    args = parser.parse_args()

    # Call the analysis function with the provided CSV file path
    analyze_signal_quality(args.csv_file)