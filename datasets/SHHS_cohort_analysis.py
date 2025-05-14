#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse
from typing import NoReturn # Use NoReturn to indicate function does not return a value

def analyze_cohort(csv_file_path: str) -> NoReturn:
    """
    Analyzes the SHHS2 cohort demographics (Age, Gender, Race) and AHI distribution
    from the main SHHS2 dataset CSV file.

    The analysis cohort is defined by individuals with non-missing AHI ('ahi_a0h3a') values.
    Statistical summaries for the cohort and AHI severity distribution based on standard
    clinical thresholds are printed to the console.

    Args:
        csv_file_path: Path to the SHHS2 dataset CSV file (e.g., 'shhs2-dataset-0.21.0.csv').
                       This file is typically available from the NSRR website.
    """
    print(f"--- Analyzing SHHS2 Cohort from: {csv_file_path} ---")
    print("="*70)

    # --- Configuration: Define column names used in the analysis ---
    # Based on SHHS2 dataset documentation from NSRR
    TARGET_AHI_COL: str = 'ahi_a0h3a' # Apnea-Hypopnea Index (events/hour)
    TARGET_AGE_COL: str = 'age_s2'    # Age at SHHS2 visit
    TARGET_GENDER_COL: str = 'gender' # Gender (1=Male, 2=Female)
    TARGET_RACE_COL: str = 'race'     # Race (1=White, 2=Black or African American, 3=Other)

    try:
        # Load the CSV file into a pandas DataFrame
        # Using encoding='latin1' and low_memory=False as often required for large SHHS CSVs
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
        # Note: The number can vary slightly based on NSRR dataset versions or minor exclusions.
        # Expected SHHS2 N with non-missing ahi_a0h3a is around 2651 based on common data versions.
        print("="*70)

        # --- 2. Calculate Statistics for the Cohort (N={num_cohort}) ---

        # Age Statistics
        print(f"\n--- Age Statistics for Cohort (N={num_cohort}) ---")
        if TARGET_AGE_COL in cohort_df.columns:
            age_series = cohort_df[TARGET_AGE_COL].dropna()
            print(f"N (non-missing age in cohort): {len(age_series)}")
            if not age_series.empty:
                # Ensure data is numeric before calculating stats
                age_series_numeric = pd.to_numeric(age_series, errors='coerce').dropna()
                if not age_series_numeric.empty:
                    print(f"Mean Age: {age_series_numeric.mean():.1f} ± {age_series_numeric.std():.1f} years")
                    print(f"Median Age: {age_series_numeric.median():.1f} years")
                    print(f"Age Range: {age_series_numeric.min():.1f} - {age_series_numeric.max():.1f} years")
                else:
                    print("No valid numeric age data after handling missing values.")
            else:
                print("No age data available in the defined cohort.")
        else:
            print(f"Warning: Age column '{TARGET_AGE_COL}' not found in cohort data.")
        print("-" * 30)

        # Gender Statistics
        print(f"\n--- Gender Statistics for Cohort (N={num_cohort}) ---")
        if TARGET_GENDER_COL in cohort_df.columns:
            gender_series = cohort_df[TARGET_GENDER_COL].dropna()
            print(f"N (non-missing gender in cohort): {len(gender_series)}")
            if not gender_series.empty:
                gender_counts = gender_series.value_counts().sort_index()
                gender_percentages = gender_series.value_counts(normalize=True).sort_index() * 100
                print("Counts & Percentages:")
                # Mapping based on NSRR SHHS2 documentation: 1.0=Male, 2.0=Female
                print(f"  Male (1.0):   {gender_counts.get(1.0, 0):<5} ({gender_percentages.get(1.0, 0.0):.1f}%)")
                print(f"  Female (2.0): {gender_counts.get(2.0, 0):<5} ({gender_percentages.get(2.0, 0.0):.1f}%)")
                # Print any other unexpected gender codes
                for val, count in gender_counts.items():
                    if val not in [1.0, 2.0]:
                        print(f"  Unknown Code ({val}): {count:<5} ({gender_percentages.get(val, 0.0):.1f}%)")
            else:
                print("No gender data available in the defined cohort.")
        else:
            print(f"Warning: Gender column '{TARGET_GENDER_COL}' not found in cohort data.")
        print("-" * 30)

        # Race Statistics
        print(f"\n--- Race Statistics for Cohort (N={num_cohort}) ---")
        if TARGET_RACE_COL in cohort_df.columns:
            race_series = cohort_df[TARGET_RACE_COL].dropna()
            print(f"N (non-missing race in cohort): {len(race_series)}")
            if not race_series.empty:
                race_counts = race_series.value_counts().sort_index()
                race_percentages = race_series.value_counts(normalize=True).sort_index() * 100
                print("Counts & Percentages:")
                # Mapping based on NSRR SHHS2 documentation: 1.0=White, 2.0=Black or African American, 3.0=Other
                print(f"  White (1.0):                      {race_counts.get(1.0, 0):<5} ({race_percentages.get(1.0, 0.0):.1f}%)")
                print(f"  Black or African American (2.0): {race_counts.get(2.0, 0):<5} ({race_percentages.get(2.0, 0.0):.1f}%)")
                print(f"  Other (3.0):                      {race_counts.get(3.0, 0):<5} ({race_percentages.get(3.0, 0.0):.1f}%)")
                # Print any other unexpected race codes
                for val, count in race_counts.items():
                    if val not in [1.0, 2.0, 3.0]:
                        print(f"  Unknown Code ({val}): {count:<5} ({race_percentages.get(val, 0.0):.1f}%)")
            else:
                print("No race data available in the defined cohort.")
        else:
            print(f"Warning: Race column '{TARGET_RACE_COL}' not found in cohort data.")
        print("-" * 30)


        # AHI Overall Statistics and Categorization
        print(f"\n--- AHI ('{TARGET_AHI_COL}') Statistics for Cohort (N={num_cohort}) ---")
        ahi_series_cohort = cohort_df[TARGET_AHI_COL].dropna()
        # Note: AHI was used to define the cohort, so dropna() here should keep all cohort members.
        # Adding a check just in case, though.
        if not ahi_series_cohort.empty:
            # Ensure AHI data is numeric
            ahi_series_cohort_numeric = pd.to_numeric(ahi_series_cohort, errors='coerce').dropna()

            if not ahi_series_cohort_numeric.empty:
                print(f"N (non-missing numeric AHI): {len(ahi_series_cohort_numeric)}")
                print(f"Mean AHI: {ahi_series_cohort_numeric.mean():.1f} ± {ahi_series_cohort_numeric.std():.1f} events/hour")
                print(f"Median AHI: {ahi_series_cohort_numeric.median():.1f} events/hour")
                print(f"AHI Range: {ahi_series_cohort_numeric.min():.1f} - {ahi_series_cohort_numeric.max():.1f} events/hour")

                # AHI Categorization (standard clinical thresholds from R. Berry et al., 2012, J Clin Sleep Med)
                print("\n--- AHI Categories (Standard Clinical Thresholds) ---")
                conditions = [
                    (cohort_df[TARGET_AHI_COL] < 5),
                    (cohort_df[TARGET_AHI_COL] >= 5) & (cohort_df[TARGET_AHI_COL] < 15),
                    (cohort_df[TARGET_AHI_COL] >= 15) & (cohort_df[TARGET_AHI_COL] < 30),
                    (cohort_df[TARGET_AHI_COL] >= 30)
                ]
                categories = [
                    'Normal (AHI < 5.0)', # Added .0 for consistency
                    'Mild OSA (AHI 5.0-14.9)',
                    'Moderate OSA (AHI 15.0-29.9)',
                    'Severe OSA (AHI >= 30.0)'
                ]
                # Apply categorization, ensuring the AHI column is numeric for comparison
                cohort_df['ahi_severity_category'] = np.select(
                    [pd.to_numeric(cohort_df[TARGET_AHI_COL], errors='coerce').fillna(-1)[cond.index].loc[cond] for cond in conditions],
                    categories,
                    default='Unknown_AHI_Value' # Changed from Error_AHI_Value for clarity
                )

                # Ensure categories are ordered correctly in the output
                category_counts = cohort_df['ahi_severity_category'].value_counts().reindex(categories + ['Unknown_AHI_Value'], fill_value=0)
                category_percentages = cohort_df['ahi_severity_category'].value_counts(normalize=True).reindex(categories + ['Unknown_AHI_Value'], fill_value=0) * 100

                print("AHI Severity Distribution in Cohort:")
                for category_name in categories:
                    abs_count = category_counts[category_name]
                    percentage = category_percentages[category_name]
                    print(f"  {category_name:<25}: {abs_count:<5} ({percentage:.1f}%)")

                if 'Unknown_AHI_Value' in category_counts and category_counts['Unknown_AHI_Value'] > 0:
                    error_count = category_counts['Unknown_AHI_Value']
                    error_percentage = category_percentages['Unknown_AHI_Value']
                    print(f"  Unknown AHI Value        : {error_count:<5} ({error_percentage:.1f}%) - Could be non-numeric or unexpected values.")

            else:
                print("No valid numeric AHI data after handling missing values.")
        else:
            print(f"No AHI data available in the defined cohort (should not happen if cohort is defined by non-missing AHI).")

    except FileNotFoundError:
        print(f"\nError: The file '{csv_file_path}' was not found.")
        print("Please ensure the SHHS2 dataset CSV is in the correct location or provide the full path using the --csv_file argument.")
    except KeyError as e:
        print(f"\nKeyError: A specified column name does not exist in the CSV: {e}")
        print(f"Please verify column names ({TARGET_AHI_COL}, {TARGET_AGE_COL}, {TARGET_GENDER_COL}, {TARGET_RACE_COL}) against your SHHS2 CSV file.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during analysis: {e}")

    print("\n" + "="*70)
    print("--- Cohort Analysis Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze SHHS2 cohort demographics and AHI distribution from the main SHHS2 dataset CSV."
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
    analyze_cohort(args.csv_file)