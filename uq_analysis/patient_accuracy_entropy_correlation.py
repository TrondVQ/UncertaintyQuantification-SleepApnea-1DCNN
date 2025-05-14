#!/usr/bin/env python3
import sys
import pandas as pd
from scipy.stats import pearsonr
import os
import argparse
from typing import Tuple, Optional, NoReturn # Import types

# --- Configuration ---
# Default path to the patient-level summary CSV file for MC Dropout results -> from analyze_MCD_patient_level.py
DEFAULT_MCD_CSV_PATH: str = ''

# Default path to the patient-level summary CSV file for Deep Ensemble results -> from analyze_DE_patient_level.py
DEFAULT_DE_CSV_PATH: str = ''

# Default column names to use for correlation calculation
DEFAULT_X_COL: str = 'mean_entropy'     # Default X variable (e.g., a UQ metric)
DEFAULT_Y_COL: str = 'patient_accuracy' # Default Y variable (e.g., a performance metric)


# --- Function to Load Data and Calculate Correlation ---
def calculate_and_print_correlation(
        csv_path: str,
        method_name: str, # Name of the method (e.g., "MC Dropout", "Deep Ensemble") for printing
        x_col: str,
        y_col: str
) -> Tuple[Optional[float], Optional[float]]:
    """
    Loads data from a CSV file, calculates the Pearson correlation coefficient
    and p-value between two specified columns, and prints the results.

    Args:
        csv_path: Path to the input CSV file.
        method_name: A descriptive name for the method or dataset being analyzed.
        x_col: The name of the column to use as the independent variable (X).
        y_col: The name of the column to use as the dependent variable (Y).

    Returns:
        A tuple containing the Pearson correlation coefficient (r) and the p-value.
        Returns (None, None) if the file is not found, required columns are missing,
        or if there's not enough valid data for calculation.
    """
    print(f"--- Analyzing {method_name} ---")

    if not os.path.exists(csv_path):
        print(f"ERROR: Input file not found - '{csv_path}'")
        return None, None

    try:
        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data from: '{csv_path}' with shape: {df.shape}")

        # Check if required columns exist
        if x_col not in df.columns or y_col not in df.columns:
            print(f"ERROR: Required columns ('{x_col}', '{y_col}') not found in '{csv_path}'")
            print(f"Available columns are: {list(df.columns)}")
            return None, None

        # Drop rows with NaN values in the relevant columns to ensure valid correlation calculation
        initial_row_count = len(df)
        df_clean = df[[x_col, y_col]].dropna()
        rows_dropped = initial_row_count - len(df_clean)

        if rows_dropped > 0:
            print(f"Warning: Dropped {rows_dropped} rows due to missing values (NaN) in '{x_col}' or '{y_col}'.")

        # Check if there are enough data points after dropping NaNs
        if len(df_clean) < 2:
            print(f"ERROR: Not enough valid data points ({len(df_clean)}) for correlation calculation after handling missing values.")
            return None, None
        elif len(df_clean) < 3:
            print(f"Warning: Only {len(df_clean)} data points available. Correlation may not be reliable.")


        # Calculate Pearson correlation coefficient and p-value
        correlation, p_value = pearsonr(df_clean[x_col], df_clean[y_col])

        print(f"\nCorrelation Results for {method_name}:")
        print(f"  Correlation between '{x_col}' and '{y_col}':")
        print(f"    Pearson r = {correlation:.4f}")
        # Use .4g format for p-value to handle very small values clearly
        print(f"    P-value = {p_value:.4g}")
        print("-" * 30) # Use 30 dashes for consistency
        return correlation, p_value

    except Exception as e:
        print(f"An unexpected error occurred while processing '{csv_path}': {e}")
        return None, None

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Patient-Level Correlation Calculation Script ---")

    # Setup argparse for command-line configuration
    parser = argparse.ArgumentParser(
        description="Calculate Pearson correlation between specified metrics from patient-level UQ summary CSVs."
    )
    parser.add_argument(
        "--mcd_csv",
        type=str,
        default=DEFAULT_MCD_CSV_PATH,
        help=f"Path to the patient-level summary CSV for MC Dropout results (default: '{DEFAULT_MCD_CSV_PATH}')."
    )
    parser.add_argument(
        "--de_csv",
        type=str,
        default=DEFAULT_DE_CSV_PATH,
        help=f"Path to the patient-level summary CSV for Deep Ensemble results (default: '{DEFAULT_DE_CSV_PATH}')."
    )
    parser.add_argument(
        "--x_col",
        type=str,
        default=DEFAULT_X_COL,
        help=f"Name of the column for the X variable (default: '{DEFAULT_X_COL}')."
    )
    parser.add_argument(
        "--y_col",
        type=str,
        default=DEFAULT_Y_COL,
        help=f"Name of the column for the Y variable (default: '{DEFAULT_Y_COL}')."
    )
    # Optional flag to skip MCD analysis if not needed/available
    parser.add_argument(
        "--skip_mcd",
        action='store_true', # This makes it a boolean flag, default is False
        help="Set this flag to skip the MC Dropout correlation analysis."
    )
    # Optional flag to skip DE analysis if not needed/available
    parser.add_argument(
        "--skip_de",
        action='store_true',
        help="Set this flag to skip the Deep Ensemble correlation analysis."
    )


    args = parser.parse_args()

    # Validate that the user has provided necessary paths if not skipping
    if not args.skip_mcd and not args.mcd_csv:
        print("\nERROR: MC Dropout CSV path (--mcd_csv) must be specified unless --skip_mcd is used, as the default is empty.")
        parser.print_help()
        sys.exit(1)
    if not args.skip_de and not args.de_csv:
        print("\nERROR: Deep Ensemble CSV path (--de_csv) must be specified unless --skip_de is used, as the default is empty.")
        parser.print_help()
        sys.exit(1)
    if args.skip_mcd and args.skip_de:
        print("\nWarning: Both --skip_mcd and --skip_de flags are set. No analysis will be performed.")


    # --- Perform Calculations and Print Results ---
    print("Calculating Patient-Level Correlations...")
    print(f"Using X variable: '{args.x_col}'")
    print(f"Using Y variable: '{args.y_col}'")
    print("-" * 30)

    mcd_correlation, mcd_p_value = None, None
    de_correlation, de_p_value = None, None


    # Calculate for MC Dropout if not skipped
    if not args.skip_mcd:
        mcd_correlation, mcd_p_value = calculate_and_print_correlation(
            csv_path=args.mcd_csv,
            method_name="MC Dropout",
            x_col=args.x_col,
            y_col=args.y_col
        )
    else:
        print("\n--- Skipping MC Dropout Analysis as requested ---")


    # Calculate for Deep Ensemble if not skipped
    if not args.skip_de:
        de_correlation, de_p_value = calculate_and_print_correlation(
            csv_path=args.de_csv,
            method_name="Deep Ensemble",
            x_col=args.x_col,
            y_col=args.y_col
        )
    else:
        print("\n--- Skipping Deep Ensemble Analysis as requested ---")


    # --- Conclusion ---
    print("\n--- Patient-Level Correlation Calculation Complete ---")
    print("Correlation coefficients (Pearson r) and p-values printed above.")

    # Added specific printout to guide thesis writing, similar to original script
    print("\nGuide for Thesis Sections:")
    if mcd_correlation is not None:
        print(f"  Pearson r for MC Dropout (correlation between '{args.x_col}' and '{args.y_col}'): {mcd_correlation:.4f}")
        print(f"  P-value for MC Dropout: {mcd_p_value:.4g}")
        print("    -> Use these values for your analysis of MC Dropout UQ vs Performance.")
    if de_correlation is not None:
        print(f"  Pearson r for Deep Ensemble (correlation between '{args.x_col}' and '{args.y_col}'): {de_correlation:.4f}")
        print(f"  P-value for Deep Ensemble: {de_p_value:.4g}")
        print("    -> Use these values for your analysis of Deep Ensemble UQ vs Performance.")

    # IMPORTANT NOTE ON MCD ACCURACY (Kept from original code)
    print("\n--- IMPORTANT NOTE ON MC Dropout Accuracy ---")
    print(f"REMINDER: The '{DEFAULT_Y_COL}' in the MC Dropout summary CSV typically corresponds to the accuracy based on the MEAN MC prediction (>0.5 threshold).")
    print("This might be different from the accuracy obtained from a single deterministic model pass (training=False).")
    print("Ensure you interpret the MC Dropout correlation value within this specific context.")
    print("-" * 30)

    print("Script finished.")