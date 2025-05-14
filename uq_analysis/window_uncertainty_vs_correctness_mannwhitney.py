#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np # Keep numpy, might be used by pandas or scipy internally
from scipy.stats import mannwhitneyu
import os
import argparse
from typing import Optional, Tuple, NoReturn, List

# --- Configuration ---
# Default path to the detailed per-window MC Dropout results CSV -> created from analyze_mcd_patient_level.py
DEFAULT_MCD_DETAIL_CSV: str = ''

# Default path to the detailed per-window Deep Ensemble results CSV -> created from analyze_DE_patient_level.py
DEFAULT_DE_DETAIL_CSV: str = ''

# Default column names for the uncertainty metric and true/predicted labels
DEFAULT_UNCERTAINTY_COL: str = 'Predictive_Entropy' # Column containing the uncertainty metric
DEFAULT_TRUE_LABEL_COL: str = 'True_Label'
DEFAULT_PREDICTED_LABEL_COL: str = 'Predicted_Label'

# Significance level (alpha) for interpreting the p-value
DEFAULT_ALPHA: float = 0.05

# --- Function to Perform Mann-Whitney U Test ---
def perform_mannwhitneyu_test(
        df: pd.DataFrame,
        method_name: str, # Name of the method (e.g., "MC Dropout", "Deep Ensemble")
        uncertainty_col: str,
        true_label_col: str,
        predicted_label_col: str,
        alpha: float = DEFAULT_ALPHA
) -> Optional[Tuple[float, float]]:
    """
    Performs the Mann-Whitney U test to compare the distribution of an
    uncertainty metric between correctly and incorrectly classified windows.

    The null hypothesis (H0) is that the distribution of the uncertainty metric
    is the same for both groups (correct and incorrect predictions).
    The alternative hypothesis (H1) is that the distribution of the uncertainty
    metric for incorrect predictions is GREATER than for correct predictions.

    Args:
        df: pandas DataFrame containing the detailed per-window results.
        method_name: A descriptive name for the method being analyzed.
        uncertainty_col: The name of the column containing the uncertainty metric.
        true_label_col: The name of the column with true labels.
        predicted_label_col: The name of the column with predicted labels.
        alpha: The significance level for the test.

    Returns:
        A tuple containing the Mann-Whitney U statistic and the p-value.
        Returns None if essential columns are missing, or if there are not
        enough valid data points in one or both groups to perform the test.
    """
    print(f"--- Analyzing {method_name} ---")

    # Check if essential columns exist
    required_cols = [uncertainty_col, true_label_col, predicted_label_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Required columns ('{missing_cols}') not found in the DataFrame for {method_name}.")
        return None

    # Ensure 'Correct' column exists
    if 'Correct' not in df.columns:
        df['Correct'] = (df[true_label_col] == df[predicted_label_col])
        print(f"Added 'Correct' column for {method_name} analysis.")


    # Separate the uncertainty metric values for correct and incorrect predictions
    # Use .loc for label-based indexing and .dropna() to remove potential NaNs
    try:
        uncertainty_correct = df.loc[df['Correct'] == True, uncertainty_col].dropna()
        uncertainty_incorrect = df.loc[df['Correct'] == False, uncertainty_col].dropna()

        print(f"Number of correctly classified windows: {len(uncertainty_correct)}")
        print(f"Number of incorrectly classified windows: {len(uncertainty_incorrect)}")

        # Check if there are enough data points in both groups for the test
        # Mann-Whitney U test requires at least 1 sample in each group
        if len(uncertainty_correct) == 0 or len(uncertainty_incorrect) == 0:
            print("Warning: Not enough data points in one or both groups (correct/incorrect) to perform the test.")
            return None
        # For robust results, more samples are usually needed, but the test technically runs with small N.


        # Perform the Mann-Whitney U test
        # We are testing if the uncertainty for incorrect predictions is greater than for correct predictions.
        # H0: Distribution of uncertainty is the same for both groups.
        # H1: Distribution of uncertainty for incorrect predictions is stochastically greater than for correct predictions.
        # So, we test 'incorrect' against 'correct' with alternative='greater'.
        stat, p_value = mannwhitneyu(uncertainty_incorrect, uncertainty_correct, alternative='greater')

        print(f"\n--- Mann-Whitney U Test Results ({uncertainty_col}: Incorrect > Correct) ---")
        print(f"Method: {method_name}")
        print(f"  U Statistic: {stat:.4f}")
        # Use .4g format for p-value to handle very small values clearly
        print(f"  P-value: {p_value:.4g}")

        # Interpret the result based on the significance level
        if p_value < alpha:
            print(f"Conclusion: The difference is statistically significant at alpha = {alpha}.")
            print(f"  We reject the null hypothesis (H0). Incorrect predictions have significantly higher {uncertainty_col}.")
        else:
            print(f"Conclusion: The difference is not statistically significant at alpha = {alpha}.")
            print(f"  We fail to reject the null hypothesis (H0).")

        print("-" * 30) # Use 30 dashes for consistency
        return float(stat), float(p_value) # Return as standard floats

    except Exception as e:
        print(f"An unexpected error occurred during the Mann-Whitney U test for {method_name}: {e}")
        return None # Return None if any error occurs


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Mann-Whitney U Test Script ---")

    # Setup argparse for command-line configuration
    parser = argparse.ArgumentParser(
        description="Perform Mann-Whitney U test to compare uncertainty distribution for correct vs. incorrect predictions."
    )
    parser.add_argument(
        "--mcd_csv",
        type=str,
        default=DEFAULT_MCD_DETAIL_CSV,
        help=f"Path to the detailed per-window MC Dropout results CSV (default: '{DEFAULT_MCD_DETAIL_CSV}')."
    )
    parser.add_argument(
        "--de_csv",
        type=str,
        default=DEFAULT_DE_DETAIL_CSV,
        help=f"Path to the detailed per-window Deep Ensemble results CSV (default: '{DEFAULT_DE_DETAIL_CSV}')."
    )
    parser.add_argument(
        "--uncertainty_col",
        type=str,
        default=DEFAULT_UNCERTAINTY_COL,
        help=f"Name of the column containing the uncertainty metric (default: '{DEFAULT_UNCERTAINTY_COL}')."
    )
    parser.add_argument(
        "--true_label_col",
        type=str,
        default=DEFAULT_TRUE_LABEL_COL,
        help=f"Name of the column containing true labels (default: '{DEFAULT_TRUE_LABEL_COL}')."
    )
    parser.add_argument(
        "--predicted_label_col",
        type=str,
        default=DEFAULT_PREDICTED_LABEL_COL,
        help=f"Name of the column containing predicted labels (default: '{DEFAULT_PREDICTED_LABEL_COL}')."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Significance level for interpreting the p-value (default: {DEFAULT_ALPHA})."
    )
    # Optional flag to skip MCD analysis if not needed/available
    parser.add_argument(
        "--skip_mcd",
        action='store_true', # This makes it a boolean flag, default is False
        help="Set this flag to skip the MC Dropout analysis."
    )
    # Optional flag to skip DE analysis if not needed/available
    parser.add_argument(
        "--skip_de",
        action='store_true',
        help="Set this flag to skip the Deep Ensemble analysis."
    )


    args = parser.parse_args()

    # Validate that the user has provided necessary paths if not skipping
    if not args.skip_mcd and not args.mcd_csv:
        print(f"\nERROR: MC Dropout CSV path (--mcd_csv) must be specified unless --skip_mcd is used, as the default is empty ('{DEFAULT_MCD_DETAIL_CSV}').")
        parser.print_help()
        sys.exit(1)
    if not args.skip_de and not args.de_csv:
        print(f"\nERROR: Deep Ensemble CSV path (--de_csv) must be specified unless --skip_de is used, as the default is empty ('{DEFAULT_DE_DETAIL_CSV}').")
        parser.print_help()
        sys.exit(1)
    if args.skip_mcd and args.skip_de:
        print("\nWarning: Both --skip_mcd and --skip_de flags are set. No analysis will be performed.")


    print(f"Uncertainty Metric Column: '{args.uncertainty_col}'")
    print(f"True Label Column: '{args.true_label_col}'")
    print(f"Predicted Label Column: '{args.predicted_label_col}'")
    print(f"Significance Level (alpha): {args.alpha}")
    print("-" * 30)


    # --- Perform Test for MC Dropout ---
    mcd_stat, mcd_p_value = None, None
    if not args.skip_mcd:
        try:
            mcd_detail_df = pd.read_csv(args.mcd_csv)
            mcd_stat, mcd_p_value = perform_mannwhitneyu_test(
                df=mcd_detail_df,
                method_name="MC Dropout",
                uncertainty_col=args.uncertainty_col,
                true_label_col=args.true_label_col,
                predicted_label_col=args.predicted_label_col,
                alpha=args.alpha
            )
        except FileNotFoundError:
            print(f"ERROR: MC Dropout detailed results CSV not found at '{args.mcd_csv}'.")
            mcd_stat, mcd_p_value = None, None # Explicitly set to None on FileNotFoundError
        except Exception as e:
            print(f"An error occurred during MC Dropout analysis setup or test: {e}")
            mcd_stat, mcd_p_value = None, None


    # --- Perform Test for Deep Ensemble ---
    de_stat, de_p_value = None, None
    if not args.skip_de:
        try:
            de_detail_df = pd.read_csv(args.de_csv)
            de_stat, de_p_value = perform_mannwhitneyu_test(
                df=de_detail_df,
                method_name="Deep Ensemble",
                uncertainty_col=args.uncertainty_col,
                true_label_col=args.true_label_col,
                predicted_label_col=args.predicted_label_col,
                alpha=args.alpha
            )
        except FileNotFoundError:
            print(f"ERROR: Deep Ensemble detailed results CSV not found at '{args.de_csv}'.")
            de_stat, de_p_value = None, None # Explicitly set to None on FileNotFoundError
        except Exception as e:
            print(f"An error occurred during Deep Ensemble analysis setup or test: {e}")
            de_stat, de_p_value = None, None


    # --- Conclusion ---
    print("\n--- Mann-Whitney U Test Script Complete ---")
    print("Results printed above.")
    print("Script finished.")