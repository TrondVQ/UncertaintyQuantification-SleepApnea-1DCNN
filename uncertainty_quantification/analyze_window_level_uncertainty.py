#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import argparse
from typing import List, NoReturn, Dict, Any, Optional  # Import types
import sys

# --- Configuration ---
# Define default parameters (can be overridden by command-line args)

# Default path to the input CSV file containing detailed window-level UQ results -> from either analyze_DE_window_level.py or analyze_MCD_window_level.py
DEFAULT_DETAILED_RESULTS_CSV: str = ""

# Default number of bins to use for analyzing metrics across uncertainty levels
NUM_BINS: int = 10

# Define required column names in the input CSV
REQUIRED_COLUMNS: List[str] = [
    'True_Label',        # The true binary label for the window (e.g., 0 or 1)
    'Predicted_Label',   # The model's predicted binary label for the window (e.g., 0 or 1)
    'Predictive_Variance', # Window-level predictive variance
    'Predictive_Entropy' # Window-level predictive entropy
    # 'Patient_ID', # Optional column, not used in window-level aggregation here
    # 'Window_Index', # Optional column, not used in window-level aggregation here
    # 'Predicted_Probability', # Optional, but useful to have
    # 'Expected_Aleatoric_Entropy', # Optional, useful if present
    # 'Mutual_Information' # Optional, useful if present
]


# --- Main Analysis Function ---

def analyze_window_uq(
        detailed_results_csv: str,
        num_bins: int = NUM_BINS,
        output_binned_csv: Optional[str] = None # Optional path to save binned results CSV
) -> NoReturn:
    """
    Analyzes overall window-level uncertainty quantification (UQ) metrics and their
    relationship with classification correctness.

    Loads detailed window-level UQ results from a CSV, calculates overall accuracy,
    compares uncertainty metrics for correct vs. incorrect predictions, and bins
    data by Predictive Entropy to show accuracy/error rate trends across bins.
    Prints textual summaries and optionally saves the binned results to a CSV.

    Args:
        detailed_results_csv: Path to the input CSV file with window-level UQ results.
        num_bins: The number of bins to use for analyzing trends across uncertainty levels.
        output_binned_csv: Optional path (including filename) to save the binned
                           accuracy/error rate results as a CSV file. If None, the
                           binned results are only printed.
    """
    print(f"--- Starting Overall Window-Level UQ Analysis ---")
    print(f"Input Detailed Results CSV: {detailed_results_csv}")
    print(f"Number of Bins for Analysis: {num_bins}")
    if output_binned_csv:
        print(f"Binned results will be saved to: {output_binned_csv}")
    else:
        print("Binned results will be printed but NOT saved to file.")
    print("="*70)

    # --- Load Detailed UQ Results ---
    try:
        print("Loading detailed window-level UQ results...")
        uq_results_df = pd.read_csv(detailed_results_csv)
        print(f"Successfully loaded data with shape: {uq_results_df.shape}")
    except FileNotFoundError:
        print(f"ERROR: Input CSV file not found at '{detailed_results_csv}'.")
        print("Please ensure the file exists or provide the correct path using the --input_csv argument.")
        return # Exit if file not found
    except Exception as e:
        print(f"Error loading CSV file '{detailed_results_csv}': {e}")
        return # Exit on other loading errors


    # --- Data Checks and Preparation ---
    print("\nPerforming data checks for required columns...")
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in uq_results_df.columns]
    if missing_cols:
        print(f"ERROR: Missing one or more required columns in the input CSV: {missing_cols}")
        print(f"Required columns include: {REQUIRED_COLUMNS}")
        return # Exit if essential columns are missing
    print("All required columns are present.")

    if uq_results_df.empty:
        print("Warning: Input DataFrame is empty. Cannot perform analysis.")
        return # Exit if dataframe is empty

    # Ensure 'Correct' column exists
    if 'Correct' not in uq_results_df.columns:
        uq_results_df['Correct'] = (uq_results_df['True_Label'] == uq_results_df['Predicted_Label'])


    print(f"\nTotal number of windows analyzed: {len(uq_results_df)}")
    # Calculate and print overall accuracy
    overall_accuracy = uq_results_df['Correct'].mean()
    print(f"Overall accuracy across all windows: {overall_accuracy:.4f}")


    # --- Textual Summary of Overall Window-Level Trends ---
    print("\n--- Textual Summary of Overall Window-Level Uncertainty vs. Correctness ---")

    # Ensure there are both correct and incorrect predictions before describing
    correct_df = uq_results_df[uq_results_df['Correct'] == True]
    incorrect_df = uq_results_df[uq_results_df['Correct'] == False]

    # Compare uncertainty for Correct vs Incorrect predictions
    print("\nStatistics for CORRECTLY Classified Windows:")
    if not correct_df.empty:
        # Select relevant uncertainty columns (Predictive_Entropy, Predictive_Variance, maybe others if present)
        uncertainty_cols = [col for col in ['Predictive_Entropy', 'Predictive_Variance', 'Expected_Aleatoric_Entropy', 'Mutual_Information'] if col in uq_results_df.columns]
        if uncertainty_cols:
            print(correct_df[uncertainty_cols].describe().to_string())
        else:
            print("No standard uncertainty columns found in correctly classified windows.")
    else:
        print("No correctly classified windows found to describe.")


    print("\nStatistics for INCORRECTLY Classified Windows:")
    if not incorrect_df.empty:
        uncertainty_cols = [col for col in ['Predictive_Entropy', 'Predictive_Variance', 'Expected_Aleatoric_Entropy', 'Mutual_Information'] if col in uq_results_df.columns]
        if uncertainty_cols:
            print(incorrect_df[uncertainty_cols].describe().to_string())
        else:
            print("No standard uncertainty columns found in incorrectly classified windows.")
    else:
        print("No incorrectly classified windows found to describe.")


    # --- Calculate and Print Binned Accuracy/Error Rate vs. Predictive Entropy ---
    metric_to_bin = 'Predictive_Entropy'
    if metric_to_bin not in uq_results_df.columns:
        print(f"\nWarning: Binning by '{metric_to_bin}' skipped as column not found.")
    elif num_bins <= 0:
        print(f"\nWarning: Binning skipped as num_bins is {num_bins} (must be > 0).")
    else:
        print(f"\n--- Binned Accuracy/Error Rate vs. {metric_to_bin} ---")

        # Calculate bin edges
        min_metric = uq_results_df[metric_to_bin].min()
        max_metric = uq_results_df[metric_to_bin].max()

        # Handle case where min == max (all values are the same)
        if min_metric == max_metric:
            print(f"Warning: All '{metric_to_bin}' values are the same ({min_metric}). Cannot create multiple bins.")
            # Create a single bin for all data
            bins = [min_metric, max_metric + 1e-9] # Create a bin edge slightly above max
            labels = [f'{min_metric:.3f}'] # Single label
            print(f"Created 1 bin: {labels[0]}")
            num_bins_actual = 1
        else:
            # Create bins using linspace, ensuring max is included by adding a small epsilon
            bins = np.linspace(min_metric, max_metric + 1e-9, num_bins + 1)
            # Create labels for bins for easier reading
            # Format labels as ranges [start, end) except for the last bin [start, end]
            labels = [f'[{bins[i]:.3f}-{bins[i+1]:.3f})' for i in range(num_bins)]
            labels[-1] = f'[{bins[-2]:.3f}-{bins[-1]:.3f}]' # Correct the last label to include the max value
            print(f"Created {num_bins} bins ranging from {min_metric:.3f} to {max_metric:.3f}")
            num_bins_actual = num_bins


        # Assign each window to a bin based on its metric value
        # Use include_lowest=True to make the first bin inclusive of the minimum value
        # Use right=False if bins are [start, end), True if (start, end] - standard is right=True, but linspace with epsilon works well with right=False
        # Let's stick to right=False and ensure the last bin label is correct.
        try:
            uq_results_df[f'{metric_to_bin}_Bin'] = pd.cut(
                uq_results_df[metric_to_bin],
                bins=bins,
                labels=labels if num_bins_actual > 1 else labels, # Use labels if multiple bins
                include_lowest=True, # Include the minimum value in the first bin
                right=False # Bins are [a, b)
            )
        except ValueError as e:
            print(f"Error creating bins: {e}. Check bin edges and labels.")
            # Print bin edges and labels for debugging
            print(f"Bins: {bins}")
            print(f"Labels: {labels}")
            return # Exit if binning fails


        # Group by bin and calculate metrics
        # observed=False ensures all bins from the 'categories' are included, even if empty
        binned_results = uq_results_df.groupby(f'{metric_to_bin}_Bin', observed=False).agg(
            window_count=('Correct', 'size'),
            accuracy=('Correct', 'mean')
        )
        binned_results['error_rate'] = 1 - binned_results['accuracy'] # Error rate is 1 - accuracy

        # Print the binned results table
        print(f"\nAccuracy and Error Rate per {metric_to_bin} Bin:")
        if not binned_results.empty:
            # Format columns for cleaner output
            binned_results_formatted = binned_results.copy()
            binned_results_formatted['accuracy'] = binned_results_formatted['accuracy'].map('{:.4f}'.format)
            binned_results_formatted['error_rate'] = binned_results_formatted['error_rate'].map('{:.4f}'.format)

            print(binned_results_formatted.to_string())
        else:
            print("No binned results to display.")


        # --- Optional: Save Binned Results to CSV ---
        if output_binned_csv:
            try:
                # Ensure directory exists before saving
                output_dir = os.path.dirname(output_binned_csv)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"Created output directory for binned results: {output_dir}")

                binned_results.to_csv(output_binned_csv) # Save the raw numerical values, not the formatted ones
                print(f"Binned accuracy and error rate results saved to: {output_binned_csv}")
            except Exception as e:
                print(f"ERROR saving binned results CSV to '{output_binned_csv}': {e}")


    print("\n--- Overall Window-Level Textual Analysis Complete ---")
    print(f"Analysis performed on: {detailed_results_csv}")
    if output_binned_csv:
        print(f"Binned results saved to: {output_binned_csv}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze overall window-level uncertainty quantification metrics and their relationship with classification correctness."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=DEFAULT_DETAILED_RESULTS_CSV,
        help=f"Path to the input CSV file with detailed window-level UQ results (default: '{DEFAULT_DETAILED_RESULTS_CSV}'). "
             "E.g., ./uq_results/.../detailed_results_CNN_DE_Unbalanced.csv"
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=NUM_BINS,
        help=f"Number of bins to use for analyzing trends across uncertainty levels (default: {NUM_BINS})."
    )
    parser.add_argument(
        "--output_binned_csv",
        type=str,
        default=None, # Default is None, meaning don't save
        help="Optional path (including filename) to save the binned accuracy/error rate results as a CSV file. If not specified, results are only printed."
    )


    args = parser.parse_args()

    # Validate that the user has provided an input CSV path if the default is empty
    if not args.input_csv:
        print("\nERROR: Input CSV file path (--input_csv) must be specified, as the default is empty.")
        parser.print_help()
        sys.exit(1)


    # Run the analysis function with parameters from argparse
    analyze_window_uq(
        detailed_results_csv=args.input_csv,
        num_bins=args.num_bins,
        output_binned_csv=args.output_binned_csv # Pass the optional save path
    )