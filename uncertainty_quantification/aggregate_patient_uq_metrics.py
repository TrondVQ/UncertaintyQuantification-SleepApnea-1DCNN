#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import argparse
from typing import List, NoReturn, Dict, Any, Tuple # Import types

# --- Configuration ---
# Define default paths and parameters (can be overridden by command-line args)

# Default path to the input CSV file containing detailed window-level UQ results -> from either analyze_DE_patient_level.py or analyze_MCD_patient_level.py
DEFAULT_INPUT_CSV_FILE: str = ""

# Default directory to save the aggregated patient-level results
DEFAULT_OUTPUT_DIR: str = ""

# Default base filename for the aggregated patient-level summary CSV
DEFAULT_OUTPUT_SUMMARY_FILENAME_BASE: str = "patient_summary.csv" # A generic base name

# Number of example patients to display for high/low uncertainty
NUM_EXAMPLE_PATIENTS: int = 5

# Define required column names in the input CSV
REQUIRED_COLUMNS: List[str] = [
    'Patient_ID',        # Identifier for each patient
    'True_Label',        # The true binary label for the window (e.g., 0 or 1)
    'Predicted_Label',   # The model's predicted binary label for the window (e.g., 0 or 1)
    'Predictive_Variance', # Window-level predictive variance
    'Predictive_Entropy' # Window-level predictive entropy
]


# --- Main Analysis Function ---

def analyze_patient_uq(
        input_csv_file: str,
        output_dir: str,
        summary_filename_base: str = DEFAULT_OUTPUT_SUMMARY_FILENAME_BASE,
        num_example_patients: int = NUM_EXAMPLE_PATIENTS
) -> NoReturn:
    """
    Analyzes patient-level uncertainty quantification (UQ) and performance metrics.

    Loads window-level UQ results, aggregates them to the patient level,
    calculates summary statistics per patient, identifies example patients
    with high/low uncertainty, compares these groups, and saves the
    aggregated patient summary to a CSV file.

    Args:
        input_csv_file: Path to the input CSV file with window-level UQ results.
        output_dir: Directory where the aggregated patient-level results will be saved.
        summary_filename_base: The base filename for the output patient summary CSV.
                               The file will be saved as os.path.join(output_dir, summary_filename_base).
        num_example_patients: The number of example patients to display for high and low uncertainty groups.
    """
    print(f"--- Starting Patient-Level UQ Analysis ---")
    print(f"Input CSV File: {input_csv_file}")
    print(f"Output Directory: {output_dir}")
    print(f"Summary Filename Base: {summary_filename_base}")
    print(f"Number of Example Patients to Display: {num_example_patients}")
    print("="*70)

    # Construct the full path for the output summary file
    summary_file_path = os.path.join(output_dir, summary_filename_base)
    print(f"Aggregated patient summary will be saved to: {summary_file_path}")

    # --- Create Output Directory ---
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory '{output_dir}' ensured to exist.")
    except Exception as e:
        print(f"Error creating output directory '{output_dir}': {e}")
        return # Exit if directory creation fails


    # --- Load your detailed UQ results ---
    try:
        print("\nLoading detailed window-level UQ results...")
        uq_results_df = pd.read_csv(input_csv_file)
        print(f"Successfully loaded data with shape: {uq_results_df.shape}")
    except FileNotFoundError:
        print(f"ERROR: Input CSV file not found at '{input_csv_file}'.")
        print("Please ensure the file exists or provide the correct path using the --input_csv argument.")
        return # Exit if file not found
    except Exception as e:
        print(f"Error loading CSV file '{input_csv_file}': {e}")
        return # Exit on other loading errors


    # --- Data Checks: Verify required columns ---
    print("\nPerforming data checks for required columns...")
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in uq_results_df.columns]
    if missing_cols:
        print(f"ERROR: Missing one or more required columns in the input CSV: {missing_cols}")
        print(f"Required columns are: {REQUIRED_COLUMNS}")
        return # Exit if essential columns are missing
    print("All required columns are present.")

    if uq_results_df.empty:
        print("Warning: Input DataFrame is empty. Cannot perform analysis.")
        return # Exit if dataframe is empty


    # --- Patient-Level Aggregation ---
    print("\n--- Performing Patient-Level Aggregation ---")

    # Calculate if each window prediction was correct
    uq_results_df['Correct'] = (uq_results_df['True_Label'] == uq_results_df['Predicted_Label'])

    # Aggregate metrics per patient
    # Note: std for single-window patients will be NaN, handled below
    try:
        patient_summary = uq_results_df.groupby('Patient_ID').agg(
            mean_variance=('Predictive_Variance', 'mean'),
            median_variance=('Predictive_Variance', 'median'),
            std_variance=('Predictive_Variance', 'std'),
            mean_entropy=('Predictive_Entropy', 'mean'),
            median_entropy=('Predictive_Entropy', 'median'),
            std_entropy=('Predictive_Entropy', 'std'),
            patient_accuracy=('Correct', 'mean'), # Mean of 'Correct' gives patient-level accuracy
            num_windows=('Patient_ID', 'size'),  # Count number of windows per patient
        ).reset_index() # Convert groupby output back to DataFrame

        # Replace NaN std values with 0 for patients with only one window
        # Use .loc to avoid SettingWithCopyWarning
        patient_summary.loc[patient_summary['num_windows'] == 1, 'std_variance'] = 0
        patient_summary.loc[patient_summary['num_windows'] == 1, 'std_entropy'] = 0

        print(f"\nAggregation complete. Summary for {len(patient_summary)} unique patients created.")

    except Exception as e:
        print(f"Error during patient-level aggregation: {e}")
        return # Exit on aggregation error


    print("\nPatient Summary DataFrame Head:")
    # Use .to_string() to ensure entire head is printed without truncation
    print(patient_summary.head().to_string())
    print(f"\nTotal unique patients summarized: {len(patient_summary)}")

    # --- Save Aggregated Patient Summary Data ---
    try:
        print(f"\nSaving aggregated patient summary to '{summary_file_path}'...")
        patient_summary.to_csv(summary_file_path, index=False)
        print("Patient summary metrics saved successfully.")
    except Exception as e:
        print(f"ERROR saving patient summary CSV to '{summary_file_path}': {e}")


    # --- Textual Summary of Patient-Level Results ---
    print("\n--- Textual Summary of Patient-Level Distributions (Across all patients) ---")

    # Describe key metrics across ALL patients in the summary
    # Select the columns relevant for the overall description
    overall_summary_cols = ['mean_entropy', 'mean_variance', 'std_entropy', 'std_variance', 'patient_accuracy', 'num_windows']
    print("\nOverall Patient Statistics:")
    if not patient_summary.empty:
        print(patient_summary[overall_summary_cols].describe().to_string())
    else:
        print("No patient summary data available to describe.")


    # --- Identify and Display Example Patients (High/Low Uncertainty) ---
    print(f"\n--- Example Patients (High vs Low Mean Entropy) ---")

    if len(patient_summary) >= num_example_patients * 2: # Ensure enough patients for both lists
        # Sort by mean entropy to find highest and lowest
        patient_summary_sorted = patient_summary.sort_values(by='mean_entropy', ascending=False)

        # Select the top N patients for highest entropy and bottom N for lowest
        high_uncertainty_patients_df = patient_summary_sorted.head(num_example_patients).copy() # Use .copy()
        low_uncertainty_patients_df = patient_summary_sorted.tail(num_example_patients).copy() # Use .copy()

        # Define columns to display for example patients
        example_cols_display = ['Patient_ID', 'mean_entropy', 'mean_variance', 'patient_accuracy', 'num_windows']

        print(f"\nTop {num_example_patients} Patients with HIGHEST Mean Entropy:")
        print(high_uncertainty_patients_df[example_cols_display].to_string(index=False)) # Use index=False for cleaner output

        print(f"\nTop {num_example_patients} Patients with LOWEST Mean Entropy:")
        print(low_uncertainty_patients_df[example_cols_display].to_string(index=False)) # Use index=False for cleaner output

        # --- Compare High vs Low Uncertainty Groups ---
        print("\n--- Statistics for High vs Low Mean Entropy Patient Groups ---")
        compare_cols = ['mean_entropy', 'mean_variance', 'std_entropy', 'std_variance', 'patient_accuracy', 'num_windows']

        print("\nStatistics for HIGHEST Mean Entropy Group:")
        print(high_uncertainty_patients_df[compare_cols].describe().to_string())

        print("\nStatistics for LOWEST Mean Entropy Group:")
        print(low_uncertainty_patients_df[compare_cols].describe().to_string())

    else:
        print(f"\nNot enough patients ({len(patient_summary)}) to display {num_example_patients} examples each for high and low uncertainty.")


    print("\n--- Patient-Level UQ Analysis Complete ---")
    print(f"Aggregated patient summary saved to: {summary_file_path}")
    print(f"Detailed console output provided above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze patient-level uncertainty quantification (UQ) and performance metrics."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=DEFAULT_INPUT_CSV_FILE,
        help=f"Path to the input CSV file with window-level UQ results (default: '{DEFAULT_INPUT_CSV_FILE}'). "
             "E.g., ./detail_patient_MCD.csv or ./detail_patient_DE.csv."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save the aggregated patient-level results (default: '{DEFAULT_OUTPUT_DIR}'). "
             "E.g., ./patient_level_uq_analysis_MCD or ./patient_level_uq_analysis_DE."
    )
    parser.add_argument(
        "--summary_filename",
        type=str,
        default=DEFAULT_OUTPUT_SUMMARY_FILENAME_BASE,
        help=f"Base filename for the output patient summary CSV (default: '{DEFAULT_OUTPUT_SUMMARY_FILENAME_BASE}'). "
             "The file will be saved inside the output_dir."
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=NUM_EXAMPLE_PATIENTS,
        help=f"Number of example patients to display for high/low uncertainty groups (default: {NUM_EXAMPLE_PATIENTS})."
    )

    args = parser.parse_args()

    # Validate that the user has provided necessary paths if defaults are empty
    if not args.input_csv:
        print("\nERROR: Input CSV file path (--input_csv) must be specified, as the default is empty.")
        parser.print_help()
        exit()
    if not args.output_dir:
        print("\nERROR: Output directory path (--output_dir) must be specified, as the default is empty.")
        parser.print_help()
        exit()


    # Run the analysis function with parameters from argparse
    analyze_patient_uq(
        input_csv_file=args.input_csv,
        output_dir=args.output_dir,
        summary_filename_base=args.summary_filename,
        num_example_patients=args.num_examples
    )