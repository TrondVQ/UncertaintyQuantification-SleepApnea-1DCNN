#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import pandas as pd
import io
import numpy as np
import argparse
import os
from typing import List, Dict, Any, Tuple, NoReturn, Optional # Import types

# --- Configuration ---

# Default path to the CSV file containing convergence data
# This CSV should have columns like: 'N', 'Variance_Unbalanced', 'Variance_Balanced'.
# 'N' represents the number of forward passes (for MCD) or ensemble members (for DE).
# These values need to be obtained from the UQ method's output.
# ** USER MUST CONFIGURE THIS PATH OR PROVIDE VIA COMMAND-LINE ARGUMENTS **
DEFAULT_CONVERGENCE_DATA_CSV: str = ""

# Define default output plot filename
DEFAULT_OUTPUT_PLOT_FILENAME: str = "variance_convergence_plot.png"

# Default values for plot appearance
PLOT_STYLE: str = 'seaborn-v0_8-whitegrid'
FIGSIZE: Tuple[int, int] = (8, 5)
DPI: int = 300 # Dots Per Inch for saved image

# --- Main Plotting Function ---

def plot_variance_convergence(
        convergence_data_csv: str,
        output_plot_filename: str = DEFAULT_OUTPUT_PLOT_FILENAME,
        method: str = 'mcd' # 'mcd' or 'de' to adjust labels/title
) -> NoReturn:
    """
    Loads convergence data from a CSV and plots the trend of Overall Mean Variance
    against the number of passes (for MCD) or ensemble members (for DE).

    Args:
        convergence_data_csv: Path to the CSV file containing the convergence data.
                              Expected columns: 'N', 'Variance_Unbalanced', 'Variance_Balanced'.
                              'N' should be the number of passes/members.
        output_plot_filename: Path (including filename) to save the generated plot.
        method: Specifies the method ('mcd' or 'de') to adjust plot title and
                x-axis label appropriately. Defaults to 'mcd'.
    """
    print(f"--- Generating Variance Convergence Plot ---")
    print(f"Input Data CSV: {convergence_data_csv}")
    print(f"Output Plot File: {output_plot_filename}")
    print(f"Method (for labels): {method.upper()}")
    print("="*70)

    # 1. Load Convergence Data
    try:
        print("\nLoading convergence data...")
        df = pd.read_csv(convergence_data_csv)
        print(f"Loaded data shape: {df.shape}")
        # Data Check: Ensure required columns are present
        required_cols = ['N', 'Variance_Unbalanced', 'Variance_Balanced']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Input CSV missing required columns: {missing_cols}")
            print(f"Expected columns: {required_cols}")
            return # Exit if essential columns are missing
        if df.empty:
            print("Warning: Input DataFrame is empty. Cannot generate plot.")
            return # Exit if empty DataFrame

    except FileNotFoundError:
        print(f"ERROR: Input CSV file not found at '{convergence_data_csv}'.")
        print("Please ensure the file exists or provide the correct path using the --input_csv argument.")
        return # Exit if file not found
    except Exception as e:
        print(f"Error loading CSV file '{convergence_data_csv}': {e}")
        return # Exit on other loading errors

    # --- Adjust Plot Labels and Title Based on Method ---
    if method.lower() == 'mcd':
        plot_title = 'MC Dropout: Overall Mean Variance Convergence'
        xlabel = 'Number of Forward Passes (N)'
    elif method.lower() == 'de':
        plot_title = 'Deep Ensemble: Overall Mean Variance Convergence'
        xlabel = 'Number of Ensemble Members'
    else:
        print(f"Warning: Unknown method '{method}'. Using generic labels.")
        plot_title = 'Overall Mean Variance Convergence'
        xlabel = 'N'


    # 2. Plotting
    print("\nGenerating plot...")
    try:
        plt.style.use(PLOT_STYLE)
        fig, ax = plt.subplots(figsize=FIGSIZE)

        # Plot Variance for Unbalanced dataset
        ax.plot(df['N'], df['Variance_Unbalanced'], marker='^', linestyle='-', color='forestgreen', label='Variance (Unbalanced)')

        # Plot Variance for Balanced dataset
        ax.plot(df['N'], df['Variance_Balanced'], marker='v', linestyle='--', color='firebrick', label='Variance (Balanced)')

        # 3. Formatting the plot
        ax.set_title(plot_title, fontsize=13, pad=15)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Overall Mean Predictive Variance', fontsize=11) # Use full metric name
        ax.legend(fontsize=9)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Set x-axis ticks to match the actual N values for clarity
        ax.set_xticks(df['N'])
        ax.tick_params(axis='both', which='major', labelsize=9)

        # Ensure y-axis starts at 0 if appropriate for variance
        ax.set_ylim(bottom=0)

        plt.tight_layout() # Adjust layout

        # 4. Save the Plot
        print(f"\nSaving plot to: {output_plot_filename}")
        # Ensure the output directory exists before saving
        output_dir = os.path.dirname(output_plot_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory for plot: {output_dir}")

        plt.savefig(output_plot_filename, dpi=DPI, bbox_inches='tight') # Thesis quality DPI
        print("Plot saved successfully.")

        # plt.show() # Uncomment if you also want to display the plot interactively


    except Exception as e:
        print(f"ERROR generating or saving plot: {e}")
        # Ensure plot figure is closed in case of error
        plt.close(fig)
        return # Exit on plotting error

    # Close the figure to free memory
    plt.close(fig)
    print("\n--- Plotting Complete ---")
    print("="*70)


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot the convergence of Overall Mean Predictive Variance with increasing passes/members for UQ methods."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=DEFAULT_CONVERGENCE_DATA_CSV,
        help=f"Path to the CSV file containing convergence data (default: '{DEFAULT_CONVERGENCE_DATA_CSV}'). "
             "Expected columns: 'N', 'Variance_Unbalanced', 'Variance_Balanced'."
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default=DEFAULT_OUTPUT_PLOT_FILENAME,
        help=f"Path (including filename) to save the generated plot (default: '{DEFAULT_OUTPUT_PLOT_FILENAME}')."
    )
    parser.add_argument(
        "--method",
        type=str,
        default='mcd',
        choices=['mcd', 'de'],
        help="Specify the UQ method ('mcd' for MC Dropout or 'de' for Deep Ensemble) "
             "to adjust plot labels and title appropriately (default: 'mcd')."
    )

    args = parser.parse_args()

    # Validate that the user has provided an input CSV path if the default is empty
    if not args.input_csv:
        print("\nERROR: Input CSV file path (--input_csv) must be specified, as the default is empty.")
        parser.print_help()
        sys.exit(1)


    # Run the plotting function with parameters from argparse
    plot_variance_convergence(
        convergence_data_csv=args.input_csv,
        output_plot_filename=args.output_plot,
        method=args.method
    )