#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr # For adding correlation to scatter plot
import argparse
from typing import List, Dict, Any, NoReturn, Optional, Tuple # Import types

# --- Configuration ---
# Define default paths and parameters (can be overridden by command-line args)

# Default paths to DETAILED per-window results CSVs (from UQ evaluation scripts)
DEFAULT_MCD_DETAIL_CSV: str = './uq_results/mc_dropout/detailed_results_CNN_MCD_Unbalanced.csv' # from -> analyze_MCD_patient_level.py
DEFAULT_DE_DETAIL_CSV: str = './uq_results/deep_ensemble/detailed_results_CNN_DE_Unbalanced.csv' # from -> analyze_DE_patient_level.py

# Default paths to patient SUMMARY CSVs (from patient-level analysis script)
# These typically contain columns: Patient_ID, mean_variance, median_variance,
# mean_entropy, median_entropy, patient_accuracy, num_windows, etc.
DEFAULT_MCD_SUMMARY_CSV: str = './uq_results/mc_dropout/patient_level_analysis/patient_summary_CNN_MCD_Unbalanced.csv' # from -> analyze_MCD_patient_level.py
DEFAULT_DE_SUMMARY_CSV: str = './uq_results/deep_ensemble/patient_level_analysis/patient_summary_CNN_DE_Unbalanced.csv' #from -> analyze_DE_patient_level.py

# Default output directory for saving the generated plots
DEFAULT_PLOT_OUTPUT_DIR: str = './final_thesis_plots' # Choose a directory

# Default number of bins for binned accuracy plot
DEFAULT_NUM_BINS: int = 10

# Default values for plot appearance and saving
PLOT_STYLE: str = 'seaborn-v0_8-whitegrid'
DPI: int = 300 # Dots Per Inch for saved image


# --- Helper Function to Load Data ---
def load_data(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Loads data from a CSV file into a pandas DataFrame. Handles file not found errors.

    Args:
        csv_path: Path to the input CSV file.

    Returns:
        A pandas DataFrame containing the loaded data, or None if the file
        is not found or an error occurs during loading.
    """
    if not os.path.exists(csv_path):
        print(f"ERROR: Input file not found - '{csv_path}'")
        return None
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded '{os.path.basename(csv_path)}', shape: {df.shape}")
        if df.empty:
            print(f"Warning: Loaded DataFrame from '{os.path.basename(csv_path)}' is empty.")
        return df
    except Exception as e:
        print(f"Error loading CSV file '{csv_path}': {e}")
        return None

# --- Main Plot Generation Function ---
def generate_comparison_plots(
        mcd_detail_csv: str,
        de_detail_csv: str,
        mcd_summary_csv: str,
        de_summary_csv: str,
        plot_output_dir: str = DEFAULT_PLOT_OUTPUT_DIR,
        num_bins: int = DEFAULT_NUM_BINS
) -> NoReturn:
    """
    Loads detailed and summary UQ results for MC Dropout and Deep Ensemble
    and generates comparative plots.

    Plots generated include:
    1. Distribution of Mean Predictive Entropy per Patient (Histograms)
    2. Patient Accuracy vs. Mean Predictive Entropy (Scatter Plots with Correlation)
    3. Predictive Entropy Distribution for Correct vs. Incorrect Windows (Box Plots)
    4. Accuracy across Predictive Entropy Bins (Line Plots with Annotations)

    Plots are saved to the specified output directory.

    Args:
        mcd_detail_csv: Path to the detailed per-window MCD results CSV.
        de_detail_csv: Path to the detailed per-window DE results CSV.
        mcd_summary_csv: Path to the patient summary MCD results CSV.
        de_summary_csv: Path to the patient summary DE results CSV.
        plot_output_dir: Directory to save the generated plot files.
        num_bins: Number of bins for the binned accuracy plot.
    """
    print(f"--- Generating Comparison Plots ---")
    print(f"MCD Detailed CSV: {mcd_detail_csv}")
    print(f"DE Detailed CSV: {de_detail_csv}")
    print(f"MCD Summary CSV: {mcd_summary_csv}")
    print(f"DE Summary CSV: {de_summary_csv}")
    print(f"Output Plot Directory: {plot_output_dir}")
    print(f"Number of Bins for Binned Plot: {num_bins}")
    print("="*70)

    # --- Create Output Directory ---
    try:
        os.makedirs(plot_output_dir, exist_ok=True)
        print(f"\nOutput plot directory '{plot_output_dir}' ensured to exist.")
    except Exception as e:
        print(f"Error creating output plot directory '{plot_output_dir}': {e}")
        return # Exit if directory creation fails


    # --- Load Data ---
    print("\nLoading necessary data files...")
    mcd_detail_df = load_data(mcd_detail_csv)
    de_detail_df = load_data(de_detail_csv)
    mcd_summary_df = load_data(mcd_summary_csv)
    de_summary_df = load_data(de_summary_csv)


    # --- Data Preparation: Add 'Correct' column if missing ---
    # This column is essential for correctness-related plots.
    print("\nEnsuring 'Correct' column exists in detailed data...")
    for df, method_name in zip([mcd_detail_df, de_detail_df], ['MC Dropout Detailed', 'Deep Ensemble Detailed']):
        if df is not None and 'Correct' not in df.columns:
            if 'True_Label' in df.columns and 'Predicted_Label' in df.columns:
                df['Correct'] = (df['True_Label'] == df['Predicted_Label'])
                print(f"Added 'Correct' column to {method_name} data.")
            else:
                print(f"Warning: Cannot create 'Correct' column for {method_name}, missing 'True_Label' or 'Predicted_Label'. Some plots may be skipped.")


    # --- Generate Plots ---
    plt.style.use(PLOT_STYLE)


    # Plot 1: Patient Mean Entropy Histograms
    # Requires MCD Summary and DE Summary DataFrames
    if mcd_summary_df is not None and de_summary_df is not None and not mcd_summary_df.empty and not de_summary_df.empty:
        print("\nGenerating Patient Mean Entropy Histograms...")
        # Check for essential columns
        if 'mean_entropy' not in mcd_summary_df.columns or 'mean_entropy' not in de_summary_df.columns:
            print("Skipping Patient Entropy Histograms - Missing 'mean_entropy' column in summary data.")
        else:
            try:
                fig_hist, axes_hist = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

                sns.histplot(data=mcd_summary_df, x='mean_entropy', ax=axes_hist[0], bins=num_bins, kde=True) # Use num_bins
                axes_hist[0].set_title('a) MC Dropout')
                axes_hist[0].set_xlabel('Mean Predictive Entropy per Patient')
                axes_hist[0].set_ylabel('Number of Patients')

                sns.histplot(data=de_summary_df, x='mean_entropy', ax=axes_hist[1], bins=num_bins, kde=True) # Use num_bins
                axes_hist[1].set_title('b) Deep Ensemble')
                axes_hist[1].set_xlabel('Mean Predictive Entropy per Patient')

                fig_hist.suptitle('Distribution of Mean Predictive Entropy Across Patients (Unbalanced Set)')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
                plot_filename = os.path.join(plot_output_dir, 'patient_entropy_histograms_final.png')
                plt.savefig(plot_filename, dpi=DPI)
                print(f"Saved plot: {plot_filename}")
                plt.close(fig_hist) # Close figure
            except Exception as e:
                print(f"Error generating Patient Entropy Histograms: {e}")
                if 'fig_hist' in locals() and fig_hist: plt.close(fig_hist) # Close if created


    # Plot 2: Patient Accuracy vs Entropy Scatter Plot
    # Requires MCD Summary and DE Summary DataFrames
    if mcd_summary_df is not None and de_summary_df is not None and not mcd_summary_df.empty and not de_summary_df.empty:
        print("\nGenerating Patient Accuracy vs Entropy Scatter Plots...")
        # Check for essential columns
        if 'mean_entropy' not in mcd_summary_df.columns or 'patient_accuracy' not in mcd_summary_df.columns or \
                'mean_entropy' not in de_summary_df.columns or 'patient_accuracy' not in de_summary_df.columns:
            print("Skipping Patient Accuracy vs Entropy Scatter Plots - Missing essential columns in summary data.")
        else:
            try:
                fig_scatter, axes_scatter = plt.subplots(1, 2, figsize=(12, 5.5)) # Slightly taller

                # Calculate correlations for adding text annotations
                # Use .dropna() before pearsonr in case of NaNs in the columns
                mcd_corr_df = mcd_summary_df[['mean_entropy', 'patient_accuracy']].dropna()
                de_corr_df = de_summary_df[['mean_entropy', 'patient_accuracy']].dropna()

                mcd_r = pearsonr(mcd_corr_df['mean_entropy'], mcd_corr_df['patient_accuracy'])[0] if len(mcd_corr_df) >= 2 else np.nan
                de_r = pearsonr(de_corr_df['mean_entropy'], de_corr_df['patient_accuracy'])[0] if len(de_corr_df) >= 2 else np.nan


                sns.scatterplot(data=mcd_summary_df, x='mean_entropy', y='patient_accuracy', ax=axes_scatter[0], alpha=0.6)
                axes_scatter[0].set_title('a) MC Dropout (~77% Accuracy Baseline)') # Note: Hardcoded baseline accuracy
                axes_scatter[0].set_xlabel('Mean Predictive Entropy per Patient')
                axes_scatter[0].set_ylabel('Patient Accuracy')
                # Add correlation text if calculated
                if not np.isnan(mcd_r):
                    axes_scatter[0].text(0.95, 0.05, f'Pearson r = {mcd_r:.3f}', transform=axes_scatter[0].transAxes,
                                         fontsize=9, verticalalignment='bottom', horizontalalignment='right')
                axes_scatter[0].grid(True, linestyle='--', alpha=0.6)
                axes_scatter[0].set_ylim(-0.05, 1.05) # Ensure accuracy limits


                sns.scatterplot(data=de_summary_df, x='mean_entropy', y='patient_accuracy', ax=axes_scatter[1], alpha=0.6)
                axes_scatter[1].set_title('b) Deep Ensemble (~88% Accuracy Baseline)') # Note: Hardcoded baseline accuracy
                axes_scatter[1].set_xlabel('Mean Predictive Entropy per Patient')
                axes_scatter[1].set_ylabel('Patient Accuracy')
                # Add correlation text if calculated
                if not np.isnan(de_r):
                    axes_scatter[1].text(0.95, 0.05, f'Pearson r = {de_r:.3f}', transform=axes_scatter[1].transAxes,
                                         fontsize=9, verticalalignment='bottom', horizontalalignment='right')
                axes_scatter[1].grid(True, linestyle='--', alpha=0.6)
                axes_scatter[1].set_ylim(-0.05, 1.05) # Ensure accuracy limits


                fig_scatter.suptitle('Patient Accuracy vs. Mean Predictive Entropy (Unbalanced Set)')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plot_filename = os.path.join(plot_output_dir, 'patient_accuracy_vs_entropy_final.png')
                plt.savefig(plot_filename, dpi=DPI)
                print(f"Saved plot: {plot_filename}")
                plt.close(fig_scatter) # Close figure
            except Exception as e:
                print(f"Error generating Patient Accuracy vs Entropy Scatter Plots: {e}")
                if 'fig_scatter' in locals() and fig_scatter: plt.close(fig_scatter) # Close if created


    # Plot 3: Window Correctness Box Plots
    # Requires MCD Detailed and DE Detailed DataFrames with 'Correct' column
    if mcd_detail_df is not None and de_detail_df is not None and not mcd_detail_df.empty and not de_detail_df.empty:
        print("\nGenerating Window Correctness Box Plots...")
        # Check for essential columns
        if 'Correct' not in mcd_detail_df.columns or 'Predictive_Entropy' not in mcd_detail_df.columns or \
                'Correct' not in de_detail_df.columns or 'Predictive_Entropy' not in de_detail_df.columns:
            print("Skipping Window Correctness Box Plots - Missing essential columns ('Correct' or 'Predictive_Entropy') in detailed data.")
        else:
            try:
                fig_box, axes_box = plt.subplots(1, 2, figsize=(10, 5))

                # Create 'Correct_Label' for plotting
                mcd_detail_df['Correct_Label'] = mcd_detail_df['Correct'].map({False: 'Incorrect', True: 'Correct'})
                de_detail_df['Correct_Label'] = de_detail_df['Correct'].map({False: 'Incorrect', True: 'Correct'})

                # Ensure data exists for both categories before plotting
                if not mcd_detail_df['Correct_Label'].value_counts().empty:
                    sns.boxplot(data=mcd_detail_df, x='Correct_Label', y='Predictive_Entropy', ax=axes_box[0], order=['Incorrect', 'Correct']) # Specify order
                    axes_box[0].set_title('a) MC Dropout')
                    axes_box[0].set_xlabel('Prediction Correct')
                    axes_box[0].set_ylabel('Predictive Entropy')
                else:
                    print("Warning: No data for MC Dropout Box Plot.")


                if not de_detail_df['Correct_Label'].value_counts().empty:
                    sns.boxplot(data=de_detail_df, x='Correct_Label', y='Predictive_Entropy', ax=axes_box[1], order=['Incorrect', 'Correct']) # Specify order
                    axes_box[1].set_title('b) Deep Ensemble')
                    axes_box[1].set_xlabel('Prediction Correct')
                    axes_box[1].set_ylabel('Predictive Entropy')
                else:
                    print("Warning: No data for Deep Ensemble Box Plot.")


                fig_box.suptitle('Predictive Entropy Distribution for Correct vs. Incorrect Windows (Unbalanced Set)')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plot_filename = os.path.join(plot_output_dir, 'window_correctness_boxplots_final.png')
                plt.savefig(plot_filename, dpi=DPI)
                print(f"Saved plot: {plot_filename}")
                plt.close(fig_box) # Close figure
            except Exception as e:
                print(f"Error generating Window Correctness Box Plots: {e}")
                if 'fig_box' in locals() and fig_box: plt.close(fig_box) # Close if created


    # Plot 4: Binned Accuracy Line Plot
    # Requires MCD Detailed and DE Detailed DataFrames with 'Correct' and 'Predictive_Entropy' columns
    if mcd_detail_df is not None and de_detail_df is not None and not mcd_detail_df.empty and not de_detail_df.empty:
        print("\nGenerating Binned Accuracy Plots...")
        # Check for essential columns
        if 'Correct' not in mcd_detail_df.columns or 'Predictive_Entropy' not in mcd_detail_df.columns or \
                'Correct' not in de_detail_df.columns or 'Predictive_Entropy' not in de_detail_df.columns:
            print("Skipping Binned Accuracy Plots - Missing essential columns ('Correct' or 'Predictive_Entropy') in detailed data.")
        elif num_bins <= 0:
            print(f"Skipping Binned Accuracy Plots - num_bins ({num_bins}) must be positive.")
        else:
            try:
                fig_binned, axes_binned = plt.subplots(1, 2, figsize=(14, 6), sharey=True) # Slightly wider for annotations
                metric_to_bin = 'Predictive_Entropy'

                for i, (df, method_name, ax) in enumerate(zip(
                        [mcd_detail_df, de_detail_df],
                        ['MC Dropout', 'Deep Ensemble'],
                        axes_binned)):

                    if df.empty or metric_to_bin not in df.columns or 'Correct' not in df.columns:
                        print(f"Warning: Skipping binned accuracy plot for {method_name} - Data not available or missing columns.")
                        ax.set_title(f'{("a)" if i == 0 else "b)")} {method_name}\n(Data Unavailable)', fontsize=11)
                        ax.set_xlabel('Predictive Entropy Bin', fontsize=10)
                        if i == 0: ax.set_ylabel('Accuracy', fontsize=10)
                        ax.grid(True, linestyle='--', alpha=0.7)
                        ax.set_ylim(0, 1.05) # Set a default y-limit
                        continue # Skip plotting for this method


                    # Calculate bin edges
                    min_metric = df[metric_to_bin].min()
                    max_metric = df[metric_to_bin].max()

                    # Handle case where min == max
                    if min_metric == max_metric:
                        print(f"Warning: All '{metric_to_bin}' values are the same ({min_metric}) for {method_name}. Creating a single bin.")
                        bins = [min_metric, max_metric + 1e-9]
                    else:
                        bins = np.linspace(min_metric, max_metric, num_bins + 1)
                        bins[-1] = bins[-1] + 1e-9 # Ensure max value is included

                    # Assign windows to bins
                    df[f'{metric_to_bin}_Bin_Intervals'] = pd.cut(df[metric_to_bin], bins=bins, include_lowest=True)

                    # Group by bin and calculate accuracy
                    binned_results = df.groupby(f'{metric_to_bin}_Bin_Intervals', observed=False).agg(
                        window_count=('Correct', 'size'),
                        accuracy=('Correct', 'mean')
                    ).reset_index()

                    # Remove bins with no data if desired (optional, depends on desired visualization)
                    # binned_results = binned_results[binned_results['window_count'] > 0].copy()

                    if binned_results.empty:
                        print(f"Warning: No data points found in any bin for {method_name}.")
                        ax.set_title(f'{("a)" if i == 0 else "b)")} {method_name}\n(No Data in Bins)', fontsize=11)
                        ax.set_xlabel('Predictive Entropy Bin', fontsize=10)
                        if i == 0: ax.set_ylabel('Accuracy', fontsize=10)
                        ax.grid(True, linestyle='--', alpha=0.7)
                        ax.set_ylim(0, 1.05)
                        continue # Skip plotting if no binned results


                    # Plot the line
                    line = sns.lineplot(x=range(len(binned_results)), y=binned_results['accuracy'], ax=ax, marker='o', linewidth=2)

                    # Create x-tick labels from bin intervals
                    tick_labels = [f'[{b.left:.2f}-{b.right:.2f})' for b in binned_results[f'{metric_to_bin}_Bin_Intervals']]
                    # Correct the last label if needed
                    if len(tick_labels) > 0 and binned_results[f'{metric_to_bin}_Bin_Intervals'].iloc[-1].right > binned_results[f'{metric_to_bin}_Bin_Intervals'].iloc[-1].left: # Avoid issue if single bin
                        last_interval = binned_results[f'{metric_to_bin}_Bin_Intervals'].iloc[-1]
                        tick_labels[-1] = f'[{last_interval.left:.2f}-{last_interval.right:.2f}]'


                    ax.set_xticks(range(len(binned_results)))
                    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)

                    # --- Annotation for the first bin ---
                    # Ensure there's data in the first bin before annotating
                    if binned_results['window_count'].iloc[0] > 0:
                        accuracy_first_bin = binned_results['accuracy'].iloc[0]
                        count_first_bin = binned_results['window_count'].iloc[0]

                        # Add a distinct marker for the first point
                        ax.plot(0, accuracy_first_bin, marker='*', markersize=12,
                                color=line.get_lines()[0].get_color(), markeredgecolor='black', zorder=5)

                        # Add text annotation
                        annotation_text = f"Acc: {accuracy_first_bin:.3f}\nN: {count_first_bin:,}" # Format N with comma
                        # Adjust annotation position to avoid overlap
                        text_x = 0.05 # Position relative to bin (first bin is index 0)
                        text_y = accuracy_first_bin - 0.08 # Position relative to accuracy
                        # Simple check to keep annotation within bounds
                        if text_y < 0.5: text_y = 0.5

                        ax.text(text_x, text_y, annotation_text,
                                transform=ax.transData, # Use data coordinates
                                ha='left', va='top', fontsize=9,
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))


                    # Add overall accuracy of the UQ run to the title
                    overall_uq_run_accuracy = df['Correct'].mean() if 'Correct' in df.columns else np.nan
                    title_suffix = f'\n(Overall Acc: {overall_uq_run_accuracy:.2f})' if not np.isnan(overall_uq_run_accuracy) else ''
                    ax.set_title(f'{("a)" if i == 0 else "b)")} {method_name}{title_suffix}', fontsize=11)

                    ax.set_xlabel('Predictive Entropy Bin', fontsize=10)
                    if i == 0:
                        ax.set_ylabel('Accuracy', fontsize=10)
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.set_ylim(0, 1.05) # Ensure y-axis is suitable for accuracy (0 to 1, slightly above 1)
                    ax.tick_params(axis='both', which='major', labelsize=9)


                fig_binned.suptitle('Accuracy across Predictive Entropy Bins (Unbalanced Set)', fontsize=14)
                plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust rect for suptitle
                plot_filename = os.path.join(plot_output_dir, 'binned_accuracy_plot_final_annotated.png')
                plt.savefig(plot_filename, dpi=DPI)
                print(f"Saved plot: {plot_filename}")
                plt.close(fig_binned) # Close figure
            except Exception as e:
                print(f"Error generating Binned Accuracy Plots: {e}")
                if 'fig_binned' in locals() and fig_binned: plt.close(fig_binned) # Close if created


    print("\nPlot generation complete.")
    print(f"Plots saved in directory: {plot_output_dir}")
    print("="*70)


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Comparison Plot Generation Script ---")

    # Setup argparse for command-line configuration
    parser = argparse.ArgumentParser(
        description="Generate comparative plots for MC Dropout and Deep Ensemble UQ results."
    )
    parser.add_argument(
        "--mcd_detail_csv",
        type=str,
        default=DEFAULT_MCD_DETAIL_CSV,
        help=f"Path to the detailed per-window MC Dropout results CSV (default: '{DEFAULT_MCD_DETAIL_CSV}')."
    )
    parser.add_argument(
        "--de_detail_csv",
        type=str,
        default=DEFAULT_DE_DETAIL_CSV,
        help=f"Path to the detailed per-window Deep Ensemble results CSV (default: '{DEFAULT_DE_DETAIL_CSV}')."
    )
    parser.add_argument(
        "--mcd_summary_csv",
        type=str,
        default=DEFAULT_MCD_SUMMARY_CSV,
        help=f"Path to the patient summary MC Dropout results CSV (default: '{DEFAULT_MCD_SUMMARY_CSV}')."
    )
    parser.add_argument(
        "--de_summary_csv",
        type=str,
        default=DEFAULT_DE_SUMMARY_CSV,
        help=f"Path to the patient summary Deep Ensemble results CSV (default: '{DEFAULT_DE_SUMMARY_CSV}')."
    )
    parser.add_argument(
        "--plot_output_dir",
        type=str,
        default=DEFAULT_PLOT_OUTPUT_DIR,
        help=f"Directory to save the generated plot files (default: '{DEFAULT_PLOT_OUTPUT_DIR}')."
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=DEFAULT_NUM_BINS,
        help=f"Number of bins for the binned accuracy plot (default: {DEFAULT_NUM_BINS})."
    )

    args = parser.parse_args()

    # Validate that the user has provided necessary paths (unless they intend for some plots to be skipped)
    # It's hard to validate definitively here, so we just check if defaults are empty and warn.
    if not args.mcd_detail_csv: print(f"Warning: Default MCD detailed CSV path is empty ('{DEFAULT_MCD_DETAIL_CSV}'). Plots requiring this data may be skipped.")
    if not args.de_detail_csv: print(f"Warning: Default DE detailed CSV path is empty ('{DEFAULT_DE_DETAIL_CSV}'). Plots requiring this data may be skipped.")
    if not args.mcd_summary_csv: print(f"Warning: Default MCD summary CSV path is empty ('{DEFAULT_MCD_SUMMARY_CSV}'). Plots requiring this data may be skipped.")
    if not args.de_summary_csv: print(f"Warning: Default DE summary CSV path is empty ('{DEFAULT_DE_SUMMARY_CSV}'). Plots requiring this data may be skipped.")


    # Run the plot generation function with parameters from argparse
    generate_comparison_plots(
        mcd_detail_csv=args.mcd_detail_csv,
        de_detail_csv=args.de_detail_csv,
        mcd_summary_csv=args.mcd_summary_csv,
        de_summary_csv=args.de_summary_csv,
        plot_output_dir=args.plot_output_dir,
        num_bins=args.num_bins
    )

print("--- Comparison Plot Generation Script Finished ---")