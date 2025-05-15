import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pearsonr # For adding correlation to scatter plot

# --- Configuration ---

# Paths to DETAILED per-window results CSVs (change filenames if needed)
MCD_DETAIL_CSV = './detail_patient_MCD.csv'
DE_DETAIL_CSV = './detail_patient_DE.csv'

# Paths to patient SUMMARY CSVs
MCD_SUMMARY_CSV = './patient_level_uq_analysis_MCD/patient_summary_metrics_MCD.csv'
DE_SUMMARY_CSV = './patient_level_uq_analysis_DE/patient_summary_metrics_DE.csv'

# Output directory for plots
PLOT_OUTPUT_DIR = './final_thesis_plots' # Choose a directory

# --- Create Output Directory ---
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
print(f"Plots will be saved to: {PLOT_OUTPUT_DIR}")

# --- Helper Function to Load Data ---
def load_data(csv_path):
    """Loads CSV, handles file not found."""
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found - {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {csv_path}, shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None

# --- Load Data ---
mcd_detail_df = load_data(MCD_DETAIL_CSV)
de_detail_df = load_data(DE_DETAIL_CSV)
mcd_summary_df = load_data(MCD_SUMMARY_CSV)
de_summary_df = load_data(DE_SUMMARY_CSV)

# Add 'Correct' column if missing (should exist based on previous scripts)
for df in [mcd_detail_df, de_detail_df]:
    if df is not None and 'Correct' not in df.columns:
        if 'True_Label' in df.columns and 'Predicted_Label' in df.columns:
            df['Correct'] = (df['True_Label'] == df['Predicted_Label'])
            print("Added 'Correct' column.")
        else:
            print("Warning: Cannot create 'Correct' column, missing True/Predicted labels.")


# --- Generate Plots ---

# Plot 1: Patient Entropy Histograms (Fig 4.X / fig:patient_entropy_histograms_final)
if mcd_summary_df is not None and de_summary_df is not None:
    print("Generating Patient Entropy Histograms...")
    fig_hist, axes_hist = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    sns.histplot(data=mcd_summary_df, x='mean_entropy', ax=axes_hist[0], bins=20, kde=True)
    axes_hist[0].set_title('a) MC Dropout')
    axes_hist[0].set_xlabel('Mean Predictive Entropy per Patient')
    axes_hist[0].set_ylabel('Number of Patients')

    sns.histplot(data=de_summary_df, x='mean_entropy', ax=axes_hist[1], bins=20, kde=True)
    axes_hist[1].set_title('b) Deep Ensemble')
    axes_hist[1].set_xlabel('Mean Predictive Entropy per Patient')

    fig_hist.suptitle('Distribution of Mean Predictive Entropy Across Patients (Unbalanced Set)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plot_filename = os.path.join(PLOT_OUTPUT_DIR, 'patient_entropy_histograms_final.png')
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved: {plot_filename}")
    plt.close(fig_hist)

# Plot 2: Patient Accuracy vs Entropy Scatter Plot (Fig 4.Y / fig:patient_accuracy_vs_entropy_final)
if mcd_summary_df is not None and de_summary_df is not None:
    print("Generating Patient Accuracy vs Entropy Scatter Plots...")
    fig_scatter, axes_scatter = plt.subplots(1, 2, figsize=(12, 5.5)) # Slightly taller

    # Calculate correlations again for adding text
    mcd_r, _ = pearsonr(mcd_summary_df['mean_entropy'].dropna(), mcd_summary_df['patient_accuracy'].dropna())
    de_r, _ = pearsonr(de_summary_df['mean_entropy'].dropna(), de_summary_df['patient_accuracy'].dropna())

    sns.scatterplot(data=mcd_summary_df, x='mean_entropy', y='patient_accuracy', ax=axes_scatter[0], alpha=0.6)
    axes_scatter[0].set_title('a) MC Dropout (~77% Accuracy Baseline)')
    axes_scatter[0].set_xlabel('Mean Predictive Entropy per Patient')
    axes_scatter[0].set_ylabel('Patient Accuracy')
    axes_scatter[0].text(0.95, 0.05, f'Pearson r = {mcd_r:.3f}', transform=axes_scatter[0].transAxes,
                         fontsize=9, verticalalignment='bottom', horizontalalignment='right')
    axes_scatter[0].grid(True, linestyle='--', alpha=0.6)


    sns.scatterplot(data=de_summary_df, x='mean_entropy', y='patient_accuracy', ax=axes_scatter[1], alpha=0.6)
    axes_scatter[1].set_title('b) Deep Ensemble (~88% Accuracy Baseline)')
    axes_scatter[1].set_xlabel('Mean Predictive Entropy per Patient')
    axes_scatter[1].set_ylabel('Patient Accuracy')
    axes_scatter[1].text(0.95, 0.05, f'Pearson r = {de_r:.3f}', transform=axes_scatter[1].transAxes,
                         fontsize=9, verticalalignment='bottom', horizontalalignment='right')
    axes_scatter[1].grid(True, linestyle='--', alpha=0.6)

    fig_scatter.suptitle('Patient Accuracy vs. Mean Predictive Entropy (Unbalanced Set)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(PLOT_OUTPUT_DIR, 'patient_accuracy_vs_entropy_final.png')
    plt.savefig(plot_filename, dpi=300)
    print(f"Saved: {plot_filename}")
    plt.close(fig_scatter)


# Plot 3: Window Correctness Box Plots (Fig 4.Z / fig:window_correctness_boxplots_final)
if mcd_detail_df is not None and de_detail_df is not None:
    print("Generating Window Correctness Box Plots...")
    # Ensure 'Correct' column exists
    if 'Correct' not in mcd_detail_df.columns or 'Correct' not in de_detail_df.columns:
        print("Skipping Box Plots - Missing 'Correct' column in detailed data.")
    else:
        fig_box, axes_box = plt.subplots(1, 2, figsize=(10, 5))
        mcd_detail_df['Correct_Label'] = mcd_detail_df['Correct'].map({False: 'Incorrect', True: 'Correct'})
        de_detail_df['Correct_Label'] = de_detail_df['Correct'].map({False: 'Incorrect', True: 'Correct'})

        sns.boxplot(data=mcd_detail_df, x='Correct_Label', y='Predictive_Entropy', ax=axes_box[0])
        axes_box[0].set_title('a) MC Dropout')
        axes_box[0].set_xlabel('Prediction Correct')
        axes_box[0].set_ylabel('Predictive Entropy')
        # axes_box[0].set_xticklabels(['Incorrect', 'Correct'])

        sns.boxplot(data=de_detail_df, x='Correct_Label', y='Predictive_Entropy', ax=axes_box[1])
        axes_box[1].set_title('b) Deep Ensemble')
        axes_box[1].set_xlabel('Prediction Correct')
        axes_box[1].set_ylabel('Predictive Entropy')
        #axes_box[1].set_xticklabels(['Incorrect', 'Correct'])

        fig_box.suptitle('Predictive Entropy Distribution for Correct vs. Incorrect Windows (Unbalanced Set)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = os.path.join(PLOT_OUTPUT_DIR, 'window_correctness_boxplots_final.png')
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved: {plot_filename}")
        plt.close(fig_box)


# Plot 4: Binned Accuracy Line Plot (Fig 4.AA / fig:binned_accuracy_plot_final)
if mcd_detail_df is not None and de_detail_df is not None:
    print("Generating Binned Accuracy Plots...")
    if 'Correct' not in mcd_detail_df.columns or 'Correct' not in de_detail_df.columns:
        print("Skipping Binned Accuracy Plots - Missing 'Correct' column in detailed data.")
    else:
        fig_binned, axes_binned = plt.subplots(1, 2, figsize=(14, 6), sharey=True) # Slightly wider for annotations
        NUM_BINS = 10
        metric_to_bin = 'Predictive_Entropy'

        for i, (df, method_name, ax) in enumerate(zip(
                [mcd_detail_df, de_detail_df],
                ['MC Dropout', 'Deep Ensemble'],
                axes_binned)):

            min_metric = df[metric_to_bin].min()
            max_metric = df[metric_to_bin].max()
            bins = np.linspace(min_metric, max_metric, NUM_BINS + 1)
            bins[-1] = bins[-1] + 1e-9 # Ensure max value is included

            df[f'{metric_to_bin}_Bin_Intervals'] = pd.cut(df[metric_to_bin], bins=bins, include_lowest=True)

            binned_results = df.groupby(f'{metric_to_bin}_Bin_Intervals', observed=False).agg(
                window_count=('Correct', 'size'),
                accuracy=('Correct', 'mean')
            ).reset_index() # Reset index to make plotting easier if bin intervals are used for ticks later

            # Plot the line
            line = sns.lineplot(x=range(len(binned_results)), y=binned_results['accuracy'], ax=ax, marker='o', linewidth=2)

            # Create x-tick labels
            tick_labels = [f'{b.left:.2f}-{b.right:.2f}' for b in binned_results[f'{metric_to_bin}_Bin_Intervals']]
            ax.set_xticks(range(len(binned_results)))
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)

            # --- Annotation for the first bin ---
            accuracy_first_bin = binned_results['accuracy'].iloc[0]
            count_first_bin = binned_results['window_count'].iloc[0]

            # Add a distinct marker for the first point
            ax.plot(0, accuracy_first_bin, marker='*', markersize=12,
                    color=line.get_lines()[0].get_color(), markeredgecolor='black', zorder=5)

            # Add text annotation
            annotation_text = f"Acc: {accuracy_first_bin:.3f}\nN: {count_first_bin:,}" # Format N with comma
            ax.text(0.05, accuracy_first_bin - 0.08, annotation_text, # Adjust x, y for positioning
                    transform=ax.transData, # Use data coordinates
                    ha='left', va='top', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            ax.set_title(f'{("a)" if i == 0 else "b)")} {method_name}\n(Overall UQ run Acc: {df["Correct"].mean():.2f})', fontsize=11) # Add UQ run acc
            ax.set_xlabel('Predictive Entropy Bin', fontsize=10)
            if i == 0:
                ax.set_ylabel('Accuracy', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_ylim(0.5, 1.05) # Zoom in slightly if accuracies are mostly high, but ensure lowest is visible
            ax.tick_params(axis='both', which='major', labelsize=9)

        fig_binned.suptitle('Accuracy across Predictive Entropy Bins (Unbalanced Set)', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust rect for suptitle
        plot_filename = os.path.join(PLOT_OUTPUT_DIR, 'binned_accuracy_plot_final_annotated.png')
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved: {plot_filename}")
        plt.close(fig_binned)

print("\nPlot generation complete.")