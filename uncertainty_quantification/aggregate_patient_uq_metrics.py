import pandas as pd
import numpy as np
import os

# This script aggregates patient-level uncertainty quantification (UQ) metrics

INPUT_CSV_FILE = './detail_patient_MCD.csv'
OUTPUT_DIR = './patient_level_uq_analysis_MCD'
SUMMARY_FILE = os.path.join(OUTPUT_DIR, 'patient_summary_metrics_MCD.csv')
# --- Create Output Directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output will be saved to: {OUTPUT_DIR}")

# --- Load your detailed UQ results ---
try:
    uq_results_df = pd.read_csv(INPUT_CSV_FILE)
    print(f"Loaded detailed UQ results from: {INPUT_CSV_FILE}")
except FileNotFoundError:
    print(f"ERROR: '{INPUT_CSV_FILE}' not found.")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- Data Checks ---
required_cols = ['Patient_ID', 'True_Label', 'Predicted_Label',
                 'Predictive_Variance', 'Predictive_Entropy']
if not all(col in uq_results_df.columns for col in required_cols):
    print(f"ERROR: Missing one or more required columns: {required_cols}")
    exit()

# --- Patient-Level Aggregation ---
print("\n--- Performing Patient-Level Aggregation ---")
uq_results_df['Correct'] = (uq_results_df['True_Label'] == uq_results_df['Predicted_Label'])
patient_summary = uq_results_df.groupby('Patient_ID').agg(
    mean_variance=('Predictive_Variance', 'mean'),
    median_variance=('Predictive_Variance', 'median'),
    std_variance=('Predictive_Variance', 'std'),
    mean_entropy=('Predictive_Entropy', 'mean'),
    median_entropy=('Predictive_Entropy', 'median'),
    std_entropy=('Predictive_Entropy', 'std'),
    patient_accuracy=('Correct', 'mean'),
    num_windows=('Patient_ID', 'size'),
).reset_index()
patient_summary['std_variance'] = patient_summary.apply(lambda row: row['std_variance'] if row['num_windows'] > 1 else 0, axis=1)
patient_summary['std_entropy'] = patient_summary.apply(lambda row: row['std_entropy'] if row['num_windows'] > 1 else 0, axis=1)

print("\nPatient Summary DataFrame Head:")
print(patient_summary.head().to_string())
print(f"\nNumber of unique patients in summary: {len(patient_summary)}")

# --- Save Aggregated Patient Summary Data ---
try:
    patient_summary.to_csv(SUMMARY_FILE, index=False)
    print(f"\nSaved patient summary metrics to: {SUMMARY_FILE}")
except Exception as e:
    print(f"ERROR saving patient summary CSV: {e}")

# --- Textual Summary of Patient-Level Results ---
print("\n--- Textual Summary of Patient-Level Distributions ---")

# Describe key metrics across ALL patients
print("\nOverall Patient Statistics:")
print(patient_summary[['mean_entropy', 'mean_variance', 'std_entropy', 'std_variance', 'patient_accuracy']].describe().to_string())

# --- Identify Example Patients (High/Low Uncertainty) ---
patient_summary_sorted = patient_summary.sort_values(by='mean_entropy', ascending=False)
N_examples = 5
high_uncertainty_patients_df = patient_summary_sorted.head(N_examples)
low_uncertainty_patients_df = patient_summary_sorted.tail(N_examples)

print("\n--- Example Patients (Same as before) ---")
print(f"\nTop {N_examples} Patients with HIGHEST Mean Entropy:")
print(high_uncertainty_patients_df[['Patient_ID', 'mean_entropy', 'mean_variance', 'patient_accuracy', 'num_windows']].to_string())
print(f"\nTop {N_examples} Patients with LOWEST Mean Entropy:")
print(low_uncertainty_patients_df[['Patient_ID', 'mean_entropy', 'mean_variance', 'patient_accuracy', 'num_windows']].to_string())

# --- Compare High vs Low Uncertainty Groups ---
print("\n--- Comparison of High vs Low Uncertainty Patient Groups ---")
print("\nStatistics for HIGHEST Mean Entropy Group:")
print(high_uncertainty_patients_df[['mean_entropy', 'mean_variance', 'std_entropy', 'std_variance', 'patient_accuracy']].describe().to_string())
print("\nStatistics for LOWEST Mean Entropy Group:")
print(low_uncertainty_patients_df[['mean_entropy', 'mean_variance', 'std_entropy', 'std_variance', 'patient_accuracy']].describe().to_string())

print("\n--- Patient-Level Textual Analysis Complete ---")
print(f"Outputs saved in directory: {OUTPUT_DIR}")
