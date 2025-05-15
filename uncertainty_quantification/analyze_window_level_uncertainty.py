#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

# --- Configuration ---

DETAILED_RESULTS_CSV = './detail_patient_DE.csv' # CHANGE THIS FOR MCD
NUM_BINS = 10

# --- Load Detailed UQ Results ---
try:
    uq_results_df = pd.read_csv(DETAILED_RESULTS_CSV)
    print(f"Loaded detailed UQ results from: {DETAILED_RESULTS_CSV}")
except FileNotFoundError:
    print(f"ERROR: '{DETAILED_RESULTS_CSV}' not found.")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- Data Checks and Preparation ---
required_cols = ['True_Label', 'Predicted_Label',
                 'Predictive_Variance', 'Predictive_Entropy']
if not all(col in uq_results_df.columns for col in required_cols):
    print(f"ERROR: Missing one or more required columns: {required_cols}")
    exit()

if 'Correct' not in uq_results_df.columns:
    uq_results_df['Correct'] = (uq_results_df['True_Label'] == uq_results_df['Predicted_Label'])

print(f"\nTotal number of windows analyzed: {len(uq_results_df)}")
print(f"Overall accuracy across all windows: {uq_results_df['Correct'].mean():.4f}")

# --- Textual Summary of Overall Window-Level Trends ---
print("\n--- Textual Summary of Overall Window-Level Uncertainty vs. Correctness ---")

# Compare uncertainty for Correct vs Incorrect predictions
print("\nStatistics for CORRECTLY Classified Windows:")
print(uq_results_df[uq_results_df['Correct'] == True][['Predictive_Entropy', 'Predictive_Variance']].describe().to_string())

print("\nStatistics for INCORRECTLY Classified Windows:")
print(uq_results_df[uq_results_df['Correct'] == False][['Predictive_Entropy', 'Predictive_Variance']].describe().to_string())

# --- Calculate and Print Binned Accuracy/Error Rate ---
metric_to_bin = 'Predictive_Entropy'
print(f"\n--- Binned Accuracy/Error Rate vs. {metric_to_bin} (Numerical) ---")

min_metric = uq_results_df[metric_to_bin].min()
max_metric = uq_results_df[metric_to_bin].max()
bins = np.linspace(min_metric, max_metric + 1e-9, NUM_BINS + 1)
# Create labels for bins for easier reading
labels = [f'{bins[i]:.3f}-{bins[i+1]:.3f}' for i in range(NUM_BINS)]

uq_results_df[f'{metric_to_bin}_Bin'] = pd.cut(uq_results_df[metric_to_bin],
                                               bins=bins, labels=labels, right=False)

# Group by bin and calculate metrics
binned_results = uq_results_df.groupby(f'{metric_to_bin}_Bin', observed=False).agg(
    window_count=('Correct', 'size'),
    accuracy=('Correct', 'mean')
)
binned_results['error_rate'] = 1 - binned_results['accuracy']

print(f"\nAccuracy and Error Rate per {metric_to_bin} Bin:")
print(binned_results.to_string(float_format="%.4f"))


print("\n--- Overall Window-Level Textual Analysis Complete ---")

