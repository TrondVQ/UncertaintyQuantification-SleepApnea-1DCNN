import pandas as pd
from scipy.stats import pearsonr
import os

# --- Configuration ---
# Adjust these paths if needed based on where your scripts saved the files
MCD_CSV_PATH = './patient_level_uq_analysis_MCD/patient_summary_metrics_MCD.csv'
DE_CSV_PATH = './patient_level_uq_analysis_DE/patient_summary_metrics_DE.csv'

# Columns to correlate
xcol = 'mean_entropy'
ycol = 'patient_accuracy'

# --- Function to Load and Calculate ---
def calculate_and_print_correlation(csv_path, method_name, x_col, y_col):
    """Loads data and calculates Pearson correlation."""
    print(f"--- Analyzing {method_name} ---")
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found - {csv_path}")
        return None, None

    try:
        df = pd.read_csv(csv_path)
        if x_col not in df.columns or y_col not in df.columns:
            print(f"ERROR: Required columns ('{x_col}', '{y_col}') not found in {csv_path}")
            return None, None

        # Drop rows with NaN in relevant columns if any exist
        df_clean = df[[x_col, y_col]].dropna()
        if len(df_clean) < len(df):
            print(f"Warning: Dropped {len(df) - len(df_clean)} rows with NaN values.")
        if len(df_clean) < 2:
            print("ERROR: Not enough valid data points for correlation.")
            return None, None

        correlation, p_value = pearsonr(df_clean[x_col], df_clean[y_col])

        print(f"Correlation between '{x_col}' and '{y_col}':")
        print(f"Pearson r = {correlation:.4f}")
        print(f"P-value = {p_value:.4g}") # Use general format for small p-values
        print("-" * 20)
        return correlation, p_value

    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None, None

# --- Main Execution ---
if __name__ == "__main__":
    print("Calculating Patient-Level Correlations...")

    mcd_r, mcd_p = calculate_and_print_correlation(MCD_CSV_PATH, "MC Dropout", xcol, ycol)
    de_r, de_p = calculate_and_print_correlation(DE_CSV_PATH, "Deep Ensemble", xcol, ycol)

    print("\nCalculation complete.")
    print("Insert the 'Pearson r' values into Sections 4.4.2 and 5.3.3/5.6 of your thesis.")
    if mcd_r is not None:
        print(f"Value for [Insert MCD r]: {mcd_r:.4f}")
    if de_r is not None:
        print(f"Value for [Insert DE r]: {de_r:.4f}")

    # IMPORTANT NOTE ON MCD ACCURACY
    print("\nREMINDER: The 'patient_accuracy' in the MCD summary CSV corresponds to the ~77% baseline observed")
    print("during the MCD UQ run (mean prediction thresholding), not the ~88% deterministic baseline.")
    print("Interpret the MCD correlation value with this context in mind.")