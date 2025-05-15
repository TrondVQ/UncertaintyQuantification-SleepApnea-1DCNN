import pandas as pd
from scipy.stats import mannwhitneyu # Import the function

# Load your detailed results CSV (assuming it's generated and correct)
try:
    mcd_detail_df = pd.read_csv('./detail_patient_DE.csv') # CHANGE THIS FOR MCD
    # Ensure 'Correct' column exists
    if 'Correct' not in mcd_detail_df.columns:
        mcd_detail_df['Correct'] = (mcd_detail_df['True_Label'] == mcd_detail_df['Predicted_Label'])

    # Separate the entropy scores for correct and incorrect predictions
    entropy_correct = mcd_detail_df.loc[mcd_detail_df['Correct'] == True, 'Predictive_Entropy'].dropna()
    entropy_incorrect = mcd_detail_df.loc[mcd_detail_df['Correct'] == False, 'Predictive_Entropy'].dropna()

    # Perform the Mann-Whitney U test
    # alternative='greater' tests if the first group (incorrect) has higher values
    if len(entropy_correct) > 0 and len(entropy_incorrect) > 0:
        stat, p_value = mannwhitneyu(entropy_incorrect, entropy_correct, alternative='greater')

        print(f"\n--- Mann-Whitney U Test Results (Predictive Entropy: Incorrect > Correct) ---")
        print(f"Method: Deep Ensembles")
        print(f"U Statistic: {stat}")
        print(f"P-value: {p_value: .4g}") # Format p-value nicely

        if p_value < 0.05:
            print("Conclusion: The difference is statistically significant (p < 0.05). Incorrect predictions have significantly higher entropy.")
        else:
            print("Conclusion: The difference is not statistically significant (p >= 0.05).")

    else:
        print("Not enough data in one or both groups to perform the test.")

except FileNotFoundError:
    print("ERROR: Detailed MCD results CSV not found.")
except Exception as e:
    print(f"An error occurred during statistical testing: {e}")

# --- Repeat the process for Deep Ensemble using its detailed CSV ---
# try:
#    de_detail_df = pd.read_csv('./uq_results_patient_DE_new/detailed_results_CNN_DE_Unbalanced.csv')
#    ... (calculate 'Correct' column) ...
#    entropy_correct_de = ...
#    entropy_incorrect_de = ...
#    stat_de, p_value_de = mannwhitneyu(entropy_incorrect_de, entropy_correct_de, alternative='greater')
#    print("\nMethod: Deep Ensemble")
#    print(f"P-value: {p_value_de:.4g}")
#    ... (print conclusion) ...
# except ...