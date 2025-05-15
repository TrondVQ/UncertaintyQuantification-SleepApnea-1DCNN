import sys
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from uq_techniques import mc_dropout_predict, evaluate_uq_methods as evaluate_uq_from_preds
from sklearn.metrics import accuracy_score

# This file is designed to run the MCD evaluation on the patient-level data
# It is a key file to generate the uncertainty quantification (UQ) metrics for the MCD method which other analysis can be derived from.


# --- Configuration ---
SEED = 2025
# Directory where finalize_datasets.py saved the .npy files
PROCESSED_DATA_DIR = "./processed_datasets"
# Paths to the trained models (ensure these models have dropout layers)
CNN_MODEL_PATH = "./AlCNN1D_no_pool.keras"
# Parameters for MC Dropout
N_MC_PASSES = 50
# Parameters for Bootstrap CIs (within evaluate_uq_methods)
N_BOOTSTRAP = 100
# Directory for saving UQ plots
OUTPUT_PLOT_DIR = "./uq_plots_patient/mc_dropout_no_pool"
# Directory for saving detailed results CSV
OUTPUT_CSV_DIR = "./uq_results_patient_no_pool"

# Set seed for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Suppress excessive TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Create output directories if they don't exist
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True) # Create CSV output dir

# --- 1. Load Pre-processed Test Data ---
print("Loading pre-processed test datasets...")
try:
    # Load Unbalanced Test Set
    X_test_std_unbalanced = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test_std_unbalanced.npy'))
    y_test_unbalanced = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test_unbalanced.npy'))
    # --- Load Patient IDs for the unbalanced test set ---
    patient_ids_test_unbalanced = np.load(os.path.join(PROCESSED_DATA_DIR, 'patient_ids_test_unbalanced.npy'))
    print(f"Loaded {len(patient_ids_test_unbalanced)} patient IDs for unbalanced test set.")

    # Load Balanced (RUS) Test Set
    X_test_std_rus = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test_std_rus.npy'))
    y_test_rus = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test_rus.npy'))
    # --- Optional: Load Patient IDs for the balanced test set if needed ---
    # If you saved patient IDs corresponding to the RUS set, load them here.
    # Otherwise, you might need to generate/track them during the RUS process.
    # For now, we'll focus on saving detailed results for the unbalanced set.
    # patient_ids_test_rus = np.load(os.path.join(PROCESSED_DATA_DIR, 'patient_ids_test_rus.npy')) # Example

    print("Test datasets loaded successfully.")
    print(f"X_test_std_unbalanced shape: {X_test_std_unbalanced.shape}")
    print(f"X_test_std_rus shape: {X_test_std_rus.shape}")

except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure 'finalize_datasets.py' has been run successfully and files are in the correct directory.")
    sys.exit(1) # Exit if data isn't found
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    sys.exit(1)


def evaluate_mc_dropout(model, X_data, y_data, patient_ids, model_eval_label: str, save_detailed_csv=False):
    """
    Runs MC Dropout prediction, calculates per-window UQ metrics, saves detailed results,
    and evaluates aggregated UQ metrics.

    Args:
        model: Loaded Keras model with Dropout layers.
        X_data: Test features (samples, steps, features).
        y_data: True test labels (samples,).
        patient_ids: Patient IDs corresponding to X_data (samples,).
        model_eval_label (str): Unique label for this evaluation run (e.g., "CNN_MCD_Unbalanced").
        save_detailed_csv (bool): Whether to save the detailed per-window results CSV.

    Returns:
        Dictionary of aggregated UQ metrics or None if fails.
    """
    print(f"\n===== Running MC Dropout Evaluation for: {model_eval_label} =====")

    # --- Step 1: Get MC Dropout raw predictions/probabilities ---
    # mc_dropout_predict function needs to return the probabilities from each pass
    # Let's assume it returns a numpy array of shape (n_pred, samples, 1) for binary classification
    print(f"Running MC Dropout with {N_MC_PASSES} passes...")
    mc_probabilities = mc_dropout_predict(model, X_data, n_pred=N_MC_PASSES) # Assuming this returns probabilities

    if mc_probabilities is None or mc_probabilities.shape[0] != N_MC_PASSES or mc_probabilities.shape[1] != len(y_data):
        print(f"MC Dropout prediction failed or returned unexpected shape for {model_eval_label}")
        return None
    print(f"MC Dropout probabilities shape: {mc_probabilities.shape}") # Should be (N_MC_PASSES, num_windows, 1)
    try:
        np.save(f"./mc_raw_pred0505_{model_eval_label}.npy", mc_probabilities)
        print(f"Saved raw MC predictions (shape: {mc_probabilities.shape}) to")
    except Exception as e:
        print(f"ERROR saving raw MC predictions: {e}")
        # Decide if you want to stop or continue if saving fails

    # --- Step 2: Calculate Per-Window UQ Metrics ---
    print("Calculating per-window UQ metrics...")
    # Mean probability across passes for each window
    mean_probs_per_window = np.mean(mc_probabilities, axis=0).flatten() # Shape: (num_windows,)
    # Predictive variance across passes for each window
    pred_variance_per_window = np.var(mc_probabilities, axis=0).flatten() # Shape: (num_windows,)
    # Predictive entropy from the mean probability (binary cross-entropy)
    epsilon = 1e-9 # To avoid log(0)
    pred_entropy_per_window = - (mean_probs_per_window * np.log2(mean_probs_per_window + epsilon) + \
                                 (1 - mean_probs_per_window) * np.log2(1 - mean_probs_per_window + epsilon))
    # Final predicted label based on mean probability
    final_predicted_labels = (mean_probs_per_window > 0.5).astype(int)
    accuracy_from_mean_mcd = accuracy_score(y_data, final_predicted_labels)
    print(f"Accuracy based on mean MCD probability (>0.5): {accuracy_from_mean_mcd:.4f}")
    # This should match the ~77% value if the issue is purely the prediction method difference

    # --- Step 3: Create Detailed Results DataFrame ---
    print("Creating detailed results DataFrame...")
    try:
        # Ensure all arrays have the same length (number of windows)
        if patient_ids is None and save_detailed_csv:
            print("Warning: Patient IDs are None. Cannot save detailed CSV with patient IDs.")
            # Decide if you want to proceed without patient IDs or stop
        elif patient_ids is not None:
            assert len(patient_ids) == len(y_data) == len(final_predicted_labels) == \
                   len(mean_probs_per_window) == len(pred_variance_per_window) == \
                   len(pred_entropy_per_window), "Mismatch in array lengths!"

        detailed_results_df = pd.DataFrame({
            'Patient_ID': patient_ids if patient_ids is not None else np.nan, # Handle missing IDs
            'Window_Index': np.arange(len(y_data)), # Simple index for each window
            'True_Label': y_data,
            'Predicted_Label': final_predicted_labels,
            'Predicted_Probability': mean_probs_per_window,
            'Predictive_Variance': pred_variance_per_window,
            'Predictive_Entropy': pred_entropy_per_window
            # Add 'Mutual_Information' here if you calculate it per window
            # Note: MI calculation might differ slightly for DE vs MCD in some frameworks
        })
        print(detailed_results_df.head())

        # --- Step 4: Save Detailed Results (Optional) ---
        if save_detailed_csv and patient_ids is not None: # Only save if IDs are available
            csv_filename = f"detailed_results_{model_eval_label}.csv" # e.g., detailed_results_CNN_DE_Unbalanced.csv
            csv_filepath = os.path.join(OUTPUT_CSV_DIR, csv_filename)
            print(f"Saving detailed results to: {csv_filepath}")
            detailed_results_df.to_csv(csv_filepath, index=False)
            print("Detailed results saved.")
        elif save_detailed_csv and patient_ids is None:
            print("Skipping detailed CSV save because patient IDs were not provided.")


    except AssertionError as e:
        print(f"ERROR creating DataFrame: {e}")
        # Decide how to handle: maybe return None or just skip saving
        return None
    except Exception as e:
        print(f"ERROR during DataFrame creation or saving: {e}")
        return None


    # --- Step 5: Evaluate Aggregated UQ Metrics (using your existing function) ---
    # Pass the raw MC probabilities to the evaluation function if it expects them
    # Or pass the calculated per-window metrics if it expects those.
    # Adjust based on what evaluate_uq_from_preds needs.
    # Assuming it needs the raw probabilities:
    print("Calculating aggregated UQ metrics...")
    metrics = evaluate_uq_from_preds(
        predictions=mc_probabilities,      # Pass the raw probabilities (n_pred, samples, 1)
        y_test=y_data,
        evaluation_label=model_eval_label, # Pass label for plots/output
        n_bootstrap=N_BOOTSTRAP,          # Set number of bootstrap samples
        random_state=SEED,                # Pass seed for reproducibility
        output_plot_dir=os.path.join(OUTPUT_PLOT_DIR, model_eval_label) # Save plots in subdirs
    )

    # Print key aggregated metrics summary (using keys from the refined evaluate_uq_methods)
    if metrics:
        print(f"\nAggregated Uncertainty Metrics Summary for {model_eval_label}:")
        print(f"- Overall Mean Variance: {metrics.get('overall_mean_variance_mean', metrics.get('overall_mean_variance', 'N/A')):.6f}")
        print(f"- Mean Total Predictive Entropy: {metrics.get('mean_total_pred_entropy_mean', metrics.get('mean_total_pred_entropy', 'N/A')):.4f}")
        print(f"- Mean Mutual Information: {metrics.get('mean_mutual_info_mean', metrics.get('mean_mutual_info', 'N/A')):.6f}")
        print(f"- Mean Expected Aleatoric Entropy: {metrics.get('mean_expected_aleatoric_entropy_mean', metrics.get('mean_expected_aleatoric_entropy', 'N/A')):.4f}")
    else:
        print(f"Aggregated UQ metric calculation failed for {model_eval_label}")

    #  metrics = {}
    return metrics # Return the aggregated metrics dictionary

# --- 4. CNN MC Dropout Evaluation ---
try:
    print("\n" + "="*50)
    print(f"Loading CNN model for MC Dropout from {CNN_MODEL_PATH}...")
    cnn_model = load_model(CNN_MODEL_PATH)
    print(f"Model loaded successfully: {CNN_MODEL_PATH}")

    print("\n--- Performing quick deterministic accuracy check ---")
    try:
        deterministic_probs_ub = cnn_model(X_test_std_unbalanced, training=False).numpy().flatten()
        deterministic_labels_ub = (deterministic_probs_ub > 0.5).astype(int)
        deterministic_accuracy_ub = accuracy_score(y_test_unbalanced, deterministic_labels_ub)
        print(f"Deterministic Accuracy (training=False) on Unbalanced Set: {deterministic_accuracy_ub:.4f}")
        # This should ideally match your Section 4.2 result (~88%)
    except Exception as e_det:
        print(f"Could not perform deterministic check: {e_det}")
    print("--- End deterministic accuracy check ---\n")
    # Evaluate on Unbalanced Test Set - Save detailed CSV for this one
    cnn_metrics_mcd_ub = evaluate_mc_dropout(
        model=cnn_model,
        X_data=X_test_std_unbalanced,
        y_data=y_test_unbalanced,
        patient_ids=patient_ids_test_unbalanced, # Pass patient IDs
        model_eval_label="CNN_MCD_Unbalanced",
        save_detailed_csv=True # Set to True to save the file
    )

    # Evaluate on Balanced (RUS) Test Set - Optionally save detailed CSV
    # Note: Ensure you have corresponding patient_ids_test_rus if saving details
    cnn_metrics_mcd_bal = evaluate_mc_dropout(
        model=cnn_model,
        X_data=X_test_std_rus,
        y_data=y_test_rus,
        patient_ids= None, # Set to None or load appropriate IDs if needed
        model_eval_label="CNN_MCD_Balanced_RUS",
        save_detailed_csv=False # Set to False or True if needed
    )

except FileNotFoundError:
    print(f"ERROR: CNN model file not found at {CNN_MODEL_PATH}")
    cnn_metrics_mcd_ub, cnn_metrics_mcd_bal = None, None
except Exception as e:
    print(f"An error occurred during CNN MC Dropout evaluation: {e}")
    cnn_metrics_mcd_ub, cnn_metrics_mcd_bal = None, None

# --- 6. Conclusion ---
print("\n" + "="*50)
print("MC Dropout Evaluation complete.")
print(f"Aggregated metric plots saved in subdirectories under: {OUTPUT_PLOT_DIR}")
print(f"Detailed results CSV saved in: {OUTPUT_CSV_DIR}")
# You can now use the dictionaries (e.g., cnn_metrics_mcd_ub) to report numerical results.
