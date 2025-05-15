import sys
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import List
from uq_techniques import deep_ensembles_predict, evaluate_uq_methods as evaluate_uq_from_preds

# This file is designed to run the Deep Ensemble evaluation on the patient-level data
# It is a key file to generate the uncertainty quantification (UQ) metrics for the Deep Ensemble method which other analysis can be derived from.

# --- Configuration ---
SEED = 2025
# Directory where finalize_datasets.py saved the .npy files
PROCESSED_DATA_DIR = "./processed_datasets"

ENSEMBLE_MODEL_DIR = "./models/ensemble_cnn_no_pool" # Directory containing ensemble models
ENSEMBLE_MODEL_PATTERN = "AlCNN_smote_seed{}.keras" # Pattern like model_seed_*.keras
NUM_ENSEMBLE_MEMBERS = 5 # Number of models in the ensemble

# Parameters for Bootstrap CIs (within evaluate_uq_methods)
N_BOOTSTRAP = 100
# Directory for saving UQ plots
OUTPUT_PLOT_DIR = "./uq_plots_patient/deep_ensemble_no_pool" # Changed output dir
# Directory for saving detailed results CSV
OUTPUT_CSV_DIR = "./uq_results_patient_DE_new"  # Keep same CSV output dir, filename will differ

# Set seed for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Suppress excessive TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Create output directories if they don't exist
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)

# --- Helper function to load ensemble models ---
def load_ensemble(model_dir: str, pattern: str, num_members: int) -> List[tf.keras.Model]:
    """Loads ensemble models based on a directory and pattern."""
    ensemble_models = []
    print(f"Loading {num_members} ensemble models from {model_dir} with pattern '{pattern}'...")
    for i in range(num_members):
        model_filename = pattern.format(i+5)
        model_path = os.path.join(model_dir, model_filename)
        try:
            model = load_model(model_path)
            print(f"Loaded: {model_filename}")
            ensemble_models.append(model)
        except Exception as e:
            print(f"ERROR loading model {model_path}: {e}")
            # Decide whether to raise error or continue with fewer models
            raise # Re-raise error to stop execution if a model is missing
    if len(ensemble_models) != num_members:
        raise ValueError(f"Expected {num_members} models, but only loaded {len(ensemble_models)}")
    print(f"Successfully loaded {len(ensemble_models)} ensemble models.")
    return ensemble_models

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
    # patient_ids_test_rus = np.load(os.path.join(PROCESSED_DATA_DIR, 'patient_ids_test_rus.npy')) # Load if available and needed

    print("Test datasets loaded successfully.")
    print(f"X_test_std_unbalanced shape: {X_test_std_unbalanced.shape}")
    print(f"X_test_std_rus shape: {X_test_std_rus.shape}")

except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure 'finalize_datasets.py' has been run successfully and files are in the correct directory.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    sys.exit(1)


def evaluate_deep_ensemble(ensemble_models: List[tf.keras.Model], X_data, y_data, patient_ids, model_eval_label: str, save_detailed_csv=False):
    """
    Runs Deep Ensemble prediction, calculates per-window UQ metrics, saves detailed results,
    and evaluates aggregated UQ metrics.

    Args:
        ensemble_models: List of loaded Keras models forming the ensemble.
        X_data: Test features (samples, steps, features).
        y_data: True test labels (samples,).
        patient_ids: Patient IDs corresponding to X_data (samples,).
        model_eval_label (str): Unique label for this evaluation run (e.g., "CNN_DE_Unbalanced").
        save_detailed_csv (bool): Whether to save the detailed per-window results CSV.

    Returns:
        Dictionary of aggregated UQ metrics or None if fails.
    """
    print(f"\n===== Running Deep Ensemble Evaluation for: {model_eval_label} =====")
    num_models = len(ensemble_models)
    if num_models == 0:
        print("ERROR: No models provided in the ensemble list.")
        return None

    # --- Step 1: Get Deep Ensemble raw predictions/probabilities ---
    # Use the function from uq_techniques.py
    print(f"Running Deep Ensemble prediction with {num_models} models...")
    # This function should return shape (num_models, samples, 1) or similar
    ensemble_probabilities = deep_ensembles_predict(ensemble_models, X_data)

    if ensemble_probabilities is None or ensemble_probabilities.shape[0] != num_models or ensemble_probabilities.shape[1] != len(y_data):
        print(f"Deep Ensemble prediction failed or returned unexpected shape for {model_eval_label}")
        return None
    print(f"Deep Ensemble probabilities shape: {ensemble_probabilities.shape}") # Should be (num_models, num_windows, 1)

    # --- Step 2: Calculate Per-Window UQ Metrics ---
    print("Calculating per-window UQ metrics...")
    # Mean probability across models for each window
    mean_probs_per_window = np.mean(ensemble_probabilities, axis=0).flatten() # Shape: (num_windows,)
    # Predictive variance across models for each window
    pred_variance_per_window = np.var(ensemble_probabilities, axis=0).flatten() # Shape: (num_windows,)
    # Predictive entropy from the mean probability (binary cross-entropy)
    epsilon = 1e-9 # To avoid log(0)
    pred_entropy_per_window = - (mean_probs_per_window * np.log2(mean_probs_per_window + epsilon) + \
                                 (1 - mean_probs_per_window) * np.log2(1 - mean_probs_per_window + epsilon))
    # Final predicted label based on mean probability
    final_predicted_labels = (mean_probs_per_window > 0.5).astype(int)

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
        return None
    except Exception as e:
        print(f"ERROR during DataFrame creation or saving: {e}")
        return None

    # --- Step 5: Evaluate Aggregated UQ Metrics (Optional - Uncomment if needed) ---
    # This uses the same evaluation function as before, passing the ensemble probabilities

    print("Calculating aggregated UQ metrics...")
    metrics = evaluate_uq_from_preds(
        predictions=ensemble_probabilities, # Pass the raw probabilities (num_models, samples, 1)
        y_test=y_data,
        evaluation_label=model_eval_label,
        n_bootstrap=N_BOOTSTRAP,
        random_state=SEED,
        output_plot_dir=os.path.join(OUTPUT_PLOT_DIR, model_eval_label)
    )

    # Print key aggregated metrics summary
    if metrics:
        print(f"\nAggregated Uncertainty Metrics Summary for {model_eval_label}:")
        print(f"- Overall Mean Variance: {metrics.get('overall_mean_variance_mean', metrics.get('overall_mean_variance', 'N/A')):.6f}")
        print(f"- Mean Total Predictive Entropy: {metrics.get('mean_total_pred_entropy_mean', metrics.get('mean_total_pred_entropy', 'N/A')):.4f}")
        print(f"- Mean Mutual Information: {metrics.get('mean_mutual_info_mean', metrics.get('mean_mutual_info', 'N/A')):.6f}") # Note: MI interpretation might differ for DE
        print(f"- Mean Expected Aleatoric Entropy: {metrics.get('mean_expected_aleatoric_entropy_mean', metrics.get('mean_expected_aleatoric_entropy', 'N/A')):.4f}")
    else:
        print(f"Aggregated UQ metric calculation failed for {model_eval_label}")

    return metrics

# --- Main Execution Block ---
try:
    print("\n" + "="*50)
    # --- Load the Ensemble Models ---
    ensemble_models = load_ensemble(ENSEMBLE_MODEL_DIR, ENSEMBLE_MODEL_PATTERN, NUM_ENSEMBLE_MEMBERS)

    # --- Evaluate Deep Ensemble ---
    # Evaluate on Unbalanced Test Set - Save detailed CSV for this one
    cnn_metrics_de_ub = evaluate_deep_ensemble(
        ensemble_models=ensemble_models,
        X_data=X_test_std_unbalanced,
        y_data=y_test_unbalanced,
        patient_ids=patient_ids_test_unbalanced, # Pass patient IDs
        model_eval_label="CNN_DE_Unbalanced", # Updated label
        save_detailed_csv=True # Set to True to save the file
    )

    # Evaluate on Balanced (RUS) Test Set - Optionally save detailed CSV
    cnn_metrics_de_bal = evaluate_deep_ensemble(
        ensemble_models=ensemble_models,
        X_data=X_test_std_rus,
        y_data=y_test_rus,
        patient_ids=None, # Set to None or load appropriate IDs if needed
        model_eval_label="CNN_DE_Balanced_RUS", # Updated label
        save_detailed_csv=False # Set to False or True if needed
    )

except FileNotFoundError:
    print(f"ERROR: One or more DE model files not found in {ENSEMBLE_MODEL_DIR} with pattern {ENSEMBLE_MODEL_PATTERN}")
    cnn_metrics_de_ub, cnn_metrics_de_bal = None, None
except Exception as e:
    print(f"An error occurred during Deep Ensemble evaluation: {e}")
    cnn_metrics_de_ub, cnn_metrics_de_bal = None, None

# --- Conclusion ---
print("\n" + "="*50)
print("Deep Ensemble Evaluation complete.")
print(f"Aggregated metric plots saved in subdirectories under: {OUTPUT_PLOT_DIR}") # Uncomment if Step 5 is activ
print(f"Detailed results CSV saved in: {OUTPUT_CSV_DIR}")
