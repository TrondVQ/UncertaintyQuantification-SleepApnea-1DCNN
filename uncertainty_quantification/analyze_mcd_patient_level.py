#!/usr/bin/env python3

import sys
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import argparse
from sklearn.metrics import accuracy_score
from typing import List, NoReturn, Dict, Any, Optional, Tuple

try:
    # Adjust import path if uq_techniques.py is in a different location
    from uq_techniques import mc_dropout_predict, evaluate_uq_methods as evaluate_aggregated_uq
except ImportError:
    print("Error: Could not import 'uq_techniques.py'. Please ensure it's in the correct path.")
    print("This script requires functions: mc_dropout_predict and evaluate_uq_methods.")
    sys.exit(1)


# --- Configuration ---
# Default random seed for reproducibility (used for bootstrap sampling in evaluation)
SEED: int = 2025

# --- Define default paths and parameters (can be overridden by command-line args) ---
# Default directory where the processed .npy datasets (from prepare_final_datasets.py) are located
PROCESSED_DATA_DIR: str = "./final_processed_datasets"

# Default path to the trained model file (should have Dropout layers)
CNN_MODEL_PATH: str = "./alarcon_cnn_model.keras" # Assuming this is the model trained for MCD

# Number of Monte Carlo passes to perform for each window prediction
N_MC_PASSES: int = 50

# Number of bootstrap samples for Confidence Interval calculation in aggregated UQ evaluation
N_BOOTSTRAP: int = 100

# Default directory for saving detailed per-window UQ results CSV
OUTPUT_CSV_DIR: str = "./uq_results/mc_dropout" # More generic name

# Default directory for saving raw MC Dropout predictions (optional) -> this can be used for AHI calculations
RAW_PREDICTIONS_SAVE_DIR: str = "./raw_predictions/mc_dropout" # Directory to save raw .npy predictions

# Base filename for the detailed per-window results CSV
DETAILED_CSV_FILENAME_BASE: str = "detailed_results_{}.csv" # {} will be formatted with eval_label

# Base filename for saving raw MC Dropout predictions
RAW_PREDICTIONS_FILENAME_BASE: str = "raw_mc_predictions_{}.npy" # {} will be formatted with eval_label


# --- Set seed for reproducibility (for numpy and tensorflow) ---
# This affects operations outside of specific model training, e.g., bootstrap sampling.
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Suppress excessive TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0=all, 1=info, 2=warnings, 3=errors


def evaluate_mc_dropout_and_uq(
        model: Model,
        X_data: np.ndarray,
        y_data: np.ndarray,
        patient_ids: Optional[np.ndarray], # Patient IDs are optional for evaluation logic
        model_eval_label: str,
        n_mc_passes: int,
        output_csv_dir: str,
        raw_preds_save_dir: Optional[str] = None, # Optional directory to save raw predictions
        save_detailed_csv: bool = False,
        n_bootstrap: int = N_BOOTSTRAP,
        random_state: int = SEED
) -> Optional[Dict[str, Any]]:
    """
    Runs MC Dropout prediction, calculates per-window UQ metrics, optionally
    saves detailed results and raw predictions, and evaluates aggregated UQ metrics.

    Args:
        model: Loaded Keras model with Dropout layers that should be active during prediction.
        X_data: Test features as a NumPy array (samples, time_steps, features).
        y_data: True test labels as a NumPy array (samples,).
        patient_ids: Patient IDs corresponding to X_data as a NumPy array (samples,) or None.
        model_eval_label: Unique label for this evaluation run (e.g., "CNN_MCD_Unbalanced").
                          Used for output filenames
        n_mc_passes: Number of Monte Carlo passes to run for each window.
        output_csv_dir: Directory to save the detailed per-window results CSV (if save_detailed_csv is True).
        raw_preds_save_dir: Optional directory to save the raw MC Dropout predictions (.npy file).
        save_detailed_csv: Whether to save the detailed per-window results CSV.
        n_bootstrap: Number of bootstrap samples for Confidence Interval calculation.
        random_state: Random seed for reproducibility of bootstrap sampling.

    Returns:
        A dictionary containing aggregated UQ metrics from evaluate_aggregated_uq,
        or None if any step fails.
    """
    print(f"\n===== Running MC Dropout Evaluation for: {model_eval_label} =====")

    if model is None:
        print("Error: No model provided for evaluation.")
        return None
    if X_data is None or y_data is None or X_data.size == 0 or y_data.size == 0:
        print("Error: Input data (features or labels) is empty.")
        return None
    if patient_ids is not None and len(patient_ids) != len(y_data):
        print(f"Warning: Mismatch in length between patient_ids ({len(patient_ids)}) and y_data ({len(y_data)}).")
        # Decide if this should be an error or just a warning. Proceeding with warning.
    if n_mc_passes <= 0:
        print(f"Error: Number of MC passes must be positive, but got {n_mc_passes}.")
        return None


    # --- Step 1: Get MC Dropout raw predictions/probabilities ---
    # Use the mc_dropout_predict function. It should activate Dropout layers during prediction.
    # It is expected to return a numpy array of shape (n_pred, samples, 1) for binary classification.
    print(f"Running MC Dropout with {n_mc_passes} passes on {len(y_data)} samples...")
    mc_probabilities: Optional[np.ndarray] = None
    try:
        mc_probabilities = mc_dropout_predict(model, X_data, n_pred=n_mc_passes)

        if mc_probabilities is None or mc_probabilities.ndim != 3 or \
                mc_probabilities.shape[0] != n_mc_passes or mc_probabilities.shape[1] != len(y_data) or \
                mc_probabilities.shape[2] != 1:
            print(f"Error: mc_dropout_predict failed or returned unexpected shape: {mc_probabilities.shape if mc_probabilities is not None else 'None'}")
            print(f"Expected shape: ({n_mc_passes}, {len(y_data)}, 1).")
            return None

        # Ensure probabilities are within [0, 1] range due to potential floating point issues
        mc_probabilities = np.clip(mc_probabilities, 0.0, 1.0)
        print(f"MC Dropout probabilities shape: {mc_probabilities.shape}") # Should be (n_mc_passes, num_windows, 1)

        # --- Optional: Save Raw MC Dropout Predictions ---
        if raw_preds_save_dir:
            try:
                os.makedirs(raw_preds_save_dir, exist_ok=True)
                raw_preds_filename = RAW_PREDICTIONS_FILENAME_BASE.format(model_eval_label)
                raw_preds_filepath = os.path.join(raw_preds_save_dir, raw_preds_filename)
                print(f"Saving raw MC predictions to: {raw_preds_filepath}")
                np.save(raw_preds_filepath, mc_probabilities)
                print("Raw MC predictions saved.")
            except Exception as e:
                print(f"Warning: Failed to save raw MC predictions to '{raw_preds_save_dir}': {e}")


    except Exception as e:
        print(f"Error during MC Dropout prediction passes: {e}")
        return None


    # --- Step 2: Calculate Per-Window UQ Metrics from MC Samples ---
    print("Calculating per-window UQ metrics...")
    try:
        # Mean probability across MC passes for each window (for final prediction and expected value)
        mean_probs_per_window: np.ndarray = np.mean(mc_probabilities, axis=0).flatten() # Shape: (num_windows,)

        # Predictive variance across MC passes for each window (Variance of means)
        pred_variance_per_window: np.ndarray = np.var(mc_probabilities, axis=0).flatten() # Shape: (num_windows,)

        # Predictive entropy from the mean probability (Total Predictive Entropy approximation for binary)
        # H(Y|x) approx H(E[Y|x]) for binary classification
        epsilon = 1e-9 # Small value to avoid log(0) issues
        # Ensure values are within (0, 1) before log
        mean_probs_clipped = np.clip(mean_probs_per_window, epsilon, 1 - epsilon)
        pred_entropy_per_window: np.ndarray = - (mean_probs_clipped * np.log2(mean_probs_clipped) + \
                                                 (1 - mean_probs_clipped) * np.log2(1 - mean_probs_clipped))

        # Calculate Expected Aleatoric Entropy for each window: E[H(Y|x, w)] approx 1/T sum(H(Y|x, wt))
        # H(Y|x, wt) for binary is - (pt*log2(pt) + (1-pt)*log2(1-pt)) where pt is probability from pass t
        # Need to calculate entropy for each MC pass prediction per window, then average.
        # Shape of mc_probabilities is (n_mc_passes, num_windows, 1). We need to calculate entropy along axis=2, then mean along axis=0.
        entropies_per_pass_per_window: np.ndarray = - (mc_probabilities * np.log2(mc_probabilities + epsilon) + \
                                                       (1 - mc_probabilities) * np.log2(1 - mc_probabilities + epsilon)) # Shape: (n_mc_passes, num_windows, 1)
        expected_aleatoric_entropy_per_window: np.ndarray = np.mean(entropies_per_pass_per_window, axis=0).flatten() # Shape: (num_windows,)


        # Calculate Mutual Information for each window: MI = H(Y|x) - E[H(Y|x, w)]
        # This represents Epistemic Uncertainty
        mutual_information_per_window: np.ndarray = pred_entropy_per_window - expected_aleatoric_entropy_per_window
        # Ensure MI is not negative due to floating point errors
        mutual_information_per_window[mutual_information_per_window < 0] = 0


        # Final predicted label based on the mean probability across MC passes
        final_predicted_labels: np.ndarray = (mean_probs_per_window > 0.5).astype(int)

        # Calculate accuracy based on the mean predictions
        accuracy_from_mean_mcd = accuracy_score(y_data, final_predicted_labels)
        print(f"Accuracy based on mean MCD probability (>0.5): {accuracy_from_mean_mcd:.4f}")
        # This accuracy should ideally match the deterministic accuracy if model weights are fixed.


    except Exception as e:
        print(f"Error calculating per-window UQ metrics: {e}")
        return None


    # --- Step 3: Create Detailed Results DataFrame ---
    print("Creating detailed results DataFrame...")
    try:
        # Ensure all core arrays have the same length (number of windows being evaluated)
        expected_len = len(y_data)
        if not all(len(arr) == expected_len for arr in [
            final_predicted_labels, mean_probs_per_window, pred_variance_per_window,
            pred_entropy_per_window, expected_aleatoric_entropy_per_window, mutual_information_per_window
        ]):
            print("Error: Mismatch in length of calculated per-window metric arrays.")
            return None

        # Prepare patient IDs for DataFrame
        patient_ids_for_df = patient_ids if patient_ids is not None else [np.nan] * expected_len

        if len(patient_ids_for_df) != expected_len:
            print(f"Error: Mismatch in length between patient_ids array prepared for DataFrame ({len(patient_ids_for_df)}) and expected length ({expected_len}).")
            return None


        detailed_results_df = pd.DataFrame({
            'Patient_ID': patient_ids_for_df,
            'Window_Index': np.arange(expected_len), # Simple index for each window within this dataset split
            'True_Label': y_data,
            'Predicted_Label': final_predicted_labels,
            'Predicted_Probability': mean_probs_per_window, # This is the mean probability across MC passes
            'Predictive_Variance': pred_variance_per_window,
            'Predictive_Entropy': pred_entropy_per_window, # Total Predictive Entropy
            'Expected_Aleatoric_Entropy': expected_aleatoric_entropy_per_window,
            'Mutual_Information': mutual_information_per_window # Epistemic Uncertainty component
        })

        print("\nDetailed Results DataFrame Head:")
        print(detailed_results_df.head().to_string()) # Use .to_string() for full view of head


        # --- Step 4: Save Detailed Results (Optional) ---
        if save_detailed_csv:
            # Ensure output CSV directory exists (should be done earlier, but safe check)
            os.makedirs(output_csv_dir, exist_ok=True)
            csv_filename = DETAILED_CSV_FILENAME_BASE.format(model_eval_label) # Format filename with eval label
            csv_filepath = os.path.join(output_csv_dir, csv_filename)

            print(f"\nSaving detailed per-window results to: {csv_filepath}")
            detailed_results_df.to_csv(csv_filepath, index=False)
            print("Detailed results CSV saved.")

    except Exception as e:
        print(f"ERROR during Detailed Results DataFrame creation or saving: {e}")
        return None

    # --- Step 5: Evaluate and Plot Aggregated UQ Metrics ---
    # This uses the evaluate_uq_methods function from uq_techniques.py.
    # Pass the raw MC probabilities and true labels.
    # This function is expected to calculate various aggregated metrics,
    # potentially including Brier Score, ECE, calibration curves, etc.,

    print("\nCalculating and evaluating aggregated UQ metrics (requires evaluate_uq_methods)...")
    aggregated_metrics: Optional[Dict[str, Any]] = None
    try:

        aggregated_metrics = evaluate_aggregated_uq(
            predictions=mc_probabilities.squeeze(-1), # evaluate_uq_methods might expect (n_pred, samples)
            y_true=y_data,
            evaluation_label=model_eval_label,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )

        # Print a brief summary of key aggregated metrics returned by evaluate_aggregated_uq
        if aggregated_metrics:
            print(f"\nAggregated Uncertainty Metrics Summary for {model_eval_label}:")
            # Print specific metrics if they are expected to be in the returned dict
            # Adapt these print statements based on what evaluate_aggregated_uq actually returns
            print(f"- Mean Predictive Variance: {aggregated_metrics.get('overall_mean_pred_variance', 'N/A'):.6f}")
            print(f"- Mean Total Predictive Entropy: {aggregated_metrics.get('overall_mean_total_pred_entropy', 'N/A'):.4f}")
            print(f"- Mean Mutual Information: {aggregated_metrics.get('overall_mean_mutual_information', 'N/A'):.6f}")
            print(f"- Mean Expected Aleatoric Entropy: {aggregated_metrics.get('overall_mean_expected_aleatoric_entropy', 'N/A'):.4f}")
            # Add other relevant metrics like Brier Score, ECE if returned
            print(f"- Overall Brier Score: {aggregated_metrics.get('overall_brier_score', 'N/A'):.4f}")
            print(f"- Overall ECE: {aggregated_metrics.get('overall_ece', 'N/A'):.4f}")
        else:
            print(f"Aggregated UQ metric evaluation function returned None or empty for {model_eval_label}.")

    except Exception as e:
        print(f"ERROR during aggregated UQ metric evaluation: {e}")
        aggregated_metrics = None # Ensure None is returned on error

    # Return the dictionary of aggregated metrics
    return aggregated_metrics


# --- Main Execution Block ---
def run_mc_dropout_evaluation(
        processed_data_dir: str = PROCESSED_DATA_DIR,
        cnn_model_path: str = CNN_MODEL_PATH,
        n_mc_passes: int = N_MC_PASSES,
        output_csv_dir: str = OUTPUT_CSV_DIR,
        raw_preds_save_dir: Optional[str] = RAW_PREDICTIONS_SAVE_DIR, # Optional directory to save raw predictions
        n_bootstrap: int = N_BOOTSTRAP,
        seed: int = SEED,
        evaluate_unbalanced: bool = True, # Flag to control evaluation on unbalanced set
        evaluate_balanced: bool = True,   # Flag to control evaluation on balanced set
        save_unbalanced_csv: bool = False, # Flag to control saving detailed CSV for unbalanced set
        save_balanced_csv: bool = False    # Flag to control saving detailed CSV for balanced set
) -> NoReturn:
    """
    Main function to load data and a trained model, perform MC Dropout
    evaluation, calculate UQ metrics, and save results.

    Args:
        processed_data_dir: Directory containing the processed test data (.npy files).
        cnn_model_path: Path to the trained Keras model file.
        n_mc_passes: Number of Monte Carlo passes for prediction.
        output_csv_dir: Directory to save detailed per-window results CSV.
        raw_preds_save_dir: Optional directory to save the raw MC Dropout predictions (.npy file).
        n_bootstrap: Number of bootstrap samples for CI calculation.
        seed: Random seed for reproducibility.
        evaluate_unbalanced: Whether to evaluate on the unbalanced test set.
        evaluate_balanced: Whether to evaluate on the balanced test set.
        save_unbalanced_csv: Whether to save detailed CSV for unbalanced test set.
        save_balanced_csv: Whether to save detailed CSV for balanced test set.
    """
    print("\n" + "="*70)
    print("--- Starting MC Dropout Evaluation Script ---")
    print(f"Processed Data Dir: {processed_data_dir}")
    print(f"CNN Model Path: {cnn_model_path}")
    print(f"Number of MC Passes: {n_mc_passes}")
    print(f"Output CSV Dir: {output_csv_dir}")
    print(f"Raw Predictions Save Dir: {raw_preds_save_dir if raw_preds_save_dir else 'Not saving'}")
    print(f"Bootstrap Samples for CI: {n_bootstrap}")
    print(f"Random Seed: {seed}")
    print(f"Evaluate Unbalanced Set: {evaluate_unbalanced}")
    print(f"Evaluate Balanced Set: {evaluate_balanced}")
    print(f"Save Unbalanced CSV: {save_unbalanced_csv}")
    print(f"Save Balanced CSV: {save_balanced_csv}")
    print("="*70)

    # --- Create Output Directories ---
    try:
        os.makedirs(output_csv_dir, exist_ok=True)
        print(f"Output CSV directory '{output_csv_dir}' ensured to exist.")
        if raw_preds_save_dir:
            os.makedirs(raw_preds_save_dir, exist_ok=True)
            print(f"Raw predictions save directory '{raw_preds_save_dir}' ensured to exist.")

    except Exception as e:
        print(f"Error creating output directories: {e}")
        sys.exit(1)


    # --- Load Pre-processed Test Data ---
    print("\nLoading pre-processed test datasets...")
    X_test_std_unbalanced: Optional[np.ndarray] = None
    y_test_unbalanced: Optional[np.ndarray] = None
    patient_ids_test_unbalanced: Optional[np.ndarray] = None
    X_test_std_rus: Optional[np.ndarray] = None
    y_test_rus: Optional[np.ndarray] = None

    try:
        # Construct full data paths
        X_test_unbalanced_path = os.path.join(processed_data_dir, 'X_test_win_std_unbalanced.npy')
        y_test_unbalanced_path = os.path.join(processed_data_dir, 'y_test_unbalanced.npy')
        patient_ids_test_unbalanced_path = os.path.join(processed_data_dir, 'patient_ids_test_unbalanced.npy') # Patient IDs for Unbalanced set

        X_test_rus_path = os.path.join(processed_data_dir, 'X_test_win_std_rus.npy')
        y_test_rus_path = os.path.join(processed_data_dir, 'y_test_rus.npy')
        # Patient IDs for RUS set are typically not available or meaningful after RUS


        # Load the data - check existence before loading
        if os.path.exists(X_test_unbalanced_path) and os.path.exists(y_test_unbalanced_path) and os.path.exists(patient_ids_test_unbalanced_path):
            X_test_std_unbalanced = np.load(X_test_unbalanced_path)
            y_test_unbalanced = np.load(y_test_unbalanced_path)
            patient_ids_test_unbalanced = np.load(patient_ids_test_unbalanced_path) # Load Patient IDs
            print("Unbalanced test datasets loaded successfully.")
            print(f"X_test_std_unbalanced shape: {X_test_std_unbalanced.shape}")
            print(f"y_test_unbalanced shape: {y_test_unbalanced.shape}")
            print(f"Patient IDs (Unbalanced) shape: {patient_ids_test_unbalanced.shape}")
            # Basic validation
            if X_test_std_unbalanced.size == 0 or y_test_unbalanced.size == 0 or patient_ids_test_unbalanced.size == 0:
                print("Warning: Unbalanced test data or patient IDs are empty after loading.")
        else:
            print("Warning: Unbalanced test data or patient IDs files not found. Skipping load.")


        if os.path.exists(X_test_rus_path) and os.path.exists(y_test_rus_path):
            X_test_std_rus = np.load(X_test_rus_path)
            y_test_rus = np.load(y_test_rus_path)
            print("Balanced (RUS) test datasets loaded successfully.")
            print(f"X_test_std_rus shape: {X_test_std_rus.shape}")
            print(f"y_test_rus shape: {y_test_rus.shape}")
            # Basic validation
            if X_test_std_rus.size == 0 or y_test_rus.size == 0:
                print("Warning: RUS balanced test data is empty after loading. Evaluation on RUS set may be skipped.")
        else:
            print("Warning: Balanced (RUS) test data files not found. Skipping load.")


        if (X_test_std_unbalanced is None or X_test_std_unbalanced.size == 0) and (X_test_std_rus is None or X_test_std_rus.size == 0):
            print("Error: No test data (unbalanced or RUS) was loaded.")
            sys.exit(1)


    except Exception as e:
        print(f"An unexpected error occurred during test data loading: {e}")
        sys.exit(1)


    # --- Load the Trained CNN Model ---
    print(f"\nLoading CNN model for MC Dropout from {cnn_model_path}...")
    cnn_model: Optional[Model] = None
    if not os.path.exists(cnn_model_path):
        print(f"Error: CNN model file not found at '{cnn_model_path}'.")
        print("Ensure 'train_cnn.py' has run successfully and saved the model to this path.")
        sys.exit(1)

    try:
        # It's crucial that this model was trained with Dropout layers
        cnn_model = load_model(cnn_model_path)
        print(f"Model loaded successfully: {cnn_model_path}")
        # Optional: Print model summary to verify Dropout layers
        # cnn_model.summary()
    except Exception as e:
        print(f"An unexpected error occurred during model loading from '{cnn_model_path}': {e}")
        sys.exit(1)

    # --- Quick Deterministic Accuracy Check ---
    # This helps confirm the model loads and predicts correctly in deterministic mode
    # before running MC Dropout. Should match performance from standard evaluation.
    print("\n--- Performing quick deterministic accuracy check (training=False) ---")
    if X_test_std_unbalanced is not None and X_test_std_unbalanced.size > 0:
        try:
            # Use training=False to disable dropout layers deterministically
            deterministic_probs_ub = cnn_model(X_test_std_unbalanced, training=False).numpy().flatten()
            deterministic_labels_ub = (deterministic_probs_ub > 0.5).astype(int)
            deterministic_accuracy_ub = accuracy_score(y_test_unbalanced, deterministic_labels_ub)
            print(f"Deterministic Accuracy on Unbalanced Test Set: {deterministic_accuracy_ub:.4f}")
            # This value should align with the standard classification evaluation accuracy on the unbalanced set.
        except Exception as e_det:
            print(f"Could not perform deterministic check on Unbalanced Set: {e_det}")
    else:
        print("Skipping deterministic check: Unbalanced test data not available.")
    print("--- End deterministic accuracy check ---\n")


    # --- Evaluate MC Dropout ---

    # Evaluate on the original UNBALANCED Test Set (if requested and data available)
    if evaluate_unbalanced and X_test_std_unbalanced is not None and X_test_std_unbalanced.size > 0:
        print("\n" + "="*50)
        print("--- Evaluating MC Dropout on UNBALANCED Test Set ---")
        cnn_metrics_mcd_ub = evaluate_mc_dropout_and_uq(
            model=cnn_model,
            X_data=X_test_std_unbalanced,
            y_data=y_test_unbalanced,
            patient_ids=patient_ids_test_unbalanced, # Pass patient IDs for saving CSV
            model_eval_label="CNN_MCD_Unbalanced", # Label for outputs
            n_mc_passes=n_mc_passes,
            output_csv_dir=output_csv_dir,
            raw_preds_save_dir=raw_preds_save_dir,
            save_detailed_csv=save_unbalanced_csv, # Use flag from argparse
            n_bootstrap=n_bootstrap,
            random_state=seed
        )
    else:
        print("\n" + "="*50)
        print("--- Skipping MC Dropout Evaluation on UNBALANCED Test Set ---")
        if not evaluate_unbalanced: print("Evaluation skipped by user flag.")
        if X_test_std_unbalanced is None or X_test_std_unbalanced.size == 0: print("Evaluation skipped: Data not available.")
        cnn_metrics_mcd_ub = None


    # Evaluate on the BALANCED (RUS) Test Set (if requested and data available)
    # Note: Patient IDs are typically not available or meaningful after RUS.
    if evaluate_balanced and X_test_std_rus is not None and X_test_std_rus.size > 0:
        print("\n" + "="*50)
        print("--- Evaluating MC Dropout on BALANCED (RUS) Test Set ---")
        cnn_metrics_mcd_bal = evaluate_mc_dropout_and_uq(
            model=cnn_model,
            X_data=X_test_std_rus,
            y_data=y_test_rus,
            patient_ids=None, # Patient IDs not available for RUS data
            model_eval_label="CNN_MCD_Balanced_RUS", # Label for outputs
            n_mc_passes=n_mc_passes,
            output_csv_dir=output_csv_dir,
            raw_preds_save_dir=raw_preds_save_dir,
            save_detailed_csv=save_balanced_csv, # Use flag from argparse
            n_bootstrap=n_bootstrap,
            random_state=seed
        )
    else:
        print("\n" + "="*50)
        print("--- Skipping MC Dropout Evaluation on BALANCED (RUS) Test Set ---")
        if not evaluate_balanced: print("Evaluation skipped by user flag.")
        if X_test_std_rus is None or X_test_std_rus.size == 0: print("Evaluation skipped: Data not available.")
        cnn_metrics_mcd_bal = None


    # --- Conclusion ---
    print("\n" + "="*70)
    print("MC Dropout Evaluation Script Finished.")
    print(f"Detailed results CSV saved in: '{output_csv_dir}' (if save_detailed_csv was True)")
    print(f"Raw MC predictions saved in: '{raw_preds_save_dir}' (if raw_preds_save_dir was specified)")
    print("="*70)


if __name__ == "__main__":
    # Setup argparse for command-line configuration
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CNN model using Monte Carlo Dropout for classification performance and uncertainty quantification."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=PROCESSED_DATA_DIR,
        help=f"Directory containing the processed .npy test datasets (default: '{PROCESSED_DATA_DIR}')"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=CNN_MODEL_PATH,
        help=f"Path to the trained CNN Keras model file (default: '{CNN_MODEL_PATH}'). Model must have Dropout layers."
    )
    parser.add_argument(
        "--n_mc_passes",
        type=int,
        default=N_MC_PASSES,
        help=f"Number of Monte Carlo passes to perform for prediction (default: {N_MC_PASSES})."
    )
    parser.add_argument(
        "--output_csv_dir",
        type=str,
        default=OUTPUT_CSV_DIR,
        help=f"Directory to save the detailed per-window UQ results CSV (default: '{OUTPUT_CSV_DIR}')"
    )
    parser.add_argument(
        "--raw_preds_save_dir",
        type=str,
        default=RAW_PREDICTIONS_SAVE_DIR,
        help=f"Optional directory to save the raw MC Dropout predictions (.npy file). Set to '' to not save. (default: '{RAW_PREDICTIONS_SAVE_DIR}')"
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help=f"Number of bootstrap samples for Confidence Interval calculation (default: {N_BOOTSTRAP})."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducibility of bootstrap sampling (default: {SEED})."
    )
    parser.add_argument(
        "--evaluate_unbalanced",
        action='store_true', # This creates a boolean flag, default is False
        help="Set this flag to run evaluation on the UNBALANCED test set."
    )
    parser.add_argument(
        "--no_evaluate_unbalanced",
        action='store_false', # If this flag is present, evaluate_unbalanced becomes False
        dest='evaluate_unbalanced', # Ties to the same variable as --evaluate_unbalanced
        help="Set this flag to SKIP evaluation on the UNBALANCED test set."
    )
    parser.set_defaults(evaluate_unbalanced=True) # Default to evaluate unbalanced if neither flag is set

    parser.add_argument(
        "--evaluate_balanced",
        action='store_true',
        help="Set this flag to run evaluation on the BALANCED (RUS) test set."
    )
    parser.add_argument(
        "--no_evaluate_balanced",
        action='store_false',
        dest='evaluate_balanced',
        help="Set this flag to SKIP evaluation on the BALANCED (RUS) test set."
    )
    parser.set_defaults(evaluate_balanced=True) # Default to evaluate balanced if neither flag is set

    parser.add_argument(
        "--save_unbalanced_csv",
        action='store_true',
        help="Set this flag to save the detailed per-window UQ results CSV for the UNBALANCED test set."
    )
    parser.set_defaults(save_unbalanced_csv=False) # Default to NOT save unbalanced CSV unless flag is set

    parser.add_argument(
        "--save_balanced_csv",
        action='store_true',
        help="Set this flag to save the detailed per-window UQ results CSV for the BALANCED (RUS) test set."
             "Note: Patient IDs are typically not available for the RUS set, so saving Patient_ID column will result in NaNs."
    )
    parser.set_defaults(save_balanced_csv=False) # Default to NOT save balanced CSV unless flag is set


    args = parser.parse_args()

    # Run the evaluation with parameters from argparse
    # Pass the boolean flags directly to the evaluation function
    run_mc_dropout_evaluation(
        processed_data_dir=args.data_dir,
        cnn_model_path=args.model_path,
        n_mc_passes=args.n_mc_passes,
        output_csv_dir=args.output_csv_dir,
        raw_preds_save_dir=args.raw_preds_save_dir if args.raw_preds_save_dir else None, # Pass None if empty string
        n_bootstrap=args.n_bootstrap,
        seed=args.seed, # Pass the argparse seed to the main function
        evaluate_unbalanced=args.evaluate_unbalanced,
        evaluate_balanced=args.evaluate_balanced,
        save_unbalanced_csv=args.save_unbalanced_csv,
        save_balanced_csv=args.save_balanced_csv
    )