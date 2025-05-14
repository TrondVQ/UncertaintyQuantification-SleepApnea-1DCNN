#!/usr/bin/env python3

import sys
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import argparse
from typing import List, Dict, Optional, Any, NoReturn, NoReturn, Tuple
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
PROCESSED_DATA_DIR: str = "./final_processed_datasets" # Should match OUTPUT_DIR in prepare_final_datasets.py

# Default path to the trained model file (should have Dropout layers)
CNN_MODEL_PATH: str = "./alarcon_cnn_model.keras" # Assuming this is a trained model

# Number of Monte Carlo passes to perform for each window prediction
N_MC_PASSES: int = 50

# Number of bootstrap samples for Confidence Interval calculation in aggregated UQ evaluation
N_BOOTSTRAP: int = 100

# Default directory for saving raw MC Dropout predictions (optional)
# Set to None by default if you don't want to save raw predictions
RAW_PREDICTIONS_SAVE_DIR: Optional[str] = "./raw_predictions/mc_dropout"


# --- Set seed for reproducibility (for numpy and tensorflow) ---
# This affects operations outside of specific model training, e.g., bootstrap sampling.
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Suppress excessive TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0=all, 1=info, 2=warnings, 3=errors


def evaluate_mc_dropout_and_uq_aggregated(
        model: Model,
        X_data: np.ndarray, # Expecting 3D data (samples, steps, features)
        y_data: np.ndarray,
        model_eval_label: str,
        n_mc_passes: int,
        raw_preds_save_dir: Optional[str] = None, # Optional directory to save raw predictions
        n_bootstrap: int = N_BOOTSTRAP,
        random_state: int = SEED
) -> Optional[Dict[str, Any]]:
    """
    Runs MC Dropout prediction, calculates aggregated UQ metrics using evaluate_uq_methods. Does NOT save per-window UQ results CSV.

    Args:
        model: Loaded Keras model with Dropout layers that should be active during prediction.
        X_data: Test features as a NumPy array, expected shape (samples, time_steps, features).
        y_data: True test labels as a NumPy array (samples,).
        model_eval_label: Unique label for this evaluation run (e.g., "CNN_MCD_Unbalanced").
                          Used for output filenames
        n_mc_passes: Number of Monte Carlo passes to run for each window.
        raw_preds_save_dir: Optional directory to save the raw MC Dropout predictions (.npy file). If None, raw predictions are not saved.
        n_bootstrap: Number of bootstrap samples for Confidence Interval calculation.
        random_state: Random seed for reproducibility of bootstrap sampling.

    Returns:
        A dictionary containing aggregated UQ metrics calculated by evaluate_uq_methods,
        or None if any step fails.
    """
    print(f"\n===== Running MC Dropout Aggregated Evaluation for: {model_eval_label} =====")

    if model is None:
        print("Error: No model provided for evaluation.")
        return None
    if X_data is None or y_data is None or X_data.size == 0 or y_data.size == 0:
        print("Error: Input data (features or labels) is empty.")
        return None
    if X_data.ndim != 3:
        print(f"Error: Expected 3D input data (samples, steps, features), but got shape {X_data.shape}.")
        print("Please ensure you are loading the correctly processed temporal data.")
        return None
    if len(X_data) != len(y_data):
        print(f"Error: Mismatch in number of samples between X_data ({len(X_data)}) and y_data ({len(y_data)}).")
        return None
    if n_mc_passes <= 0:
        print(f"Error: Number of MC passes must be positive, but got {n_mc_passes}.")
        return None


    # --- Step 1: Get MC Dropout raw predictions/probabilities ---
    # Use the mc_dropout_predict function. It should activate Dropout layers during prediction.
    # It is expected to return a numpy array of shape (n_pred, samples, 1) for binary classification.
    print(f"Running MC Dropout with {n_mc_passes} passes on {len(y_data)} samples...")
    mc_probabilities: Optional[np.ndarray] = None
    try:
        # Pass the 3D data directly to mc_dropout_predict
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
                raw_preds_filename = f"raw_mc_predictions_{model_eval_label}.npy" # Format filename with eval label
                raw_preds_filepath = os.path.join(raw_preds_save_dir, raw_preds_filename)
                print(f"\nSaving raw MC predictions to: {raw_preds_filepath}")
                np.save(raw_preds_filepath, mc_probabilities)
                print("Raw MC predictions saved.")
            except Exception as e:
                print(f"Warning: Failed to save raw MC predictions to '{raw_preds_save_dir}': {e}")


    except Exception as e:
        print(f"Error during MC Dropout prediction passes: {e}")
        return None


    # --- Step 2: Evaluate Aggregated UQ Metrics ---
    # This uses the evaluate_uq_methods function from uq_techniques.py.
    # Pass the raw MC probabilities and true labels.
    # This function is expected to calculate various aggregated metrics,
    # potentially including Brier Score, ECE, calibration curves, etc.,


    print("\nCalculating and evaluating aggregated UQ metrics (requires evaluate_uq_methods)...")
    aggregated_metrics: Optional[Dict[str, Any]] = None
    try:

        # Assuming evaluate_uq_methods expects predictions of shape (n_pred, samples)
        # We need to pass mc_probabilities squeezed to (n_mc_passes, num_windows)
        if mc_probabilities.ndim == 3 and mc_probabilities.shape[2] == 1:
            predictions_for_eval = mc_probabilities.squeeze(-1) # Shape becomes (n_mc_passes, num_windows)
        else:
            print(f"Warning: Unexpected shape for predictions before passing to evaluate_uq_methods: {mc_probabilities.shape}. Passing as is.")
            predictions_for_eval = mc_probabilities # Pass as is, hope evaluate_uq_methods handles it

        if predictions_for_eval.ndim != 2 or predictions_for_eval.shape[0] != n_mc_passes or predictions_for_eval.shape[1] != len(y_data):
            print(f"Error: Predictions prepared for evaluate_uq_methods have unexpected shape: {predictions_for_eval.shape}. Expected ({n_mc_passes}, {len(y_data)}).")
            return None


        aggregated_metrics = evaluate_aggregated_uq(
            predictions=predictions_for_eval,  # Pass predictions (n_mc_passes, samples)
            y_true=y_data,                     # Pass true labels (samples,)
            evaluation_label=model_eval_label, # Pass label for output
            n_bootstrap=n_bootstrap,           # Set number of bootstrap samples for CIs
            random_state=random_state,         # Pass seed for reproducibility
        )

        # Print a brief summary of key aggregated metrics returned by evaluate_aggregated_uq
        if aggregated_metrics:
            print(f"\nAggregated Uncertainty Metrics Summary for {model_eval_label}:")
            # Using mean keys which account for CI calculation if bootstrap was successful
            # Adapt these print statements based on what evaluate_uq_methods actually returns
            print(f"- Overall Mean Variance: {aggregated_metrics.get('overall_mean_pred_variance_mean', aggregated_metrics.get('overall_mean_pred_variance', 'N/A')):.6f}")
            print(f"- Mean Total Predictive Entropy: {aggregated_metrics.get('overall_mean_total_pred_entropy_mean', aggregated_metrics.get('overall_mean_total_pred_entropy', 'N/A')):.4f}")
            print(f"- Mean Mutual Information: {aggregated_metrics.get('overall_mean_mutual_information_mean', aggregated_metrics.get('overall_mean_mutual_information', 'N/A')):.6f}")
            print(f"- Mean Expected Aleatoric Entropy: {aggregated_metrics.get('overall_mean_expected_aleatoric_entropy_mean', aggregated_metrics.get('overall_mean_expected_aleatoric_entropy', 'N/A')):.4f}")
            # Add other relevant metrics like Brier Score, ECE if returned
            print(f"- Overall Brier Score: {aggregated_metrics.get('overall_brier_score', 'N/A'):.4f}")
            print(f"- Overall ECE: {aggregated_metrics.get('overall_ece', 'N/A'):.4f}")


        else:
            print(f"Aggregated UQ metric evaluation function returned None or empty for {model_eval_label}.")

    except Exception as e:
        print(f"Error during aggregated UQ metric evaluation {e}")
        aggregated_metrics = None # Ensure None is returned on error

    # Return the dictionary of aggregated metrics
    return aggregated_metrics


# --- Main Execution Block ---
def run_mc_dropout_aggregated_evaluation(
        processed_data_dir: str = PROCESSED_DATA_DIR,
        cnn_model_path: str = CNN_MODEL_PATH,
        n_mc_passes: int = N_MC_PASSES,
        raw_preds_save_dir: Optional[str] = RAW_PREDICTIONS_SAVE_DIR, # Optional directory to save raw predictions
        n_bootstrap: int = N_BOOTSTRAP,
        seed: int = SEED,
        evaluate_unbalanced: bool = True, # Flag to control evaluation on unbalanced set
        evaluate_balanced: bool = True    # Flag to control evaluation on balanced set
) -> NoReturn:
    """
    Main function to load data and a trained model, perform MC Dropout
    evaluation, calculate aggregated UQ metrics, and save results.
    This script does NOT save per-window UQ metrics to a CSV or use Patient IDs.

    Args:
        processed_data_dir: Directory containing the processed test data (.npy files).
        cnn_model_path: Path to the trained Keras model file.
        n_mc_passes: Number of Monte Carlo passes for prediction.
        raw_preds_save_dir: Optional directory to save the raw MC Dropout predictions (.npy file). If None, raw predictions are not saved.
        n_bootstrap: Number of bootstrap samples for CI calculation.
        seed: Random seed for reproducibility.
        evaluate_unbalanced: Whether to evaluate on the unbalanced test set.
        evaluate_balanced: Whether to evaluate on the balanced test set.
    """
    print("\n" + "="*70)
    print("--- Starting MC Dropout Aggregated Evaluation Script ---")
    print(f"Processed Data Dir: {processed_data_dir}")
    print(f"CNN Model Path: {cnn_model_path}")
    print(f"Number of MC Passes: {n_mc_passes}")
    print(f"Raw Predictions Save Dir: {raw_preds_save_dir if raw_preds_save_dir else 'Not saving'}")
    print(f"Bootstrap Samples for CI: {n_bootstrap}")
    print(f"Random Seed: {seed}")
    print(f"Evaluate Unbalanced Set: {evaluate_unbalanced}")
    print(f"Evaluate Balanced Set: {evaluate_balanced}")
    print("="*70)

    # --- Create Output Directories ---
    try:
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
    X_test_std_rus: Optional[np.ndarray] = None
    y_test_rus: Optional[np.ndarray] = None

    try:
        # Construct full data paths
        X_test_unbalanced_path = os.path.join(processed_data_dir, 'X_test_win_std_unbalanced.npy')
        y_test_unbalanced_path = os.path.join(processed_data_dir, 'y_test_unbalanced.npy')

        X_test_rus_path = os.path.join(processed_data_dir, 'X_test_win_std_rus.npy')
        y_test_rus_path = os.path.join(processed_data_dir, 'y_test_rus.npy')


        # Load data - check existence before loading
        if os.path.exists(X_test_unbalanced_path) and os.path.exists(y_test_unbalanced_path):
            X_test_std_unbalanced = np.load(X_test_unbalanced_path)
            y_test_unbalanced = np.load(y_test_unbalanced_path)
            print("Unbalanced test datasets loaded successfully.")
            print(f"X_test_std_unbalanced shape: {X_test_std_unbalanced.shape}") # Should be (samples, 60, 4)
            print(f"y_test_unbalanced shape: {y_test_unbalanced.shape}")
            # Basic validation
            if X_test_std_unbalanced.size == 0 or y_test_unbalanced.size == 0:
                print("Warning: Unbalanced test data is empty after loading.")
                X_test_std_unbalanced, y_test_unbalanced = None, None # Set to None if empty
            elif X_test_std_unbalanced.ndim != 3:
                print(f"Error: Expected 3D Unbalanced test data, but got shape {X_test_std_unbalanced.shape}.")
                print("Skipping evaluation on Unbalanced set.")
                X_test_std_unbalanced, y_test_unbalanced = None, None # Set to None if shape is wrong
        else:
            print(f"Warning: Unbalanced test data files not found in '{processed_data_dir}'. Skipping load.")


        if os.path.exists(X_test_rus_path) and os.path.exists(y_test_rus_path):
            X_test_std_rus = np.load(X_test_rus_path)
            y_test_rus = np.load(y_test_rus_path)
            print("Balanced (RUS) test datasets loaded successfully.")
            print(f"X_test_std_rus shape: {X_test_std_rus.shape}") # Should be (samples, 60, 4)
            print(f"y_test_rus shape: {y_test_rus.shape}")
            # Basic validation
            if X_test_std_rus.size == 0 or y_test_rus.size == 0:
                print("Warning: RUS balanced test data is empty after loading. Evaluation on RUS set may be skipped.")
                X_test_std_rus, y_test_rus = None, None # Set to None if empty
            elif X_test_std_rus.ndim != 3:
                print(f"Error: Expected 3D RUS test data, but got shape {X_test_std_rus.shape}.")
                print("Skipping evaluation on RUS set.")
                X_test_std_rus, y_test_rus = None, None # Set to None if shape is wrong
        else:
            print(f"Warning: Balanced (RUS) test data files not found in '{processed_data_dir}'. Skipping load.")


        if (X_test_std_unbalanced is None or X_test_std_unbalanced.size == 0) and (X_test_std_rus is None or X_test_std_rus.size == 0):
            print("Error: No valid test data (unbalanced or RUS) was loaded.")
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
        # and that load_model correctly handles custom objects/layers if any.
        cnn_model = load_model(cnn_model_path)
        print(f"Model loaded successfully: {cnn_model_path}")
        # Optional: Print model summary to verify Dropout layers are present
        # cnn_model.summary()
    except Exception as e:
        print(f"An unexpected error occurred during model loading from '{cnn_model_path}': {e}")
        sys.exit(1)


    # --- Evaluate MC Dropout ---

    # Evaluate on the original UNBALANCED Test Set (if requested and data available)
    if evaluate_unbalanced and X_test_std_unbalanced is not None and X_test_std_unbalanced.size > 0:
        print("\n" + "="*50)
        print("--- Evaluating MC Dropout on UNBALANCED Test Set (Aggregated UQ) ---")
        cnn_metrics_mcd_ub = evaluate_mc_dropout_and_uq_aggregated(
            model=cnn_model,
            X_data=X_test_std_unbalanced,
            y_data=y_test_unbalanced,
            model_eval_label="CNN_MCD_Unbalanced", # Label for outputs
            n_mc_passes=n_mc_passes,
            raw_preds_save_dir=raw_preds_save_dir,
            n_bootstrap=n_bootstrap,
            random_state=seed
        )
    else:
        print("\n" + "="*50)
        print("--- Skipping MC Dropout Evaluation on UNBALANCED Test Set ---")
        if not evaluate_unbalanced: print("Evaluation skipped by user flag.")
        if X_test_std_unbalanced is None or X_test_std_unbalanced.size == 0: print("Evaluation skipped: Data not available or has incorrect shape.")
        cnn_metrics_mcd_ub = None


    # Evaluate on the BALANCED (RUS) Test Set (if requested and data available)
    if evaluate_balanced and X_test_std_rus is not None and X_test_std_rus.size > 0:
        print("\n" + "="*50)
        print("--- Evaluating MC Dropout on BALANCED (RUS) Test Set (Aggregated UQ) ---")
        cnn_metrics_mcd_bal = evaluate_mc_dropout_and_uq_aggregated(
            model=cnn_model,
            X_data=X_test_std_rus,
            y_data=y_test_rus,
            model_eval_label="CNN_MCD_Balanced_RUS", # Label for outputs
            n_mc_passes=n_mc_passes,
            raw_preds_save_dir=raw_preds_save_dir,
            n_bootstrap=n_bootstrap,
            random_state=seed
        )
    else:
        print("\n" + "="*50)
        print("--- Skipping MC Dropout Evaluation on BALANCED (RUS) Test Set ---")
        if not evaluate_balanced: print("Evaluation skipped by user flag.")
        if X_test_std_rus is None or X_test_std_rus.size == 0: print("Evaluation skipped: Data not available or has incorrect shape.")
        cnn_metrics_mcd_bal = None


    # --- Conclusion ---
    print("\n" + "="*70)
    print("MC Dropout Aggregated Evaluation Script Finished.")
    print(f"Raw MC predictions saved in: '{raw_preds_save_dir}' (if raw_preds_save_dir was specified)")
    print("="*70)

    # The aggregated metrics are stored in cnn_metrics_mcd_ub and cnn_metrics_mcd_bal dictionaries
    # These can be used programmatically later if needed.