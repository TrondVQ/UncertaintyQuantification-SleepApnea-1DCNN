#!/usr/bin/env python3

import sys
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import argparse
from typing import List, Dict, Optional, Any, Tuple
try:
    # Adjust import path if uq_techniques.py is in a different location
    from uq_techniques import deep_ensembles_predict, evaluate_uq_methods
except ImportError:
    print("Error: Could not import 'uq_techniques.py'. Please ensure it's in the correct path.")
    print("This script requires functions: deep_ensembles_predict and evaluate_uq_methods.")
    sys.exit(1)


# --- Configuration ---
# Default random seed for reproducibility (used for bootstrap sampling in evaluation)
SEED: int = 2025

# --- Define default paths and parameters (can be overridden by command-line args) ---
# Default directory where the processed .npy datasets (from prepare_final_datasets.py) are located
PROCESSED_DATA_DIR: str = "./final_processed_datasets" # Should match OUTPUT_DIR in prepare_final_datasets.py

# Default directory containing the trained CNN ensemble models
ENSEMBLE_MODEL_DIR: str = "./models/cnn_ensemble_no_pool" # Example directory

# Default prefix for the CNN ensemble model filenames (e.g., "AlCNN_smote_seed")
# The full path will be os.path.join(ENSEMBLE_MODEL_DIR, f"{CNN_MODEL_PREFIX}{i}.keras")
# where 'i' is the model index (0 to num_models-1).
# ** Ensure this prefix and the indexing logic match how your models were saved during training **
CNN_MODEL_PREFIX: str = "AlCNN_smote_seed" # Example prefix

# Default number of individual CNN models trained in the ensemble
NUM_CNN_ENSEMBLE_MEMBERS: int = 5

# Parameters for Bootstrap CIs (passed to evaluate_uq_methods)
N_BOOTSTRAP: int = 100

# --- Set seed for reproducibility (for numpy and tensorflow) ---
# This affects operations outside of specific model training, e.g., bootstrap sampling.
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Suppress excessive TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0=all, 1=info, 2=warnings, 3=errors


# --- Helper Function to Load Ensemble Models ---
def load_ensemble(
        model_dir: str,
        prefix: str,
        num_models: int
) -> List[tf.keras.Model]:
    """
    Loads a specified number of ensemble models from a directory based on a filename prefix and index.

    Assumes model files are named like '{prefix}0.keras', '{prefix}1.keras', ..., '{prefix}{num_models-1}.keras'.

    Args:
        model_dir: Directory containing the ensemble model files.
        prefix: The filename prefix for the models (e.g., "AlCNN_smote_seed").
        num_models: The expected number of models in the ensemble.

    Returns:
        A list of loaded TensorFlow Keras Model instances.

    Raises:
        FileNotFoundError: If the specified model directory does not exist or if any expected model file is not found.
        Exception: For other errors during model loading.
    """
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Ensemble model directory not found: {model_dir}")

    ensemble_models: List[tf.keras.Model] = []
    print(f"Loading {num_models} models from '{model_dir}' with prefix '{prefix}'...")

    for i in range(num_models):
        # Construct the model filename based on the prefix and index
        model_filename = f"{prefix}{i}.keras"
        model_path = os.path.join(model_dir, model_filename)

        if not os.path.exists(model_path):
            # Raise FileNotFoundError if a specific model file is missing
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            # Load the model. Use compile=False if you only need predictions
            # and don't want to re-compile (can be faster).
            # However, default load_model with compile=True is usually fine.
            model = load_model(model_path)
            print(f"Loaded model: {os.path.basename(model_path)}")
            ensemble_models.append(model)
        except Exception as e:
            # Catch other potential loading errors (e.g., corrupted file, custom object issues)
            print(f"ERROR loading model from {model_path}: {e}")
            raise # Re-raise the exception to indicate failure

    # Final check to ensure all models were loaded
    if len(ensemble_models) != num_models:
        # This case should ideally be caught by FileNotFoundError above, but adding for robustness.
        raise ValueError(f"Expected to load {num_models} models, but only successfully loaded {len(ensemble_models)}.")

    print(f"Successfully loaded all {len(ensemble_models)} ensemble models.")
    return ensemble_models


# --- Helper Function to Evaluate Ensemble and Calculate Aggregated UQ ---
def evaluate_ensemble_and_uq(
        models: List[tf.keras.Model],
        X_data: np.ndarray, # Expecting 3D data (samples, steps, features)
        y_data: np.ndarray,
        evaluation_label: str, # Unique label for this evaluation run
        n_bootstrap: int = N_BOOTSTRAP,
        random_state: int = SEED
) -> Optional[Dict[str, Any]]:
    """
    Evaluates a Deep Ensemble model on provided data, calculates aggregated UQ metrics,

    Args:
        models: List of loaded TensorFlow Keras Model instances forming the ensemble.
        X_data: Test features as a NumPy array, expected shape (samples, time_steps, features).
        y_data: True test labels as a NumPy array (samples,).
        evaluation_label: Unique label for this evaluation run (e.g., "CNN_Ensemble_Unbalanced").
                          Used for output file/directory naming by evaluate_uq_methods.
        n_bootstrap: Number of bootstrap samples for Confidence Interval calculation
                     (passed to evaluate_uq_methods).
        random_state: Random seed for reproducibility (passed to evaluate_uq_methods).

    Returns:
        A dictionary containing aggregated UQ metrics calculated by evaluate_uq_methods,
        or None if any step fails.
    """
    print(f"\n===== Evaluating Deep Ensemble: {evaluation_label} =====")

    if not models:
        print("Error: No models provided in the ensemble list.")
        return None
    if X_data is None or y_data is None or X_data.size == 0 or y_data.size == 0:
        print("Error: Input data (features or labels) is empty.")
        return None
    if X_data.ndim != 3:
        print(f"Error: Expected 3D input data (samples, steps, features), but got shape {X_data.shape}")
        return None
    if len(X_data) != len(y_data):
        print(f"Error: Mismatch in number of samples between X_data ({len(X_data)}) and y_data ({len(y_data)}).")
        return None


    # Get predictions from all ensemble members
    # deep_ensembles_predict function is expected to handle the ensemble prediction
    # and return predictions in a format suitable for evaluate_uq_methods (likely (n_models, samples))
    print(f"Running Deep Ensemble prediction with {len(models)} models on {len(y_data)} samples...")
    predictions: Optional[np.ndarray] = None
    try:
        # Pass the 3D data directly to deep_ensembles_predict
        # deep_ensembles_predict should return shape (n_models, samples, 1) or similar probabilities
        raw_ensemble_predictions = deep_ensembles_predict(models, X_data)

        if raw_ensemble_predictions is None:
            print("Error: deep_ensembles_predict returned None.")
            return None

        # evaluate_uq_methods likely expects probabilities of shape (n_predictions, n_samples) or (n_samples, n_predictions)
        # Assuming raw_ensemble_predictions shape is (n_models, n_samples, 1)
        # We need to pass (n_models, n_samples) by squeezing the last dimension
        if raw_ensemble_predictions.ndim == 3 and raw_ensemble_predictions.shape[2] == 1:
            predictions = raw_ensemble_predictions.squeeze(-1) # Shape becomes (n_models, n_samples)
        else:
            print(f"Warning: Unexpected shape from deep_ensembles_predict: {raw_ensemble_predictions.shape}. Proceeding but evaluate_uq_methods might fail.")
            predictions = raw_ensemble_predictions # Use as is, hope evaluate_uq_methods handles it

        if predictions.ndim != 2 or predictions.shape[0] != len(models) or predictions.shape[1] != len(y_data):
            print(f"Error: Predictions prepared for evaluate_uq_methods have unexpected shape: {predictions.shape}. Expected ({len(models)}, {len(y_data)}).")
            return None

        # Ensure probabilities are within [0, 1] range
        predictions = np.clip(predictions, 0.0, 1.0)

        print(f"Predictions prepared for evaluate_uq_methods have shape: {predictions.shape}")


    except Exception as e:
        print(f"Error during Deep Ensemble prediction: {e}")
        return None


    # Evaluate with comprehensive UQ metrics using the evaluate_uq_methods function
    # This function is expected to calculate aggregated metrics (like Brier, ECE,
    # various means of uncertainty metrics)
    print("\nCalculating and evaluating aggregated UQ metrics (requires evaluate_uq_methods)...")
    aggregated_metrics: Optional[Dict[str, Any]] = None
    try:
        aggregated_metrics = evaluate_uq_methods(
            predictions=predictions,           # Pass predictions (n_models, samples)
            y_true=y_data,                     # Pass true labels (samples,)
            evaluation_label=evaluation_label, # Pass label for output
            n_bootstrap=n_bootstrap,           # Set number of bootstrap samples for CIs
            random_state=random_state,         # Pass seed for reproducibility
        )

        # Print a brief summary of key aggregated metrics returned by evaluate_uq_methods
        if aggregated_metrics:
            print(f"\nAggregated Uncertainty Metrics Summary for {evaluation_label}:")
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
            print(f"Aggregated UQ metric evaluation function returned None or empty for {evaluation_label}.")

    except Exception as e:
        print(f"Error during aggregated UQ metric evaluation: {e}")
        aggregated_metrics = None # Ensure None is returned on error


    return aggregated_metrics


# ================== Main Execution ==================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("--- Starting Deep Ensemble Aggregated Evaluation Script ---")
    print("="*70)

    # Setup argparse for command-line configuration
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Deep Ensemble model and calculate aggregated uncertainty quantification metrics."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=PROCESSED_DATA_DIR,
        help=f"Directory containing the processed .npy test datasets (default: '{PROCESSED_DATA_DIR}')"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=ENSEMBLE_MODEL_DIR,
        help=f"Directory containing the trained CNN ensemble model files (default: '{ENSEMBLE_MODEL_DIR}')"
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
        default=CNN_MODEL_PREFIX,
        help=f"Filename prefix for ensemble models (default: '{CNN_MODEL_PREFIX}'). Files are expected to be named like prefix0.keras, prefix1.keras, etc."
    )
    parser.add_argument(
        "--num_members",
        type=int,
        default=NUM_CNN_ENSEMBLE_MEMBERS,
        help=f"Number of models in the CNN ensemble (default: {NUM_CNN_ENSEMBLE_MEMBERS})."
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


    args = parser.parse_args()

    # Set seeds again within main execution block for safety and consistency
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


    # --- Load Pre-processed Test Data ---
    print("\n=== Loading Pre-processed Window-Standardized Test Datasets ===")
    X_test_std_unbalanced: Optional[np.ndarray] = None
    y_test_unbalanced: Optional[np.ndarray] = None
    X_test_std_rus: Optional[np.ndarray] = None
    y_test_rus: Optional[np.ndarray] = None

    try:
        # Construct full data paths
        X_test_unbalanced_path = os.path.join(args.data_dir, 'X_test_win_std_unbalanced.npy')
        y_test_unbalanced_path = os.path.join(args.data_dir, 'y_test_unbalanced.npy')

        X_test_rus_path = os.path.join(args.data_dir, 'X_test_win_std_rus.npy')
        y_test_rus_path = os.path.join(args.data_dir, 'y_test_rus.npy')

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
        else:
            print(f"Warning: Unbalanced test data files not found in '{args.data_dir}'. Skipping load.")


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
        else:
            print(f"Warning: Balanced (RUS) test data files not found in '{args.data_dir}'. Skipping load.")


        if (X_test_std_unbalanced is None or X_test_std_unbalanced.size == 0) and (X_test_std_rus is None or X_test_std_rus.size == 0):
            print("Error: No test data (unbalanced or RUS) was loaded.")
            sys.exit(1)

    except Exception as e:
        print(f"An unexpected error occurred during test data loading: {e}")
        sys.exit(1)


    # --- Load CNN Ensemble Models ---
    print("\n=== Loading Trained CNN Ensemble Models ===")
    cnn_ensemble: Optional[List[tf.keras.Model]] = None
    try:
        # Load the ensemble using the helper function and argparse parameters
        cnn_ensemble = load_ensemble(
            model_dir=args.model_dir,
            prefix=args.model_prefix,
            num_models=args.num_members
        )
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"\nERROR loading ensemble models: {e}")
        print(f"Ensure the directory '{args.model_dir}' exists and contains {args.num_members} models matching the prefix '{args.model_prefix}' named like '{args.model_prefix}0.keras', '{args.model_prefix}1.keras', etc.")
        print("Ensure 'train_ensemble_cnn.py' has run successfully and saved models to this location with the correct naming.")
        sys.exit(1) # Exit if model loading fails


    # --- Evaluate CNN Ensemble (Aggregated UQ) ---
    cnn_metrics_ub: Optional[Dict[str, Any]] = None
    cnn_metrics_bal: Optional[Dict[str, Any]] = None

    # Evaluate on Unbalanced Test Set (if data available)
    if cnn_ensemble and X_test_std_unbalanced is not None and X_test_std_unbalanced.size > 0:
        print("\n" + "="*50)
        print("--- Evaluating Deep Ensemble on UNBALANCED Test Set (Aggregated UQ) ---")
        cnn_metrics_ub = evaluate_ensemble_and_uq(
            models=cnn_ensemble,
            X_data=X_test_std_unbalanced,
            y_data=y_test_unbalanced,
            evaluation_label="CNN_Ensemble_Unbalanced", # Label for output
            n_bootstrap=args.n_bootstrap, # Use bootstrap samples from argparse
            random_state=args.seed # Use seed from argparse
        )
    else:
        print("\n" + "="*50)
        print("--- Skipping Deep Ensemble Evaluation on UNBALANCED Test Set ---")
        if not cnn_ensemble: print("Evaluation skipped: Ensemble models not loaded.")
        if X_test_std_unbalanced is None or X_test_std_unbalanced.size == 0: print("Evaluation skipped: Data not available.")


    # Evaluate on Balanced (RUS) Test Set (if data available)
    if cnn_ensemble and X_test_std_rus is not None and X_test_std_rus.size > 0:
        print("\n" + "="*50)
        print("--- Evaluating Deep Ensemble on BALANCED (RUS) Test Set (Aggregated UQ) ---")
        cnn_metrics_bal = evaluate_ensemble_and_uq(
            models=cnn_ensemble,
            X_data=X_test_std_rus,
            y_data=y_test_rus,
            evaluation_label="CNN_Ensemble_Balanced_RUS", # Label for output
            n_bootstrap=args.n_bootstrap, # Use bootstrap samples from argparse
            random_state=args.seed # Use seed from argparse
        )
    else:
        print("\n" + "="*50)
        print("--- Skipping Deep Ensemble Evaluation on BALANCED (RUS) Test Set ---")
        if not cnn_ensemble: print("Evaluation skipped: Ensemble models not loaded.")
        if X_test_std_rus is None or X_test_std_rus.size == 0: print("Evaluation skipped: Data not available.")


    # --- Conclusion ---
    print("\n" + "="*70)
    print("Deep Ensemble Aggregated Evaluation Script Finished.")
    print("="*70)

    # The aggregated metrics are stored in cnn_metrics_ub and cnn_metrics_bal dictionaries
    # These can be used programmatically later if needed, but for this script, the analysis