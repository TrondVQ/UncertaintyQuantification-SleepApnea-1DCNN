#!/usr/bin/env python3

import sys
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
from typing import List, NoReturn, Dict, Any, Optional, Tuple

try:
    # Adjust import path if uq_techniques.py is in a different location
    from uq_techniques import deep_ensembles_predict, evaluate_uq_methods as evaluate_aggregated_uq
except ImportError:
    print("Error: Could not import 'uq_techniques.py'. Please ensure it's in the correct path.")
    print("This script requires functions: deep_ensembles_predict and evaluate_uq_methods.")
    sys.exit(1)


# --- Configuration ---
# Default random seed for reproducibility (used for bootstrap sampling in evaluation)
SEED: int = 2025

# --- Define default paths and parameters (can be overridden by command-line args) ---
# Default directory where the processed .npy datasets (from prepare_final_datasets.py) are located
PROCESSED_DATA_DIR: str = "./final_processed_datasets"

# Default directory containing the trained ensemble models
ENSEMBLE_MODEL_DIR: str = "./models/cnn_ensemble_no_pool" # Should match save_dir in train_ensemble_cnn.py

# Pattern for the ensemble model filenames (e.g., "AlCNN_smote_seed{seed}.keras")
# The {seed} placeholder will be formatted with the actual seed number used during training.
# Note: The original code used pattern.format(i+5), suggesting seeds might start from 5.
# Ensure this pattern matches how your ensemble models were saved.
ENSEMBLE_MODEL_PATTERN: str = "AlCNN_smote_seed{}.keras"

# Number of individual models trained in the ensemble
NUM_ENSEMBLE_MEMBERS: int = 5

# Number of bootstrap samples for Confidence Interval calculation in aggregated UQ evaluation
N_BOOTSTRAP: int = 100


# Default directory for saving detailed per-window UQ results CSV
OUTPUT_CSV_DIR: str = "./uq_results/deep_ensemble" # More generic name

# Base filename for the detailed per-window results CSV
DETAILED_CSV_FILENAME_BASE: str = "detailed_results_{}.csv" # {} will be formatted with eval_label

# --- Set seed for reproducibility (for numpy and tensorflow) ---
# This affects operations outside of specific model training, e.g., bootstrap sampling.
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Suppress excessive TensorFlow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 0=all, 1=info, 2=warnings, 3=errors


# --- Helper function to load ensemble models ---
def load_ensemble(
        model_dir: str,
        pattern: str,
        num_members: int,
        seed_base_offset: int = 0 # Offset to apply to seed index if pattern uses non-zero base (e.g., seeds 5-9)
) -> List[tf.keras.Model]:
    """
    Loads a specified number of ensemble models from a directory based on a filename pattern.

    Args:
        model_dir: Directory containing the ensemble model files.
        pattern: Filename pattern with a single '{}' placeholder for the seed number.
        num_members: The expected number of models in the ensemble.
        seed_base_offset: An integer offset added to the loop counter (0 to num_members-1)
                          to get the seed used in the filename pattern. Use if seeds
                          in filenames are not sequential starting from 0 (e.g., seeds 5, 6, ...).

    Returns:
        A list of loaded TensorFlow Keras Model instances.

    Raises:
        FileNotFoundError: If any expected model file is not found.
        ValueError: If the number of successfully loaded models does not match num_members.
        Exception: For other errors during model loading.
    """
    ensemble_models: List[tf.keras.Model] = []
    print(f"Loading {num_members} ensemble models from '{model_dir}' with pattern '{pattern}'...")

    for i in range(num_members):
        # Construct the model filename based on the pattern and the seed index + offset
        seed_in_filename = i + seed_base_offset
        model_filename = pattern.format(seed_in_filename)
        model_path = os.path.join(model_dir, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            # Load the model
            model = load_model(model_path)
            print(f"Loaded model: {os.path.basename(model_path)}")
            ensemble_models.append(model)
        except Exception as e:
            # Catch other potential loading errors (e.g., corrupted file)
            print(f"ERROR loading model from {model_path}: {e}")
            raise # Re-raise the exception to indicate failure

    # Final check to ensure all models were loaded
    if len(ensemble_models) != num_members:
        # This case should ideally be caught by FileNotFoundError above, but adding for robustness.
        raise ValueError(f"Expected to load {num_members} models, but only successfully loaded {len(ensemble_models)}.")

    print(f"Successfully loaded all {len(ensemble_models)} ensemble models.")
    return ensemble_models


def evaluate_deep_ensemble_and_uq(
        ensemble_models: List[tf.keras.Model],
        X_data: np.ndarray,
        y_data: np.ndarray,
        patient_ids: Optional[np.ndarray], # Patient IDs are optional for evaluation logic
        model_eval_label: str,
        output_csv_dir: str,
        save_detailed_csv: bool = False,
        n_bootstrap: int = N_BOOTSTRAP,
        random_state: int = SEED
) -> Optional[Dict[str, Any]]:
    """
    Runs Deep Ensemble prediction, calculates per-window UQ metrics, optionally
    saves detailed results, and evaluates aggregated UQ metrics.

    Args:
        ensemble_models: List of loaded Keras models forming the ensemble.
        X_data: Test features as a NumPy array (samples, time_steps, features).
        y_data: True test labels as a NumPy array (samples,).
        patient_ids: Patient IDs corresponding to X_data as a NumPy array (samples,) or None.
        model_eval_label: Unique label for this evaluation run (e.g., "CNN_DE_Unbalanced").
                          Used for output filenames.
        output_csv_dir: Directory to save the detailed per-window results CSV (if save_detailed_csv is True).
        save_detailed_csv: Whether to save the detailed per-window results CSV.
        n_bootstrap: Number of bootstrap samples for Confidence Interval calculation.
        random_state: Random seed for reproducibility of bootstrap sampling.

    Returns:
        A dictionary containing aggregated UQ metrics from evaluate_aggregated_uq,
        or None if any step fails.
    """
    print(f"\n===== Running Deep Ensemble Evaluation for: {model_eval_label} =====")
    num_models = len(ensemble_models)
    if num_models == 0:
        print("ERROR: No models provided in the ensemble list.")
        return None
    if X_data is None or y_data is None or X_data.size == 0 or y_data.size == 0:
        print("Error: Input data (features or labels) is empty.")
        return None
    if patient_ids is not None and len(patient_ids) != len(y_data):
        print(f"Warning: Mismatch in length between patient_ids ({len(patient_ids)}) and y_data ({len(y_data)}).")
        # Decide if this should be an error or just a warning. Proceeding with warning.


    # --- Step 1: Get Deep Ensemble raw predictions/probabilities ---
    # Use the function from uq_techniques.py
    print(f"Running Deep Ensemble prediction with {num_models} models on {len(y_data)} samples...")
    try:
        # deep_ensembles_predict should return shape (num_models, num_windows, 1)
        ensemble_probabilities: np.ndarray = deep_ensembles_predict(ensemble_models, X_data)

        if ensemble_probabilities is None or ensemble_probabilities.ndim != 3 or \
                ensemble_probabilities.shape[0] != num_models or ensemble_probabilities.shape[1] != len(y_data) or \
                ensemble_probabilities.shape[2] != 1:
            print(f"Error: deep_ensembles_predict failed or returned unexpected shape: {ensemble_probabilities.shape if ensemble_probabilities is not None else 'None'}")
            print(f"Expected shape: ({num_models}, {len(y_data)}, 1).")
            return None

        # Ensure probabilities are within [0, 1] range due to potential floating point issues
        ensemble_probabilities = np.clip(ensemble_probabilities, 0.0, 1.0)
        print(f"Deep Ensemble probabilities shape: {ensemble_probabilities.shape}") # Should be (num_models, num_windows, 1)

    except Exception as e:
        print(f"Error during Deep Ensemble prediction: {e}")
        return None

    # --- Step 2: Calculate Per-Window UQ Metrics from Ensemble Probabilities ---
    print("Calculating per-window UQ metrics...")
    try:
        # Mean probability across models for each window (for final prediction and expected value)
        mean_probs_per_window: np.ndarray = np.mean(ensemble_probabilities, axis=0).flatten() # Shape: (num_windows,)

        # Predictive variance across models for each window (Variance of means)
        pred_variance_per_window: np.ndarray = np.var(ensemble_probabilities, axis=0).flatten() # Shape: (num_windows,)

        # Predictive entropy from the mean probability (Total Predictive Entropy approximation for binary)
        # H(Y|x) approx H(E[Y|x]) for binary classification
        epsilon = 1e-9 # Small value to avoid log(0) issues
        # Ensure values are within (0, 1) before log
        mean_probs_clipped = np.clip(mean_probs_per_window, epsilon, 1 - epsilon)
        pred_entropy_per_window: np.ndarray = - (mean_probs_clipped * np.log2(mean_probs_clipped) + \
                                                 (1 - mean_probs_clipped) * np.log2(1 - mean_probs_clipped))

        # Calculate Expected Aleatoric Entropy for each window: E[H(Y|x, w)] approx 1/M sum(H(Y|x, wi))
        # H(Y|x, wi) for binary is - (pi*log2(pi) + (1-pi)*log2(1-pi)) where pi is probability from model i
        # Need to calculate entropy for each model's prediction per window, then average.
        entropies_per_model_per_window: np.ndarray = - (ensemble_probabilities * np.log2(ensemble_probabilities + epsilon) + \
                                                        (1 - ensemble_probabilities) * np.log2(1 - ensemble_probabilities + epsilon)) # Shape: (num_models, num_windows, 1)
        expected_aleatoric_entropy_per_window: np.ndarray = np.mean(entropies_per_model_per_window, axis=0).flatten() # Shape: (num_windows,)

        # Calculate Mutual Information for each window: MI = H(Y|x) - E[H(Y|x, w)]
        mutual_information_per_window: np.ndarray = pred_entropy_per_window - expected_aleatoric_entropy_per_window
        # Ensure MI is not negative due to floating point errors
        mutual_information_per_window[mutual_information_per_window < 0] = 0


        # Final predicted label based on the mean probability across models
        final_predicted_labels: np.ndarray = (mean_probs_per_window > 0.5).astype(int)

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
            'Predicted_Probability': mean_probs_per_window, # This is the mean probability across the ensemble
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
    # Pass the raw ensemble probabilities and true labels.
    # This function is expected to calculate various aggregated metrics,
    # potentially including Brier Score, ECE, ROC AUC, AUC-PR, etc.,
    # in relation to uncertainty quantiles/bins

    print("\nCalculating and evaluating aggregated UQ metrics (requires evaluate_uq_methods)...")
    aggregated_metrics: Optional[Dict[str, Any]] = None
    try:

        aggregated_metrics = evaluate_aggregated_uq(
            predictions=ensemble_probabilities.squeeze(-1), # evaluate_uq_methods might expect (num_models, samples)
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


    # Return the dictionary of aggregated metrics and/or the detailed DataFrame if needed by calling script
    # For this script, we primarily save/print, so returning metrics is optional but good practice.
    return aggregated_metrics


# --- Main Execution Block ---
def run_ensemble_evaluation(
        processed_data_dir: str = PROCESSED_DATA_DIR,
        ensemble_model_dir: str = ENSEMBLE_MODEL_DIR,
        ensemble_model_pattern: str = ENSEMBLE_MODEL_PATTERN,
        num_ensemble_members: int = NUM_ENSEMBLE_MEMBERS,
        seed_base_offset: int = 0, # Offset for loading models if seed in filename isn't 0-based
        output_csv_dir: str = OUTPUT_CSV_DIR,
        n_bootstrap: int = N_BOOTSTRAP,
        seed: int = SEED
) -> NoReturn:
    """
    Main function to load data, load ensemble models, perform Deep Ensemble
    evaluation, calculate UQ metrics, and save results.

    Args:
        processed_data_dir: Directory containing the processed test data (.npy files).
        ensemble_model_dir: Directory containing the trained ensemble model files.
        ensemble_model_pattern: Filename pattern for ensemble models.
        num_ensemble_members: Number of models in the ensemble.
        seed_base_offset: Offset for model seed indexing in filenames.
        output_csv_dir: Directory to save detailed per-window results CSV.
        n_bootstrap: Number of bootstrap samples for CI calculation.
        seed: Random seed for reproducibility.
    """
    print("\n" + "="*70)
    print("--- Starting Deep Ensemble Evaluation Script ---")
    print(f"Processed Data Dir: {processed_data_dir}")
    print(f"Ensemble Model Dir: {ensemble_model_dir}")
    print(f"Ensemble Model Pattern: {ensemble_model_pattern}")
    print(f"Number of Ensemble Members: {num_ensemble_members}")
    print(f"Model Seed Offset: {seed_base_offset}")
    print(f"Output CSV Dir: {output_csv_dir}")
    print(f"Bootstrap Samples for CI: {n_bootstrap}")
    print(f"Random Seed: {seed}")
    print("="*70)

    # --- Create Output Directories ---
    try:
        os.makedirs(output_csv_dir, exist_ok=True)
        print(f"Output CSV directory '{output_csv_dir}' ensured to exist.")
    except Exception as e:
        print(f"Error creating output directories: {e}")
        sys.exit(1)


    # --- Load Pre-processed Test Data ---
    print("\nLoading pre-processed test datasets...")
    try:
        # Construct full data paths
        X_test_unbalanced_path = os.path.join(processed_data_dir, 'X_test_win_std_unbalanced.npy')
        y_test_unbalanced_path = os.path.join(processed_data_dir, 'y_test_unbalanced.npy')
        patient_ids_test_unbalanced_path = os.path.join(processed_data_dir, 'patient_ids_test_unbalanced.npy') # Patient IDs for Unbalanced set

        X_test_rus_path = os.path.join(processed_data_dir, 'X_test_win_std_rus.npy')
        y_test_rus_path = os.path.join(processed_data_dir, 'y_test_rus.npy')
        # Patient IDs for RUS set might be needed if you evaluate RUS at patient level later


        # Load the data
        X_test_std_unbalanced: np.ndarray = np.load(X_test_unbalanced_path)
        y_test_unbalanced: np.ndarray = np.load(y_test_unbalanced_path)
        patient_ids_test_unbalanced: np.ndarray = np.load(patient_ids_test_unbalanced_path) # Load Patient IDs

        # Load Balanced (RUS) Test Set
        X_test_std_rus: np.ndarray = np.load(X_test_rus_path)
        y_test_rus: np.ndarray = np.load(y_test_rus_path)
        # patient_ids_test_rus = np.load(os.path.join(processed_data_dir, 'patient_ids_test_rus.npy')) # Load if available and needed

        print("Test datasets loaded successfully.")
        print(f"X_test_std_unbalanced shape: {X_test_std_unbalanced.shape}")
        print(f"y_test_unbalanced shape: {y_test_unbalanced.shape}")
        print(f"Patient IDs (Unbalanced) shape: {patient_ids_test_unbalanced.shape}")
        print(f"X_test_std_rus shape: {X_test_std_rus.shape}")
        print(f"y_test_rus shape: {y_test_rus.shape}")

        # Basic validation
        if X_test_std_unbalanced.size == 0 or y_test_unbalanced.size == 0 or patient_ids_test_unbalanced.size == 0:
            print("Error: Unbalanced test data or patient IDs are empty after loading.")
            sys.exit(1)
        if X_test_std_rus.size == 0 or y_test_rus.size == 0:
            print("Warning: RUS balanced test data is empty after loading. Evaluation on RUS set may be skipped.")


    except FileNotFoundError as e:
        print(f"\nError loading test data: {e}")
        print(f"Ensure the directory '{processed_data_dir}' exists and contains the expected .npy files.")
        print("Ensure 'prepare_final_datasets.py' has been run successfully.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during test data loading: {e}")
        sys.exit(1)


    # --- Load the Ensemble Models ---
    try:
        ensemble_models: List[tf.keras.Model] = load_ensemble(
            model_dir=ensemble_model_dir,
            pattern=ensemble_model_pattern,
            num_members=num_ensemble_members,
            seed_base_offset=seed_base_offset # Pass the offset if needed
        )
    except FileNotFoundError as e:
        print(f"\nError loading ensemble models: {e}")
        print(f"Ensure the directory '{ensemble_model_dir}' exists and contains {num_ensemble_members} models matching the pattern '{ensemble_model_pattern}'.")
        print("Ensure 'train_ensemble_cnn.py' has run successfully.")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError loading ensemble models: {e}")
        print("Check if the number of ensemble members matches the models found or if the pattern/offset is correct.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during ensemble model loading: {e}")
        sys.exit(1)


    # --- Evaluate Deep Ensemble ---

    # Evaluate on the original UNBALANCED Test Set
    # Save detailed CSV for this one as it contains patient IDs
    print("\n" + "="*50)
    print("--- Evaluating Deep Ensemble on UNBALANCED Test Set ---")
    if X_test_std_unbalanced.size > 0:
        cnn_metrics_de_ub = evaluate_deep_ensemble_and_uq(
            ensemble_models=ensemble_models,
            X_data=X_test_std_unbalanced,
            y_data=y_test_unbalanced,
            patient_ids=patient_ids_test_unbalanced, # Pass patient IDs for saving CSV
            model_eval_label="CNN_DE_Unbalanced", # Label for outputs
            output_csv_dir=output_csv_dir,
            save_detailed_csv=True, # Save the detailed per-window CSV for the unbalanced set
            n_bootstrap=n_bootstrap,
            random_state=seed
        )
    else:
        print("Skipping evaluation on UNBALANCED Test Set: Data is empty.")
        cnn_metrics_de_ub = None


    # Evaluate on the BALANCED (RUS) Test Set
    # Detailed CSV usually not needed for this, as patient IDs are not preserved by RUS
    print("\n" + "="*50)
    print("--- Evaluating Deep Ensemble on BALANCED (RUS) Test Set ---")
    if X_test_std_rus.size > 0:
        cnn_metrics_de_bal = evaluate_deep_ensemble_and_uq(
            ensemble_models=ensemble_models,
            X_data=X_test_std_rus,
            y_data=y_test_rus,
            patient_ids=None, # Patient IDs not available for RUS data
            model_eval_label="CNN_DE_Balanced_RUS", # Label for outputs
            output_csv_dir=output_csv_dir, # Still pass, even if not saving CSV
            save_detailed_csv=False, # Usually False for RUS balanced set
            n_bootstrap=n_bootstrap,
            random_state=seed
        )
    else:
        print("Skipping evaluation on BALANCED (RUS) Test Set: Data is empty.")
        cnn_metrics_de_bal = None


    # --- Conclusion ---
    print("\n" + "="*70)
    print("Deep Ensemble Evaluation Script Finished.")
    print(f"Detailed results CSV saved in: '{output_csv_dir}' (if save_detailed_csv was True)")
    print("="*70)


if __name__ == "__main__":
    # Setup argparse for command-line configuration
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Deep Ensemble model for classification performance and uncertainty quantification."
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
        help=f"Directory containing the trained ensemble Keras model files (default: '{ENSEMBLE_MODEL_DIR}')"
    )
    parser.add_argument(
        "--model_pattern",
        type=str,
        default=ENSEMBLE_MODEL_PATTERN,
        help=f"Filename pattern for ensemble models (default: '{ENSEMBLE_MODEL_PATTERN}'). Use '{{}}' as a placeholder for the seed/index."
    )
    parser.add_argument(
        "--num_members",
        type=int,
        default=NUM_ENSEMBLE_MEMBERS,
        help=f"Number of models in the ensemble (default: {NUM_ENSEMBLE_MEMBERS})."
    )
    parser.add_argument(
        "--model_seed_offset",
        type=int,
        default=0, # Default offset 0 assumes seeds in filename are 0, 1, 2...
        help="Integer offset to apply when generating model filenames from the loop index (0 to num_members-1). "
             "Use if seeds in filenames are not 0-based (e.g., use --model_seed_offset 5 for seeds 5, 6,...). (default: 0)"
    )
    parser.add_argument(
        "--output_csv_dir",
        type=str,
        default=OUTPUT_CSV_DIR,
        help=f"Directory to save the detailed per-window UQ results CSV (default: '{OUTPUT_CSV_DIR}')"
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
        "--save_unbalanced_csv",
        action='store_true', # This creates a boolean flag, default is False
        help="Set this flag to save the detailed per-window UQ results CSV for the UNBALANCED test set."
    )
    parser.add_argument(
        "--save_balanced_csv",
        action='store_true',
        help="Set this flag to save the detailed per-window UQ results CSV for the BALANCED (RUS) test set."
             "Note: Patient IDs are typically not available for the RUS set."
    )


    args = parser.parse_args()

    # Run the evaluation with parameters from argparse
    run_ensemble_evaluation(
        processed_data_dir=args.data_dir,
        ensemble_model_dir=args.model_dir,
        ensemble_model_pattern=args.model_pattern,
        num_ensemble_members=args.num_members,
        seed_base_offset=args.model_seed_offset,
        output_csv_dir=args.output_csv_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed # Pass the argparse seed to the main function
    )

    # Note: The save_unbalanced_csv and save_balanced_csv flags are handled
    # directly within the evaluate_deep_ensemble_and_uq calls based on args.
    # You might pass args directly to the evaluation function if it uses more flags.
    # For now, explicitly passing the save flags below the main function call.


    # Re-evaluate specifically to save CSVs if flags are set
    # Load data and models again if needed, or pass them from the main function
    # For simplicity here, I'll structure it so the main function handles passing the flag
    # I'll modify evaluate_deep_ensemble_and_uq to accept the save flag and remove the
    # separate calls here. The flags will be passed when evaluate_deep_ensemble_and_uq is called inside run_ensemble_evaluation.