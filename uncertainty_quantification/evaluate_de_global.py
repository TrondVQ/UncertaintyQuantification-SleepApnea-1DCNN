import sys
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from typing import List, Dict, Optional
from uq_techniques import deep_ensembles_predict, evaluate_uq_methods

# --- Configuration ---
SEED = 2025
NUM_MODELS_PER_TYPE = 20 # Number of CNN ensemble models
PROCESSED_DATA_DIR = "./processed_datasets"
# Output directory for UQ plots
OUTPUT_PLOT_DIR = "./uq_plots/deep_ensemble20"

# --- Model Loading ---
# Ensure this prefix matches how you saved your CNN ensemble models
CNN_MODEL_PREFIX = "./models/ensemble_cnn/AlCNN_smote_seed"

# Parameters for Bootstrap CIs (within evaluate_uq_methods)
N_BOOTSTRAP = 100

# --- Helper Functions ---
def load_ensemble(model_prefix: str, num_models: int) -> List[tf.keras.Model]:
    """Load ensemble models from disk based on prefix and index."""
    models = []
    print(f"Loading {num_models} models with prefix: {model_prefix}...")
    for i in range(num_models):
        model_path = f"{model_prefix}{i}.keras" # Assumes files like prefix0.keras, prefix1.keras,...
        try:
            print(f"  Loading: {model_path}")
            # Ensure custom objects are handled if needed, though likely not for basic CNN/Dropout
            models.append(load_model(model_path))
        except Exception as e:
            print(f"ERROR loading model {model_path}: {str(e)}")
            raise FileNotFoundError(f"Failed to load all ensemble models for prefix {model_prefix}") from e
    print(f"Loaded {len(models)} models successfully.")
    return models

def evaluate_ensemble(models: List[tf.keras.Model],
                      X_data: np.ndarray, # Expecting 3D data now
                      y_data: np.ndarray,
                      model_eval_label: str) -> Optional[Dict]:
    """Evaluates a single Deep Ensemble model type."""
    print(f"\n===== Evaluating Ensemble: {model_eval_label} =====")

    # Ensure data is 3D (samples, steps, features)
    if X_data.ndim != 3:
        print(f"ERROR: Expected 3D input data (samples, steps, features) for model, but got shape {X_data.shape}")
        return None

    # Get predictions from all ensemble members
    # deep_ensembles_predict uses model.predict which expects 3D input
    predictions = deep_ensembles_predict(models, X_data) # Pass 3D data directly

    if predictions is None:
        print(f"Prediction failed for {model_eval_label}")
        return None

    # Evaluate with comprehensive UQ metrics using the corrected evaluate_uq_methods
    metrics = evaluate_uq_methods(
        predictions=predictions,           # Shape (n_models, samples) after potential squeeze
        y_test=y_data,
        evaluation_label=model_eval_label, # Pass label for plots/output
        n_bootstrap=N_BOOTSTRAP,
        random_state=SEED,
        output_plot_dir=os.path.join(OUTPUT_PLOT_DIR, model_eval_label) # Save plots in subdirs
    )

    # --- ADDED: Optional Summary Printing (using corrected keys) ---
    if metrics:
        print(f"\nAggregated Uncertainty Metrics Summary for {model_eval_label}:")
        # Using mean values which account for CI calculation if bootstrap was successful
        print(f"- Overall Mean Variance: {metrics.get('overall_mean_variance_mean', metrics.get('overall_mean_variance', 'N/A')):.6f}")
        print(f"- Mean Total Predictive Entropy: {metrics.get('mean_total_pred_entropy_mean', metrics.get('mean_total_pred_entropy', 'N/A')):.4f}")
        print(f"- Mean Mutual Information: {metrics.get('mean_mutual_info_mean', metrics.get('mean_mutual_info', 'N/A')):.6f}")
        # Optionally print aleatoric part:
        print(f"- Mean Expected Aleatoric Entropy: {metrics.get('mean_expected_aleatoric_entropy_mean', metrics.get('mean_expected_aleatoric_entropy', 'N/A')):.4f}")
    else:
        print(f"UQ metric calculation failed for {model_eval_label}")
    # --- End Added Block ---

    return metrics

# ================== Main Execution ==================
if __name__ == "__main__":
    # Set seeds
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF info logs

    # --- Load Pre-processed Test Data ---
    print("\n=== Loading Pre-processed Window-Standardized Test Datasets ===")
    try:
        # MODIFIED: Load the window-standardized files
        X_test_std_unbalanced = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test_std_unbalanced.npy'))
        y_test_unbalanced = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test_unbalanced.npy'))
        X_test_std_rus = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test_std_rus.npy'))
        y_test_rus = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test_rus.npy'))
        print("Test datasets loaded successfully.")
        print(f"X_test_std_unbalanced shape: {X_test_std_unbalanced.shape}") # Should be (samples, 60, 4)
        print(f"X_test_std_rus shape: {X_test_std_rus.shape}") # Should be (samples, 60, 4)

        # --- Validation: Ensure data is 3D ---
        if X_test_std_unbalanced.ndim != 3 or X_test_std_rus.ndim != 3:
            raise ValueError("Loaded test data is not 3-dimensional (samples, steps, features). Check preprocessing script.")

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(f"Ensure corrected preprocessing script ('finalize_datasets.py' performing window-std) has run and files are in: {PROCESSED_DATA_DIR}")
        sys.exit(1)
    except ValueError as e:
        print(f"Data shape error: {e}")
        sys.exit(1)


    try:
        # --- Load Ensembles ---
        print("\n=== Loading Trained CNN Ensemble Models ===")
        # Ensure the prefix and NUM_MODELS_PER_TYPE match your trained models
        cnn_ensemble = load_ensemble(CNN_MODEL_PREFIX, NUM_MODELS_PER_TYPE)

        predictions = deep_ensembles_predict(cnn_ensemble, X_test_std_unbalanced)

        if predictions is None:
            print(f"DE prediction failed")
        """ 
        try:
            np.save("./de_raw_pred.npy", predictions)
            print(f"Saved raw MC predictions (shape: {predictions.shape}) to de_raw_pred")
        except Exception as e:
            print(f"ERROR saving raw MC predictions: {e}")
        """
        # --- Evaluate CNN Ensemble ---
        # Pass the 3D data directly
        cnn_metrics_ub = evaluate_ensemble(
            cnn_ensemble, X_test_std_unbalanced, y_test_unbalanced, "CNN_Ensemble_Unbalanced"
        )
        cnn_metrics_bal = evaluate_ensemble(
            cnn_ensemble, X_test_std_rus, y_test_rus, "CNN_Ensemble_Balanced_RUS"
        )

    except Exception as e:
        print(f"\nAn ERROR occurred during evaluation: {str(e)}")
        sys.exit(1)

    # --- Completion ---
    print("\n" + "="*50)
    print("Deep Ensemble Evaluation complete.")
    print(f"Plots saved in subdirectories under: {OUTPUT_PLOT_DIR}")
    # Added print statement example to show where results are stored:
    print("\nExample accessing results:")
    #if cnn_metrics_ub:
    #   print(f"CNN Ensemble Unbalanced - Overall Mean Variance: {cnn_metrics_ub.get('overall_mean_variance_mean', cnn_metrics_ub.get('overall_mean_variance', 'N/A')):.6f}")