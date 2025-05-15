import sys
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from uq_techniques import mc_dropout_predict, evaluate_uq_methods as evaluate_uq_from_preds

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
OUTPUT_PLOT_DIR = "./uq_plots/mc_dropout_new"

# Set seed for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Suppress excessive TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- 1. Load Pre-processed Test Data ---
print("Loading pre-processed test datasets...")
try:
    # Load Unbalanced Test Set
    X_test_std_unbalanced = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test_std_unbalanced.npy'))
    y_test_unbalanced = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test_unbalanced.npy'))
    # Load Balanced (RUS) Test Set
    X_test_std_rus = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test_std_rus.npy'))
    y_test_rus = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test_rus.npy'))
    print("Test datasets loaded successfully.")
    print(f"X_test_std_unbalanced shape: {X_test_std_unbalanced.shape}")
    print(f"X_test_std_rus shape: {X_test_std_rus.shape}")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure 'finalize_datasets.py' has been run successfully and files are in the correct directory.")
    sys.exit(1) # Exit if data isn't found


def evaluate_mc_dropout(model, X_data_reshaped, y_data, model_eval_label: str):
    """
    Runs MC Dropout prediction and evaluates UQ metrics.

    Args:
        model: Loaded Keras model with Dropout layers.
        X_data_reshaped: Reshaped test features (samples, 1, features).
        y_data: True test labels.
        model_eval_label (str): Unique label for this evaluation run (e.g., "CNN_MCD_Unbalanced").

    Returns:
        Dictionary of UQ metrics or None if fails.
    """
    print(f"\n===== Running MC Dropout Evaluation for: {model_eval_label} =====")

    # Get MC Dropout predictions
    # mc_dropout_predict function expects (samples, steps, features)
    predictions = mc_dropout_predict(model, X_data_reshaped, n_pred=N_MC_PASSES)

    if predictions is None:
        print(f"MC Dropout prediction failed for {model_eval_label}")
        return None

    # Evaluate with comprehensive UQ metrics
    # Assuming evaluate_uq_from_preds is the updated function from uq_utils
    metrics = evaluate_uq_from_preds(
        predictions=predictions,          # Shape (n_pred, samples, 1) or (n_pred, samples)
        y_test=y_data,
        evaluation_label=model_eval_label, # Pass label for plots/output
        n_bootstrap=N_BOOTSTRAP,          # Set number of bootstrap samples
        random_state=SEED,                # Pass seed for reproducibility
        output_plot_dir=os.path.join(OUTPUT_PLOT_DIR, model_eval_label) # Save plots in subdirs
    )

    # Print key metrics summary (using keys from the refined evaluate_uq_methods)
    if metrics:
        print(f"\nAggregated Uncertainty Metrics Summary for {model_eval_label}:")
        # Using mean values which account for CI calculation if bootstrap was successful
        print(f"- Overall Mean Variance: {metrics.get('overall_mean_variance_mean', metrics.get('overall_mean_variance', 'N/A')):.6f}")
        # VVV Use the key for TOTAL predictive entropy VVV
        print(f"- Mean Total Predictive Entropy: {metrics.get('mean_total_pred_entropy_mean', metrics.get('mean_total_pred_entropy', 'N/A')):.4f}")
        # ^^^ Use the key for TOTAL predictive entropy ^^^
        print(f"- Mean Mutual Information: {metrics.get('mean_mutual_info_mean', metrics.get('mean_mutual_info', 'N/A')):.6f}") # This key should be correct now
        # Optionally, print aleatoric entropy too for comparison:
        print(f"- Mean Expected Aleatoric Entropy: {metrics.get('mean_expected_aleatoric_entropy_mean', metrics.get('mean_expected_aleatoric_entropy', 'N/A')):.4f}")

    else:
        print(f"UQ metric calculation failed for {model_eval_label}")

    return metrics

# --- 4. CNN MC Dropout Evaluation ---
try:
    print("\n" + "="*50)
    print(f"Loading CNN model for MC Dropout from {CNN_MODEL_PATH}...")
    # Ensure this model was trained WITH dropout layers active appropriately
    # If using functional API, dropout needs to be specified with training=True/False argument
    # If using Sequential, dropout is typically active during model(..., training=True)
    cnn_model = load_model(CNN_MODEL_PATH)
    predictions = mc_dropout_predict(cnn_model, X_test_std_unbalanced, n_pred=N_MC_PASSES)

    if predictions is None:
        print(f"MC Dropout prediction failed")

    try:
        np.save("./mc_raw_pred0205.npy", predictions)
        print(f"Saved raw MC predictions (shape: {predictions.shape}) to")
    except Exception as e:
        print(f"ERROR saving raw MC predictions: {e}")
        # Decide if you want to stop or continue if saving fails

    #Evaluate on Unbalanced Test Set
    cnn_metrics_mcd_ub = evaluate_mc_dropout(
        cnn_model, X_test_std_unbalanced, y_test_unbalanced, "CNN_MCD_Unbalanced"
    )

    # Evaluate on Balanced (RUS) Test Set
    cnn_metrics_mcd_bal = evaluate_mc_dropout(
        cnn_model, X_test_std_rus, y_test_rus, "CNN_MCD_Balanced_RUS"
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
print(f"Plots saved in subdirectories under: {OUTPUT_PLOT_DIR}")
# You can now use the dictionaries (e.g., cnn_metrics_mcd_ub) to report numerical results.