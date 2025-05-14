#!/usr/bin/env python3

import time
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
from scipy.stats import entropy
from tensorflow.keras.models import Model
import tensorflow as tf
import os
import pandas as pd

# --- Configuration ---
# Epsilon for numerical stability in logarithm calculations (e.g., entropy)
EPSILON: float = 1e-10

# Default number of stochastic forward passes for MC Dropout prediction
DEFAULT_N_MC_PASSES: int = 50

# Default number of bootstrap samples for Confidence Interval calculation
DEFAULT_N_BOOTSTRAP: int = 100

# Default significance level (alpha) for Confidence Interval calculation (e.g., 0.05 for 95% CI)
DEFAULT_ALPHA: float = 0.05

# ===================== Core Prediction Functions =====================

def mc_dropout_predict(
        model: tf.keras.Model,
        x_test_data: np.ndarray, # Expected shape (samples, time_steps, features)
        n_pred: int = DEFAULT_N_MC_PASSES
) -> Optional[np.ndarray]:
    """
    Performs Monte Carlo Dropout predictions using a Keras model with Dropout layers.

    Activates Dropout layers during prediction (training=True).

    Args:
        model: A pre-trained TensorFlow Keras model with Dropout layers.
               These layers must be configured to be active during prediction when training=True.
        x_test_data: The input data for prediction. Expected shape
                     (number of samples, time_steps, number of features).
        n_pred: The number of stochastic forward passes to perform for each sample.

    Returns:
        A NumPy array of predictions from all passes.
        Shape: (n_pred, number of samples, number of outputs).
        Returns None if input data is empty or model is invalid.
    """
    if model is None:
        print("Error: No model provided for MC Dropout prediction.")
        return None
    if x_test_data is None or x_test_data.size == 0:
        print("Error: Input data is empty for MC Dropout prediction.")
        return None
    if n_pred <= 0:
        print(f"Error: Number of prediction passes must be positive, but got {n_pred}.")
        return None

    print(f"Starting MC Dropout prediction with {n_pred} passes...")
    start_time = time.time()
    mc_predictions: List[np.ndarray] = []
    try:
        # Ensure Dropout layers are active during prediction
        # For Sequential and functional models, training=True is the standard way to do this.
        for _ in range(n_pred):
            # Pass the data and explicitly set training=True
            pred = model(x_test_data, training=True).numpy() # Get numpy output
            mc_predictions.append(pred)

        # Stack predictions along a new axis (axis=0)
        mc_predictions_stacked = np.stack(mc_predictions, axis=0) # Shape: (n_pred, samples, outputs)

        print(f"MC Dropout prediction completed in {time.time()-start_time:.2f}s ({n_pred} passes)")
        return mc_predictions_stacked

    except Exception as e:
        print(f"Error during MC Dropout prediction: {e}")
        return None


def deep_ensembles_predict(
        ensemble_models: List[tf.keras.Model],
        x_test_data: np.ndarray # Expected shape (samples, time_steps, features)
) -> Optional[np.ndarray]:
    """
    Performs predictions using an ensemble of trained models.

    Each model's prediction is obtained deterministically (Dropout layers are
    typically inactive by default during model.predict, matching ensemble inference).

    Args:
        ensemble_models: A list of loaded TensorFlow Keras Model instances.
        x_test_data: The input data for prediction. Expected shape
                     (number of samples, time_steps, number of features).

    Returns:
        A NumPy array of predictions from all ensemble members.
        Shape: (number of models, number of samples, number of outputs).
        Returns None if input data is empty or ensemble list is empty.
    """
    if not ensemble_models:
        print("Error: Ensemble models list is empty for Deep Ensemble prediction.")
        return None
    if x_test_data is None or x_test_data.size == 0:
        print("Error: Input data is empty for Deep Ensemble prediction.")
        return None

    print(f"Starting Deep Ensemble prediction with {len(ensemble_models)} models...")
    start_time = time.time()
    de_predictions: List[np.ndarray] = []
    try:
        # Use model.predict for deterministic inference from each ensemble member
        for i, model in enumerate(ensemble_models):
            # Use verbose=0 to suppress progress bar for each model
            pred = model.predict(x_test_data, verbose=0)
            de_predictions.append(pred)
            if (i + 1) % 5 == 0 or (i + 1) == len(ensemble_models):
                print(f"  Predicted with {i+1}/{len(ensemble_models)} models.")

        # Stack predictions along a new axis (axis=0)
        de_predictions_stacked = np.stack(de_predictions, axis=0) # Shape: (n_models, samples, outputs)

        print(f"Deep Ensemble prediction completed in {time.time()-start_time:.2f}s ({len(ensemble_models)} models)")
        return de_predictions_stacked

    except Exception as e:
        print(f"Error during Deep Ensemble prediction: {e}")
        return None


# ===================== Uncertainty Metrics =====================

def safe_entropy(probs: np.ndarray, axis: int = -1, epsilon: float = EPSILON) -> np.ndarray:
    """
    Numerically stable entropy calculation for probability arrays.

    Args:
        probs: A NumPy array of probabilities. The calculation is performed along the specified axis.
               If the shape along the axis is 1, binary entropy H(p, 1-p) is calculated.
               Otherwise, standard categorical entropy -sum(p_i log2(p_i)) is calculated.
        axis: The axis along which to calculate entropy. Defaults to the last axis.
        epsilon: Small value for numerical stability in log calculation.

    Returns:
        A NumPy array containing the entropy values. The shape is the input shape
        with the specified axis removed.
    """
    # Clip probabilities to avoid log(0) or log(1) issues
    clipped_probs = np.clip(probs, epsilon, 1.0 - epsilon)

    if clipped_probs.shape[axis] == 1:
        # Assuming input is shape (..., 1) for probability of class 1 in binary case
        p1 = np.squeeze(clipped_probs, axis=axis) # Get probability of class 1
        # Calculate binary entropy H(p, 1-p) = - (p log2(p) + (1-p) log2(1-p))
        return - (p1 * np.log2(p1) + (1.0 - p1) * np.log2(1.0 - p1)) # Shape will be (...,)
    else:
        # Assuming input is shape (..., num_classes) where sum along axis is 1
        # Calculate standard categorical entropy -sum(p_i log2(p_i))
        return -np.sum(clipped_probs * np.log2(clipped_probs), axis=axis) # Shape will be (...,)


def uq_evaluation_dist(
        uq_predictions: np.ndarray, # Expected shape (n_models_or_passes, n_samples, n_outputs)
        y_true: np.ndarray         # Expected shape (n_samples,)
) -> Optional[Dict[str, np.ndarray]]:
    """
    Computes per-sample uncertainty metrics from a distribution of predictions.

    Args:
        uq_predictions: Array of predictions from multiple passes/models.
                        Expected shape: (n_models_or_passes, n_samples, n_outputs).
                        Values assumed to be probabilities [0, 1]. For binary classification,
                        n_outputs is typically 1 (prob of class 1) or 2 (prob of class 0, class 1).
        y_true: Ground truth labels. Expected shape: (n_samples,).

    Returns:
        Dictionary containing per-sample uncertainty metric arrays:
        - 'mean_pred': Mean prediction probability (for class 1 if binary) per sample. Shape: (n_samples,).
        - 'pred_variance': Predictive variance (of probabilities) per sample. Shape: (n_samples,).
        - 'total_pred_entropy': Total Predictive Entropy per sample. Shape: (n_samples,).
        - 'expected_aleatoric_entropy': Expected Aleatoric Entropy per sample. Shape: (n_samples,).
        - 'mutual_info': Mutual Information (Epistemic Uncertainty) per sample. Shape: (n_samples,).
        Returns None if input data is invalid or calculations fail.
    """
    if uq_predictions is None or y_true is None:
        print("Error: Input predictions or true labels are None for per-sample UQ metrics calculation.")
        return None
    if uq_predictions.ndim < 2:
        print(f"Error: Expected prediction shape with at least 2 dimensions (passes/models, samples, ...), but got {uq_predictions.shape}.")
        return None

    n_passes_or_models = uq_predictions.shape[0]
    n_samples = uq_predictions.shape[1]
    n_outputs = uq_predictions.shape[2] if uq_predictions.ndim > 2 else 1 # Assume 1 output if 2D

    if n_samples != len(y_true):
        print(f"Error: Mismatch between prediction samples ({n_samples}) and label samples ({len(y_true)}) for per-sample UQ metrics.")
        return None
    if n_samples == 0:
        print("Warning: No samples provided for per-sample UQ metrics calculation.")
        # Return zeroed arrays with correct shapes (0,)
        zero_arr_samples = np.zeros(n_samples)
        return {
            "mean_pred": zero_arr_samples,
            "pred_variance": zero_arr_samples,
            "total_pred_entropy": zero_arr_samples,
            "expected_aleatoric_entropy": zero_arr_samples,
            "mutual_info": zero_arr_samples
        }
    if n_passes_or_models == 0:
        print("Warning: No prediction passes/models provided for per-sample UQ metrics calculation.")
        # Cannot calculate variance or MI, return zeros for those, mean pred is NaN
        nan_arr_samples = np.full(n_samples, np.nan)
        zero_arr_samples = np.zeros(n_samples)
        return {
            "mean_pred": nan_arr_samples, # Cannot calculate mean without passes
            "pred_variance": zero_arr_samples, # Variance is 0 with 0 passes
            "total_pred_entropy": zero_arr_samples, # Entropy is 0 without predictions
            "expected_aleatoric_entropy": zero_arr_samples,
            "mutual_info": zero_arr_samples
        }

    # Ensure probabilities are in [0, 1] range
    uq_predictions_clipped = np.clip(uq_predictions, 0.0, 1.0)


    try:
        # 1. Mean prediction probability for each sample (Expected value E[Y|x])
        # If binary (n_outputs=1), mean_pred is shape (n_samples,)
        # If multi-class (n_outputs > 1), mean_pred is shape (n_samples, n_outputs)
        mean_pred = np.mean(uq_predictions_clipped, axis=0) # Shape: (n_samples, n_outputs) or (n_samples,) if n_outputs=1

        # 2. Predictive variance for each sample (Variance of the prediction probability)
        # If binary (n_outputs=1), pred_variance is shape (n_samples,)
        # If multi-class (n_outputs > 1), calculate variance independently for each output
        # For binary, Var(p_hat) over passes/models
        if n_outputs == 1:
            pred_variance = np.var(uq_predictions_clipped.squeeze(axis=-1), axis=0) # Shape: (n_samples,)
        else:
            # Variance per class probability across passes/models
            pred_variance = np.var(uq_predictions_clipped, axis=0) # Shape: (n_samples, n_outputs)
            # A single variance value per sample might be desired, e.g., sum of variances
            # pred_variance = np.sum(pred_variance, axis=-1) # Option: sum variances across classes

        # 3. Total Predictive Entropy H(Y|x)
        # For binary, approx H(E[Y|x]) where E[Y|x] is mean_pred (shape (n_samples,))
        # For multi-class, calculate H(E[Y|x]), where E[Y|x] is mean_pred (shape (n_samples, n_outputs))
        if n_outputs == 1: # Binary case, mean_pred is shape (n_samples,)
            total_pred_entropy = safe_entropy(mean_pred.reshape(n_samples, 1), axis=-1, epsilon=EPSILON) # Shape: (n_samples,)
        else: # Multi-class case, mean_pred is shape (n_samples, n_outputs)
            total_pred_entropy = safe_entropy(mean_pred, axis=-1, epsilon=EPSILON) # Shape: (n_samples,)


        # 4. Expected Aleatoric Entropy E[H(Y|x, w)]
        # For binary, approx 1/N_passes * Sum(H(Y|x, wi))
        # For multi-class, approx 1/N_passes * Sum(H(Y|x, wi))
        # Calculate entropy for each pass/model's prediction, then average
        # predictions_per_pass_or_model shape is (n_passes/models, n_samples, n_outputs)
        entropies_per_prediction_source = safe_entropy(uq_predictions_clipped, axis=-1, epsilon=EPSILON) # Shape (n_passes_or_models, n_samples)
        expected_aleatoric_entropy = np.mean(entropies_per_prediction_source, axis=0) # Shape (n_samples,)

        # 5. Mutual Information (Epistemic Uncertainty) I(Y; w|x)
        # MI = H(Y|x) - E[H(Y|x, w)] = Total Entropy - Expected Aleatoric Entropy
        mutual_info = total_pred_entropy - expected_aleatoric_entropy
        # Mutual Information should theoretically be non-negative. Clip to 0 to handle floating point errors.
        mutual_info = np.maximum(mutual_info, 0) # Shape: (n_samples,)

        # --- Return Metrics ---
        # For binary case where pred_variance is (n_samples,), return directly
        if n_outputs == 1:
            return {
                "mean_pred": mean_pred.squeeze(), # *** Squeeze mean_pred here ***
                "pred_variance": pred_variance,
                "total_pred_entropy": total_pred_entropy,
                "expected_aleatoric_entropy": expected_aleatoric_entropy,
                "mutual_info": mutual_info
            }
        # For multi-class case where pred_variance is (n_samples, n_outputs)
        else:
            return {
                "mean_pred": mean_pred, # Keep multi-output shape
                "pred_variance": pred_variance,
                "total_pred_entropy": total_pred_entropy,
                "expected_aleatoric_entropy": expected_aleatoric_entropy,
                "mutual_info": mutual_info
            }

    except Exception as e:
        print(f"Error during per-sample UQ metrics calculation in uq_evaluation_dist: {e}")
        return None


# ===================== Confidence Intervals =====================

def bootstrap_metrics(
        uq_predictions_2d: np.ndarray, # Expected shape (n_models_or_passes, n_samples)
        y_true: np.ndarray,           # Expected shape (n_samples,)
        n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
        random_state: Optional[int] = None
) -> Optional[List[Dict[str, float]]]:
    """
    Performs bootstrap resampling on samples and recalculates aggregate UQ metrics
    and accuracy for each bootstrap sample.

    Args:
        uq_predictions_2d: Array of predictions (probabilities of positive class for binary).
                           Expected shape: (n_models_or_passes, n_samples).
                           Values assumed to be probabilities [0, 1].
        y_true: Ground truth labels. Expected shape: (n_samples,).
        n_bootstrap: Number of bootstrap samples to generate.
        random_state: Seed for reproducibility of bootstrap resampling.

    Returns:
        List of dictionaries, each containing aggregate UQ metrics and accuracy
        for one bootstrap sample. Returns None if input is invalid or
        if no bootstrap results are generated due to errors.
        Returns an empty list if n_bootstrap is 0 or no samples are available.
    """
    if uq_predictions_2d is None or y_true is None:
        print("Error: Input predictions or true labels are None for bootstrapping.")
        return None
    if uq_predictions_2d.ndim != 2:
        print(f"Error: Expected 2D prediction shape (n_passes/models, n_samples) for bootstrapping, but got {uq_predictions_2d.shape}.")
        return None
    if uq_predictions_2d.shape[1] != len(y_true):
        print(f"Error: Mismatch between prediction samples ({uq_predictions_2d.shape[1]}) and label samples ({len(y_true)}) for bootstrapping.")
        return None
    n_samples = uq_predictions_2d.shape[1]
    if n_samples == 0:
        print("Warning: No samples provided for bootstrapping.")
        return [] # Return empty list if no samples
    if n_bootstrap <= 0:
        print("Warning: n_bootstrap is 0 or negative. Skipping bootstrapping.")
        return []


    if random_state is not None:
        np.random.seed(random_state)

    bootstrap_results: List[Dict[str, float]] = []

    print(f"Starting bootstrap resampling with {n_bootstrap} iterations on {n_samples} samples...")
    start_time = time.time()
    for i in range(n_bootstrap):
        # Sample indices with replacement
        idx = np.random.choice(n_samples, n_samples, replace=True)

        # Select predictions and labels for this bootstrap sample
        bs_preds_2d = uq_predictions_2d[:, idx] # Shape: (n_models_or_passes, n_samples)
        bs_y = y_true[idx]                 # Shape: (n_samples,)

        # Recalculate *per-sample* UQ metrics for this bootstrap sample
        # uq_evaluation_dist expects (n_passes/models, n_samples, 1), so add back the dim for binary
        # Need to handle multi-class case too if uq_evaluation_dist supports it
        # Assuming binary (n_outputs=1) for bootstrapping context
        bs_uq_metrics_per_sample = uq_evaluation_dist(bs_preds_2d[:, :, np.newaxis], bs_y)

        if bs_uq_metrics_per_sample: # Check if per-sample calculation succeeded
            try:
                # Calculate *aggregate* metrics for this bootstrap sample's per-sample results
                # These are the scalars we want CIs for
                bs_agg_metrics: Dict[str, float] = {}

                # Include aggregate of all metrics returned by uq_evaluation_dist
                for metric_name, metric_values in bs_uq_metrics_per_sample.items():
                    if metric_values.ndim == 1: # Aggregated if per-sample value is scalar (e.g., variance for binary)
                        bs_agg_metrics[f"overall_mean_{metric_name}"] = float(np.mean(metric_values))
                        # Add class-specific means for variance if available and binary
                        if metric_name == "pred_variance" and np.any(bs_y == 0):
                            bs_agg_metrics["mean_variance_class_0"] = float(np.mean(metric_values[bs_y == 0]))
                        if metric_name == "pred_variance" and np.any(bs_y == 1):
                            bs_agg_metrics["mean_variance_class_1"] = float(np.mean(metric_values[bs_y == 1]))

                    # Add handling for multi-output metrics if needed (e.g., variance shape (samples, outputs))
                    # elif metric_values.ndim == 2 and metric_name == "pred_variance":
                    #    bs_agg_metrics["overall_mean_pred_variance_sum"] = float(np.mean(np.sum(metric_values, axis=-1))) # Example aggregate sum

                # Also include performance metric (accuracy) for CI
                # Accuracy based on mean prediction probability (>0.5 threshold)
                bs_mean_pred = bs_uq_metrics_per_sample.get("mean_pred") # Get mean_pred from per-sample metrics
                if bs_mean_pred is not None and bs_mean_pred.ndim == 1: # Ensure it's the per-sample scalar mean prob
                    bs_predicted_labels = (bs_mean_pred > 0.5).astype(int)
                    bs_agg_metrics["overall_accuracy"] = float(np.mean(bs_predicted_labels == bs_y))
                else:
                    # If mean_pred is not a scalar per sample (e.g., multi-class (samples, outputs)),
                    # accuracy calculation would need adjustment, or skip adding accuracy CI here.
                    print(f"Warning: Cannot calculate scalar overall accuracy for bootstrap due to mean_pred shape {bs_mean_pred.shape if bs_mean_pred is not None else 'None'}. Skipping accuracy CI.")


                bootstrap_results.append(bs_agg_metrics)

            except Exception as e:
                print(f"Warning: Error calculating aggregate metrics for bootstrap iteration {i+1}: {e}. Skipping iteration.")
                # Continue to next iteration if one fails

        else:
            print(f"Warning: Per-sample UQ metrics calculation failed for bootstrap iteration {i+1}. Skipping iteration.")


        if (i + 1) % 10 == 0 or (i + 1) == n_bootstrap: # Print progress every 10 iterations or at the end
            print(f"  Completed {i+1}/{n_bootstrap} iterations.")

    if not bootstrap_results and n_bootstrap > 0:
        print("Error: No valid bootstrap results generated.")
        return None
    elif not bootstrap_results and n_bootstrap == 0:
        print("Bootstrap skipped as n_bootstrap was 0.")
        return []


    print(f"Bootstrap finished in {time.time()-start_time:.2f}s.")
    return bootstrap_results


def compute_confidence_intervals(
        bootstrap_results: List[Dict[str, float]],
        alpha: float = DEFAULT_ALPHA
) -> Dict[str, float]:
    """
    Calculate mean, lower, and upper confidence intervals for scalar metrics
    from bootstrap results using the percentile method.

    Args:
        bootstrap_results: List of dictionaries, each containing aggregate metrics
                           from one bootstrap sample (output of bootstrap_metrics).
                           Expected values are floats.
        alpha: The significance level (e.g., 0.05 for a 95% confidence interval).

    Returns:
        Dictionary containing the mean, lower CI bound, and upper CI bound
        for each metric found in the bootstrap results. Keys are formatted as
        '{metric_name}_mean', '{metric_name}_ci_lower', '{metric_name}_ci_upper'.
        Returns an empty dictionary if no bootstrap results are provided.
    """
    ci_results: Dict[str, float] = {}
    if not bootstrap_results:
        print("Warning: No bootstrap results provided for CI calculation.")
        return ci_results

    # Get metric names from the keys of the dictionaries in the list
    # Assuming all dictionaries have the same keys (based on how bootstrap_metrics creates them)
    metric_names = bootstrap_results[0].keys()
    print(f"Calculating confidence intervals for metrics: {list(metric_names)}")

    for metric_name in metric_names:
        try:
            # Extract the values for this metric across all bootstrap samples
            values = [m.get(metric_name) for m in bootstrap_results if m.get(metric_name) is not None] # Filter out None values

            if not values:
                print(f"Warning: No valid values found for metric '{metric_name}' across bootstrap samples. Skipping CI calculation.")
                continue
            if len(values) < 2:
                print(f"Warning: Need at least 2 bootstrap samples to calculate CI for '{metric_name}', found {len(values)}. Skipping CI.")
                ci_results[f"{metric_name}_mean"] = float(np.mean(values)) # Still include the mean
                continue


            # Calculate mean and percentile-based confidence intervals
            # Use float() to ensure the result is a standard float, not a numpy scalar
            ci_results[f"{metric_name}_mean"] = float(np.mean(values))
            ci_results[f"{metric_name}_ci_lower"] = float(np.percentile(values, 100 * alpha / 2))
            ci_results[f"{metric_name}_ci_upper"] = float(np.percentile(values, 100 * (1 - alpha / 2)))

        except Exception as e:
            print(f"Error calculating CI for metric '{metric_name}': {e}")
            # Continue to the next metric if calculation fails for one

    return ci_results

# ===================== Main Evaluation Orchestration Function =====================

def evaluate_uq_methods(
        predictions: np.ndarray, # Expected shape (n_models_or_passes, n_samples, n_outputs)
        y_true: np.ndarray,     # Expected shape (n_samples,)
        evaluation_label: str = "UQ Evaluation",
        n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
        random_state: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Complete uncertainty quantification evaluation pipeline .

    Calculates per-sample and aggregate UQ metrics and computes confidence intervals
    via bootstrapping. Does NOT generate visualizations or save per-window CSVs.

    Args:
        predictions: Array of predictions from multiple passes/models.
                     Expected shape: (n_models_or_passes, n_samples, n_outputs).
                     Values assumed to be probability of positive class (1), in [0, 1].
                     For multi-class, n_outputs > 1.
        y_true: Ground truth labels. Expected shape: (n_samples,).
        evaluation_label: Name for this evaluation run (e.g., "CNN MC Dropout Unbalanced").
        n_bootstrap: Number of bootstrap samples for CI calculation. Set to 0 to skip CI computation.
        random_state: Seed for bootstrap reproducibility.

    Returns:
        Dictionary containing aggregated UQ metrics and their confidence intervals (if n_bootstrap > 0).
        Returns None if evaluation fails.
        The keys in the dictionary include:
        - Basic aggregates: 'overall_mean_pred_variance', 'overall_mean_total_pred_entropy', etc.
        - Bootstrap means: '{metric_name}_mean' (if n_bootstrap > 0)
        - Bootstrap CIs: '{metric_name}_ci_lower', '{metric_name}_ci_upper' (if n_bootstrap > 1)
    """
    print(f"\n=== Evaluating Uncertainty: {evaluation_label} (No Plotting) ===")
    print(f"Input prediction shape: {predictions.shape}, Samples: {len(y_true)}")

    # --- Input Validation ---
    if predictions is None or y_true is None:
        print("Error: Input predictions or labels are None.")
        return None

    # Ensure predictions shape is suitable
    if predictions.ndim < 2:
        print(f"Error: Unexpected prediction shape {predictions.shape}. Expected at least 2D.")
        return None
    if predictions.shape[1] != len(y_true):
        print(f"Error: Mismatch between prediction samples ({predictions.shape[1]}) and label samples ({len(y_true)}).")
        return None
    if predictions.shape[1] == 0:
        print("Warning: No samples to evaluate UQ metrics on.")
        return {} # Return empty dict if no samples

    n_passes_or_models = predictions.shape[0]
    n_samples = predictions.shape[1]
    n_outputs = predictions.shape[2] if predictions.ndim > 2 else 1

    if n_passes_or_models == 0:
        print("Warning: No prediction passes/models provided for UQ metrics calculation.")
        # Calculate basic aggregates that are possible (which is basically nothing meaningful)
        # and return an empty dictionary or N/A values if preferred.
        # Let's return an empty dict as no calculation is possible.
        return {}


    # --- Core Metrics Calculation (Per-Sample) ---
    print("\nCalculating per-sample UQ metrics...")
    # uq_evaluation_dist expects (n_passes/models, n_samples, n_outputs)
    uq_metrics_per_sample = uq_evaluation_dist(predictions, y_true)

    if uq_metrics_per_sample is None:
        print("Error: Failed to calculate per-sample UQ metrics.")
        return None

    # Calculate and report basic aggregate metrics from the per-sample results
    overall_mean_pred_variance = float(np.mean(uq_metrics_per_sample.get('pred_variance', np.nan))) if uq_metrics_per_sample.get('pred_variance') is not None else np.nan
    mean_variance_class_0 = float(np.mean(uq_metrics_per_sample['pred_variance'][y_true == 0])) if uq_metrics_per_sample.get('pred_variance') is not None and np.any(y_true == 0) else np.nan
    mean_variance_class_1 = float(np.mean(uq_metrics_per_sample['pred_variance'][y_true == 1])) if uq_metrics_per_sample.get('pred_variance') is not None and np.any(y_true == 1) else np.nan
    overall_mean_total_pred_entropy = float(np.mean(uq_metrics_per_sample.get('total_pred_entropy', np.nan))) if uq_metrics_per_sample.get('total_pred_entropy') is not None else np.nan
    overall_mean_expected_aleatoric_entropy = float(np.mean(uq_metrics_per_sample.get('expected_aleatoric_entropy', np.nan))) if uq_metrics_per_sample.get('expected_aleatoric_entropy') is not None else np.nan
    overall_mean_mutual_information = float(np.mean(uq_metrics_per_sample.get('mutual_info', np.nan))) if uq_metrics_per_sample.get('mutual_info') is not None else np.nan

    print(f"- Overall Mean Variance (basic): {overall_mean_pred_variance:.6f}")
    print(f"- Mean Variance True Class 0 (basic): {mean_variance_class_0:.6f}")
    print(f"- Mean Variance True Class 1 (basic): {mean_variance_class_1:.6f}")
    print(f"- Overall Mean Predictive Entropy (Total, basic): {overall_mean_total_pred_entropy:.4f}")
    print(f"- Overall Mean Expected Aleatoric Entropy (basic): {overall_mean_expected_aleatoric_entropy:.4f}")
    print(f"- Overall Mean Mutual Info (Epistemic, basic): {overall_mean_mutual_information:.6f}")


    # --- Confidence Intervals (via Bootstrapping) ---
    final_metrics_aggregated: Dict[str, Any] = {} # Dictionary to store aggregated results and CIs

    if n_bootstrap > 0 and predictions.shape[1] > 0: # Only run bootstrap if positive and samples exist
        print(f"\nComputing CIs (n_bootstrap={n_bootstrap}, random_state={random_state})...")
        start_time_ci = time.time()
        # Pass the 2D predictions (n_passes/models, n_samples) to bootstrap_metrics
        # uq_evaluation_dist expects (n_passes/models, n_samples, 1) -> add the dim back temporarily for it
        # Need to handle multi-output prediction case for bootstrapping if relevant
        if predictions.ndim == 3 and predictions.shape[2] > 1:
            print("Warning: Input predictions are multi-output (e.g., multi-class probabilities). Bootstrapping logic may need adjustment for multi-output metrics.")
            # Proceeding assuming bootstrap_metrics handles the structure or focuses on scalar aggregates


        # Bootstrap works on samples, so it needs the predictions per sample (n_passes/models, n_samples, n_outputs)
        # Convert predictions back to (n_passes/models, n_samples) for bootstrap_metrics if needed, assuming binary.
        # Based on bootstrap_metrics expecting (n_passes/models, n_samples), let's pass the squeezed version.
        if predictions.ndim == 3 and predictions.shape[2] == 1:
            predictions_for_bootstrap = predictions.squeeze(-1) # Shape (n_passes/models, n_samples)
        else:
            # For multi-output, the shape passed to bootstrap_metrics might need review
            predictions_for_bootstrap = predictions # Pass as is


        bootstrap_results = bootstrap_metrics(predictions_for_bootstrap, y_true, n_bootstrap, random_state)

        if bootstrap_results:
            print(f"Successfully generated {len(bootstrap_results)} bootstrap samples.")
            # Compute CIs from the collected bootstrap results
            ci_metrics = compute_confidence_intervals(bootstrap_results, alpha=DEFAULT_ALPHA)
            final_metrics_aggregated.update(ci_metrics) # Add CI bounds and CI means (e.g., 'metric_mean', 'metric_ci_lower/upper')

            print(f"Confidence Intervals computed in {time.time()-start_time_ci:.2f}s")

            # Add the basic overall mean metrics to the final dictionary with standard keys
            # These might be slightly different from the bootstrap means ('metric_mean')
            # Include all basic aggregate metrics
            final_metrics_aggregated['overall_mean_pred_variance'] = overall_mean_pred_variance
            final_metrics_aggregated['mean_variance_class_0'] = mean_variance_class_0
            final_metrics_aggregated['mean_variance_class_1'] = mean_variance_class_1
            final_metrics_aggregated['overall_mean_total_pred_entropy'] = overall_mean_total_pred_entropy
            final_metrics_aggregated['overall_mean_expected_aleatoric_entropy'] = overall_mean_expected_aleatoric_entropy
            final_metrics_aggregated['overall_mean_mutual_information'] = overall_mean_mutual_information


        else:
            print("Warning: Bootstrap failed or returned no results. CIs not computed.")
            # Store just the basic aggregate metrics if bootstrap fails
            final_metrics_aggregated['overall_mean_pred_variance'] = overall_mean_pred_variance
            final_metrics_aggregated['mean_variance_class_0'] = mean_variance_class_0
            final_metrics_aggregated['mean_variance_class_1'] = mean_variance_class_1
            final_metrics_aggregated['overall_mean_total_pred_entropy'] = overall_mean_total_pred_entropy
            final_metrics_aggregated['overall_mean_expected_aleatoric_entropy'] = overall_mean_expected_aleatoric_entropy
            final_metrics_aggregated['overall_mean_mutual_information'] = overall_mean_mutual_information


    else:
        print("\nSkipping CI computation: n_bootstrap is 0 or negative, or no samples available.")
        # Store just the basic aggregate metrics
        final_metrics_aggregated['overall_mean_pred_variance'] = overall_mean_pred_variance
        final_metrics_aggregated['mean_variance_class_0'] = mean_variance_class_0
        final_metrics_aggregated['mean_variance_class_1'] = mean_variance_class_1
        final_metrics_aggregated['overall_mean_total_pred_entropy'] = overall_mean_total_pred_entropy
        final_metrics_aggregated['overall_mean_expected_aleatoric_entropy'] = overall_mean_expected_aleatoric_entropy
        final_metrics_aggregated['overall_mean_mutual_information'] = overall_mean_mutual_information


    print(f"\n=== Evaluation Complete: {evaluation_label} ===")
    # Return the dictionary containing aggregated metrics and CIs
    return final_metrics_aggregated

# ===================== Example Usage =====================
# The if __name__ == '__main__': block contains example code to demonstrate
# how the evaluate_uq_methods function can be used with dummy data.
# This block is only executed when the script is run directly.

if __name__ == '__main__':
    print("Running UQ Techniques Script Demo (No Plotting)...")

    # --- Generate Dummy Data ---
    N_MODELS_OR_PASSES = 5
    N_SAMPLES = 1000
    N_OUTPUTS = 1 # Binary classification
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED) # Set TF seed if dummy model is used

    print(f"\nGenerating dummy data for {N_SAMPLES} samples from {N_MODELS_OR_PASSES} sources...")

    # Simulate predictions (probabilities for class 1)
    # Shape needs to be (n_sources, n_samples, n_outputs) -> (N_MODELS_OR_PASSES, N_SAMPLES, 1)
    dummy_predictions_3d = np.random.rand(N_MODELS_OR_PASSES, N_SAMPLES, N_OUTPUTS)

    # Simulate true labels (imbalanced)
    dummy_true_labels = (np.random.rand(N_SAMPLES) > 0.7).astype(int) # ~30% class 1

    print(f"Generated dummy data: Predictions shape {dummy_predictions_3d.shape}, Labels shape {dummy_true_labels.shape}")
    print(f"Dummy label distribution:\n{pd.Series(dummy_true_labels).value_counts(normalize=True)}")


    # --- Run Evaluation using evaluate_uq_methods ---
    # Note: The example directly calls evaluate_uq_methods.
    # In your main evaluation scripts (evaluate_deep_ensemble.py, evaluate_mc_dropout.py),
    # you would first call deep_ensembles_predict or mc_dropout_predict to get the
    # predictions array, then pass that array to evaluate_uq_methods.

    print("\n--- Running evaluate_uq_methods with dummy data (No Plotting) ---")
    dummy_uq_results_aggregated = evaluate_uq_methods(
        predictions=dummy_predictions_3d, # Pass the 3D dummy predictions
        y_true=dummy_true_labels,
        evaluation_label="Dummy UQ Evaluation Demo",
        n_bootstrap=50, # Use fewer bootstraps for demo speed
        random_state=RANDOM_SEED
    )

    # --- Print Aggregated Results ---
    if dummy_uq_results_aggregated:
        print("\n--- Aggregated UQ Results (from evaluate_uq_methods) ---")
        # Print all key-value pairs in the returned dictionary
        for key, value in dummy_uq_results_aggregated.items():
            # Format float values
            if isinstance(value, float) or isinstance(value, np.number):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")


    else:
        print("\nUQ Evaluation demo failed.")

    print("\nDemo finished.")