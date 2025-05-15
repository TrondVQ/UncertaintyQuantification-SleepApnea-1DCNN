import time
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from tensorflow.keras.models import Model
import tensorflow as tf
import os
import pandas as pd

# ===================== Core Prediction Functions =====================
def mc_dropout_predict(model: tf.keras.Model , x_test_data: np.ndarray, n_pred: int = 50) -> np.ndarray:
    """
    :param model: Pre-trained Keras model with dropout layers
    :param x_test_data: Evaluation data
    :param n_pred: Number of stochastic forward passes
    :return: Array of predictions (shape: [n_pred, samples, outputs])
    """

    """MC Dropout predictions with timing."""
    start_time = time.time()
    mc_predictions = np.stack([model(x_test_data, training=True) for pred in range(n_pred)])
    print(f"MC Dropout completed in {time.time()-start_time:.1f}s ({n_pred} passes)")
    return mc_predictions

def deep_ensembles_predict(ensemble_models: List, x_test_data: np.ndarray) -> np.ndarray:
    """Deep ensemble predictions with timing."""
    start_time = time.time()
    de_predictions = np.stack([model.predict(x_test_data, verbose=0)
                               for model in ensemble_models])
    print(f"Deep Ensemble completed in {time.time()-start_time:.1f}s ({len(ensemble_models)} models)")
    return de_predictions

# ===================== Uncertainty Metrics =====================
def safe_entropy(probs: np.ndarray, axis: int = 1, epsilon: float = 1e-10) -> np.ndarray:
    """Numerically stable entropy calculation for probability arrays."""
    clipped_probs = np.clip(probs, epsilon, 1 - epsilon)
    return entropy(clipped_probs, axis=axis)

def uq_evaluation_dist(uq_predictions: np.ndarray, y_true: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Computes uncertainty metrics from a distribution of predictions.

    Args:
        uq_predictions (np.ndarray): Array of predictions from multiple passes/models.
                                     Shape: (n_models_or_passes, n_samples).
                                     Assumes values are probabilities for the positive class (1).
        y_true (np.ndarray): Ground truth labels. Shape: (n_samples,).

    Returns:
        Dictionary containing uncertainty metrics:
        - 'mean_pred': Mean prediction probability per sample. Shape: (n_samples,).
        - 'pred_variance': Predictive variance per sample. Shape: (n_samples,).
        - 'total_pred_entropy': Average predictive entropy per sample (total uncertainty). Shape: (n_samples,).
        - 'expected_aleatoric_entropy': Entropy of the mean prediction per sample (aleatoric approx.). Shape: (n_samples,).
        - 'mutual_info': Mutual information per sample (epistemic approx.). Shape: (n_samples,).
        - 'overall_mean_variance': Average predictive variance across all samples (scalar).
        - 'mean_variance_class_0': Average predictive variance for true class 0 samples (scalar).
        - 'mean_variance_class_1': Average predictive variance for true class 1 samples (scalar).
    """
    # Ensure predictions are 2D (n_models_or_passes, n_samples)
    predictions = np.squeeze(uq_predictions)
    if predictions.ndim == 1: # Handle case of single model/pass input (variance/MI will be 0)
        predictions = predictions.reshape(1, -1)
    if predictions.shape[0] == 1:
        print("Warning: Only one set of predictions provided. Variance and Mutual Info will be zero.")

    # Mean prediction probability for each sample
    mean_pred = np.mean(predictions, axis=0) # Shape: (n_samples,)

    # Predictive variance for each sample
    pred_variance = np.var(predictions, axis=0) # Shape: (n_samples,)

    # Calculate entropy metrics (requires converting prob(class=1) to [prob(class=0), prob(class=1)])
    # Entropy of the mean prediction (proxy for aleatoric uncertainty)
    # Create probability array [prob_class0, prob_class1] for mean predictions
    # H[E[p]]
    mean_probs = np.stack([1 - mean_pred, mean_pred], axis=-1) # Shape: (n_samples, 2)
    total_pred_entropy = safe_entropy(mean_probs, axis=1) # Shape: (n_samples,)

    # Expected Entropy (Mean of the entropies - Aleatoric proxy)
    # E[H(p)]
    entropies_per_prediction = []
    for p in predictions: # Iterate through each pass/model
        probs = np.stack([1 - p, p], axis=-1) # Shape: (n_samples, 2)
        entropies_per_prediction.append(safe_entropy(probs, axis=1)) # Shape: (n_samples,)
    expected_aleatoric_entropy = np.mean(entropies_per_prediction, axis=0) # Shape: (n_samples,)

    # Mutual Information (Epistemic proxy) = Total - Aleatoric
    # MI = H[E[p]] - E[H(p)]
    mutual_info = np.maximum(total_pred_entropy - expected_aleatoric_entropy, 0) # Corrected order

    # Overall mean variance across all samples
    overall_mean_variance = np.mean(pred_variance)

    # Class-specific mean variances
    class0_mask = (y_true == 0)
    class1_mask = (y_true == 1)

    mean_variance_class_0 = np.mean(pred_variance[class0_mask]) if np.any(class0_mask) else 0.0
    mean_variance_class_1 = np.mean(pred_variance[class1_mask]) if np.any(class1_mask) else 0.0

    return {
        "mean_pred": mean_pred,
        "pred_variance": pred_variance, # Per-sample variance
        "total_pred_entropy": total_pred_entropy,   # Per-sample mean entropy (Total Approx.)
        "expected_aleatoric_entropy": expected_aleatoric_entropy, # Per-sample entropy of mean (Aleatoric Approx.)
        "mutual_info": mutual_info,     # Per-sample MI (Epistemic Approx.)
        "overall_mean_variance": overall_mean_variance, # Overall average variance
        "mean_variance_class_0": mean_variance_class_0,
        "mean_variance_class_1": mean_variance_class_1
    }

# ===================== Confidence Intervals =====================

def bootstrap_metrics(uq_predictions: np.ndarray, y_true: np.ndarray,
                      n_bootstrap: int = 100, random_state: Optional[int] = None) -> Optional[List[Dict]]:
    """
    Performs bootstrap resampling on predictions and recalculates UQ metrics.

    Args:
        uq_predictions (np.ndarray): Shape (n_models_or_passes, n_samples).
        y_true (np.ndarray): Shape (n_samples,).
        n_bootstrap (int): Number of bootstrap samples.
        random_state (Optional[int]): Seed for reproducibility.

    Returns:
        List of dictionaries, each containing UQ metrics for one bootstrap sample, or None.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = uq_predictions.shape[1]
    bootstrap_results = []

    print(f"Starting bootstrap with {n_bootstrap} iterations...")
    for i in range(n_bootstrap):
        if (i + 1) % 10 == 0: # Print progress
            print(f"  Bootstrap iteration {i+1}/{n_bootstrap}")
        try:
            # Sample indices with replacement
            idx = np.random.choice(n_samples, n_samples, replace=True)
            # Select predictions and labels for this bootstrap sample
            bs_preds = uq_predictions[:, idx]
            bs_y = y_true[idx]
            # Recalculate UQ metrics for the bootstrap sample
            bs_uq_metrics = uq_evaluation_dist(bs_preds, bs_y)
            if bs_uq_metrics: # Check if calculation succeeded
                # We need the *aggregate* metrics for CI calculation
                agg_metrics = {
                    "overall_mean_variance": bs_uq_metrics["overall_mean_variance"],
                    "mean_variance_class_0": bs_uq_metrics["mean_variance_class_0"],
                    "mean_variance_class_1": bs_uq_metrics["mean_variance_class_1"],
                    "mean_total_pred_entropy": np.mean(bs_uq_metrics["total_pred_entropy"]), # Mean of per-sample metric
                    "mean_expected_aleatoric_entropy": np.mean(bs_uq_metrics["expected_aleatoric_entropy"]),
                    "mean_mutual_info": np.mean(bs_uq_metrics["mutual_info"]),
                }
                bootstrap_results.append(agg_metrics)
            else:
                print(f"Warning: Skipping bootstrap iteration {i+1} due to calculation error.")

        except Exception as e:
            print(f"Error during bootstrap iteration {i+1}: {e}")
            # Continue to next iteration if one fails, or return None
            # return None # Option: Fail completely if one iteration errors

    if not bootstrap_results:
        print("Error: No bootstrap results generated.")
        return None

    print("Bootstrap finished.")
    return bootstrap_results


def compute_confidence_intervals(bootstrap_results: List[Dict], alpha: float = 0.05) -> Dict[str, float]:
    """
    Calculate confidence intervals (lower, upper bounds) for scalar metrics
    from bootstrap results.

    Args:
        bootstrap_results (List[Dict]): List of dictionaries from bootstrap_metrics.
        alpha (float): Significance level (e.g., 0.05 for 95% CI).

    Returns:
        Dictionary containing the mean, lower CI, and upper CI for each metric.
    """
    ci_results = {}
    if not bootstrap_results:
        return ci_results

    # Get metric names from the first result dictionary
    metric_names = bootstrap_results[0].keys()

    for metric_name in metric_names:
        try:
            # Extract the values for this metric across all bootstrap samples
            values = [m[metric_name] for m in bootstrap_results]

            # Calculate mean and percentile-based confidence intervals
            ci_results[f"{metric_name}_mean"] = float(np.mean(values)) # Use float() to ensure scalar
            ci_results[f"{metric_name}_ci_lower"] = float(np.percentile(values, 100 * alpha / 2))
            ci_results[f"{metric_name}_ci_upper"] = float(np.percentile(values, 100 * (1 - alpha / 2)))
        except Exception as e:
            print(f"Error calculating CI for metric '{metric_name}': {e}")

    return ci_results

#===================== Visualization Functions =====================

def plot_uncertainty_metric(uncertainty_values: np.ndarray,
                            uq_name: str,
                            metric_name: str,
                            output_dir: str = "./uq_plots",
                            title: Optional[str] = None,
                            max_samples: int = 5000):
    """Visualize uncertainty metric over samples using a line plot (better for large N)."""
    os.makedirs(output_dir, exist_ok=True)
    n_samples_total = len(uncertainty_values)

    indices = np.arange(n_samples_total)
    plot_data = uncertainty_values

    if n_samples_total > max_samples:
        idx_sample = np.random.choice(n_samples_total, max_samples, replace=False)
        idx_sample.sort()
        plot_data = uncertainty_values[idx_sample]
        indices = indices[idx_sample]
        print(f"Warning: Plotting line plot for {max_samples} random samples out of {n_samples_total}.")

    plt.figure(figsize=(15, 4))
    plt.plot(indices, plot_data, alpha=0.7)
    final_title = title or f"{metric_name} over Samples - {uq_name}"
    plt.title(final_title)
    plt.xlabel("Sample Index (Subsampled)" if n_samples_total > max_samples else "Sample Index")
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"line_{metric_name}_{uq_name}.png"), bbox_inches='tight')
    plt.close()


def plot_class_uncertainties(class0_unc: float, class1_unc: float, uq_name: str, output_dir: str = "./uq_plots"):
    """Visualize class-specific mean uncertainties using a bar chart."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(6, 5)) # Adjusted size
    labels = ['Normal (0)', 'Apnea/Hypopnea (1)']
    values = [class0_unc, class1_unc]
    bars = plt.bar(labels, values, color=['skyblue', 'salmon'])
    plt.bar_label(bars, fmt='%.6f') # Add values on bars
    plt.title(f"Mean Predictive Variance by True Class - {uq_name}")
    plt.ylabel("Mean Predictive Variance")
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"bar_class_variance_{uq_name}.png"), bbox_inches='tight')
    plt.close()


def plot_metric_distribution(metric_values: np.ndarray, y_true: np.ndarray, uq_name: str, metric_name: str, output_dir: str = "./uq_plots", bins: int = 30):
    """Plot distribution of a per-sample metric, separated by true class."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    # Plot histograms for each class if samples exist for both
    if np.any(y_true == 0):
        plt.hist(metric_values[y_true == 0], bins=bins, alpha=0.6, label='True Normal (0)', density=True)
    if np.any(y_true == 1):
        plt.hist(metric_values[y_true == 1], bins=bins, alpha=0.6, label='True Apnea/Hypopnea (1)', density=True)

    plt.title(f"{metric_name} Distribution by True Class - {uq_name}")
    plt.xlabel(f"{metric_name} Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"hist_{metric_name}_by_class_{uq_name}.png"), bbox_inches='tight')
    plt.close()

# ===================== Main Evaluation Function =====================
def evaluate_uq_methods(
        predictions: np.ndarray,
        y_test: np.ndarray,
        evaluation_label: str = "UQ Evaluation",
        n_bootstrap: int = 100,
        random_state: Optional[int] = None,
        output_plot_dir: str = "./uq_plots"
) -> Optional[Dict]:
    """
    Complete uncertainty quantification evaluation pipeline. Calculates metrics,
    computes confidence intervals via bootstrapping, and generates visualizations.

    Args:
        predictions (np.ndarray): Array of predictions from multiple passes/models.
                                  Shape: (n_models_or_passes, n_samples) or (..., 1).
                                  Values assumed to be probability of positive class (1).
        y_test (np.ndarray): Ground truth labels. Shape: (n_samples,).
        evaluation_label (str): Name for this evaluation run (e.g., "CNN MC Dropout Unbalanced").
        n_bootstrap (int): Number of bootstrap samples for CI calculation.
        random_state (Optional[int]): Seed for bootstrap reproducibility.
        output_plot_dir (str): Directory to save generated plots.

    Returns:
        Dictionary containing aggregated metrics and CIs, or None if evaluation fails.
        Note: Per-sample metrics are calculated but not returned in the final dict,
              only their aggregates and distributions are used/plotted.
    """
    print(f"\n=== Evaluating Uncertainty: {evaluation_label} ===")
    print(f"Input prediction shape: {predictions.shape}, Samples: {len(y_test)}")

    # --- Input Validation ---
    if predictions is None or y_test is None:
        print("Error: Input predictions or labels are None.")
        return None
    if len(y_test) != predictions.shape[1]:
        raise ValueError(f"Mismatch between prediction samples ({predictions.shape[1]}) and label samples ({len(y_test)})")

    # --- Dimension Handling ---
    if predictions.ndim == 3 and predictions.shape[-1] == 1:
        predictions = predictions[..., 0] # Remove trailing dimension of size 1
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1) # Ensure 2D

    # --- Core Metrics Calculation ---
    print("\nCalculating base UQ metrics...")
    uq_metrics_per_sample = uq_evaluation_dist(predictions, y_test)

    if uq_metrics_per_sample is None:
        print("Error: Failed to calculate base UQ metrics.")
        return None

    # Report mean of per-sample metrics
    print(f"- Overall Mean Variance: {uq_metrics_per_sample['overall_mean_variance']:.6f}")
    print(f"- Mean Variance Class 0: {uq_metrics_per_sample['mean_variance_class_0']:.6f}")
    print(f"- Mean Variance Class 1: {uq_metrics_per_sample['mean_variance_class_1']:.6f}")
    print(f"- Mean Predictive Entropy (Total): {np.mean(uq_metrics_per_sample['total_pred_entropy']):.4f}")
    print(f"- Mean Expected Entropy (Aleatoric): {np.mean(uq_metrics_per_sample['expected_aleatoric_entropy']):.4f}")
    print(f"- Mean Mutual Info (Epistemic): {np.mean(uq_metrics_per_sample['mutual_info']):.6f}")

    # --- Confidence Intervals ---
    print(f"\nComputing CIs (n_bootstrap={n_bootstrap})...")
    start_time = time.time()
    # Pass the raw predictions for bootstrapping
    bootstrap_results = bootstrap_metrics(predictions, y_test, n_bootstrap, random_state)

    final_metrics_aggregated = {} # Dictionary to store aggregated results and CIs
    if bootstrap_results:
        ci_metrics = compute_confidence_intervals(bootstrap_results)
        final_metrics_aggregated.update(ci_metrics) # Add CI bounds and CI means
        print(f"CIs computed in {time.time()-start_time:.2f}s")

        # Add non-CI aggregated metrics calculated earlier for completeness
        final_metrics_aggregated['overall_mean_variance'] = uq_metrics_per_sample['overall_mean_variance']
        final_metrics_aggregated['mean_variance_class_0'] = uq_metrics_per_sample['mean_variance_class_0']
        final_metrics_aggregated['mean_variance_class_1'] = uq_metrics_per_sample['mean_variance_class_1']
        final_metrics_aggregated['mean_total_pred_entropy'] = np.mean(uq_metrics_per_sample['total_pred_entropy'])
        final_metrics_aggregated['mean_expected_aleatoric_entropy'] = np.mean(uq_metrics_per_sample['expected_aleatoric_entropy'])
        final_metrics_aggregated['mean_mutual_info'] = np.mean(uq_metrics_per_sample['mutual_info'])

    else:
        print("Warning: Bootstrap failed, CIs not computed.")
        # Store just the aggregate metrics if bootstrap fails
        final_metrics_aggregated['overall_mean_variance'] = uq_metrics_per_sample['overall_mean_variance']
        final_metrics_aggregated['mean_variance_class_0'] = uq_metrics_per_sample['mean_variance_class_0']
        final_metrics_aggregated['mean_variance_class_1'] = uq_metrics_per_sample['mean_variance_class_1']
        final_metrics_aggregated['mean_total_pred_entropy'] = np.mean(uq_metrics_per_sample['total_pred_entropy'])
        final_metrics_aggregated['mean_expected_aleatoric_entropy'] = np.mean(uq_metrics_per_sample['expected_aleatoric_entropy'])
        final_metrics_aggregated['mean_mutual_info'] = np.mean(uq_metrics_per_sample['mutual_info'])


    # --- Visualizations ---
    print("\nGenerating visualizations...")
    # Plotting per-sample distributions or values
    plot_metric_distribution(
        uq_metrics_per_sample["pred_variance"], y_test, evaluation_label, "Predictive Variance", output_dir=output_plot_dir
    )
    plot_metric_distribution(
        uq_metrics_per_sample["total_pred_entropy"], y_test, evaluation_label, "Predictive Entropy", output_dir=output_plot_dir
    )
    plot_metric_distribution(
        uq_metrics_per_sample["mutual_info"], y_test, evaluation_label, "Mutual Information", output_dir=output_plot_dir
    )

    # Plotting aggregate/class metrics (use values from final_metrics_aggregated)
    plot_class_uncertainties(
        final_metrics_aggregated["mean_variance_class_0"], # Use direct aggregate value
        final_metrics_aggregated["mean_variance_class_1"],
        evaluation_label,
        output_dir=output_plot_dir
    )

    print("\n=== Evaluation Complete ===")
    # Return aggregated metrics and CIs
    return final_metrics_aggregated

# Example Usage:
# ===================== Example Usage =====================
if __name__ == '__main__':
    # This block is for demonstration/testing purposes

    print("Running UQ Evaluation Demo...")

    # --- Generate Dummy Data ---
    N_MODELS = 5
    N_SAMPLES = 1000
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    # Simulate predictions (probabilities for class 1)
    # Let's assume higher variance for class 1
    dummy_preds = []
    for _ in range(N_MODELS):
        p = np.random.rand(N_SAMPLES) * 0.6 + 0.2 # Base probabilities centered around 0.5
        noise = np.random.randn(N_SAMPLES) * 0.1 # Model noise
        p_noisy = np.clip(p + noise, 0.01, 0.99)
        dummy_preds.append(p_noisy)
    dummy_predictions = np.stack(dummy_preds) # Shape (N_MODELS, N_SAMPLES)

    # Simulate true labels (imbalanced)
    true_labels = (np.random.rand(N_SAMPLES) > 0.7).astype(int) # ~30% class 1

    print(f"\nGenerated dummy data: Predictions shape {dummy_predictions.shape}, Labels shape {true_labels.shape}")
    print(f"Dummy label distribution:\n{pd.Series(true_labels).value_counts(normalize=True)}")


    # --- Run Evaluation ---
    uq_results = evaluate_uq_methods(
        predictions=dummy_predictions,
        y_test=true_labels,
        evaluation_label="Dummy Ensemble Test",
        n_bootstrap=50, # Use fewer bootstraps for demo
        random_state=RANDOM_SEED,
        output_plot_dir="../Alarcon_SHHS/dummy_uq_plots"
    )

    # --- Print Aggregated Results ---
    if uq_results:
        print("\n--- Aggregated UQ Results ---")
        for key, value in uq_results.items():
            if isinstance(value, (int, float, np.number)): # Check if it's a scalar number
                print(f"{key}: {value:.6f}")
            # Optionally print CI bounds if needed here too
            # else:
            #     print(f"{key}: (Array or other type)") # Handle non-scalar if any remain

    else:
        print("\nUQ Evaluation failed.")

    print("\nDemo finished. Check the 'dummy_uq_plots' directory for visualizations.")
