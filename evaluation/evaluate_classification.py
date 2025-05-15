import numpy as np
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef, precision_recall_curve, auc
)

def evaluate_classification_model(model, X_test, y_test, evaluation_description="Evaluation"):
    """
    Evaluates a binary classification model on various metrics and prints the results.
    Assumes the model has a .predict() method returning probabilities for the positive class.

    Args:
    - model: Trained machine learning model with a .predict() method.
    - X_test (np.ndarray): Test features (should be standardized).
    - y_test (np.ndarray): True binary labels (0/1) for the test set.
    - evaluation_description (str): A label for the evaluation context
                                     (e.g., "CNN - Unbalanced Test Set",
                                      "LSTM - Balanced Test Set (RUS)").

    Returns:
    - metrics (dict): Dictionary containing calculated evaluation metrics, or None if an error occurs.
    """
    print(f"\n--- {evaluation_description} ---")

    # Get model predictions
    # Assumes model.predict outputs probabilities for the positive class (class 1)
    try:
        # Use verbose=0 for Keras models to avoid progress bars during evaluation
        # Check if the model is likely a Keras model before adding verbose=0
        if hasattr(model, 'predict') and 'keras' in str(type(model)).lower():
            y_pred_probs = model.predict(X_test, verbose=0)
        else:
            y_pred_probs = model.predict(X_test)

        # Ensure output is flattened if necessary (e.g., if output shape is (n, 1))
        if y_pred_probs.ndim > 1 and y_pred_probs.shape[1] == 1:
            y_pred_probs = y_pred_probs.flatten()

        # Ensure y_pred_probs is numpy array for consistent operations
        y_pred_probs = np.asarray(y_pred_probs)

    except Exception as e:
        print(f"Error during model prediction: {e}")
        return None

    # Apply standard threshold for class prediction (defaulting to 0.5)
    # Note: Thresholding at 0.5 might not be optimal for imbalanced datasets
    # or specific clinical objectives. This is a standard approach for basic evaluation.
    y_pred_class = (y_pred_probs > 0.5).astype(int)

    # Ensure y_test is numpy array for consistent operations
    y_test = np.asarray(y_test)

    # --- Compute Metrics ---
    print("Calculating metrics...")
    try:
        # Classification report (includes per-class precision, recall, f1, support, and accuracy)
        # Use target_names for better readability
        # zero_division=0 handles cases where a class has no true or predicted samples
        report_str = classification_report(
            y_test,
            y_pred_class,
            target_names=['Normal (0)', 'Apnea/Hypopnea (1)'],
            zero_division=0
        )
        report_dict = classification_report(
            y_test,
            y_pred_class,
            output_dict=True,
            zero_division=0
        )
        # Extract accuracy from report
        accuracy = report_dict.get('accuracy', 0.0) # Use .get for safety

        # AUC Scores
        # Check if there are samples from both classes for AUC calculation
        if len(np.unique(y_test)) < 2:
            roc_auc = np.nan # Not applicable if only one class is present
            auc_pr = np.nan
            print("Warning: Only one class present in y_test. Skipping ROC AUC and AUC-PR calculation.")
        else:
            # Ensure y_test used for AUC is numerical (float or int)
            y_test_numeric = y_test.astype(float)
            roc_auc = roc_auc_score(y_test_numeric, y_pred_probs)
            precision_vals, recall_vals, _ = precision_recall_curve(y_test_numeric, y_pred_probs)
            auc_pr = auc(recall_vals, precision_vals)


        # Other aggregate metrics
        cohen_kappa = cohen_kappa_score(y_test, y_pred_class)
        mcc = matthews_corrcoef(y_test, y_pred_class)

        # Confusion Matrix and derived metrics
        cm = confusion_matrix(y_test, y_pred_class)
        # Ensure cm is 2x2 for unpacking ravel(), pad if necessary
        # This is a minimal addition to prevent crash if ravel() fails
        if cm.shape != (2, 2):
            print(f"Warning: Confusion matrix shape is {cm.shape}, expected (2, 2). Padding matrix.")
            # Create a 2x2 matrix and fill from the calculated CM
            padded_cm = np.zeros((2, 2), dtype=int)
            # This requires care: assumes y_test contains at least one of the classes
            # Simple padding if a class is missing in predictions or true labels
            try:
                for r, c in [(0,0), (0,1), (1,0), (1,1)]: # Iterate through target indices
                    # Check if the index exists in the calculated cm before copying
                    if r < cm.shape[0] and c < cm.shape[1]:
                        padded_cm[r, c] = cm[r, c]
            except IndexError:
                print("Error during confusion matrix padding, results might be unreliable.")
                # Fallback or return None could be options, but keeping minimal change
                pass # Try to continue with potentially incorrect padded matrix
            cm = padded_cm

        tn, fp, fn, tp = cm.ravel()

        # Overall Sensitivity (Recall) = TP / (TP + FN)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        # Overall Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # --- Print Results ---
        print(f"\nClassification Report:\n{report_str}")
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}" if not np.isnan(roc_auc) else "ROC AUC: N/A (only one class in y_test)")
        print(f"AUC-PR: {auc_pr:.4f}" if not np.isnan(auc_pr) else "AUC-PR: N/A (only one class in y_test)")
        print(f"Cohen's Kappa: {cohen_kappa:.4f}")
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")
        print(f"Overall Sensitivity (Recall): {sensitivity:.4f}") # Same as recall for class 1 if calculated globally
        print(f"Overall Specificity: {specificity:.4f}") # Same as recall for class 0 if calculated globally
        print(f"Confusion Matrix:\n{cm}")
        print(f"   [[TN={tn}  FP={fp}]")
        print(f"    [FN={fn}  TP={tp}]]")

        # --- Return Metrics ---
        metrics = {
            "evaluation_description": evaluation_description,
            "classification_report_dict": report_dict,
            "accuracy": accuracy,
            "roc_auc": roc_auc if not np.isnan(roc_auc) else None, # Return None for N/A
            "auc_pr": auc_pr if not np.isnan(auc_pr) else None,   # Return None for N/A
            "cohen_kappa": cohen_kappa,
            "mcc": mcc,
            "overall_sensitivity": sensitivity,
            "overall_specificity": specificity,
            "confusion_matrix": cm, # Keep as numpy array as in original
            "tn": tn, "fp": fp, "fn": fn, "tp": tp # Also return individual CM components
        }
        print(f"--- Evaluation Complete for {evaluation_description} ---")
        return metrics

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None
