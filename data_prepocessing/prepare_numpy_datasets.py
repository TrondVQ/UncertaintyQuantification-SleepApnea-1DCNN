import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import argparse
import os

# --- Configuration ---
# Define default paths and parameters (can be overridden by command-line args)
SHHS2_CSV_ALL = "./SHHS2_ID_all.csv" # Assumes 60s windows
OUTPUT_DIR = "./processed_datasets2" # Changed output dir name
TEST_SIZE = 0.20 # Corresponds to an 80/20 train/test split
RANDOM_SEED = 2025

# Define feature constants based on previous script output
ORIGINAL_FEATURES = ["SaO2", "PR", "THOR RES", "ABDO RES"]
NUM_FEATURES = len(ORIGINAL_FEATURES)
TIME_STEPS = 60 # Should match window size used in previous script
LABEL_COL = "Apnea/Hypopnea"
GROUP_COL = "Patient_ID" # Ensure this column name matches your CSV

# Define expected flattened feature columns (ensure order matches previous script)
# Assuming 'C' order flattening was used: SaO2_t0, PR_t0,..., SaO2_t1, PR_t1,...
FEATURE_COLS = [f"{col}_t{t}" for t in range(TIME_STEPS) for col in ORIGINAL_FEATURES]


# --- Helper Functions ---

def reshape_flat_to_3d(data_flat, steps, features):
    """Reshapes flat data (samples, steps*features) to 3D (samples, steps, features)."""
    num_samples = data_flat.shape[0]
    expected_flat_features = steps * features
    if data_flat.shape[1] != expected_flat_features:
        print(f"ERROR: Cannot reshape. Expected {expected_flat_features} flat features, found {data_flat.shape[1]}.")
        raise ValueError("Incorrect number of features for reshaping.")
    # Assuming 'C' order flattening was used in the previous script to create FEATURE_COLS
    # Reshape directly based on this assumption
    try:
        reshaped = data_flat.reshape((num_samples, steps, features))
        print(f"Reshaped data from {data_flat.shape} to {reshaped.shape}")
        return reshaped
    except ValueError as e:
        print(f"ERROR during reshaping: {e}. Check TIME_STEPS={steps}, NUM_FEATURES={features}, and original flattening order.")
        raise e

def standardize_per_window(data_3d, epsilon=1e-8):
    """Applies standardization independently to each window (sample)."""
    print(f"Applying window-level standardization to data with shape {data_3d.shape}...")
    # Calculate mean and std along the time axis (axis=1)
    # Keepdims=True ensures means/stds broadcast correctly: (n_samples, 1, n_features)
    mean = np.mean(data_3d, axis=1, keepdims=True)
    std = np.std(data_3d, axis=1, keepdims=True)

    # Apply standardization: (data - mean) / (std + epsilon)
    # Epsilon prevents division by zero for features constant within a window
    standardized_data = (data_3d - mean) / (std + epsilon)
    print("Window-level standardization applied.")
    return standardized_data

# --- Main Data Preparation Function ---

def prepare_final_datasets(input_csv, output_dir, test_size, seed):
    """
    Loads flattened data, performs splitting, reshapes, applies window-level
    standardization, applies balancing (SMOTE/RUS), and saves final 3D datasets.
    """
    print(f"--- Starting Final Data Preparation (Window-Level Standardization) ---")
    print(f"Input CSV: {input_csv}")
    print(f"Output Directory: {output_dir}")
    print(f"Test Set Size: {test_size}")
    print(f"Random Seed: {seed}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load the full dataset
    try:
        full_data = pd.read_csv(input_csv)
        print(f"Loaded data shape: {full_data.shape}")
        if not all(col in full_data.columns for col in FEATURE_COLS + [LABEL_COL, GROUP_COL]):
            raise ValueError("Missing required columns in input CSV.")
    except FileNotFoundError:
        print(f"ERROR: Input CSV file not found at {input_csv}")
        return
    except Exception as e:
        print(f"ERROR: Failed to load or validate CSV: {e}")
        return

    # Handle potential NaN values
    if full_data[FEATURE_COLS].isnull().values.any():
        print(f"NaN values found! Filling with global column means (consider more sophisticated imputation if needed).")
        full_data[FEATURE_COLS] = full_data[FEATURE_COLS].fillna(full_data[FEATURE_COLS].mean())
        if full_data[FEATURE_COLS].isnull().values.any():
            print(f"ERROR: NaN values persist after filling. Check data.")
            return

    # 2. Separate features, labels, and groups
    X_flat = full_data[FEATURE_COLS]
    y = full_data[LABEL_COL]
    groups = full_data[GROUP_COL]

    # 3. Patient-Independent Split (on flat data indices)
    print(f"\nPerforming patient-independent split...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    try:
        train_idx, test_idx = next(splitter.split(X_flat, y, groups))
    except Exception as e:
        print(f"Error during splitting: {e}")
        return

    X_train_flat = X_flat.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test_flat = X_flat.iloc[test_idx]
    y_test = y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    print(f"Train shape (flat): {X_train_flat.shape}, Test shape (flat): {X_test_flat.shape}")
    train_patients = set(groups_train.unique())
    test_patients = set(groups_test.unique())
    if train_patients.intersection(test_patients):
        print("WARNING: Patient overlap detected between train and test sets!")
    else:
        print("Patient split verified: No overlap.")

    # 4. Reshape Data to 3D (Samples, TimeSteps, Features)
    print("\nReshaping data to 3D format...")
    try:
        # Convert pandas DataFrames to numpy arrays before reshaping
        X_train_3d = reshape_flat_to_3d(X_train_flat.values, TIME_STEPS, NUM_FEATURES)
        X_test_3d = reshape_flat_to_3d(X_test_flat.values, TIME_STEPS, NUM_FEATURES)
    except ValueError:
        return # Error message printed in reshape function

    # 5. Apply Window-Level Standardization
    # Apply standardization to the 3D arrays
    X_train_std_win = standardize_per_window(X_train_3d)
    X_test_std_win = standardize_per_window(X_test_3d) # This is the final unbalanced test set features

    # --- Balancing Steps (Applied AFTER window-level standardization) ---

    # 6. Apply SMOTE (to Window-Standardized Training Data ONLY)
    # SMOTE requires 2D input, so flatten -> SMOTE -> reshape back
    print("\nApplying SMOTE to window-standardized training data...")
    print(f"Original train class distribution:\n{y_train.value_counts(normalize=True)}")
    n_samples_train, steps, feats = X_train_std_win.shape
    X_train_flat_for_smote = X_train_std_win.reshape((n_samples_train, steps * feats))

    smote = SMOTE(random_state=seed, n_jobs=-1)
    try:
        X_train_smote_flat, y_train_smote = smote.fit_resample(X_train_flat_for_smote, y_train)
        print(f"SMOTE balanced train shape (flat): {X_train_smote_flat.shape}")
        print(f"SMOTE balanced train class distribution:\n{pd.Series(y_train_smote).value_counts(normalize=True)}")

        # Reshape SMOTE data back to 3D for saving
        X_train_smote_3d = reshape_flat_to_3d(X_train_smote_flat, TIME_STEPS, NUM_FEATURES)

    except Exception as e:
        print(f"ERROR during SMOTE: {e}. Skipping SMOTE balancing. Using original window-standardized training data.")
        X_train_smote_3d = X_train_std_win.copy() # Fallback
        y_train_smote = y_train.copy() # Fallback


    # 7. Create Balanced Test Set using RUS (on Window-Standardized Test Data)
    # RUS also requires 2D input: flatten -> RUS -> reshape back
    print("\nApplying Random Under-Sampling (RUS) to window-standardized test data...")
    print(f"Original test class distribution:\n{y_test.value_counts(normalize=True)}")
    n_samples_test = X_test_std_win.shape[0]
    X_test_flat_for_rus = X_test_std_win.reshape((n_samples_test, steps * feats))

    rus = RandomUnderSampler(random_state=seed)
    X_test_rus_3d = None # Initialize in case RUS fails
    y_test_rus = None
    try:
        X_test_rus_flat, y_test_rus = rus.fit_resample(X_test_flat_for_rus, y_test)
        print(f"RUS balanced test shape (flat): {X_test_rus_flat.shape}")
        print(f"RUS balanced test class distribution:\n{pd.Series(y_test_rus).value_counts(normalize=True)}")

        # Reshape RUS data back to 3D for saving
        X_test_rus_3d = reshape_flat_to_3d(X_test_rus_flat, TIME_STEPS, NUM_FEATURES)

    except Exception as e:
        print(f"ERROR during RUS: {e}. Skipping RUS balancing for balanced test set.")
        # Leave X_test_rus_3d and y_test_rus as None


    # 8. Save the processed 3D datasets
    print(f"\nSaving processed 3D datasets to {output_dir}...")

    try:
        # --- Save files with updated naming convention ---
        # Training set (Window Std + SMOTE)
        np.save(os.path.join(output_dir, 'X_train_win_std_smote.npy'), X_train_smote_3d)
        np.save(os.path.join(output_dir, 'y_train_smote.npy'), y_train_smote) # Labels from SMOTE

        # Unbalanced Test set (Window Std)
        np.save(os.path.join(output_dir, 'X_test_win_std_unbalanced.npy'), X_test_std_win)
        np.save(os.path.join(output_dir, 'y_test_unbalanced.npy'), y_test.values) # Original unbalanced labels
        #patient IDs for unbalanced test set
        np.save(os.path.join(output_dir, 'patient_ids_test_unbalanced.npy'), groups_test.values) # Save corresponding Patient IDs
        # Balanced Test set (Window Std + RUS), if RUS succeeded
        if X_test_rus_3d is not None:
            np.save(os.path.join(output_dir, 'X_test_win_std_rus.npy'), X_test_rus_3d)
            np.save(os.path.join(output_dir, 'y_test_rus.npy'), y_test_rus) # Labels from RUS
        else:
            print("Skipping save for RUS balanced test set as RUS failed or was skipped.")

        print("Datasets saved successfully.")
    except Exception as e:
        print(f"ERROR saving datasets: {e}")

    print(f"--- Final Data Preparation Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare final datasets for ML with window-level standardization.")
    parser.add_argument("--input_csv", type=str, default=SHHS2_CSV_ALL,
                        help=f"Path to the combined processed CSV file (default: {SHHS2_CSV_ALL})")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help=f"Directory to save the final .npy datasets (default: {OUTPUT_DIR})")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE,
                        help=f"Proportion of data for the test set (default: {TEST_SIZE})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help=f"Random seed for reproducibility (default: {RANDOM_SEED})")

    args = parser.parse_args()

    prepare_final_datasets(args.input_csv, args.output_dir, args.test_size, args.seed)