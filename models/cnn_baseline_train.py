#!/usr/bin/env python3

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Dense, BatchNormalization, Dropout, Input, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from typing import Tuple, Optional, NoReturn #
import argparse
# Adjust the import path if your file structure is different.
from evaluation.evaluate_classification import evaluate_classification_model


# --- Configuration ---
# Random seed for reproducibility of model initialization, weights, and data shuffling
SEED: int = 2025

# --- Define default paths and parameters (can be overridden by command-line args) ---
# Default directory where the processed .npy datasets (shape: samples, time_steps, features) are located
PROCESSED_DATA_DIR: str = "./final_processed_datasets" # Should match OUTPUT_DIR in prepare_final_datasets.py

# Default path to save the trained Keras model
MODEL_SAVE_PATH: str = "./alarcon_cnn_model.keras" # More descriptive default name

# Training parameters
NUM_EPOCHS: int = 30
BATCH_SIZE: int = 1024 # Adjust based on available GPU memory

# Early Stopping configuration
EARLY_STOPPING_PATIENCE: int = 5 # Number of epochs with no improvement after which training will be stopped.


# --- Helper Functions ---

def al_1d_cnn_create_model(input_shape: Tuple[int, int]) -> Model:
    """
    Creates the 1D CNN model based on Alarcón et al.'s architecture,
    adapted for temporal input data (samples, time_steps, features).

    The architecture includes Conv1D layers followed by BatchNormalization
    and Dropout. GlobalAveragePooling1D is used before the final dense layer.
    MaxPooling1D layers from the original code were commented out and are
    not included in this model definition based on the provided code snippet.

    Args:
        input_shape: The shape of the input data EXCLUDING the batch dimension,
                     e.g., (time_steps, n_features) which is (60, 4).

    Returns:
        A compiled TensorFlow Keras Model ready for training.
    """
    print(f"\nDefining 1D CNN model with input shape: {input_shape}...")
    model = Sequential(name="Alarcon_1D_CNN_Model") # Changed name slightly for clarity
    model.add(Input(shape=input_shape, name='input_layer'))

    # 1st Hidden Layer
    model.add(Conv1D(filters=128, kernel_size=7, activation='relu', padding='same', name='conv1d_1'))
    model.add(BatchNormalization(name='batchnorm_1'))
    model.add(Dropout(0.3, name='dropout_1'))
    # 2nd Hidden Layer
    model.add(Conv1D(filters=192, kernel_size=5, activation='relu', padding='same', name='conv1d_2'))
    model.add(BatchNormalization(name='batchnorm_2'))
    model.add(Dropout(0.3, name='dropout_2'))

    # 3rd Hidden Layer
    model.add(Conv1D(filters=224, kernel_size=3, activation='relu', padding='same', name='conv1d_3'))
    model.add(BatchNormalization(name='batchnorm_3'))
    model.add(Dropout(0.4, name='dropout_3'))

    # 4th Hidden Layer
    model.add(Conv1D(filters=96, kernel_size=7, activation='relu', padding='same', name='conv1d_4'))
    model.add(BatchNormalization(name='batchnorm_4'))
    model.add(Dropout(0.2, name='dropout_4'))

    # 5th Hidden Layer
    model.add(Conv1D(filters=256, kernel_size=9, activation='relu', padding='same', name='conv1d_5'))
    model.add(BatchNormalization(name='batchnorm_5'))
    model.add(Dropout(0.3, name='dropout_5'))

    # 6th Hidden Layer
    model.add(Conv1D(filters=96, kernel_size=9, activation='relu', padding='same', name='conv1d_6'))
    model.add(BatchNormalization(name='batchnorm_6'))
    
    model.add(Dropout(0.5, name='dropout_6'))

    # Following Alarcon et al. strategy of using pooling/flattening before Dense layers
    # GlobalAveragePooling1D reduces the temporal dimension by averaging,
    # effectively flattening the output for the Dense layers.
    model.add(GlobalAveragePooling1D(name='global_avg_pooling_1d'))

    # Output Layer for Binary Classification
    model.add(Dense(units=1, activation='sigmoid', name='output_layer')) # Binary classification

    # Compile the model
    # Using Adam optimizer with a common learning rate
    # Loss function is binary_crossentropy for binary classification
    # Metrics include accuracy and AUC (Area Under the ROC Curve)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    return model


# --- Main Script Execution ---

def run_cnn_experiment(
        data_dir: str = PROCESSED_DATA_DIR,
        model_save_path: str = MODEL_SAVE_PATH,
        num_epochs: int = NUM_EPOCHS,
        batch_size: int = BATCH_SIZE,
        seed: int = SEED,
        early_stopping_patience: int = EARLY_STOPPING_PATIENCE
) -> NoReturn:
    """
    Main function to load data, create, train, save, and evaluate the 1D CNN model.

    Args:
        data_dir: Directory containing the processed .npy datasets.
        model_save_path: File path to save the trained Keras model.
        num_epochs: Number of training epochs.
        batch_size: Batch size for training.
        seed: Random seed for reproducibility.
        early_stopping_patience: Patience value for Early Stopping callback.
    """
    print(f"--- Starting 1D CNN Model Experiment ---")
    print(f"Data Directory: {data_dir}")
    print(f"Model Save Path: {model_save_path}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Random Seed: {seed}")
    print(f"Early Stopping Patience: {early_stopping_patience}")
    print("="*70)

    # Set random seeds for reproducibility
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # --- 1. Load Processed Data ---
    print(f"\nLoading pre-processed temporal datasets from {data_dir}...")
    try:
        # Construct full file paths
        X_train_path = os.path.join(data_dir, 'X_train_win_std_smote.npy') # Using names from prepare_final_datasets.py
        y_train_path = os.path.join(data_dir, 'y_train_smote.npy')
        X_test_unbalanced_path = os.path.join(data_dir, 'X_test_win_std_unbalanced.npy')
        y_test_unbalanced_path = os.path.join(data_dir, 'y_test_unbalanced.npy')
        X_test_rus_path = os.path.join(data_dir, 'X_test_win_std_rus.npy')
        y_test_rus_path = os.path.join(data_dir, 'y_test_rus.npy')

        # Load the data which should now have shape (samples, 60, 4)
        X_train_smote: np.ndarray = np.load(X_train_path)
        y_train_smote: np.ndarray = np.load(y_train_path)
        X_test_std_unbalanced: np.ndarray = np.load(X_test_unbalanced_path)
        y_test_unbalanced: np.ndarray = np.load(y_test_unbalanced_path)
        X_test_std_rus: np.ndarray = np.load(X_test_rus_path)
        y_test_rus: np.ndarray = np.load(y_test_rus_path)

        print("Datasets loaded successfully.")
        print(f"X_train_smote shape: {X_train_smote.shape}") # Should be (n_samples, 60, 4)
        print(f"y_train_smote shape: {y_train_smote.shape}")
        print(f"X_test_std_unbalanced shape: {X_test_std_unbalanced.shape}")
        print(f"y_test_unbalanced shape: {y_test_unbalanced.shape}")
        print(f"X_test_std_rus shape: {X_test_std_rus.shape}")
        print(f"y_test_rus shape: {y_test_rus.shape}")

        # --- VALIDATION: Check data shape and size ---
        expected_shape = (X_train_smote.shape[0], 60, 4) # Assuming first dim is variable batch size
        if len(X_train_smote.shape) != 3 or X_train_smote.shape[1] != 60 or X_train_smote.shape[2] != 4:
            print(f"ERROR: Expected X_train shape like (samples, 60, 4), but got {X_train_smote.shape}")
            print("Please ensure you are loading the correctly processed TEMPORAL data (shape: samples, time_steps, features).")
            exit()

        if X_train_smote.size == 0 or X_test_std_unbalanced.size == 0 or X_test_std_rus.size == 0:
            print("ERROR: One or more loaded datasets are empty.")
            exit()


    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(f"Please ensure the directory '{data_dir}' exists and contains the correct .npy files from the data preparation script.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        exit()


    # --- 2. Determine Input Shape for Model ---
    # Input shape for Conv1D layers is (time_steps, n_features)
    time_steps: int = X_train_smote.shape[1] # Should be 60
    n_features: int = X_train_smote.shape[2] # Should be 4
    input_shape: Tuple[int, int] = (time_steps, n_features) # e.g., (60, 4)

    print(f"\nDetermined input shape for Conv1D layers: {input_shape}")

    # --- 3. Create the Model ---
    model: Model = al_1d_cnn_create_model(input_shape)
    model.summary() # Print the model summary to review layers and parameters

    # --- 4. Train the Model ---
    print(f"\nTraining model on SMOTE-balanced training data (N={X_train_smote.shape[0]}) for {num_epochs} epochs...")
    # Use EarlyStopping to prevent overfitting and restore best weights
    early_stopping = EarlyStopping(
        monitor='val_loss', # Monitor validation loss
        patience=early_stopping_patience, # Number of epochs with no improvement to wait
        restore_best_weights=True # Restore weights from the epoch with the best value of the monitored quantity
    )

    history = model.fit(
        X_train_smote, y_train_smote,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.1, # Use 10% of training data for validation during training
        callbacks=[early_stopping], # Apply early stopping
        verbose=2 # Print one line per epoch
    )

    print("\nTraining complete.")

    # --- 5. Save the Trained Model ---
    print(f"Saving trained model to {model_save_path}...")
    try:
        # Ensure the directory for the model save path exists
        model_dir = os.path.dirname(model_save_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created model save directory: {model_dir}")

        model.save(model_save_path)
        print(f"Model saved successfully to '{model_save_path}'.")
    except Exception as e:
        print(f"Error saving model: {e}")


    # --- 6. Evaluate the Model ---

    # Evaluation function is imported from evaluation.evaluate_classification
    print("\n" + "="*70)
    print("--- Evaluating the Trained Model ---")

    # Evaluate on the original UNBALANCED Test Set
    if X_test_std_unbalanced.size > 0:
        print("\nEvaluating on UNBALANCED Test Set...")
        evaluate_classification_model(
            model=model,
            X_test=X_test_std_unbalanced,
            y_test=y_test_unbalanced,
            evaluation_description="CNN - Unbalanced Test Set"
        )
    else:
        print("\nSkipping evaluation on UNBALANCED Test Set: Data is empty.")


    # Evaluate on the BALANCED Test Set (using Random Under-Sampling)
    if X_test_std_rus.size > 0:
        print("\nEvaluating on BALANCED (RUS) Test Set...")
        evaluate_classification_model(
            model=model,
            X_test=X_test_std_rus,
            y_test=y_test_rus,
            evaluation_description="CNN - Balanced Test Set (RUS)"
        )
    else:
        print("\nSkipping evaluation on BALANCED (RUS) Test Set: Data is empty (RUS may have failed or was skipped).")


    print("\n" + "="*70)
    print("--- 1D CNN Model Experiment Script Finished ---")


if __name__ == "__main__":
    # Setup argparse to allow configuring paths and parameters from the command line
    parser = argparse.ArgumentParser(
        description="Train, save, and evaluate a 1D CNN model on processed time-series data."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=PROCESSED_DATA_DIR,
        help=f"Directory containing the processed .npy datasets (default: {PROCESSED_DATA_DIR})"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default=MODEL_SAVE_PATH,
        help=f"File path to save the trained Keras model (default: {MODEL_SAVE_PATH})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for training (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed for reproducibility (default: {SEED})"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help=f"Patience value for Early Stopping callback (default: {EARLY_STOPPING_PATIENCE})"
    )

    args = parser.parse_args()

    # Run the experiment with parameters from argparse
    run_cnn_experiment(
        data_dir=args.data_dir,
        model_save_path=args.model_save_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience
    )