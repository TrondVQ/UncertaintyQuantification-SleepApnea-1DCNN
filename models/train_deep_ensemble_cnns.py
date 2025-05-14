#!/usr/bin/env python3

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling1D, Input # Ensure all needed layers are imported
from tensorflow.keras.models import Sequential, Model # Import Model for type hinting
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import argparse
from typing import Tuple, Optional, NoReturn, List # Import types

# --- Configuration ---
# Base random seed for ensemble training; seeds for individual models will be seed_base + i
SEED_BASE: int = 2025

# Number of individual models to train for the ensemble
NUM_MODELS_TO_TRAIN: int = 5 # Note: The comment in the original code "already have 5 trained" suggests this is the target number.

# Directory where the processed .npy datasets (shape: samples, time_steps, features) are located
PROCESSED_DATA_DIR: str = "./final_processed_datasets" # Should match OUTPUT_DIR in prepare_final_datasets.py

# Directory to save the trained Keras models for the ensemble
MODEL_SAVE_DIR_CNN: str = "./models/cnn_ensemble_no_pool" # Descriptive save directory

# Training parameters for each model
NUM_EPOCHS: int = 50
BATCH_SIZE: int = 1024 # Adjust based on available GPU memory

# Early Stopping configuration for individual model training
EARLY_STOPPING_PATIENCE: int = 5 # Number of epochs with no improvement on validation loss after which training will be stopped
VALIDATION_SPLIT_PERCENTAGE: float = 0.1 # Proportion of the training data to use for validation during fit


# --- Model Definition (Duplicated) ---

def al_1d_cnn_create_model(input_shape: Tuple[int, int]) -> Model:
    """
    Creates a single instance of the 1D CNN model based on Alarcón et al.'s architecture,
    adapted for temporal input data (samples, time_steps, features).

    This function is designed to be called multiple times to create models
    with potentially different initial weights due to random seeds.

    The architecture includes Conv1D layers followed by BatchNormalization
    and Dropout. GlobalAveragePooling1D is used before the final dense layer.
    MaxPooling1D layers from the original code were commented out and are
    not included in this model definition.

    Args:
        input_shape: The shape of the input data EXCLUDING the batch dimension,
                     e.g., (time_steps, n_features) which is (60, 4).

    Returns:
        A compiled TensorFlow Keras Model ready for training.
    """
    print(f"\nDefining 1D CNN model instance with input shape: {input_shape}...")
    model = Sequential(name="Alarcon_1D_CNN_Model_Instance") # Unique name for instances to avoid conflicts if summary is printed
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

    # GlobalAveragePooling1D averages the time steps, preparing for the Dense layer
    model.add(GlobalAveragePooling1D(name='global_avg_pooling_1d'))

    # Output Layer for Binary Classification
    model.add(Dense(units=1, activation='sigmoid', name='output_layer')) # Binary classification

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]) # Include AUC metric

    return model


# --- Main Training Function ---
def train_ensemble_cnn(
        num_models: int = NUM_MODELS_TO_TRAIN,
        seed_base: int = SEED_BASE,
        data_dir: str = PROCESSED_DATA_DIR,
        save_dir: str = MODEL_SAVE_DIR_CNN,
        num_epochs: int = NUM_EPOCHS,
        batch_size: int = BATCH_SIZE,
        validation_split: float = VALIDATION_SPLIT_PERCENTAGE,
        early_stopping_patience: int = EARLY_STOPPING_PATIENCE
) -> NoReturn:
    """
    Trains an ensemble of 1D CNN models with varying random seeds.

    Loads training data, creates multiple instances of the CNN model with
    different seeds, trains each model with validation and early stopping,
    and saves the best version of each trained model.

    Args:
        num_models: The number of individual models to train for the ensemble.
        seed_base: The starting random seed. Each subsequent model uses seed_base + i.
        data_dir: Directory containing the SMOTE-balanced training data.
        save_dir: Directory to save the trained model files (.keras format).
        num_epochs: Maximum number of training epochs for each model.
        batch_size: Batch size for training.
        validation_split: Proportion of training data to use for validation during training.
        early_stopping_patience: Patience value for Early Stopping callback.
    """
    print(f"--- Starting Ensemble Training for {num_models} CNN Models ---")
    print(f"Base Random Seed: {seed_base}")
    print(f"Data Directory: {data_dir}")
    print(f"Model Save Directory: {save_dir}")
    print(f"Max Epochs per Model: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Validation Split: {validation_split * 100}%")
    print(f"Early Stopping Patience: {early_stopping_patience}")
    print("="*70)


    # 1. Load Training Data
    print("Loading SMOTE-balanced training data...")
    try:
        X_train_path = os.path.join(data_dir, 'X_train_win_std_smote.npy') # Using names from prepare_final_datasets.py
        y_train_path = os.path.join(data_dir, 'y_train_smote.npy')

        # Load the data which should have shape (samples, time_steps, features)
        X_train_smote: np.ndarray = np.load(X_train_path)
        y_train_smote: np.ndarray = np.load(y_train_path)

        print(f"Training data loaded successfully. Shape: {X_train_smote.shape}")

        # --- VALIDATION: Check data shape and size ---
        if X_train_smote.ndim != 3:
            print(f"Error: Expected 3D data (samples, time_steps, features), but loaded data has {X_train_smote.ndim} dimensions.")
            print(f"Please ensure '{X_train_path}' contains data in the correct format.")
            return # Exit if data shape is wrong
        if X_train_smote.size == 0:
            print(f"Error: Loaded training data from '{X_train_path}' is empty.")
            return # Exit if data is empty

        # Determine input shape for the model dynamically from the loaded data
        # Correct input shape for a Keras model is (time_steps, n_features)
        time_steps: int = X_train_smote.shape[1]
        n_features: int = X_train_smote.shape[2]
        correct_input_shape: Tuple[int, int] = (time_steps, n_features)
        print(f"Inferred model input shape from data: {correct_input_shape} (time_steps, features)")

    except FileNotFoundError as e:
        print(f"Error loading training data: {e}")
        print(f"Ensure the data directory '{data_dir}' exists and contains '{os.path.basename(X_train_path)}' and '{os.path.basename(y_train_path)}'.")
        print("Ensure 'prepare_final_datasets.py' has run successfully with the correct output directory.")
        return # Exit on file not found
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return # Exit on other loading errors


    # 2. Create Save Directory
    try:
        os.makedirs(save_dir, exist_ok=True) # Create save directory if it doesn't exist
        print(f"Model save directory '{save_dir}' ensured to exist.")
    except Exception as e:
        print(f"Error creating model save directory '{save_dir}': {e}")
        return # Exit if directory creation fails


    # 3. Loop, Train, and Save Each Model Instance
    for i in range(num_models):
        # Calculate the random seed for the current model
        current_seed = seed_base + i
        # Define the save path for the current model
        # Naming convention: AlCNN_smote_seed[base + i].keras
        model_save_path = os.path.join(save_dir, f"AlCNN_smote_seed{current_seed}.keras")

        # --- Skip training if model file already exists ---
        if os.path.exists(model_save_path):
            print(f"\n--- Model {i+1}/{num_models} (Seed: {current_seed}) already exists at '{os.path.basename(model_save_path)}'. Skipping training. ---")
            continue # Move to the next model in the loop


        print(f"\n--- Training Model {i+1}/{num_models} (Seed: {current_seed}) ---")
        print(f"Saving trained model to: {model_save_path}")

        # Set random seed for reproducibility for this specific model BEFORE model creation and training
        # This affects weight initialization and data shuffling during training.
        tf.random.set_seed(current_seed)
        np.random.seed(current_seed)
        # Note: For absolute determinism across different environments/hardware,
        # additional steps might be needed depending on the TensorFlow version and operations used.
        # tf.config.experimental.enable_op_determinism() # Available in TF 2.8+


        # Create a fresh model instance for this seed
        # Pass the determined input shape from the loaded data
        model = al_1d_cnn_create_model(correct_input_shape)
        # model.summary() # Optional: uncomment to print summary for each model instance


        # Define Early Stopping callback for this specific training run
        early_stopping = EarlyStopping(
            monitor='val_loss',        # Metric to monitor (validation loss is common)
            patience=early_stopping_patience, # How many epochs to wait for improvement
            restore_best_weights=True # Restore weights from the epoch with the best value of the monitored quantity
        )

        # Train the model using the SMOTE-balanced training data
        print(f"Training model {i+1} with Early Stopping (Patience={early_stopping_patience}) and {validation_split*100}% validation split...")
        history = model.fit(
            X_train_smote, y_train_smote, # Use training data
            epochs=num_epochs,             # Maximum number of epochs
            batch_size=batch_size,
            validation_split=validation_split, # Use a portion of training data for validation
            callbacks=[early_stopping],    # Apply early stopping
            verbose=2                      # Show progress per epoch (0=silent, 1=progress bar, 2=one line per epoch)
        )

        # Save the trained model
        # If early stopping was triggered, 'model' now holds the weights from the best epoch.
        try:
            model.save(model_save_path)
            print(f"Model {i+1} saved successfully to '{os.path.basename(model_save_path)}' (Trained for {len(history.history.get('loss', []))} epochs).")
        except Exception as e:
            print(f"ERROR saving model {i+1} to '{os.path.basename(model_save_path)}': {e}")

        # Clear the Keras session to free up memory before creating the next model instance
        tf.keras.backend.clear_session()
        del model # Explicitly delete the model instance to help garbage collection

    print(f"\n=== Finished Training {num_models} CNN Ensemble Models ===")


if __name__ == "__main__":
    print("--- Running the CNN Ensemble Training Script ---")

    # Setup argparse to allow configuring paths and parameters from the command line
    parser = argparse.ArgumentParser(
        description="Train a specified number of CNN ensemble models with different random seeds."
    )
    # Note: The --model_type argument exists in the original code, but this script
    # is hardcoded for CNNs. Keeping the argument for compatibility but adding a note.
    parser.add_argument(
        "--model_type",
        type=str,
        default="cnn",
        help="Type of model to train (only 'cnn' is supported by this script). This argument is primarily for script compatibility.",
        choices=["cnn"] # Restrict choices to 'cnn' to be explicit
    )
    parser.add_argument(
        "--num_models",
        type=int,
        default=NUM_MODELS_TO_TRAIN,
        help=f"Number of individual models to train for the ensemble (default: {NUM_MODELS_TO_TRAIN})."
    )
    parser.add_argument(
        "--seed_base",
        type=int,
        default=SEED_BASE,
        help=f"Base random seed. Seeds for individual models will be seed_base, seed_base+1, ... (default: {SEED_BASE})."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=PROCESSED_DATA_DIR,
        help=f"Directory containing the processed .npy training datasets (default: {PROCESSED_DATA_DIR})"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=MODEL_SAVE_DIR_CNN,
        help=f"Directory to save the trained Keras model files (default: {MODEL_SAVE_DIR_CNN})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Maximum number of training epochs for each model (default: {NUM_EPOCHS})"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for training (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=VALIDATION_SPLIT_PERCENTAGE,
        help=f"Proportion of training data to use for validation during training (default: {VALIDATION_SPLIT_PERCENTAGE})"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help=f"Patience value for Early Stopping callback (default: {EARLY_STOPPING_PATIENCE})"
    )


    args = parser.parse_args()

    # Check if the model_type is 'cnn' as this script only supports CNNs
    if args.model_type.lower() != 'cnn':
        print(f"\nWarning: This script is designed to train only 'cnn' models. Ignoring model_type '{args.model_type}' and proceeding with CNN training.")
        # Optionally exit here if you strictly want to enforce the 'choices' in argparse
        # parser.error("This script only supports training 'cnn' models.")


    # Run the ensemble training with parameters from argparse
    train_ensemble_cnn(
        num_models=args.num_models,
        seed_base=args.seed_base,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        early_stopping_patience=args.early_stopping_patience
    )

    print(f"=== CNN Ensemble Training Script Finished ===")