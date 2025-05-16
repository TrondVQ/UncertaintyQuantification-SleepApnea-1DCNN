import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import argparse


# --- Configuration ---
SEED_BASE = 2025
NUM_MODELS_TO_TRAIN = 5
PROCESSED_DATA_DIR = "./processed_datasets"
MODEL_SAVE_DIR_CNN = "./models/ensemble_cnn_no_pool"
NUM_EPOCHS = 50
BATCH_SIZE = 1024

# Early Stopping Parameters (adjust patience as needed)
EARLY_STOPPING_PATIENCE = 5 # Number of epochs with no improvement after which training will be stopped
VALIDATION_SPLIT_PERCENTAGE = 0.1 # Use 10% of the SMOTE training data for validation


def al_1d_cnn_create_model(input_shape_arg):

    #Creates the 1D CNN model based on Alarcón et al. parameters,
    # adapted for temporal input.

    model = Sequential(name="Alarcon_1D_CNN_Model_4")
    model.add(Input(shape=input_shape_arg)) # Use (60, 4) shape

    # 1st Hidden Layer
    model.add(Conv1D(filters=128, kernel_size=7, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # 2nd Hidden Layer
    model.add(Conv1D(filters=192, kernel_size=5, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # 3rd Hidden Layer
    model.add(Conv1D(filters=224, kernel_size=3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))

    # 4th Hidden Layer
    model.add(Conv1D(filters=96, kernel_size=7, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # 5th Hidden Layer
    model.add(Conv1D(filters=256, kernel_size=9, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # 6th Hidden Layer
    model.add(Conv1D(filters=96, kernel_size=9, activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    # Following alarcon
    model.add(GlobalAveragePooling1D())
    model.add(Dense(units=1, activation='sigmoid'))  # Binary classification

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model


# --- Main Training Function ---
def train_ensemble(model_type: str, num_models: int, seed_base: int):
    """Trains an ensemble of models with different random seeds, validation, and early stopping."""

    # Added check to ensure only 'cnn' is processed
    if model_type.lower() != 'cnn':
        print(f"Error: This script is configured to train only 'cnn' models, but received '{model_type}'.")
        return

    print(f"\n=== Training {num_models} CNN Ensemble Models ===")

    # 1. Load Training Data
    print("Loading SMOTE-balanced training data...")
    try:
        X_train_smote = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train_std_smote.npy'))
        y_train_smote = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train_smote.npy'))
        print(f"Training data shape: {X_train_smote.shape}")
    except FileNotFoundError as e:
        print(f"Error loading training data: {e}")
        print("Ensure 'finalize_datasets.py' has run successfully.")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return


    # 2. Determine Input Shape for the Model
    # Correct input shape for a Keras model is (time_steps, features)
    # The loaded data is already (samples, time_steps, features)
    if X_train_smote.ndim != 3:
        print(f"Error: Expected 3D data (samples, time_steps, features), but got {X_train_smote.ndim} dimensions.")
        return
    # Get time_steps (dimension 1) and features (dimension 2)
    correct_input_shape = (X_train_smote.shape[1], X_train_smote.shape[2])
    print(f"Model input shape will be: {correct_input_shape} (time_steps, features)")

    # 3. Select Model Creation Function and Save Directory (Simplified for only CNN)
    create_fn = al_1d_cnn_create_model
    save_dir = MODEL_SAVE_DIR_CNN
    model_prefix = "AlCNN_smote_seed"

    os.makedirs(save_dir, exist_ok=True)


    # 4. Loop, Train, and Save Each Model
    for i in range(num_models):
        current_seed = seed_base + i
        model_save_path = os.path.join(save_dir, f"{model_prefix}{21+i}.keras")

        # Skip training if model file already exists
        if os.path.exists(model_save_path):
            print(f"\n--- Model {i+1}/{num_models} (Seed: {current_seed}) already exists. Skipping training. ---")
            continue # Move to the next model in the loop

        print(f"\n--- Training Model {i+1}/{num_models} (Seed: {current_seed}) ---")
        print(f"Model save path: {model_save_path}")

        # Set seed for reproducibility for this specific model BEFORE model creation and training
        # This affects weight initialization and potentially data shuffling if shuffle=True (default in fit)
        tf.random.set_seed(current_seed)
        np.random.seed(current_seed)
        # Consider also setting global seed for non-determinism sources if needed:
        # tf.config.experimental.enable_op_determinism() # TensorFlow 2.8+

        # Create a fresh model instance for this seed
        model = create_fn(correct_input_shape)
        # model.summary() # Optional: print summary for first model


        # Define Callbacks for this specific model training run
        early_stopping = EarlyStopping(
            monitor='val_loss',        # Metric to monitor
            patience=EARLY_STOPPING_PATIENCE, # How many epochs to wait
            restore_best_weights=True # Restore weights from the best epoch
        )

        # Train the model using the SMOTE data
        print(f"Training model {i+1} with Early Stopping (Patience={EARLY_STOPPING_PATIENCE}) and {VALIDATION_SPLIT_PERCENTAGE*100}% validation split...")
        history = model.fit(
            X_train_smote, y_train_smote, # Use training data
            epochs=NUM_EPOCHS,             # Max epochs if not stopped early
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT_PERCENTAGE, # Use a portion of training data for validation
            callbacks=[early_stopping],    # Pass the list of callbacks
            verbose=2                      # Show progress per epoch
        )

        # Save the trained model
        # If early stopping triggered, model now holds the weights from the best epoch
        try:
            model.save(model_save_path)
            print(f"Model {i+1} saved successfully (Trained for {len(history.history['loss'])} epochs).")
        except Exception as e:
            print(f"ERROR saving model {i+1}: {e}")

        #  Clear session to free memory before creating the next model
        tf.keras.backend.clear_session()
        del model # delete the model instance

    print(f"\n=== Finished Training CNN Ensemble ===")


if __name__ == "__main__":
    print("--- Running the UPDATED script version ---")
    parser = argparse.ArgumentParser(description="Train CNN ensemble models.") # Updated description
    parser.add_argument("--model_type", type=str, default="cnn", # Only 'cnn' allowed
                        help="Type of model to train (only 'cnn' supported).") # Updated help text
    parser.add_argument("--num_models", type=int, default=NUM_MODELS_TO_TRAIN,
                        help=f"Number of models to train (default: {NUM_MODELS_TO_TRAIN}).")
    parser.add_argument("--seed_base", type=int, default=SEED_BASE,
                        help=f"Base random seed (increments for each model) (default: {SEED_BASE}).")

    args = parser.parse_args()

    print(f"\n=== Starting Ensemble Training Script ===")
    # Train the specified ensemble
    train_ensemble(args.model_type, args.num_models, args.seed_base)
    print(f"=== Ensemble Training Script Finished ===")
