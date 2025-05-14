# Uncertainty Quantification for Deep Learning in Sleep Apnea Detection

This repository contains the Python code for the Master's Thesis: "Uncertainty Quantification for Deep Learning in Sleep Apnea Detection: A Comparative Evaluation of Monte Carlo Dropout and Deep Ensembles using a 1D-CNN Model" by Trond Victor Qian, submitted to the Department of Informatics, Faculty of Mathematics and Natural Sciences, University of Oslo, Spring 2025.

**Author:** Trond Victor Qian
**Supervisors:** Thomas Peter Plagemann, Marta Quemada Lopez
**Thesis Document:** TBA

## Abstract

Obstructive Sleep Apnea (OSA) is a prevalent disorder with significant health consequences, yet its diagnosis is often hindered by resource-intensive polysomnography and variability in manual scoring. While deep learning models show promise for automated OSA event detection, their "black-box" nature and lack of inherent reliability measures limit clinical trust and adoption. This thesis addresses this gap by systematically investigating and comparing two prominent Uncertainty Quantification (UQ) techniques—Monte Carlo (MC) Dropout and Deep Ensembles (DE)—applied to a validated 1D-Convolutional Neural Network (1D-CNN) for epoch-level sleep apnea event classification using the Sleep Heart Health Study 2 (SHHS2) dataset.

This work involved replicating a benchmark 1D-CNN, implementing MCD and DE, and conducting a multi-level analysis of their uncertainty characteristics. Key findings reveal that while both methods can indicate prediction reliability, they exhibit distinct uncertainty profiles. Deep Ensembles demonstrated particular strength in identifying a large subset of predictions with very high accuracy and its uncertainty estimates showed a stronger correlation with patient-level accuracy. Conversely, MC Dropout, while more computationally efficient, yielded higher overall uncertainty estimates. Both methods confirmed that estimated aleatoric (data-related) uncertainty was dominant for this task, and a strong, statistically significant correlation was found between higher window-level uncertainty and increased prediction error. This research demonstrates the potential of UQ to enhance model transparency and support more reliable AI-assisted diagnostics in sleep medicine.

## Repository Structure

This repository contains the Python scripts to reproduce the core experiments of the thesis:
* Data preprocessing for the SHHS2 dataset.
* Implementation and training of the baseline 1D-CNN model.
* Training of the Deep Ensemble models.
* Application and evaluation of Monte Carlo Dropout and Deep Ensembles for uncertainty quantification.
* Scripts for various analyses performed (e.g., patient-level, window-level correlations).

**Key Scripts:**
* `prepro_shhs.py`: Preprocesses individual SHHS2 EDF and XML files.
* `prepdatasetml_to_numpy_files.py`: Finalizes the dataset into NumPy arrays for training/testing, including splitting and SMOTE balancing for the training set.
* `AlCNN1D.py`: Defines and trains the baseline 1D-CNN model.
* `de_train_cnn.py`: Trains the ensemble of 1D-CNN models for Deep Ensembles.
* `uq_techniques.py`: Core functions for MC Dropout, Deep Ensemble predictions, and UQ metric calculations.
* `evaluate_model.py`: Functions for standard classification model evaluation.
* `mc_dropout_cnn.py`: Main script for running and evaluating MC Dropout.
* `deep_ensemble_cnn.py`: Main script for running and evaluating Deep Ensembles.
* Analysis scripts (e.g., `mcd_patient.py`, `DE_patient.py`, `0505UQ_graphs.py`, `0505UQ_pearson.py`, `mann_w_window.py`, etc.): Scripts for specific UQ result analyses. Note: These may require adaptation if run independently as they were used to generate specific figures/tables in the thesis from saved intermediate results.

## Methodology and Citation

The 1D-CNN architecture and primary data preprocessing pipeline implemented in this repository are based on the work by:

* Alarcón, Á. S., Madrid, N. M., Seepold, R., & Ortega, J. A. (2023). Obstructive sleep apnea event detection using explainable deep learning models for a portable monitor. *Frontiers in Neuroscience, 17*, 1155900. DOI: 10.3389/fnins.2023.1155900

Please cite this original publication if you use or adapt the baseline model architecture or preprocessing methodology. This repository provides an implementation and further UQ analysis based on their described methods.

* As declared in the thesis, Generative AI tools (ChatGPT Model-4o and Gemini 2.5 Pro Experimental) were used to support the writing, analysis, and code development processes.

## Data Usage - IMPORTANT

Due to privacy regulations and the terms of use for the Sleep Heart Health Study (SHHS2) dataset, **no raw or processed patient data is included in this repository.**

To run the experiments, you will need to:
1.  Obtain access to the SHHS2 dataset through the National Sleep Research Resource (NSRR): [https://sleepdata.org/datasets/shhs](https://sleepdata.org/datasets/shhs)
2.  Download the EDF (polysomnography recordings) and XML (annotation) files.
3.  Modify the paths in the preprocessing scripts (`prepro_shhs.py` and potentially `prepdatasetml_to_numpy_files.py`) to point to your local SHHS2 data directories. The scripts assume specific file naming conventions from the NSRR.

The code provided here is for the methodology; data procurement and management are the user's responsibility.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```
2.  **Create a Python virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    A `requirements.txt` file is provided.
    ```bash
    pip install -r requirements.txt
    ```
    *Key libraries include: TensorFlow (v2.12.0), NumPy (v1.23.5), Scikit-learn (v1.3.2), Imbalanced-learn (v0.12.4), Pandas (v1.5.3), pyedflib, Matplotlib, Seaborn.*

## Running the Code

The general workflow is as follows. Please refer to individual scripts for more specific arguments or configurations. It's recommended to run scripts in sequence, as later scripts depend on the output of earlier ones.

1.  **Preprocessing:**
    * Run `prepro_shhs.py` (after configuring paths to your raw SHHS2 EDF and XML folders). This will process individual recordings and create intermediate CSV files (e.g., `SHHS2_ID_all.csv` as mentioned in `prepdatasetml_to_numpy_files.py`).
    * Run `prepdatasetml_to_numpy_files.py` (after configuring input CSV path and output directory). This will generate the final `X_train_win_std_smote.npy`, `y_train_smote.npy`, `X_test_win_std_unbalanced.npy`, `y_test_unbalanced.npy`, `patient_ids_test_unbalanced.npy`, etc., in your specified `processed_datasets` directory.

2.  **Model Training:**
    * Train the baseline 1D-CNN: `python AlCNN1D.py` (ensure `PROCESSED_DATA_DIR` points to the output of the previous step). This saves the model (e.g., `AlCNN1D_no_pool.keras`).
    * Train the Deep Ensemble models: `python de_train_cnn.py` (ensure `PROCESSED_DATA_DIR` is set and `MODEL_SAVE_DIR_CNN` is configured). This saves multiple models.

3.  **Uncertainty Quantification Evaluation:**
    * Run MC Dropout evaluation: `python mc_dropout_cnn.py` (ensure `PROCESSED_DATA_DIR` and `CNN_MODEL_PATH` are correct).
    * Run Deep Ensemble evaluation: `python deep_ensemble_cnn.py` (ensure `PROCESSED_DATA_DIR` and ensemble model paths/prefix are correct).
    * Run patient-specific UQ analyses: `python mcd_patient.py`, `python DE_patient.py` (these scripts load data and models, and generate detailed CSV outputs for patient-level uncertainty).

4.  **Further Analysis Scripts:**
    * Scripts like `0505UQ_graphs.py`, `0505UQ_pearson.py`, `mann_w_window.py` are designed to work with the CSV files generated by `mcd_patient.py` or `DE_patient.py`, or other intermediate result files. You may need to adjust paths within these scripts. *Note: As per the project's aim to share code and not generated plots, these scripts might need modification if you only intend to share the analytical logic rather than direct plot generation.*

**Example (Conceptual):**
```bash
# After setting up data and paths in scripts:
python prepro_shhs.py --num_files 10 # Process a small number of files first
python prepdatasetml_to_numpy_files.py --input_csv ./SHHS2_ID_all.csv --output_dir ./processed_datasets
python AlCNN1D.py
python mc_dropout_cnn.py
