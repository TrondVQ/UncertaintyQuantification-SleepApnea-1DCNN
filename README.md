# Uncertainty Quantification for Deep Learning in Sleep Apnea Detection

This repository contains the Python code for the Master's Thesis: "Uncertainty Quantification for Deep Learning in Sleep Apnea Detection: A Comparative Evaluation of Monte Carlo Dropout and Deep Ensembles using a 1D-CNN Model" by Trond Victor Qian, submitted to the Department of Informatics, Faculty of Mathematics and Natural Sciences, University of Oslo, Spring 2025.

**Author:** Trond Victor Qian
**Supervisors:** Thomas Peter Plagemann, Marta Quemada Lopez
**Thesis Document:** TBA

## Abstract
Obstructive Sleep Apnea (OSA) is a common sleep disorder with serious health consequences, yet its diagnosis relies on resource-intensive polysomnography and subjective manual scoring, which often leads to inter-scorer variability, limiting accessibility and consistency. Although deep learning models show promise for automating OSA event detection, their "black box" nature and lack of built-in reliability measures hinder clinical adoption.

This thesis addresses this gap by systematically evaluating two prominent uncertainty quantification (UQ) techniques: Monte Carlo (MC) Dropout and Deep Ensembles (DE), applied to a validated 1D-Convolutional Neural Network (1D-CNN) for epoch-level sleep apnea event classification using the Sleep Heart Health Study 2 (SHHS2) dataset.

Through replication of a benchmark 1D-CNN and a multi-level analysis of uncertainty characteristics, key findings reveal that both methods can indicate prediction reliability but exhibit distinct uncertainty profiles. DE demonstrated a superior ability to identify a large subset of predictions with very high accuracy (over 99\%), with its uncertainty estimates showing a stronger correlation with patient-level accuracy. MC Dropout, while more computationally efficient, produced higher overall uncertainty estimates and was more sensitive to test set imbalance. Both methods confirmed that aleatoric (data-related) uncertainty was dominant in this task, and higher window-level uncertainty strongly correlated with increased prediction error.

These findings provide empirical evidence of the practical trade-offs between MC Dropout and Deep Ensembles for UQ in sleep apnea detection, highlighting their potential to improve model transparency, identify trustworthy predictions, and support more robust, clinically translatable AI tools for sleep medicine.

## Repository Structure

This repository contains the Python scripts to reproduce the core experiments of the thesis:
* Data preprocessing for the SHHS2 dataset.
* Implementation and training of the baseline 1D-CNN model.
* Training of the Deep Ensemble models.
* Application and evaluation of Monte Carlo Dropout and Deep Ensembles for uncertainty quantification.
* Scripts for various analyses performed (e.g., patient-level, window-level correlations).

**Key Scripts:**
* `data_prepocessing/preprocess_shhs_raw.py`: Preprocesses individual SHHS2 EDF and XML files.
* `data_prepocessing/prepare_numpy_datasets.py`: Finalizes the dataset into NumPy arrays for training/testing, including splitting and SMOTE balancing for the training set.
* `models/cnn_baseline_train.py`: Defines and trains the baseline 1D-CNN model.
* `models/train_deep_ensemble_cnns.py`: Trains the ensemble of 1D-CNN models for Deep Ensembles.
* `uncertainty_quantification/uq_techniques.py`: Core functions for MC Dropout, Deep Ensemble predictions, and UQ metric calculations.
* `evaluation/evaluate_classification.py`: Functions for standard classification model evaluation.
* `uncertainty_quantification/analyze_mcd_patient_level.py`: Main script for running MC Dropout, generating detailed per-window UQ metrics, and evaluating.
* `uncertainty_quantification/analyze_de_patient_level.py`: Main script for running Deep Ensembles, generating detailed per-window UQ metrics, and evaluating.
* `uncertainty_quantification/aggregate_patient_uq_metrics.py`: Aggregates window-level UQ results to patient-level summaries.
* `uncertainty_quantification/evaluate_mcd_global.py`: Script for evaluating MC Dropout with a focus on aggregated global metrics.
* `uncertainty_quantification/evaluate_de_global.py`: Script for evaluating Deep Ensembles with a focus on aggregated global metrics.
* Analysis scripts in `uq_analysis/` (e.g., `final_plot_uq_overview_figures.py`, `patient_accuracy_entropy_correlation.py`, `window_uncertainty_vs_correctness_mannwhitney.py`): Scripts for specific UQ result analyses and plotting. Note: These may require adaptation if run independently as they were used to generate specific figures/tables in the thesis from saved intermediate results.

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
3.  Modify the paths in the preprocessing scripts (`data_prepocessing/preprocess_shhs_raw.py` and potentially `data_prepocessing/prepare_numpy_datasets.py`) to point to your local SHHS2 data directories. The scripts assume specific file naming conventions from the NSRR.

The code provided here is for the methodology; data procurement and management are the user's responsibility.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/TrondVQ/UncertaintyQuantification-SleepApnea-1DCNN.git](https://github.com/TrondVQ/UncertaintyQuantification-SleepApnea-1DCNN.git)
    cd UncertaintyQuantification-SleepApnea-1DCNN
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
    * Run `python data_prepocessing/preprocess_shhs_raw.py` (after configuring paths to your raw SHHS2 EDF and XML folders). This will process individual recordings and create an intermediate CSV file (e.g., `processed_shhs2_data.csv`).
    * Run `python data_prepocessing/prepare_numpy_datasets.py --input_csv ./processed_shhs2_data.csv --output_dir ./final_processed_datasets` (adjust paths as needed). This will generate the final `X_train_win_std_smote.npy`, `y_train_smote.npy`, `X_test_win_std_unbalanced.npy`, `y_test_unbalanced.npy`, `patient_ids_test_unbalanced.npy`, etc., in your specified output directory.

2.  **Model Training:**
    * Train the baseline 1D-CNN: `python models/cnn_baseline_train.py --data_dir ./final_processed_datasets --model_save_path ./alarcon_cnn_model.keras` (ensure paths are correct).
    * Train the Deep Ensemble models: `python models/train_deep_ensemble_cnns.py --data_dir ./final_processed_datasets --save_dir ./models/cnn_ensemble_no_pool` (ensure paths are correct).

3.  **Uncertainty Quantification Evaluation (Detailed Per-Window Analysis):**
    * Run MC Dropout evaluation: `python uncertainty_quantification/analyze_mcd_patient_level.py --data_dir ./final_processed_datasets --model_path ./alarcon_cnn_model.keras --output_csv_dir ./uq_results/mc_dropout --save_unbalanced_csv` (adjust paths and flags as needed).
    * Run Deep Ensemble evaluation: `python uncertainty_quantification/analyze_de_patient_level.py --data_dir ./final_processed_datasets --model_dir ./models/cnn_ensemble_no_pool --output_csv_dir ./uq_results/deep_ensemble --save_unbalanced_csv` (adjust paths and flags as needed).

4.  **Further Aggregation and Analysis Scripts:**
    * Aggregate patient-level metrics: `python uncertainty_quantification/aggregate_patient_uq_metrics.py --input_csv ./uq_results/mc_dropout/detailed_results_CNN_MCD_Unbalanced.csv --output_dir ./uq_results/mc_dropout/patient_level_summary` (example for MCD, adapt for DE).
    * Scripts in `uq_analysis/` (e.g., `python uq_analysis/final_plot_uq_overview_figures.py`) are designed to work with the CSV files generated by the previous steps. You may need to adjust paths within these scripts. *Note: As per the project's aim to share code and not generated plots, these scripts might need modification if you only intend to share the analytical logic rather than direct plot generation.*

**Example (Conceptual Workflow):**
```bash
# 1. Preprocessing
python data_prepocessing/preprocess_shhs_raw.py --num_files 10 # Process a small number of files first
python data_prepocessing/prepare_numpy_datasets.py --input_csv ./processed_shhs2_data.csv --output_dir ./final_processed_datasets

# 2. Model Training
python models/cnn_baseline_train.py
python models/train_deep_ensemble_cnns.py

# 3. UQ Evaluation (example for MCD)
python uncertainty_quantification/analyze_mcd_patient_level.py --save_unbalanced_csv

# 4. Further Analysis (example for MCD results)
python uncertainty_quantification/aggregate_patient_uq_metrics.py --input_csv ./uq_results/mc_dropout/detailed_results_CNN_MCD_Unbalanced.csv --output_dir ./uq_results/mc_dropout/patient_level_summary
# ... then run plotting scripts from uq_analysis/ using the generated CSVs
