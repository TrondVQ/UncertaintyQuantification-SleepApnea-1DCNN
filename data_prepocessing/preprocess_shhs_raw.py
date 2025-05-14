#!/usr/bin/env python3

import os
import pandas as pd
from pyedflib import EdfReader
from scipy.signal import resample
import xml.etree.ElementTree as ET
import numpy as np
import argparse
from typing import Dict, List, Any, Optional, Tuple

"""
This script preprocesses the SHHS2 dataset by combining EDF signal files
and XML annotation files. It extracts specific physiological signals,
resamples them to a target frequency, removes simple artifacts, and
segments the data into fixed-size windows. Each window is then labeled
based on overlap with apnea/hypopnea annotations from the XML file.
The resulting segmented and labeled data is saved as a single CSV file.

The script can process a limited number of files specified via a
command-line argument for testing or partial dataset processing.

Example Usage:
    # Process all files
    python3 preprocess_shhs_raw.py

    # Process only 100 files
    python3 preprocess_shhs_raw.py --num_files 100

Alarcon et al. Preprocessing Details (adapted for this script):
- Dataset: SHHS2
- Signals Used: SaO2, PR/H.R., THOR RES, ABDO RES
- Target Sampling Rate: 1 Hz
- Window Size: 60 seconds
- Window Overlap: 0 seconds (non-overlapping windows)
- Patient Exclusion Criteria:
    - Large number of missing values or artifacts (checked per signal).
    - Recording duration (based on "Recording Start Time" event) less than 300 minutes.
- Labeling: A window is labeled as apnea/hypopnea (1) if it overlaps
  with an "Obstructive apnea" or "Hypopnea" event for at least 10 seconds.
  Otherwise, it is labeled as normal (0).

Initial SHHS2 Distribution (as mentioned in original comments):
- Training: 994 patients with apnea, 230 without apnea.
- Test: 132 patients without apnea, 31 with apnea.
(Note: The script itself does not perform the balancing mentioned.)
"""

# --- Configuration ---
# Update these paths to your dataset location and desired output file
EDF_FOLDER: str = ""  # e.g., "../SHHS2_dataset/edf_files"
XML_FOLDER: str = ""  # e.g., "../SHHS2_dataset/annotation_files"

# Output CSV file path
COMBINED_FILE: str = "./processed_shhs2_data.csv" # Recommend a more descriptive name

# Target channels to extract from EDF files
TARGET_CHANNELS: List[str] = ["SaO2", "PR", "THOR RES", "ABDO RES"]

# Target sampling rate for resampling (in Hz)
TARGET_RATE: int = 1

# Window size for segmentation (in seconds)
WINDOW_SIZE: int = 60

# Overlap size for segmentation (in seconds)
OVERLAP_SIZE: int = 0 # Non-overlapping windows

# Threshold for considering a signal segment as having excessive artifacts/missing values
ARTIFACT_THRESHOLD: float = 0.1 # Proportion of artifacts to consider excessive

# Minimum required sleep time for a recording to be included (in seconds)
MIN_SLEEP_TIME: int = 300 * 60 # 300 minutes

# Minimum overlap duration for a window to be labeled as apnea/hypopnea (in seconds)
APNEA_OVERLAP_THRESHOLD: int = 10

# --- Helper Functions ---

def check_artifacts_and_missing_values(
        signals: Dict[str, np.ndarray],
        artifact_threshold: float = ARTIFACT_THRESHOLD
) -> bool:
    """
    Checks if any signal in the dictionary exceeds the artifact/missing value threshold.

    Args:
        signals: Dictionary where keys are channel names and values are signal arrays.
        artifact_threshold: Proportion of artifacts (NaNs) to consider excessive.

    Returns:
        True if all signals are below the threshold, False otherwise.
    """
    for channel, signal in signals.items():
        if signal is None or len(signal) == 0:
            print(f"Warning: Signal '{channel}' is empty or None.")
            return False # Consider empty/None signals as invalid

        num_missing = np.isnan(signal).sum()
        total_length = len(signal)

        if total_length == 0 or (num_missing / total_length > artifact_threshold):
            return False # Exceeds threshold or empty signal

    return True # All signals are valid


def calculate_recording_duration(
        events: List[Dict[str, Any]],
        min_sleep_time: int = MIN_SLEEP_TIME
) -> bool:
    """
    Calculates the recording duration based on the 'Recording Start Time' event
    and checks if it meets the minimum required time.

    Args:
        events: List of parsed event dictionaries from the XML.
        min_sleep_time: Minimum required recording duration in seconds.

    Returns:
        True if the recording duration is sufficient, False otherwise.
    """
    # Find the "Recording Start Time" event and get its duration
    recording_start_time_event = next(
        (event for event in events if event.get("event_concept") == "Recording Start Time"),
        None
    )

    if recording_start_time_event and recording_start_time_event.get("duration") is not None:
        total_duration = recording_start_time_event["duration"]
    else:
        total_duration = 0.0 # Default to 0 if event not found or duration is missing

    print(f"Total recording duration (based on Recording Start Time): {total_duration:.2f} seconds")

    # Check if the total duration meets the minimum required sleep time
    return total_duration >= min_sleep_time


def remove_artifacts(
        signals: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Replaces anomalous values in SaO2 and PR signals with interpolated values.

    For SaO2: Replaces values <80 or >100.
    For PR: Replaces values <40 or >200.
    Other channels are unchanged.

    Args:
        signals: Dictionary of signal arrays.

    Returns:
        The dictionary of signals with artifacts removed via interpolation.
    """
    processed_signals = {}
    for channel, signal in signals.items():
        if signal is None:
            processed_signals[channel] = None
            continue

        modified_signal = signal.copy() # Work on a copy to avoid modifying original input unexpectedl
        mask = np.zeros_like(modified_signal, dtype=bool)

        if channel == "SaO2":
            mask = (modified_signal < 80) | (modified_signal > 100)
        elif channel == "PR":
            mask = (modified_signal < 40) | (modified_signal > 200)
        else:
            # No artifact removal for other channels
            processed_signals[channel] = modified_signal
            continue

        # Interpolate only if there are invalid values and valid surrounding points
        if np.any(mask):
            modified_signal[mask] = np.nan  # Set invalid values to NaN
            # Interpolate NaNs using linear interpolation
            mask_nan = np.isnan(modified_signal)
            valid_indices = np.flatnonzero(~mask_nan)

            if len(valid_indices) > 1: # Need at least two valid points to interpolate
                modified_signal[mask_nan] = np.interp(
                    np.flatnonzero(mask_nan),
                    valid_indices,
                    modified_signal[valid_indices]
                )
            elif len(valid_indices) == 1: # Only one valid point, fill NaNs with this value
                modified_signal[mask_nan] = modified_signal[valid_indices[0]]
            # If len(valid_indices) == 0, all values are NaN, they will remain NaN

        processed_signals[channel] = modified_signal

    return processed_signals


def get_edf_channels(
        file_path: str,
        channels: List[str]
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Reads specified signal channels and their sampling rates from an EDF file.
    Handles alternative names for the 'PR' channel ('H.R.').

    Args:
        file_path: Path to the EDF file.
        channels: List of channel names to extract.

    Returns:
        A tuple containing two dictionaries:
        - signals: Dictionary of extracted signal arrays (channel name -> data).
        - sampling_rates: Dictionary of sampling rates for extracted channels
                          (channel name -> rate).
    """
    signals: Dict[str, np.ndarray] = {}
    sampling_rates: Dict[str, float] = {}
    alternative_pr_names = ["H.R."] # Alternative names for PR

    try:
        with EdfReader(file_path) as f:
            signal_labels = f.getSignalLabels()
            signal_channels_map = {label: i for i, label in enumerate(signal_labels)}

            for channel in channels:
                channel_id = signal_channels_map.get(channel)

                if channel_id is not None:
                    signals[channel] = f.readSignal(channel_id)
                    sampling_rates[channel] = f.getSampleFrequency(channel_id)
                elif channel == "PR":
                    # Try alternative names for PR
                    found_alt = False
                    for alt_name in alternative_pr_names:
                        if alt_name in signal_channels_map:
                            print(f"Found alternative channel '{alt_name}' for 'PR' in {file_path}")
                            channel_id = signal_channels_map[alt_name]
                            signals["PR"] = f.readSignal(channel_id)
                            sampling_rates["PR"] = f.getSampleFrequency(channel_id)
                            found_alt = True
                            break
                    if not found_alt:
                        print(f"Warning: Channel '{channel}' (and alternatives) not found in {file_path}. Skipping.")
                        signals[channel] = np.array([]) # Use empty array to indicate missing
                        sampling_rates[channel] = 0.0 # Use 0 rate for consistency
                else:
                    print(f"Warning: Channel '{channel}' not found in {file_path}. Skipping.")
                    signals[channel] = np.array([]) # Use empty array to indicate missing
                    sampling_rates[channel] = 0.0 # Use 0 rate for consistency

    except Exception as e:
        print(f"Error reading EDF file {file_path}: {e}")
        # Return empty dictionaries if file reading fails
        return {}, {}

    # Check if all target channels were found (even if empty)
    for channel in channels:
        if channel not in signals:
            print(f"Error: Target channel '{channel}' was not processed for {file_path}.")
            # This case should ideally be handled by the logic above, but adding a check.
            signals[channel] = np.array([])
            sampling_rates[channel] = 0.0


    return signals, sampling_rates


def resample_signals(
        signals: Dict[str, np.ndarray],
        sampling_rates: Dict[str, float],
        target_rate: int = TARGET_RATE
) -> Dict[str, np.ndarray]:
    """
    Resamples signal arrays to a target sampling rate using scipy.signal.resample.

    Args:
        signals: Dictionary of signal arrays.
        sampling_rates: Dictionary of original sampling rates for each signal.
        target_rate: The desired sampling rate in Hz.

    Returns:
        A dictionary containing the resampled signal arrays.
    """
    resampled_signals: Dict[str, np.ndarray] = {}
    for channel, signal in signals.items():
        original_rate = sampling_rates.get(channel, 0.0)

        if signal is None or original_rate == 0 or len(signal) == 0:
            print(f"Warning: Cannot resample channel '{channel}' due to missing signal, zero rate, or empty data.")
            resampled_signals[channel] = np.array([]) # Add empty array for consistency
            continue

        # Calculate target length based on time duration
        original_duration = len(signal) / original_rate
        target_length = int(original_duration * target_rate)

        if target_length <= 0:
            print(f"Warning: Target length for channel '{channel}' is zero or negative. Skipping resampling.")
            resampled_signals[channel] = np.array([]) # Add empty array for consistency
            continue

        try:
            # Use resample; it handles the case where original_rate == target_rate
            resampled_signals[channel] = resample(signal, target_length)
        except ValueError as e:
            print(f"Error resampling channel '{channel}': {e}. Original length: {len(signal)}, Target length: {target_length}")
            resampled_signals[channel] = np.array([]) # Add empty array if resampling fails


    return resampled_signals


def parse_xml_annotations(
        xml_file_path: str
) -> List[Dict[str, Any]]:
    """
    Parses relevant sleep/apnea events from an SHHS XML annotation file.
    Stops parsing when a "Stages|Stages" event is encountered.

    Args:
        xml_file_path: Path to the XML annotation file.

    Returns:
        A list of dictionaries, where each dictionary represents a scored event
        with keys 'event_type', 'event_concept', 'start', and 'duration'.
    """
    events: List[Dict[str, Any]] = []
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        for scored_event in root.findall("ScoredEvents/ScoredEvent"):
            event_type_elem = scored_event.find("EventType")
            event_concept_elem = scored_event.find("EventConcept")
            start_elem = scored_event.find("Start")
            duration_elem = scored_event.find("Duration")

            event_type = event_type_elem.text if event_type_elem is not None else None
            event_concept = event_concept_elem.text if event_concept_elem is not None else None
            start = float(start_elem.text) if start_elem is not None and start_elem.text else None
            duration = float(duration_elem.text) if duration_elem is not None and duration_elem.text else None

            # Stop parsing when sleep stages are encountered
            if event_type == "Stages|Stages":
                break

            # Only add events that have a concept, start time, and duration
            if event_concept is not None and start is not None and duration is not None:
                events.append({
                    "event_type": event_type,
                    "event_concept": event_concept,
                    "start": start,
                    "duration": duration
                })
            # else:
            # print(f"Skipping incomplete event in {xml_file_path}: Type={event_type}, Concept={event_concept}, Start={start}, Duration={duration}")


    except FileNotFoundError:
        print(f"Error: XML file not found at {xml_file_path}")
    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while parsing XML {xml_file_path}: {e}")


    return events


def segment_and_label_edf_data(
        edf_df: pd.DataFrame,
        xml_annotations: List[Dict[str, Any]], # Use list of dicts as returned by parse_xml_annotations
        patient_id: str,
        window_size: int = WINDOW_SIZE,
        overlap_size: int = OVERLAP_SIZE,
        apnea_overlap_threshold: int = APNEA_OVERLAP_THRESHOLD
) -> pd.DataFrame:
    """
    Segments EDF data (as a DataFrame) into fixed windows, flattens the time
    series data for each window, and labels each window based on overlap with
    apnea/hypopnea XML annotations.

    Args:
        edf_df: Pandas DataFrame containing resampled EDF signals, indexed by time.
                Expected columns are the target channels.
        xml_annotations: List of parsed XML event dictionaries.
        patient_id: Identifier for the patient.
        window_size: The size of each segment window in seconds.
        overlap_size: The overlap between consecutive windows in seconds.
        apnea_overlap_threshold: Minimum overlap duration (in seconds) with an
                                 apnea/hypopnea event to label a window as positive.

    Returns:
        A Pandas DataFrame where each row represents a window segment,
        flattened signal data as columns, and metadata including the label.
        Returns an empty DataFrame if no valid segments can be created.
    """
    segments_data: List[Dict[str, Any]] = []
    feature_columns = list(edf_df.columns) # Should be ['SaO2', 'PR', 'THOR RES', 'ABDO RES']
    num_features = len(feature_columns)
    data_length = len(edf_df)

    if data_length == 0 or num_features == 0:
        print(f"Warning: No data or features in EDF DataFrame for patient {patient_id}. Skipping segmentation.")
        return pd.DataFrame()

    # Define column names for the flattened output
    # Format: CHANNEL_tSECONDS (e.g., SaO2_t0, PR_t0, ..., ABDO_t59)
    flattened_cols = [f"{col}_t{t}" for t in range(window_size) for col in feature_columns]

    apnea_related_event_concepts = ["Obstructive apnea|Obstructive Apnea", "Hypopnea|Hypopnea"]

    # Calculate the step size between consecutive windows
    step_size = window_size - overlap_size
    if step_size <= 0:
        print(f"Error: Invalid window and overlap sizes for patient {patient_id}. Step size must be positive.")
        return pd.DataFrame()

    # Calculate the number of windows
    num_windows = (data_length - window_size) // step_size + 1
    if num_windows <= 0:
        print(f"Warning: No full windows can be created for patient {patient_id} with current window/overlap settings.")
        return pd.DataFrame()


    for i in range(num_windows):
        segment_start_idx = i * step_size
        segment_end_idx = segment_start_idx + window_size # End index is exclusive

        # Extract the segment DataFrame (should have shape window_size x num_features)
        # Use .iloc for index-based slicing
        segment = edf_df.iloc[segment_start_idx:segment_end_idx]

        # Check if the segment has the expected length (should be true with num_windows calculation, but defensive check)
        if len(segment) != window_size:
            # This should theoretically not happen with the num_windows calculation,
            # but adding for robustness. Incomplete segments at the end are implicitly
            # handled by the range of `i`.
            print(f"Warning: Unexpected segment length ({len(segment)}) at start index {segment_start_idx} for patient {patient_id}. Expected {window_size}. Skipping.")
            continue

        # Flatten the segment DataFrame into a 1D NumPy array
        # Default 'C' order: groups by feature first (SaO2_t0..t59, PR_t0..t59, ...)
        flattened_segment = segment.values.flatten() # Shape: (window_size * num_features,)

        # Create a dictionary for this segment's data
        # Ensure the number of flattened values matches the number of flattened column names
        if len(flattened_segment) != len(flattened_cols):
            print(f"Error: Mismatch between flattened segment size ({len(flattened_segment)}) and flattened column names count ({len(flattened_cols)}) for patient {patient_id}, window {i}. Skipping segment.")
            continue

        segment_dict = dict(zip(flattened_cols, flattened_segment))

        # Determine label based on event overlap
        label = 0 # Default label is 0 (Normal)
        for event in xml_annotations:
            # Check if the event concept is one of the apnea-related types
            if event.get("event_concept") in apnea_related_event_concepts:
                event_start = event.get("start", -1) # Use -1 if start is missing
                event_duration = event.get("duration", 0) # Use 0 if duration is missing
                event_end = event_start + event_duration

                # Calculate overlap with the current segment
                # Segment times are in seconds, starting from 0
                segment_start_time = segment_start_idx / TARGET_RATE # Convert index to time in seconds
                segment_end_time = segment_end_idx / TARGET_RATE # Convert index to time in seconds

                overlap_start = max(segment_start_time, event_start)
                overlap_end = min(segment_end_time, event_end)
                overlap_duration = overlap_end - overlap_start

                # Label as 1 if overlap duration meets the threshold
                if overlap_duration >= apnea_overlap_threshold:
                    label = 1
                    break # Found an apnea/hypopnea overlap, no need to check further events

        # Add metadata and label to the segment dictionary
        segment_dict['Start_Time_sec'] = segment_start_time
        segment_dict['End_Time_sec'] = segment_end_time
        segment_dict['Apnea/Hypopnea_Label'] = label
        segment_dict['Patient_ID'] = patient_id
        segments_data.append(segment_dict)

    # Check if any segments were successfully created
    if not segments_data:
        print(f"No valid segments created for patient {patient_id}.")
        return pd.DataFrame()

    # Create DataFrame from list of dictionaries
    # Use .copy() to avoid potential SettingWithCopyWarning later if manipulating this DataFrame
    return pd.DataFrame(segments_data).copy()

# --- Main Processing Functions ---

def process_single_file(
        edf_file_path: str,
        xml_file_path: str,
        patient_id: str,
        target_channels: List[str] = TARGET_CHANNELS,
        target_rate: int = TARGET_RATE
) -> Optional[pd.DataFrame]:
    """
    Processes a single pair of EDF and XML files for a patient.

    Args:
        edf_file_path: Path to the EDF file.
        xml_file_path: Path to the XML file.
        patient_id: Identifier for the patient.
        target_channels: List of channel names to extract.
        target_rate: Target sampling rate for resampling.

    Returns:
        A Pandas DataFrame containing segmented and labeled data for the patient,
        or None if the file pair is excluded based on quality criteria or errors.
    """
    print(f"Processing Patient {patient_id}: {edf_file_path}")

    # 1. Get EDF Channels and Sampling Rates
    edf_signals, sampling_rates = get_edf_channels(edf_file_path, target_channels)

    # Check if essential signals are missing entirely (empty arrays)
    # Consider a file invalid if any of the target channels are completely empty after extraction
    if any(len(sig) == 0 for sig in edf_signals.values()):
        missing_channels = [ch for ch, sig in edf_signals.items() if len(sig) == 0]
        print(f"Excluded {edf_file_path} due to missing essential channels: {missing_channels}.")
        return None

    # 2. Simple Artifact Removal (Interpolation)
    edf_signals_cleaned = remove_artifacts(edf_signals)

    # 3. Check for Excessive Artifacts/Missing Values after Interpolation
    if not check_artifacts_and_missing_values(edf_signals_cleaned):
        print(f"Excluded {edf_file_path} due to excessive artifacts or missing values after cleaning.")
        return None

    # 4. Parse XML Annotations
    xml_annotations = parse_xml_annotations(xml_file_path)

    # 5. Check Minimum Recording Duration
    # Need at least one event to check recording duration
    if not xml_annotations:
        print(f"Excluded {xml_file_path} as no annotations were found.")
        return None
    if not calculate_recording_duration(xml_annotations):
        print(f"Excluded {xml_file_path} due to insufficient recording duration.")
        return None

    # 6. Resample Signals
    resampled_signals = resample_signals(edf_signals_cleaned, sampling_rates, target_rate=target_rate)

    # Check if resampling resulted in empty signals for any target channel
    if any(len(sig) == 0 for sig in resampled_signals.values()):
        empty_resampled = [ch for ch, sig in resampled_signals.items() if len(sig) == 0]
        print(f"Excluded {edf_file_path} due to empty signals after resampling: {empty_resampled}.")
        return None

    # Convert resampled signals to DataFrame
    try:
        edf_df = pd.DataFrame(resampled_signals)
    except Exception as e:
        print(f"Error creating DataFrame from resampled signals for {edf_file_path}: {e}")
        return None


    # 7. Segment and Label Data
    combined_df = segment_and_label_edf_data(
        edf_df,
        xml_annotations,
        patient_id,
        window_size=WINDOW_SIZE,
        overlap_size=OVERLAP_SIZE,
        apnea_overlap_threshold=APNEA_OVERLAP_THRESHOLD
    )

    # Check if segmentation yielded any results
    if combined_df.empty:
        print(f"No segments generated for patient {patient_id}.")
        return None

    print(f"Successfully processed patient {patient_id}. Generated {len(combined_df)} segments.")
    return combined_df


def process_all_files(
        edf_folder: str,
        xml_folder: str,
        target_channels: List[str] = TARGET_CHANNELS,
        target_rate: int = TARGET_RATE,
        num_files: Optional[int] = None # Use Optional for argparse default None
) -> None:
    """
    Iterates through EDF files in the specified folder, finds corresponding
    XML files, and processes each pair. Concatenates results and saves to CSV.

    Args:
        edf_folder: Path to the folder containing EDF files.
        xml_folder: Path to the folder containing XML files.
        target_channels: List of channel names to extract.
        target_rate: Target sampling rate for resampling.
        num_files: Maximum number of files to process. Processes all if None.
    """
    combined_data: List[pd.DataFrame] = []
    file_count: int = 0

    # Get list of EDF files and sort them for consistent processing order
    edf_files = sorted([f for f in os.listdir(edf_folder) if f.endswith(".edf")])

    if not edf_files:
        print(f"No EDF files found in {edf_folder}. Exiting.")
        return

    print(f"Found {len(edf_files)} EDF files.")


    for edf_file in edf_files:
        if num_files is not None and file_count >= num_files:
            print(f"Reached the limit of {num_files} files to process.")
            break

        # Extract patient ID from the EDF filename
        # Assuming format like 'shhs2-NNNNNN.edf' or similar where ID is after first '-' and before '.'
        try:
            parts = edf_file.split("-")
            if len(parts) > 1:
                nsrr_id_part = parts[1]
                patient_id = nsrr_id_part.split(".")[0]
            else:
                print(f"Could not extract patient ID from filename '{edf_file}'. Skipping.")
                continue # Skip if ID extraction fails

            xml_file_name = f"shhs2-{patient_id}-nsrr.xml" # Assuming SHHS2 XML naming convention
            edf_file_path = os.path.join(edf_folder, edf_file)
            xml_file_path = os.path.join(xml_folder, xml_file_name)

            if os.path.exists(xml_file_path):
                print(f"--- Processing file {file_count + 1}/{num_files if num_files is not None else len(edf_files)}: {edf_file} ---")
                try:
                    combined_df = process_single_file(
                        edf_file_path,
                        xml_file_path,
                        patient_id,
                        target_channels=target_channels,
                        target_rate=target_rate
                    )
                    if combined_df is not None and not combined_df.empty:
                        combined_data.append(combined_df)
                    # Increment file_count regardless of successful processing to respect the limit
                    file_count += 1

                except Exception as e:
                    print(f"An unexpected error occurred during processing of {edf_file_path}: {e}")
                    # Increment file_count even on error to respect the limit
                    file_count += 1
                    # Continue to the next file

            else:
                print(f"XML file '{xml_file_name}' not found for '{edf_file}'. Skipping.")
                # Do not increment file_count here as this EDF file wasn't processed due to missing XML
        except Exception as e:
            print(f"An unexpected error occurred during processing of. Check formatting: {e}")
            # Increment file_count even on error to respect the limit
            file_count += 1
            # Continue to the next file


    if combined_data:
        print("\nConcatenating processed data from all files...")
        try:
            final_combined_df = pd.concat(combined_data, ignore_index=True)
            print(f"Total segments processed: {len(final_combined_df)}")

            # Ensure the output directory exists
            output_dir = os.path.dirname(COMBINED_FILE)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")


            final_combined_df.to_csv(COMBINED_FILE, index=False)
            print(f"Successfully saved combined dataset to {COMBINED_FILE}")
        except Exception as e:
            print(f"Error concatenating or saving data: {e}")
    else:
        print("\nNo data was successfully processed to be combined and saved.")


def main():
    """
    Main function to parse command-line arguments and initiate the preprocessing.
    """
    parser = argparse.ArgumentParser(
        description="Process SHHS2 dataset (EDF and XML files) for sleep apnea analysis."
    )
    parser.add_argument(
        "--num_files",
        type=int,
        default=None,
        help="Optional: Number of EDF/XML file pairs to process. "
             "If not specified, all files found in the EDF folder will be processed."
    )
    args = parser.parse_args()

    # Define dataset folder paths - **USER MUST CONFIGURE THESE**
    # Example (adjust based on your actual file structure):
    global EDF_FOLDER, XML_FOLDER # Use global to modify the constants defined earlier
    EDF_FOLDER = os.path.join("..", "SHHS2_dataset", "edf_files") # Example path
    XML_FOLDER = os.path.join("..", "SHHS2_dataset", "annotation_files") # Example path
    # Ensure these folders exist before running
    if not os.path.isdir(EDF_FOLDER):
        print(f"Error: EDF folder not found at {EDF_FOLDER}. Please update the path in the script.")
        exit()
    if not os.path.isdir(XML_FOLDER):
        print(f"Error: XML folder not found at {XML_FOLDER}. Please update the path in the script.")
        exit()

    # Define output file path - **USER MAY WANT TO CONFIGURE THIS**
    global COMBINED_FILE # Use global to modify the constant
    COMBINED_FILE = "./processed_shhs2_data.csv" # Default output path

    print(f"EDF Folder: {EDF_FOLDER}")
    print(f"XML Folder: {XML_FOLDER}")
    print(f"Output File: {COMBINED_FILE}")
    print(f"Number of files to process: {'All' if args.num_files is None else args.num_files}")


    # Start the processing
    process_all_files(
        edf_folder=EDF_FOLDER,
        xml_folder=XML_FOLDER,
        target_channels=TARGET_CHANNELS,
        target_rate=TARGET_RATE,
        num_files=args.num_files
    )


if __name__ == "__main__":
    main()