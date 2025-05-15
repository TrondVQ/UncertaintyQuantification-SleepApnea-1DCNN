import os
import pandas as pd
from pyedflib import EdfReader
from scipy.signal import resample
import xml.etree.ElementTree as ET
import numpy as np
import argparse

"""
This script preprocesses the SHHS2 dataset by combining EDF signal files
and XML annotation files. It extracts specific physiological signals,
resamples them to a target frequency, handles artifacts by interpolating
out-of-range values for SaO2 and PR, and segments the data into
fixed-size windows. Each window is then labeled based on overlap with
apnea/hypopnea annotations from the XML file.
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
    - Large number of missing values or artifacts (checked per signal after
      attempting interpolation for out-of-range SaO2 and PR values).
    - Recording duration (based on "Recording Start Time" event) less than 300 minutes.
- Labeling: A window is labeled as apnea/hypopnea (1) if it overlaps
  with an "Obstructive apnea|Obstructive Apnea" or "Hypopnea|Hypopnea"
  event for at least 10 seconds. Otherwise, it is labeled as normal (0).
"""

# Paths to folders
EDF_FOLDER =  ""  #"../SHHS2_dataset/edf_files"
XML_FOLDER = ""   #"../SHHS2_dataset/annotation_files"

COMBINED_FILE = "./SHHS2_ID_all_60.csv"

#Alacon: The patient data were randomly selected after discarding those patient data that contained a large number of missing values, a large number of artifacts or the sleep time was not longer than 300 minutes.

#Large number of missing values: Check for NaN values in the dataset

def check_artifacts_and_missing_values(signals, artifact_threshold=0.1):
    """
    Exclude signals with a high percentage of missing values or artifacts.
    Args:
        signals (dict): Dictionary of signal arrays.
        artifact_threshold (float): Proportion of artifacts to consider excessive.

    Returns:
        bool: True if data is valid, False if it should be excluded.
    """
    valid = True
    for channel, signal in signals.items():
        num_missing = np.isnan(signal).sum()
        total_length = len(signal)

        if (num_missing / total_length > artifact_threshold):
            valid = False
            break

    return valid


def calculate_sleep_time(events, min_sleep_time=300 * 60):
    """
    Calculate the total duration of the recording based on the 'Recording Start Time' event.
    Args:
        events (list): List of parsed events from the XML.
        min_sleep_time (int): Minimum sleep time in seconds.

    Returns:
        bool: True if the recording duration is valid (i.e., greater than or equal to min_sleep_time), False otherwise.
    """
    # Find the "Recording Start Time" event and get its duration
    recording_start_time_event = next((event for event in events if event["EventConcept"] == "Recording Start Time"), None)

    if recording_start_time_event:
        total_sleep_time = recording_start_time_event["Duration"]
    else:
        total_sleep_time = 0  # In case there's no "Recording Start Time" event

    print(f"Total sleep time (based on Recording Start Time): {total_sleep_time}")

    # Check if the total sleep time meets the minimum required sleep time
    return total_sleep_time >= min_sleep_time


# Artifact Removal: Interpolation of invalid values
def remove_artifacts(signals):
    """
    Remove artifacts by replacing anomalous values with interpolated values.
    For SpO2: Replace values <80 or >100.
    For PR: Replace values <40 or >200.
    """
    for channel, signal in signals.items():
        if channel == "SaO2":
            mask = (signal < 80) | (signal > 100)
        elif channel == "PR":
            mask = (signal < 40) | (signal > 200)
        else:
            continue

        # Interpolate only if there are invalid values
        if np.any(mask):
            signal[mask] = np.nan  # Set invalid values to NaN
            # Interpolate NaNs
            mask_nan = np.isnan(signal)
            signal[mask_nan] = np.interp(
                np.flatnonzero(mask_nan),
                np.flatnonzero(~mask_nan),
                signal[~mask_nan]
            )
    return signals


# Get the EDF channels and their sampling rates from the EDF file/ From marta and project 3
def get_edf_channels(file_path, channels):
    with EdfReader(file_path) as f:
        signal_channels = {chn: i for i, chn in enumerate(f.getSignalLabels())}
        sampling_rates = {chn: f.getSampleFrequency(i) for chn, i in signal_channels.items()}

        signals = {}
        for channel in channels:
            channel_id = signal_channels.get(channel)
            if channel_id is not None:
                signals[channel] = f.readSignal(channel_id)
            else:
                if channel == "PR": #PR will have alternative names.
                    print(f"PR not found. Trying Alternative H.R. in EDF file {file_path}")
                    alt_names= ["H.R."] #The file signals  PR are sometimes called H.R. Files: 107, 108
                    for alt_name in alt_names:
                        if alt_name in signal_channels:
                            print(f"Found {alt_name} in channel list.")
                            channel_id = signal_channels[alt_name]
                            signals["PR"] = f.readSignal(channel_id)
                            sampling_rates["PR"] = sampling_rates.pop(alt_name)  # Update the name in the dictionary
                            break
                else:
                    print(f"Channel {channel} not found in {file_path}")

    if "PR" not in signals:
        print(f"Warning: 'PR/H.R. channel is missing in {file_path}.")

    return signals, sampling_rates

# Resample all signals to a target sampling rate
def resample_signals(signals, sampling_rates, target_rate=1):
    resampled_signals = {}
    for channel, signal in signals.items():
        original_rate = sampling_rates[channel]
        target_length = int(len(signal) * (target_rate / original_rate))
        resampled_signals[channel] = resample(signal, target_length)
    return resampled_signals



# Parse XML annotations and extract event details
def parse_xml_annotations(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    events = []

    for scored_event in root.findall("ScoredEvents/ScoredEvent"):
        event_type = scored_event.find("EventType").text if scored_event.find("EventType") is not None else None
        if event_type == "Stages|Stages":
            break

        event_concept = scored_event.find("EventConcept").text if scored_event.find("EventConcept") is not None else None
        start = float(scored_event.find("Start").text) if scored_event.find("Start") is not None else None
        duration = float(scored_event.find("Duration").text) if scored_event.find("Duration") is not None else None

        events.append({
            "event_type": event_type,
            "event_concept": event_concept,
            "start": start,
            "duration": duration
        })

    return events

# Segment EDF data into fixed windows and label based on overlap with XML annotations
# Alacon -> Considering the above and that an OSA event can range from 10 to 40 s, a window period of 60 s is the most appropriate.
def segment_and_label_edf_data(edf_df, xml_annotations_df, patient_id, window_size=60, overlap_size=0):
    """
    Segments EDF data into fixed windows, flattens the time series data for each window,
    and labels based on overlap with XML annotations.
    """
    segments_data = []
    feature_columns = list(edf_df.columns) # Should be ['SaO2', 'PR', 'THOR RES', 'ABDO RES']
    num_features = len(feature_columns)

    # Define column names for the flattened output
    flattened_cols = [f"{col}_t{t}" for t in range(window_size) for col in feature_columns]

    apnea_related_events = ["Obstructive apnea|Obstructive Apnea", "Hypopnea|Hypopnea"]

    num_windows = (len(edf_df) - window_size + overlap_size) // (window_size - overlap_size) + 1

    for i in range(num_windows):
        segment_start_time = i * (window_size - overlap_size) # Start time in seconds from beginning of recording
        segment_end_time = segment_start_time + window_size

        # Extract the segment DataFrame (should have shape window_size x num_features, e.g., 60x4)
        segment = edf_df.iloc[segment_start_time:segment_end_time]

        # Check if the segment has the expected length (window_size)
        if len(segment) != window_size:
            print(f"Skipping segment {i} for patient {patient_id} due to unexpected length: {len(segment)}")
            continue # Skip incomplete segments (usually at the end)

        # Flatten the segment DataFrame into a 1D NumPy array
        # Order will be: SaO2_t0, PR_t0, THOR_t0, ABDO_t0, SaO2_t1, PR_t1, ... ABDO_t59
        # Use 'F' (Fortran-like) order for column-major flattening to group by time step first if desired,
        # or default 'C' (C-like) order for row-major flattening (groups by feature first: SaO2_t0..t59, PR_t0..t59, ...)
        # Let's use 'C' order first, which aligns with how pandas would naturally iterate if converted row-by-row.
        # If using 'F' order: flattened_segment = segment.values.flatten('F')
        # Default 'C' order:
        flattened_segment = segment.values.flatten() # Shape will be (window_size * num_features,), e.g., (240,)

        # Create a dictionary for this segment's data
        segment_dict = dict(zip(flattened_cols, flattened_segment))

        # Determine label based on event overlap
        label = 0
        for _, event in xml_annotations_df.iterrows():
            if event["event_concept"] in apnea_related_events:
                event_start = event["start"]
                event_end = event["start"] + event["duration"]

                # Compare event time with segment time (both in seconds)
                overlap_start = max(segment_start_time, event_start)
                overlap_end = min(segment_end_time, event_end)
                overlap_duration = overlap_end - overlap_start

                # Alarcon: "Segments were labelled as apnea/hypopnea if the overlap between the segment and an event was greater than or equal to 10 s"
                # Note: Ensure your event times and segment times are relative to the same starting point.
                if overlap_duration >= 10:
                    label = 1
                    break

        # Add metadata and label
        segment_dict['Start_Time'] = segment_start_time
        segment_dict['End_Time'] = segment_end_time
        segment_dict['Apnea/Hypopnea'] = label
        segment_dict['Patient_ID'] = patient_id
        segments_data.append(segment_dict)

    # Check if any segments were created before making DataFrame
    if not segments_data:
        return pd.DataFrame() # Return empty DataFrame if no valid segments

    return pd.DataFrame(segments_data)
# Process a single EDF and XML file pair
def process_single_file(edf_file_path, xml_file_path, patient_id, target_channels, target_rate=1):
    edf_signals, sampling_rates = get_edf_channels(edf_file_path, target_channels)
    edf_signals = remove_artifacts(edf_signals)

    if not check_artifacts_and_missing_values(edf_signals):
        print(f"Excluded {edf_file_path} due to excessive artifacts or missing values.")
        return None

    xml_annotations = parse_xml_annotations(xml_file_path)

    if not calculate_sleep_time(xml_annotations):
        print(f"Excluded {xml_file_path} due to insufficient sleep time.")
        return None

    xml_df = pd.DataFrame(xml_annotations)

    resampled_signals = resample_signals(edf_signals, sampling_rates, target_rate=target_rate)

    edf_df =  pd.DataFrame(resampled_signals)

    combined_df = segment_and_label_edf_data(edf_df, xml_df, patient_id)
    return combined_df


# Process all EDF and XML file pairs in the dataset folder
def process_all_files(edf_folder, xml_folder, target_channels, target_rate=1, num_files=None):
    combined_data = []
    file_count = 0

    for edf_file in os.listdir(edf_folder):
        if num_files is not None and file_count >= num_files:
            break

        if not (edf_file.endswith(".edf")):
            print(f"EDF file for {edf_file} does not have the correct ending. Skipping...")
            continue

        nsrr_id = edf_file.split("-")[1].split(".")[0]
        xml_file = f"shhs2-{nsrr_id}-nsrr.xml"
        edf_file_path = os.path.join(edf_folder, edf_file)
        xml_file_path = os.path.join(xml_folder, xml_file)

        patient_id = nsrr_id
        if os.path.exists(xml_file_path):
            print(f"Processing {file_count}: {edf_file_path} and {xml_file_path}")
            try:
                combined_df = process_single_file(edf_file_path, xml_file_path, patient_id, target_channels, target_rate)
                if combined_df is not None: # might have been excluded.
                    combined_data.append(combined_df)
                file_count += 1

            except Exception as e:
                print(f"Error processing {edf_file_path} and {xml_file_path}: {e}")
                file_count += 1

        else:
            print(f"XML file for {edf_file} not found. Skipping...")

    if combined_data:
        final_combined_df = pd.concat(combined_data, ignore_index=True)
        final_combined_df.to_csv(COMBINED_FILE, index=False)
        print(f"Saved combined dataset to {COMBINED_FILE}")

def main():
    parser = argparse.ArgumentParser(description="Process SHHS2 dataset.")
    parser.add_argument("--num_files", type=int,  default=None, help="Number of files to process")
    args = parser.parse_args()

    sleep_apnea_channels = ["SaO2", "PR", "THOR RES", "ABDO RES"]
    process_all_files(EDF_FOLDER, XML_FOLDER, target_channels=sleep_apnea_channels, target_rate=1, num_files=args.num_files)

if __name__ == "__main__":
    main()