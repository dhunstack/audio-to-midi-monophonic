import os
import pickle
from config import CONFIDENCE_PATH, FREQUENCY_PATH, ONSET_ACTIVATIONS_PATH, RMS_PATH, TIME_PATH

def create_dir_if_not_exist(directory):
    """
    Create directory if it does not exist

    Args:
        directory (str): Directory path

    Returns:
        None
    """

    return os.makedirs(directory) if not os.path.exists(directory) else None

def save_all_features(directory, onset_activations, time, frequency, confidence, rms):
    """
    Save all features as pickle files

    Args:
        directory (str): Directory path
        features (dict): Dictionary containing features

    Returns:
        None
    """
    create_dir_if_not_exist(directory)
    dump_pickle(directory / ONSET_ACTIVATIONS_PATH, onset_activations)
    dump_pickle(directory / TIME_PATH, time)
    dump_pickle(directory / FREQUENCY_PATH, frequency)
    dump_pickle(directory / CONFIDENCE_PATH, confidence)
    dump_pickle(directory / RMS_PATH, rms)


def load_all_features(directory):
    """
    Load all features from pickle files

    Args:
        directory (str): Directory path

    Returns:
        onset_activations (np.array): Onset activations
        time (np.array): Time
        frequency (np.array): Frequency
        confidence (np.array): Confidence
        rms (np.array): RMS energy
    """
    onset_activations = load_pickle(directory / ONSET_ACTIVATIONS_PATH)
    time = load_pickle(directory / TIME_PATH)
    frequency = load_pickle(directory / FREQUENCY_PATH)
    confidence = load_pickle(directory / CONFIDENCE_PATH)
    rms = load_pickle(directory / RMS_PATH)

    return onset_activations, time, frequency, confidence, rms


def dump_pickle(file, data):
    """
    Dump data to pickle file

    Args:
        file (str): File path
        data (any): Data to be dumped

    Returns:
        None
    """
    with open(file, "wb") as f:
        pickle.dump(data, f)
    return None

def load_pickle(file):
    """
    Load data from pickle file

    Args:
        file (str): File path

    Returns:
        any: Data from pickle file
    """
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data