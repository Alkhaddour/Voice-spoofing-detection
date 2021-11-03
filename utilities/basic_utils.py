# This file contains different utility functions used in the project
import pandas as pd
import torch
from config import FRAME_SIZE, FRAME_STEP
import scipy.io.wavfile as wav
import numpy as np
import os

from utilities.disply_utils import warn


def make_valid_path(name, is_dir=False, exist_ok=True):
    """
    This function make sure that a given path has all its parent directories created
    :param name: path name
    :param is_dir: is this path a directory and should be created also
    :param exist_ok: behaviour if the directory to be created is already existed
    :return: the same name passed to the function with its parents defined
    """
    if is_dir is True:
        parent_dir = name
    else:
        parent_dir = os.path.dirname(name)
    os.makedirs(parent_dir, exist_ok=exist_ok)
    return name


def frames_in_timeframe(duration):
    """
    Find the number of frames in a timeframe
    :param duration: timeframe duration (in ms)
    :return: number of frames
    """
    n_frames = ((duration - FRAME_SIZE) / FRAME_STEP + 1)
    return max(1, n_frames)


def samples_to_time(samples_count, sampling_rate, unit):
    """
    Convert number of samples to time (in minutes or seconds)
    :param samples_count:
    :param sampling_rate:
    :param unit: 's' for seconds, 'm' for minutes
    :return:
    """
    time_seconds = samples_count / sampling_rate
    if unit.lower() == 's':
        return time_seconds
    elif unit.lower() == 'm':
        return time_seconds / 60.0
    else:
        warn(f"Unexpected unit {unit}, use 's' for seconds or 'm' for minutes")


def index_to_one_hot(class_id, n_classes):
    vec = np.zeros((n_classes), dtype=np.int64)
    vec[class_id] = 1
    return vec


def get_audio_length(wav_file):
    """
    Given a path to audio files, find its lengths in seconds
    :param wav_file: path to wav file
    :return: wave file duration
    """
    sample_rate, signal = wav.read(wav_file)
    return len(signal) / sample_rate


def get_accelerator(device):
    """
    Get a torch device based on a string name of the target device
    :param device:
    :return:
    """
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def export_incorrect_samples_to_csv(files, y_pred, y_true, out_file):
    incorrect_samples = [i for i in range(len(files)) if y_true[i] != y_pred[i]]
    pd.DataFrame(zip([os.path.basename(files[x]) for x in incorrect_samples],
                     [y_pred[x] for x in incorrect_samples],
                     [y_true[x] for x in incorrect_samples])
                 ).to_csv(out_file, index=False, header=['Files', 'Predictions', 'Labels'])
