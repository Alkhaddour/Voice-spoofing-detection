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
    :param is_dir: True of this path is a directory and should be created also
    :param exist_ok: behaviour if the directory to be created is already existed
    :return: the same path passed to the function with its parents directories created
    """
    if is_dir is True:
        parent_dir = name
    else:
        parent_dir = os.path.dirname(name)
    os.makedirs(parent_dir, exist_ok=exist_ok)
    return name


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
    """
    Export names of incorrectly classified files to csv file along with predicted label and correct label
    :param files: List of file names
    :param y_pred: Predicted label
    :param y_true: Correct label
    :param out_file: Name of csv file to be created
    :return:
    """
    incorrect_samples = [i for i in range(len(files)) if y_true[i] != y_pred[i]]
    pd.DataFrame(zip([os.path.basename(files[x]) for x in incorrect_samples],
                     [y_pred[x] for x in incorrect_samples],
                     [y_true[x] for x in incorrect_samples])
                 ).to_csv(out_file, index=False, header=['Files', 'Predictions', 'Labels'])
