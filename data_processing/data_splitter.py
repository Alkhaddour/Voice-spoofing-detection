"""
Summary:
This files defines a function which takes a split of train set to be used later for model validation.
All parameters for input and output directories, class names used for classification are indicated in the config.py file
"""
import os
import random

from config import TRAIN_PROCESSED_DIR, VAL_PCT, VAL_PROCESSED_DIR, CLASS_CODE_MAP
from utilities.basic_utils import make_valid_path
from utilities.disply_utils import info


def split_data():
    info("Splitting data")
    classes = list(CLASS_CODE_MAP.keys())

    files_list = [[] for _ in classes]

    for i, class_ in enumerate(classes):
        files_list[i] = [file for file in os.listdir(os.path.join(TRAIN_PROCESSED_DIR, class_))]

    val_files = [[] for _ in classes]

    for i in range(len(classes)):
        # sample VAL_PCT of training data and add them to val folder
        val_files[i] = random.sample(files_list[i], int(len(files_list[i]) * VAL_PCT))

    # Move files from train folder to val folder
    for i, class_ in enumerate(classes):
        for file in val_files[i]:
            src_path = os.path.join(TRAIN_PROCESSED_DIR, class_)
            src_file = os.path.join(src_path, file)
            target_path = os.path.join(VAL_PROCESSED_DIR, class_)
            target_file = os.path.join(target_path, file)
            make_valid_path(target_path, is_dir=True)
            os.replace(src_file, target_file)

    info("Done!")
