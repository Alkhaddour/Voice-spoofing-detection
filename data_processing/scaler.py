# This file creates a scaler object based on the train data, this scaler will be used later to scale any new data.
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from utilities.basic_utils import make_valid_path
import pickle
from config import *
import numpy as np
import os


def create_scaler(files, save_path=None):
    """
    This function takes a list of files to create scaler from them
    :param files: The list of files to create the scaler
    :param save_path: if not None, then it should indicate the path where we want ot save the scaler
    :return: the fitted scaler
    """
    train_features = []
    # Reading files
    print(f"[{datetime.now()}] -- Reading files...")
    for file in files:
        part = np.load(file)
        train_features.append(part)
    # Concatenating
    print(f"[{datetime.now()}] -- Concatenating data...")
    train_features = np.concatenate(train_features, axis=0)
    # Build scaler
    print(f"[{datetime.now()}] -- Building scaler...")
    scaler = StandardScaler()
    scaler.fit(train_features)
    # save and return
    if save_path is not None:
        print(f"[{datetime.now()}] -- Saving scaler...")
        pickle.dump(scaler, open(save_path, 'wb'))
        print(f"[{datetime.now()}] -- scaler saved to: {save_path}")
    return scaler


def scale_file(scaler, source_file, target_file):
    file = np.load(source_file)
    file_scaled = scaler.transform(file)
    np.save(target_file, file_scaled)


def scale_sets():
    files_to_create_scale = []
    for class_ in os.listdir(TRAIN_PROCESSED_DIR):
        class_dir = os.path.join(TRAIN_PROCESSED_DIR, class_)
        for file in os.listdir(class_dir):
            files_to_create_scale.append(os.path.join(class_dir, file))

    # Creating scaler
    create_scaler(files_to_create_scale, SCALER_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # scale train / val
    for split_dir, scaled_dir in zip([TRAIN_PROCESSED_DIR, VAL_PROCESSED_DIR], [TRAIN_SCALED_DIR, VAL_SCALED_DIR]):
        for class_ in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_)
            for i, file in enumerate(os.listdir(class_dir)):
                if i%200 == 0:
                    print(f"[{datetime.now()}] -- {i} files processed...")
                src_file = os.path.join(class_dir, file)
                target_dir = make_valid_path(os.path.join(scaled_dir, class_), is_dir=True)
                target_file = os.path.join(target_dir, file)
                scale_file(scaler, src_file, target_file)
    # scale test
    for i, file in enumerate(os.listdir(TEST_PROCESSED_DIR)):
        if i%200 == 0:
            print(f"[{datetime.now()}] -- {i} files processed...")
        src_file = os.path.join(TEST_PROCESSED_DIR, file)
        target_file = os.path.join( make_valid_path(TEST_SCALED_DIR, is_dir=True), file)
        scale_file(scaler, src_file, target_file)
