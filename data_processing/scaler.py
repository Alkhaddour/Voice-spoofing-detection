"""
Summary:
In this file we define 3 functions
#1  create_scaler(): Creates a StandardScaler. The scaler defines mean and standard deviation of features.
#2  scale_file(): scales features in file using a pre-defined scaler
#3  scale_sets(): Create a StandardScaler object using the train set, and use it to scale all of the sets, the scaler
                  is also saved for further use using other data.
"""
from sklearn.preprocessing import StandardScaler
from utilities.basic_utils import make_valid_path
from utilities.disply_utils import info
import pickle
from config import *
import numpy as np
from tqdm import tqdm
import time
import os
import gc
import sys

def create_scaler(files, save_path=None):
    """
    This function takes a list of paths to feature files to create StandardScaler from them.
    :param files: The list of paths to files to create the scaler from them.
    :param save_path: if not None, then it should indicate the path where we want ot save the scaler
    :return: the fitted scaler
    """
    train_features = []
    # Reading files
    info(f"Reading files...")
    time.sleep(1.0)
    bar = tqdm(files, file=sys.stdout)
    for filename in bar:
        bar.set_description(os.path.basename(filename))
        part = np.load(filename)
        train_features.append(part)
    gc.collect()
    time.sleep(10.0)

    # Concatenating
    info(f"Concatenating data1...")
    train_features1 = np.concatenate(train_features[:len(train_features)//2], axis=0)
    info(f"Concatenating data2...")
    train_features2 = np.concatenate(train_features[len(train_features) // 2:], axis=0)

    info(f"Concatenating data3...")
    del train_features
    gc.collect()
    time.sleep(10.0)
    train_features = np.concatenate([train_features1, train_features2], axis=0)

    # Build scaler
    info(f"Building scaler...")
    del train_features1, train_features2
    gc.collect()
    time.sleep(10.0)
    scaler = StandardScaler()
    scaler.fit(train_features)
    # save and return
    if save_path is not None:
        info(f"Saving scaler...")
        pickle.dump(scaler, open(save_path, 'wb'))
        info(f"Scaler saved to: {save_path}")
    return scaler


def scale_file(scaler, source_file, target_file):
    """
    Scale a feature file using a scaler and save the output to a file
    :param scaler: Scaler to be used
    :param source_file: Source feature file
    :param target_file: Path to save the scaled features.
    :return:
    """
    file = np.load(source_file)
    file_scaled = scaler.transform(file)
    np.save(target_file, file_scaled)


def scale_sets():
    # Step 1: Load paths to train files to be used for creating the scaler (We use the trian data for this)
    files_to_create_scale = []
    for class_ in os.listdir(TRAIN_PROCESSED_DIR):
        class_dir = os.path.join(TRAIN_PROCESSED_DIR, class_)
        info(f"Scaling files in {class_dir}")
        time.sleep(1.0)
        for file in tqdm(os.listdir(class_dir)):
            files_to_create_scale.append(os.path.join(class_dir, file))
        time.sleep(1.0)

    # Step 2: Create scale using the train data
    info("Creating scaler")
    create_scaler(files_to_create_scale, SCALER_PATH)

    # Step 3: The scaler was saved to disk, we will re-read it from disk to check that it was saved correctly.
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    # Step 4: Scale train / val data
    for split_dir, scaled_dir in zip([TRAIN_PROCESSED_DIR, VAL_PROCESSED_DIR], [TRAIN_SCALED_DIR, VAL_SCALED_DIR]):
        for class_ in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_)
            info(f"Scaling files in {class_dir}")
            time.sleep(1.0)
            bar = tqdm(os.listdir(class_dir))
            for file in bar:
                bar.set_description(f"Processing {file}")
                src_file = os.path.join(class_dir, file)
                target_dir = make_valid_path(os.path.join(scaled_dir, class_), is_dir=True)
                target_file = os.path.join(target_dir, file)
                scale_file(scaler, src_file, target_file)
            time.sleep(1.0)

    # Step 5: Scale test data
    bar = tqdm(os.listdir(TEST_PROCESSED_DIR))
    for file in bar:
        bar.set_description(f"Processing {file}")
        src_file = os.path.join(TEST_PROCESSED_DIR, file)
        target_file = os.path.join(make_valid_path(TEST_SCALED_DIR, is_dir=True), file)
        scale_file(scaler, src_file, target_file)
    time.sleep(1.0)
