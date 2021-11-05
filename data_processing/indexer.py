"""
Summary:
This file defines a function used to create index of train/val/test sets. An index contains paths for files of extracted
features, each file represents one audio file (for details see feature_extractor.py file)

The index contains a list of pairs (the first element is a file path and the second is its label) saved as binary file
using pickle library
"""

import os
import pickle

from config import TRAIN_SCALED_DIR, CLASS_CODE_MAP, VAL_SCALED_DIR, TEST_SCALED_DIR, TRAIN_INDEX, VAL_INDEX, TEST_INDEX


def create_index():
    train_files = []
    train_labels = []
    val_files = []
    val_labels = []
    test_files = []
    test_labels = []

    classes = list(CLASS_CODE_MAP.keys())

    for class_ in classes:
        class_path = os.path.join(TRAIN_SCALED_DIR, class_)
        files = os.listdir(class_path)
        train_files = train_files + [os.path.join(class_path, file) for file in files]
        train_labels = train_labels + [CLASS_CODE_MAP[class_]] * len(train_files)

        class_path = os.path.join(VAL_SCALED_DIR, class_)
        files = os.listdir(class_path)
        val_files = val_files + [os.path.join(class_path, file) for file in files]
        val_labels = val_labels + [CLASS_CODE_MAP[class_]] * len(val_files)

    files = os.listdir(TEST_SCALED_DIR)
    test_files = test_files + [os.path.join(TEST_SCALED_DIR, file) for file in files]
    test_labels = test_labels + [None] * len(test_files)

    train_data = list(zip(train_files, train_labels))
    val_data = list(zip(val_files, val_labels))
    test_data = list(zip(test_files, test_labels))

    pickle.dump(train_data, open(TRAIN_INDEX, 'wb'))
    pickle.dump(val_data, open(VAL_INDEX, 'wb'))
    pickle.dump(test_data, open(TEST_INDEX, 'wb'))
