# This file defines Torch-based dataset class, the class handles loading data and preparing it for model

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from utilities.disply_utils import info
import numpy as np
import config
import pickle


class ReplySpoofDataset(Dataset):
    def __init__(self, index_file):
        """
        ReplySpoofDataset initializer
        :param index_file: Index for dataset, the index is pickled list of pairs (filename, label)
        """
        super(ReplySpoofDataset, self).__init__()
        with open(index_file, 'rb') as handler:
            index = pickle.load(handler)
        self.files = []     # files to examine what samples were incorrectly classified
        self.data = []      # Extracted features
        self.labels = []    # Labels
        # read files
        info(f"Loading data ({len(index)} files)...")
        for (filename, label) in index:
            self.files.append(filename)
            self.data.append(np.load(filename))
            self.labels.append(label)
        info("Done...")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # return filename, data, label
        return self.files[idx], torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx])


def collate_fn_pad(raw_batch):
    """
    DataLoader auxiliary function. Used to make sure that all sequences have the same length by padding small sequences
    using torch pad_sequence function
    :param raw_batch: list of data elements (samples from ReplySpoofDataset)
    :return: list of data elements (samples from ReplySpoofDataset) with all sequences of the same size
    """
    files = [file for file, _, _ in raw_batch]
    seqs = [seq for _, seq, _ in raw_batch]
    labels = [label for _, _, label in raw_batch]
    # Pad sequences
    seqs_padded_batched = pad_sequence(seqs, batch_first=config.BATCH_FIRST)
    # Stack labels
    labels_batched = torch.stack(labels)
    if config.BATCH_FIRST is False:
        assert seqs_padded_batched.shape[1] == len(labels_batched)
    else:
        assert seqs_padded_batched.shape[0] == len(labels_batched)
    return files, seqs_padded_batched, labels_batched
