import pickle

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import config
from utilities.disply_utils import info


class ReplySpoofDataset(Dataset):
    def __init__(self, index_file):
        super(ReplySpoofDataset, self).__init__()
        # read_index: index is list of pairs (filename, label)
        with open(index_file, 'rb') as handler:
            index = pickle.load(handler)
        self.files = []
        self.data = []
        self.labels = []
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
        return self.files[idx], torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx])


def collate_fn_pad(list_pairs_seq_target):
    files = [file for file, _, _ in list_pairs_seq_target]
    seqs = [seq for _, seq, _ in list_pairs_seq_target]
    targets = [target for _, _, target in list_pairs_seq_target]
    seqs_padded_batched = pad_sequence(seqs, batch_first=config.BATCH_FIRST)  # will pad at beginning of sequences
    targets_batched = torch.stack(targets)
    if config.BATCH_FIRST is False:
        assert seqs_padded_batched.shape[1] == len(targets_batched)
    else:
        assert seqs_padded_batched.shape[0] == len(targets_batched)
    return files, seqs_padded_batched, targets_batched


# ds = ReplySpoofDataset(VAL_INDEX)
# dl = DataLoader(ds, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn_pad)
# i = 0
# for b, l in dl:
#     l=l.reshape(-1)
#     print(b.size())
#     print(l.size())
#     if i == 2:
#         break
#     i += 1
