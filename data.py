import musdb
from scipy.signal import stft, istft

import torch
from torch.utils.data import Dataset

class MUSDBDataset(Dataset):
    def __init__(self):
        super(MUSDBDataset, self).__init__()
        self.tracks = musdb.DB(download=True)

    def __getitem__(self, idx):
        track = self.tracks[idx]

    def __len__(self):
        return len(self.tracks)

def MUSDBLoader()

if __name__ == '__main__':
    dataset = MUSDBDataset()

    print(dataset.tracks)
    print(len(dataset))
