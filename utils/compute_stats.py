"""Compute the mean and standard deviation per frequency bin over the training set
"""

from numpy import save
from pathlib import Path
import time
from tqdm import tqdm

import musdb

import torch
import torchaudio

dataset_path = (Path(__file__) / '../../datasets/musdb18').resolve()
stats_path = dataset_path / 'stats'
use_cuda = False

stats_path.mkdir(exist_ok=True)
device = torch.device('cuda' if use_cuda else 'cpu')

# STFT params
n_fft = 4096
n_hop = 1024
window = torch.hann_window(n_fft).to(device)
center = False
normalized = False
onesided = True
pad_mode = 'reflect'

t0 = time.time()

mus = musdb.DB(root=dataset_path, subsets='train', split='train')

n_bins = n_fft//2+1
A_sum = torch.zeros(n_bins).to(device)
A_sqsum = torch.zeros(n_bins).to(device)
n_samples = 0

print('Reading tracks for training set...')
for track in tqdm(mus.tracks, leave=False):
    audio = torch.from_numpy(track.audio.T).to(device)
    stft = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=n_hop,
        window=window,
        center=center,
        normalized=normalized,
        onesided=onesided,
        pad_mode=pad_mode
    )
    A = torchaudio.functional.complex_norm(stft).transpose(0,1).reshape(n_bins, -1)
    A_sum += torch.sum(A, dim=1)
    A_sqsum += torch.sum(A**2, dim=1)
    n_samples += A.size(1)

print('Computing statistics over all frequency-bins...')
A_mean = A_sum / n_samples
A_sqmean = A_sqsum / n_samples
A_std = torch.sqrt(A_sqmean - A_mean**2)

save(stats_path / f'mean_fft{n_fft:d}-hop{n_hop:d}.npy', A_mean.cpu().numpy(), allow_pickle=False)
save(stats_path / f'std_fft{n_fft:d}-hop{n_hop:d}.npy', A_std.cpu().numpy(), allow_pickle=False)

t1 = time.time()

d = t1 - t0
print('Done. Time elapsed:', '{:.0f} s.'.format(d) if d < 60 else '{:.0f} min {:.0f} s.'.format(*divmod(d, 60)))
print(f'Statistics saved in {stats_path}.')
