import os
import glob

import numpy as np

import torch
import torchaudio
import torchvision

from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import utilz

"""
goal:
train VAE to compress audio to latent space,
output of VAE encoding is a single element of LSTM sequence

"""


class WavLSTM(Dataset):
    def __init__(self, wave, sr, win_len):
        self.w, self.sr = wave, sr
        self.windows = data_windows(wave, win_len)
        self.length = len(self.windows)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.windows[idx][0]
        y = self.windows[idx][0]
        return x, y


class WaveSet(Dataset):
    # , resample_to=None):
    def __init__(self, fns: list, seconds: int, win_len: int = None, start_pct: float = 0, end_pct: float = 1):
        """
        seconds is int that is multiplied by sample rate 
        for using multiple songs at same time, im assuming same sr for rn
        TODO: make bounds distribute to each wav file before cat
            - use transforms?

        """
        # waves, srs = torchaudio.load(filepath=fn)
        self.w, srs = utilz.get_n_fix(fns)
        self.sample_rate = srs[0]

        if np.nan in self.w[0] or np.nan in self.w[1]:
            print('oh no, there are nans in this wav')

        self.wave_len = len(self.w[0])

        start_idx = int(self.wave_len * start_pct)
        end_idx = int(self.wave_len * end_pct)

        l = self.w[0][start_idx:end_idx].view(1, -1)
        r = self.w[1][start_idx:end_idx].view(1, -1)
        self.w = torch.cat([l, r], dim=0)

        if seconds is None:
            window_len = win_len
        else:
            window_len = int(seconds * self.sample_rate)

        self.length = (self.wave_len // window_len) - 2
        self.window_len = window_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = wave_cat(self.w, idx, self.window_len)
        # print(f'x.shape{x.shape}')
        return x


class Videoset(Dataset):

    def __init__(self, fn, transforms=None):
        """

        """
        frames, audio, info = torchvision.io.read_video(fn, pts_unit='sec')
        print(frames)
        self.frames = frames
        self.length = len(self.frames)
        self.transform = transforms

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = self.frames[0]
        if self.transform:
            image = self.transform(image)
        return image


def wave_cat(w: torch.tensor, idx: int, n: int, dim=0):
    l = w[0][idx*n:(idx + 1)*n].view(1, -1)
    r = w[1][idx*n:(idx + 1)*n].view(1, -1)
    x = torch.cat([l, r], dim=dim)
    return x


def data_windows(w: torch.tensor, n: int = 1000):
    # assuming stereo channels, w.shape == (2, n)
    windows = []
    length = len(w[0]) // n
    for i in range(length):
        l = w[0][i*n:(i+1)*n].view(1, -1)
        r = w[1][i*n:(i+1)*n].view(1, -1)
        elt = torch.cat([l, r])
        windows.append(elt)
    return windows


if __name__ == "__main__":
    fns = [
        '/home/sippycups/Music/2020/81 - 2 8 20.wav', 
        '/home/sippycups/Music/2019/81 - 4 3 19.wav'
    ]
    dataset = WaveSet(fns, seconds=1)
    print(dataset[0])
