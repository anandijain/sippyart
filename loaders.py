import numpy as np

import torchaudio
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

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
        return self.length - 1

    def __getitem__(self, idx):
        x = self.windows[idx][0]
        y = self.windows[idx+1][0]
        return x, y


class WaveSet(Dataset):
    def __init__(self, fn: str, seconds: int):  # , resample_to=None):
        """
        seconds is int that is multiplied by sample rate 
        """
        wave = torchaudio.load(filepath=fn)
        self.w = wave[0]
        self.sample_rate = wave[1]

        window_len = int(seconds * self.sample_rate)

        self.length = (len(self.w[0]) // window_len) - 2
        self.window_len = window_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = wave_cat(x, idx, self.window_len)
        # print(x)
        if np.nan in x:
            print('oh no')
        return x.view(1, -1)


def wave_cat(w:torch.tensor, idx:int, n:int):
    l = w[0][idx*n:(idx + 1)*n]
    r = w[1][idx*n:(idx + 1)*n]
    x = torch.cat([l, r])
    return x



def data_windows(w:torch.tensor, n: int = 1000):
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
    
    wave, sr = torchaudio.load('data/8_14_18.wav')
    windows = data_windows(wave, sr // 4)
