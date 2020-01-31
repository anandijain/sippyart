from PIL import Image
import os
import glob
import numpy as np
import pandas as pd
from skimage import io, transform
import matplotlib.pyplot as plt
import torchaudio
import torchvision
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
plt.ion()   # interactive mode
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
    def __init__(self, fn: str, seconds: int=None, win_len:int=None):  # , resample_to=None):
        """
        seconds is int that is multiplied by sample rate 
        """
        wave = torchaudio.load(filepath=fn)
        self.w = wave[0]
        self.sample_rate = wave[1]
        if seconds is None:
            window_len = win_len
        else:
            window_len = int(seconds * self.sample_rate)

        self.length = (len(self.w[0]) // window_len) - 2
        self.window_len = window_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = wave_cat(self.w, idx, self.window_len)
        # print(x)
        if np.nan in x:
            print('oh no')
        return x.view(1, -1)


class Images(Dataset):

    def __init__(self, root_dir): #, transform=None):
        """

        """
        self.root_dir = root_dir
        self.fns = glob.glob(self.root_dir + '/**.jpg')
        self.length = len(self.fns)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        fn = self.fns[idx]
        image = np.array(Image.open(fn), dtype=np.uint8)
        sample = torch.from_numpy(image)
        return sample


class Videoset(Dataset):

    def __init__(self, fn):
        """

        """
        frames, audio, info = torchvision.io.read_video(fn, pts_unit='sec')
        self.frames = frames
        self.length = len(self.frames)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = self.frames[idx]
        return image


def wave_cat(w:torch.tensor, idx:int, n:int, dim=0):
    l = w[0][idx*n:(idx + 1)*n]
    r = w[1][idx*n:(idx + 1)*n]
    x = torch.cat([l, r], dim=dim)
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
