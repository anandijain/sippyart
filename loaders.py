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
    def __init__(self, fn='data/5_8_18_2.wav'):
        self.w, self.sr = torchaudio.load(fn)
        self.length = len(self.w[0])

        
    def __len__(self):
        return self.length - 1

    def __getitem__(self, idx):
        x = torch.tensor([self.w[0][idx], self.w[1][idx]])
        y = torch.tensor([self.w[0][idx+1], self.w[1][idx+1]])
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
        l = self.w[0][idx*self.window_len:(idx + 1)*self.window_len]
        r = self.w[1][idx*self.window_len:(idx + 1)*self.window_len]
        x = torch.cat([l, r])
        # print(x)
        if np.nan in x:
            print('oh no')
        return x.view(1, -1)


# class Files(Dataset):
#     def __init__(self, dir):
#         self.fns = [dir + f for f in FILE_NAMES]
#         self.length = len(self.fns)

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         return self.fns[idx]


# def sgram(wave):
#     return torchaudio.transforms.Spectrogram()(wave)


# def wavey(fn=FILE_NAMES[0]):
#     return torchaudio.load(DIRECTORY + fn)


# def data_windows(n: int = 100000):
#     m = len(FILE_NAMES)
#     d = {}
#     for i, fn in enumerate(FILE_NAMES):
#         t = wavey(fn)
#         window = t[0][i*n:(i+1)*n]
#         if n > len(window):
#             d[fn] = i
#             break
#     return d
