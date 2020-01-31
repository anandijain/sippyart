import os
import torch
import numpy as np

import torchaudio
from torch.nn import functional as F


def sync_sample_rates(fn, fn2):
    w, sr = torchaudio.load(fn)
    w2, sr2 = torchaudio.load(fn2)
    if sr == sr2:
        pass
    elif sr > sr2:
        resampler = torchaudio.transforms.Resample(sr, sr2)
        w = resampler.forward(w)
        sr = sr2
    else:
        resampler = torchaudio.transforms.Resample(sr2, sr)
        w2 = resampler.forward(w2)
        sr2 = sr
    return w, sr, w2, sr2


def get_two(fn, fn2):
    w, sr, w2, sr2 = sync_sample_rates(fn, fn2)

    w_len = len(w[0])
    w2_len = len(w2[0])
    if w_len > w2_len:
        print('a')
        new = w[:][:w2_len]
    elif w_len < w2_len:
        print('b')
        new = w2[:][:w_len]
    return (w, sr), (w2, sr2)

def full_fn_to_name(fn):
    return fn.split('/')[-1].split('.')[0].replace(' ', '_')


def kl_loss(recon_x, x, mu, logvar):
    try:
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    except RuntimeError:
        print(f'recon: {np.unique(recon_x.cpu().detach().numpy())}' )
        print(f'x: {np.unique(x.cpu().detach().numpy())}' )
        
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
