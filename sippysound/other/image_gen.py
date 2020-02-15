import os
import scipy.misc
import time
from PIL import Image
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import skvideo.io
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from torch.utils.tensorboard import SummaryWriter

from glob import glob

import numpy as np

import image_loader
from sippysound import models
from sippysound import utilz


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

DATA_PATH = 'data/images'
T = time.asctime()
HEIGHT, WIDTH = 256, 256
CHANNELS = 3
MIDDLE = 250
BOTTLENECK = 100
EPOCHS = 500
LR = 1e-4
BATCH_SIZE = 256
edits = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((HEIGHT, WIDTH)),
    transforms.ToTensor()])

def prep():
    images = image_loader.Images(
        '/home/sippycups/audio/data/images', transforms=edits)
    data = images[0]
    dataloader = DataLoader(images, shuffle=True, batch_size=BATCH_SIZE)

    print(data.shape)
    dim = data.flatten().shape[0]

    writer = SummaryWriter(
        f"runs/image_gen_test_MID_{MIDDLE}_BOTTLE_{BOTTLENECK}_{T}")

    model = models.VAE(dim, middle=MIDDLE, bottleneck=BOTTLENECK).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    write_to = f'samples/images/image_gen_{T}'
    os.makedirs(write_to)
    d = {
        'write_to': write_to,
        'writer': writer,
        'dataloader': dataloader,
        'model': model,
        'optimizer': optimizer,
        'set': images,
        'model_fn': 'models/image_gen1.pth'
    }
    return d

def train(d):

    samples = []
    for epoch in range(EPOCHS):
        for i, data in enumerate(d['dataloader']):
            data = data.flatten().float().to(device) / 255
            d['optimizer'].zero_grad()

            recon_batch, mu, logvar = d['model'](data)
            loss = utilz.kl_loss(recon_batch, data, mu, logvar)
            loss.backward()

            idx = len(d['set']) * epoch + i
            d['writer'].add_scalar('train_loss', loss.item(), global_step=idx)
            if i % 500 == 0:
                print(
                    f'recon: {np.unique(recon_batch.cpu().detach().numpy())}')
                print(f'x: {np.unique(data.cpu().detach().numpy())}')
            d['optimizer'].step()
            with torch.no_grad():
                sample = torch.randn(1, BOTTLENECK).to(device)
                sample = d['model'].decode(sample).cpu()
                sample = sample.view(HEIGHT, WIDTH, CHANNELS)
                scipy.misc.imsave(
                    f'samples/images/image_gen_{T}/img_{idx}.png', sample.numpy())
                samples.append(sample)

    video = torch.cat(samples).view(-1, HEIGHT, WIDTH, CHANNELS) * 255
    print(f'video.shape: {video.shape}')
    torchvision.io.write_video('tv_test2.mp4', video, 60)
    torch.save(d["m"].state_dict(), d['model_fn'])

if __name__ == "__main__":
    d = prep()
    print(d)
    train(d)
