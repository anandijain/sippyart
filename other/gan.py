import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchaudio

from scipy.io.wavfile import write
import matplotlib.pyplot as plt

import utils
import models
import loaders
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

GEN_PATH = 'gen_smol.pth'
DISC_PATH = 'disc_smol.pth'
# DIRECTORY = "/home/sippycups/Music/2018/"
FN = '/home/sippycups/Music/2019/81 - 9 21 19 2.wav'

# FILE_NAMES = os.listdir(DIRECTORY)

FAKES_PATH = 'samples/sound/'

WINDOW_LEN = 4410
# GEN_LATENT = WINDOW_LEN // 100
GEN_LATENT = 4410


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def prep(load_trained=True, write_gen=True):

    time_dir = time.asctime()

    if write_gen:
        if not os.path.isdir(FAKES_PATH):
            os.mkdir(FAKES_PATH)

        os.mkdir(FAKES_PATH + time_dir)

    WAV_WRITE_PATH = FAKES_PATH + time_dir + '/'

    writer = SummaryWriter()

    ngpu = 1
    workers = 4
    batch_size = 1
    lr = 0.001
    beta1 = 0.5

    dataset = loaders.WaveSet(fn=FN, win_len=WINDOW_LEN)
    x = dataset[0]

    print(x.shape)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=workers)

    netG = models.Generator(GEN_LATENT * 2, WINDOW_LEN * 2).to(device)
    netD = models.Discriminator(WINDOW_LEN * 2).to(device)

    if device.type == 'cuda':
        netG = nn.DataParallel(netG, list(range(ngpu)))
        netD = nn.DataParallel(netD, list(range(ngpu)))

    if load_trained:
        try:
            netG.load_state_dict(torch.load(GEN_PATH))
            netD.load_state_dict(torch.load(DISC_PATH))
        except RuntimeError:
            print('applying new weights')
            netG.apply(weights_init)
            netD.apply(weights_init)
        except FileNotFoundError:
            print('applying new weights')
            netG.apply(weights_init)
            netD.apply(weights_init)

    else:
        print('applying new weights')
        netG.apply(weights_init)
        netD.apply(weights_init)

    criterion = nn.BCELoss()

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    d = {
        'wav_write_path': WAV_WRITE_PATH,
        'writer': writer,
        'dataloader': dataloader,
        'g': netG,
        'd': netD,
        'c': criterion,
        'od': optimizerD,
        'og': optimizerG,
        'device': device,
        'set': dataset
    }
    return d


def train_epoch(d, epoch, load_trained=True, write_gen=True):
    device = d['device']
    dataloader = d['dataloader']
    netG = d['g']
    netD = d['d']
    criterion = d['c']
    writer = d['writer']
    WAV_WRITE_PATH = d['wav_write_path']
    optimizerD = d['od']
    optimizerG = d['og']
    fixed_noise = torch.randn(1, GEN_LATENT, device=device)
    real_label = 1
    fake_label = 0

    fakes = []
    G_losses = []
    D_losses = []

    iters = 0

    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()

        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)

        label = torch.full((b_size,), real_label, device=device)
        output = netD(real_cpu).view(-1)

        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        writer.add_scalar('errD_real',  errD_real, iters)

        noise = torch.randn(b_size, GEN_LATENT * 2, device=device)

        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)

        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()

        global_iter = i + epoch*len(d['set'])

        writer.add_scalar('errD_real',  errD_fake, global_iter)

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        writer.add_scalar('errG',  errG, global_iter)

        if i % 50 == 0:
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        if i % 500 == 0:
            print(f'real: {real_cpu}')
            print(f'fake: {fake}')

        G_losses.append(errG.item())
        D_losses.append(errD.item())
        fakes.append(fake.detach().cpu())
        iters += 1

    torch.save(netG.state_dict(), GEN_PATH)
    torch.save(netD.state_dict(), DISC_PATH)

    return fakes, G_losses, D_losses


def train(epochs=1000):
    d = prep()
    print(d)
    print("Starting Training Loop...")
    for i in range(epochs):
        fakes, G_losses, D_losses = train_epoch(d, i)
        to_write = torch.cat(fakes)
        torchaudio.save(d['wav_write_path'] + 'fake_' +
                        str(i) + '.wav', to_write, 44100)
    return fakes, G_losses, D_losses


if __name__ == "__main__":

    f, G, D = train()
