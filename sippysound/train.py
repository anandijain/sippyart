import torch
import torchaudio

from sippysound import utilz


def train_epoch(d, epoch: int, batch_size, device, save=False):

    model = d['m']
    optimizer = d['o']
    dataset = d['data']
    train_loader = d['loader']
    sample_rate = d['sr']
    path = d['path']

    samples = []
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        # print(f'data.shape: {data.shape}')
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        recon_batch = recon_batch.view(batch_size, 2, -1)

        # print(f'recon.shape: {recon_batch.shape}')

        loss = utilz.kl_loss(recon_batch, data, mu, logvar)
        loss.backward()
        idx = len(dataset) * epoch + batch_idx
        d['writer'].add_scalar('train_loss', loss.item(), global_step=idx)

        train_loss += loss.item()
        optimizer.step()
