import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset


class VAE(nn.Module):
    def __init__(self, dim, middle=400, bottleneck=100):
        super(VAE, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, middle)
        self.fc21 = nn.Linear(middle, bottleneck)
        self.fc22 = nn.Linear(middle, bottleneck)
        self.fc3 = nn.Linear(bottleneck, middle)
        self.fc4 = nn.Linear(middle, dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEConv1d(nn.Module):
    # TODO, generalize in dim, but only doing win_len of 88100
    def __init__(self, flat_len, bottleneck=250):  #, dim=1660, middle=400, bottleneck=100):
        super(VAEConv1d, self).__init__()
        self.dim = flat_len // 2
        # w window len of 88100
        self.conv1 = nn.Conv1d(2, 2, self.dim//10, stride=4)  # o1.shape torch.Size([1, 2, 19846])
        self.conv2 = nn.Conv1d(2, 2, self.dim//5, stride=4)  # o2.shape 1, 2, 552
        self.fc21 = nn.Linear(552, bottleneck)
        self.fc22 = nn.Linear(552, bottleneck)
        self.fc3 = nn.Linear(bottleneck, 552)
        self.fc4 = nn.Linear(552, flat_len)


    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # print(f'{x.shape}')
        # h1 = x.view(-1)
        h1 = x.view(x.size(0), -1)
        # print(f'h1: {h1.shape}')
        # h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEConv2d(nn.Module):
    def __init__(self, dim=1660, middle=400, bottleneck=100):
        super(VAEConv2d, self).__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(1, 1, 3, stride=2)
        # self.conv2 = nn.Conv1d(1, 1, 3, stride=2)
        self.fc1 = nn.Linear(dim//2 - 1, middle)
        self.fc21 = nn.Linear(middle, bottleneck)
        self.fc22 = nn.Linear(middle, bottleneck)
        self.fc3 = nn.Linear(bottleneck, middle)
        self.fc4 = nn.Linear(middle, dim)

    def encode(self, x):
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = x.flatten()
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, device):
        super(LSTM, self).__init__()
        self.device = device
        self.l1 = nn.Conv1d(1, 1, 5, stride=2)
        self.l2 = nn.Conv1d(1, 1, 5, stride=2)
        self.lstm = nn.LSTM(412, 412, num_layers)
        self.p1 = nn.Linear(412, in_dim)
        self.x = torch.randn(1, 1, 2)
        self.in_dim = in_dim
        self.hid_dim = hidden_dim
        self.num_layers = num_layers
        self.reset()

    def reset(self):
        self.h = torch.zeros(self.num_layers, 1, 412).to(self.device)
        self.c = torch.zeros(self.num_layers, 1, 412).to(self.device)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        print(x.shape)
        x, (self.hn, self.cn) = self.lstm(x, (self.h, self.c))
        x = self.p1(x)
        return x, (self.hn, self.cn)


class Generator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Generator, self).__init__()
        self.l1 = nn.Linear(in_dim, in_dim)
        self.l2 = nn.Linear(in_dim, in_dim)
        # self.l3 = nn.Linear(in_dim, in_dim)
        # self.l3 = nn.Linear(WINDOW_LEN // 10, GEN_LATENT)
        self.l4 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        x = self.l4(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(in_dim, in_dim // 16)
        # self.l2 = nn.Linear(WINDOW_LEN // 4, WINDOW_LEN // 16)
        # self.l3 = nn.Linear(WINDOW_LEN // 16, WINDOW_LEN // 16)
        self.l4 = nn.Linear(in_dim // 16, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return F.softmax(x, dim=0)
