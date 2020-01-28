import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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


class DenseVAE(nn.Module):
    def __init__(self, dim, middle=400, bottleneck=100):
        super(DenseVAE, self).__init__()
        self.dim = dim
        self.middle = middle
        self.bottleneck = bottleneck
        self.l1 = nn.Linear(dim, middle)
        self.l2 = nn.Linear(middle, bottleneck)
        self.l3 = nn.Linear(bottleneck, middle)
        self.l4 = nn.Linear(middle, middle)
        self.l5 = nn.Linear(middle, dim)

    def decode(self, z):
        z = torch.tanh(self.l3(z))
        z = torch.tanh(self.l4(z))
        z = self.l5(z)
        print(z.shape)
        return torch.sigmoid(z)

    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        x = torch.tanh(self.l3(x))
        x = torch.sigmoid(self.l4(x))
        return x
