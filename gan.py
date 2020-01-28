import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Generator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Generator, self).__init__()
        self.l1 = nn.Linear(in_dim, in_dim)
        # self.l2 = nn.Linear(GEN_LATENT, WINDOW_LEN // 10)
        # self.l3 = nn.Linear(WINDOW_LEN // 10, GEN_LATENT)
        self.l4 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.l1(x))
        # x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
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

