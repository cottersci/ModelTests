import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Net_D(nn.Module):
    '''
        Discriminates between x ~ p(x) and x ~ q(x|z)
    '''
    def __init__(self):
        super(Net_D, self).__init__()
        self.NU = 600

        self.layer1 = nn.Sequential(
            nn.Linear(28 * 28,self.NU),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(self.NU, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x.view(-1,28*28))

        z = self.layer2(x)

        return z


class Net_G(nn.Module):
    '''
        Generator, models q(x|z)
    '''
    def __init__(self,z_dim):
        super(Net_G, self).__init__()
        self.z_dim = z_dim
        self.NU = 600

        self.layer1 = nn.Sequential(
            nn.Linear(self.z_dim,self.NU),
            nn.LeakyReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(self.NU,28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x.view(-1,1,28,28)
