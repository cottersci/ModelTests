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
        self.ngf = 32

        self.layer1 = self.create_layer(1, self.ngf)

        self.layer2 = self.create_layer(self.ngf, self.ngf * 2) # -> N x 8 x 8

        self.layer3 = self.create_layer(self.ngf * 2, self.ngf * 4)# -> N x 4 x 4

        self.layer_mu = nn.Sequential(
            nn.Conv2d(self.ngf * 4, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return self.layer_mu(x).view(-1,1)

    def create_layer(self, Nin, Nout, kernel=3, stride=2, padding=1):
        layer = nn.Sequential(
            nn.Conv2d(Nin, Nout, kernel, stride=stride, padding=padding,bias=False),
            nn.BatchNorm2d(Nout),
            nn.LeakyReLU()
        )

        return layer

class Net_G(nn.Module):
    '''
        Generator, models q(x|z)
    '''
    def __init__(self,z_dim):
        super(Net_G, self).__init__()
        self.ngf = 32
        self.z_dim = z_dim

        self.layer1 = self.create_layer_t(self.z_dim, self.ngf, stride=1, output_padding=0, padding=0, kernel=3)

        self.layer2 = self.create_layer_t(self.ngf, self.ngf * 2, stride=2, output_padding=0, padding=0, kernel=3)

        self.layer3 = self.create_layer_t(self.ngf * 2, self.ngf * 4, stride=2, output_padding=0, padding=0, kernel=2)

        self.output_layer = nn.Sequential(
                                nn.ConvTranspose2d(
                                   self.ngf * 4,
                                   1,
                                   kernel_size=2,
                                   stride=2,
                                   padding=0,
                                   output_padding=0,
                                   bias=False),
                                nn.Sigmoid()
                            )

    def forward(self, x):
        x = self.layer1(x.view(-1, self.z_dim, 1, 1))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output_layer(x)
        return x

    def create_layer_t(self, Nin, Nout, stride=2, output_padding=0, padding=1, kernel=3):
        layer = nn.Sequential(
            nn.ConvTranspose2d(Nin, Nout,
                               kernel_size=kernel,
                               stride=stride,
                               padding=padding,
                               output_padding=output_padding,
                               bias=False),
            nn.BatchNorm2d(Nout),
            nn.LeakyReLU(),
        )

        return layer
