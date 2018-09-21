import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class D(nn.Module):
    '''
        Discriminates between x ~ p(x) and x ~ q(x|z)
    '''
    def __init__(self,z_dim):
        super(D, self).__init__()
        self.ngf = 64
        self.z_dim = z_dim

        self.Dx = nn.Sequential(
            self.create_layer(1, self.ngf),
            self.create_layer(self.ngf, self.ngf * 2), # -> N x 8 x 8
            self.create_layer(self.ngf * 2, self.ngf * 4), # -> N x 4 x 4
        )

        self.Dz = nn.Sequential(
            self.create_layer(self.z_dim,self.ngf * 2,kernel=1,stride=1,padding=0),
            self.create_layer(self.ngf * 2,self.ngf*2,kernel=1,stride=1,padding=0)
        )

        self.I = nn.Linear(self.ngf * 4 * 4 * 4,self.z_dim)

        linear_ngf = self.ngf * 4 * 4 * 4  + self.ngf * 2
        self.Dxz = nn.Sequential(
            spectral_norm(nn.Linear(linear_ngf,linear_ngf)),
            nn.LeakyReLU(),
            spectral_norm(nn.Linear(linear_ngf,1)),
            nn.Sigmoid()
        )

    def forward(self, x, z):
        x = self.Dx(x).view(-1,self.ngf * 4 * 4 * 4)
        z = self.Dz(z.view(-1,self.z_dim,1,1)).view(-1,self.ngf * 2)
        z_pred = self.I(x)
        p = self.Dxz(torch.cat((x,z),dim=1))
        return p, z_pred

    def create_layer(self, Nin, Nout, kernel=3, stride=2, padding=1):
        layer = nn.Sequential(
            spectral_norm(nn.Conv2d(Nin, Nout, kernel, stride=stride, padding=padding,bias=False)),
            nn.LeakyReLU()
        )

        return layer

class Gz(nn.Module):
    '''

    '''
    def __init__(self,z_dim,noise_dim):
        super(Gz, self).__init__()
        self.ngf = 64
        self.z_dim = z_dim
        self.noise_dim = noise_dim

        self.layer1 = self.create_layer(1, self.ngf)
        self.layer2 = self.create_layer(self.ngf, self.ngf * 2) # -> N x 8 x 8
        self.layer3 = self.create_layer(self.ngf * 2, self.ngf * 4) # -> N x 4 x 4

        self.layer_mu = nn.Linear(self.ngf * 4 * 4 * 4 + self.noise_dim,self.z_dim)
        #self.layer_logstd = nn.Linear(self.ngf * 4 * 4 * 4 + self.noise_dim, self.z_dim)

    def forward(self, x, noise):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = torch.cat((x.view(-1,self.ngf * 4 * 4 * 4),noise),dim=1)
        mu = self.layer_mu(x)
        #logstd = self.layer_logstd(x).view(-1,self.z_dim)
        #z = self.reparametrize(mu,logstd)
        return mu, mu, mu

    def create_layer(self, Nin, Nout, kernel=3, stride=2, padding=1):
        layer = nn.Sequential(
            nn.Conv2d(Nin, Nout, kernel, stride=stride, padding=padding,bias=False),
            nn.BatchNorm2d(Nout),
            nn.LeakyReLU()
        )

        return layer

    def reparametrize(self, mu, logstd):
        std = logstd.exp()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

class Gx(nn.Module):
    '''
        Generator, models q(x|z)
    '''
    def __init__(self,z_dim,noise_dim):
        super(Gx, self).__init__()
        self.ngf = 64
        self.noise_dim = noise_dim
        self.z_dim = z_dim

        self.layer1 = self.create_layer_t(self.z_dim + self.noise_dim, self.ngf, stride=1, output_padding=0, padding=0, kernel=3)

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

    def forward(self, z, noise):
        x = torch.cat((z,noise),dim=1)
        x = self.layer1(x.view(-1, self.z_dim + self.noise_dim, 1, 1))
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
