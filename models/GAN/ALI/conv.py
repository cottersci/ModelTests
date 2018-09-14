import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

"""
    Convoluational Nerual Networks for InfoGan
"""

class D(nn.Module):
    '''
        (x,G_z(x))  --->  D(x,z) <--- (G_x(z),z)

        where x ~ p(x), z ~ p(z)
    '''

    ngf = 32

    def __init__(self,z_dim):
        """
            Initilize Discirminator network

            Args:
                z_dim: embedded space dimensions
        """
        super(D, self).__init__()
        self.z_dim = z_dim

        self.Dx = nn.Sequential( # <- 1 x 32 x 32
            self.create_layer(1, self.ngf), # -> ngf x 16 x 16
            self.create_layer(self.ngf, self.ngf * 2), # -> ngf x 8 x 8
            self.create_layer(self.ngf * 2, self.ngf * 4), # -> ngf x 4 x 4
            self.create_layer(self.ngf * 4, self.ngf * 4, kernel = 4, stride = 1, padding = 0), # -> ngf x 1 x 1
        ) # -> ngf x 1 x 1

        self.Dz = nn.Sequential( # <- z_dim x 1 x 1
            self.create_layer(self.z_dim, self.ngf * 4, kernel = 1, stride = 1, padding = 0),
            self.create_layer(self.ngf * 4, self.ngf * 4, kernel = 1, stride = 1, padding = 0),
        ) # -> ngf x 1 x 1

        linear_ngf = self.ngf * 4 + self.ngf * 4
        self.Dxz = nn.Sequential(
            self.create_layer(linear_ngf, linear_ngf, kernel = 1, stride = 1, padding = 0),
            self.create_layer(linear_ngf, linear_ngf, kernel = 1, stride = 1, padding = 0),
            nn.Conv2d(linear_ngf, 1, 1, stride = 1, padding = 0, bias=False),
            nn.Sigmoid()
        ) # -> ngf x 1 x 1

    def forward(self,x,z): #<- 1 x 32 x 32
        """
            Input is a batch_size x 1 x 32 x 32 (Channels x Height x Width).
        """
        x = self.Dx(x) # -> ngf x 4 x 4
        z = self.Dz(z.view(-1,self.z_dim,1,1))
        xz = self.Dxz(torch.cat((x,z),dim=1))
        return xz.view(-1,1)

    def create_layer(self, Nin, Nout, kernel=3, stride=2, padding=1):
        layer = nn.Sequential(
            nn.Conv2d(Nin, Nout, kernel, stride=stride, padding=padding,bias=False),
            nn.BatchNorm2d(Nout),
            #nn.Dropout2d(0.2),
            nn.LeakyReLU()
        )

        return layer

class Gz(nn.Module):
    '''
                   G_z(x)
        x ~ q(x) ---------> \hat{z} ~ q(z|x)
    '''
    ngf = 32
    z_dim = 0
    noise_dim = 0
    def __init__(self,z_dim,noise_dim):
        """
            Initilize Gz.

            Args:
                z_dim: embedded space dimensions
                noise_dim: number of gaussian noise dimenions
        """
        super(Gz, self).__init__()

        self.z_dim = z_dim
        self.noise_dim = noise_dim

        self.layer1 = self.create_layer(1, self.ngf)

        self.layer2 = self.create_layer(self.ngf, self.ngf * 2) # -> N x 8 x 8

        self.layer3 = self.create_layer(self.ngf * 2, self.ngf * 4)# -> N x 4 x 4

        #self.layer_output = nn.Conv2d(self.ngf * 4, self.z_dim, 4, stride=1, padding=0 ,bias=True)

        self.layer_mu = nn.Conv2d(self.ngf  * 4, self.z_dim, 4, stride=1, padding=0, bias=False)

        self.layer_logstd = nn.Conv2d(self.ngf  * 4, self.z_dim, 4, stride=1, padding=0, bias=False)

    def forward(self,x,noise):
        """
            Input is a batch_size x 1 x 32 x 32 (Channels x Height x Width).
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        mu = self.layer_mu(x).view(-1,self.z_dim)
        logstd = self.layer_logstd(x).view(-1,self.z_dim)

        z = self.reparametrize(mu,logstd)

        return z
        #x = x.view(-1,self.ngf * 4 * 4 * 4)
        #x = torch.cat((x,noise),dim=1)
        #return self.layer_output(x).view(-1,self.z_dim)

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
                   G_x(z)
        z ~ p(z) ---------> \hat{x} ~ q(x|z)
    '''
    ngf = 32
    z_dim = 0
    nnoise_dim = 0
    def __init__(self,z_dim,noise_dim):
        """
            Args:
                z_dim (int): Dimensionality of embedded space
                noise_dim (int): Dimensionality of noise
        """
        super(Gx, self).__init__()
        self.ngf = 32
        self.z_dim = z_dim
        self.noise_dim = noise_dim

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
        """
            Input is a batch_size x z_dim
            Output is batch_size x 1 x 32 x 32 (Channels x Height x Width)

            Args:
                z (FloatTensor): embedded space sample
                noise (FloatTensor): gaussian noise
        """
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
