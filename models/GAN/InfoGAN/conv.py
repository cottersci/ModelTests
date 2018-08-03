import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

"""
    Convoluational Nerual Networks for InfoGan
"""

class Net_D(nn.Module):
    '''
        Discirminator network
    '''
    def __init__(self,net):
        """
            Initilize Discirminator network

            Args:
                net (:class:`Net_Shared`): The network shared by both the
                    discirminator and auxiliary distirbution.
        """
        super(Net_D, self).__init__()

        self.net = net

        self.layer_output = nn.Sequential(
            nn.Conv2d(self.net.ngf * 4, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.net(x)
        return self.layer_output(x).view(-1,1)

class Net_Q(nn.Module):
    '''
        Auxiliary distribution network
    '''
    def __init__(self,z_dim,net,on_gpu):
        """
            Initilize auxiliary distribution network

            Args:
                net (:class:`Net_Shared`): The network shared by both the
                    discirminator and auxiliary distirbution.
                z_dim (int): dimsnsionality of the auxiliary distribution
        """
        super(Net_Q, self).__init__()

        self.net = net
        self.z_dim = z_dim
        self.on_gpu = on_gpu

        self.layer_mu = nn.Conv2d(self.net.ngf * 4, self.z_dim, 4, stride=1, padding=0, bias=False)

        self.layer_logstd = nn.Conv2d(self.net.ngf * 4, self.z_dim, 4, stride=1, padding=0, bias=False)

    def forward(self,x):
        x = self.net(x)

        mu = self.layer_mu(x).view(-1,self.z_dim)
        logstd = self.layer_logstd(x).view(-1,self.z_dim)

        z = self.reparametrize(mu,logstd)

        return z, mu, logstd

    def reparametrize(self, mu, logstd):
        std = logstd.exp()
        if self.on_gpu:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)

        return eps.mul(std).add_(mu)

class Net_Shared(nn.Module):
    '''
        Network shared by both net D and Q.
    '''
    def __init__(self):
        super(Net_Shared, self).__init__()
        self.ngf = 32

        self.layer1 = self.create_layer(1, self.ngf)

        self.layer2 = self.create_layer(self.ngf, self.ngf * 2) # -> N x 8 x 8

        self.layer3 = self.create_layer(self.ngf * 2, self.ngf * 4)# -> N x 4 x 4


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

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
