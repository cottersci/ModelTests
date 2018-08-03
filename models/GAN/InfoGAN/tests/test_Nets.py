import pytest

import torch
from torch.autograd import Variable

from pytorch_utils import tests

#
# Fix path
#
import sys
import os
sys.path.insert(0,os.path.abspath("."))

import conv
import linear

#
#Networks to test
#
G_networks = [ conv.Net_G, ]
D_networks = [ (conv.Net_D,conv.Net_Shared), ]
Q_networks = [ (conv.Net_Q,conv.Net_Shared), ]

#
# Network paramaters
#
@pytest.fixture()
def batch_size():
    return 2

@pytest.fixture()
def z_dim():
    return 4

@pytest.fixture()
def noise_dim():
    return 64

@pytest.fixture()
def img_width():
    return 28

@pytest.fixture()
def img_height():
    return 28

@pytest.fixture()
def img_colors():
    return 1

#
# Tests
#
@pytest.mark.parametrize("network", G_networks)
def test_weights_learn_G(network, z_dim, noise_dim, batch_size):
    '''
        All paramaters in generator networks update during learning.
    '''
    ## Test Params
    batch = torch.FloatTensor(batch_size,z_dim + noise_dim).normal_(0.1)

    ##Test
    net = network(z_dim + noise_dim)
    tests.is_learning(net,batch)

@pytest.mark.parametrize("network", G_networks)
def test_G_output_dimensions(network, z_dim, noise_dim,  batch_size,
                                    img_colors, img_width, img_height):
    '''
        Generator networks have correct output dimensionality
    '''
    ## Test Params
    batch = torch.FloatTensor(batch_size,z_dim + noise_dim).normal_(0,1)

    ##Test
    net = network(z_dim + noise_dim)
    res = net(Variable(batch))

    assert res.size() == (batch_size,img_colors,img_width,img_height)

@pytest.mark.parametrize("network,base_net", D_networks)
def test_weights_learn_D(base_net, network, batch_size,
                         img_colors, img_width, img_height):
    '''
        All paramaters in discriminator networks update during learning.
    '''
    ## Test Params
    batch = torch.FloatTensor(batch_size,
                              img_colors,
                              img_width,
                              img_height)
    batch.normal_(0,1)

    ##Test
    net = network(base_net())
    tests.is_learning(net,batch)

@pytest.mark.parametrize("network,base_net", D_networks)
def test_D_output_dimensions(base_net, network, z_dim, batch_size,
                                 img_colors, img_width, img_height):
    '''
        Discriminator networks have correct output dimensionality
    '''
    ## Test Params
    batch = torch.FloatTensor(batch_size,
                              img_colors,
                              img_width,
                              img_height)
    batch.normal_(0,1)

    ##Test
    net = network(base_net())
    res = net(Variable(batch))

    assert res.size() == torch.Size([batch_size,1])

@pytest.mark.parametrize("network,base_net", Q_networks)
def test_weights_learn_Q(base_net, network, z_dim, batch_size,
                              img_colors, img_width, img_height):
    '''
        All paramaters in auxiliary networks update during learning.
    '''
    ## Test Params
    batch = torch.FloatTensor(batch_size,
                              img_colors,
                              img_width,
                              img_height)
    batch.normal_(0,1)

    ##Test
    net = network(z_dim,base_net(),on_gpu=False)
    tests.is_learning(net,batch)

@pytest.mark.parametrize("network,base_net", Q_networks)
def test_Q_output_dimensions(base_net, network, z_dim, batch_size,
                                  img_colors, img_width, img_height):
    '''
        Auxiliary networks have correct output dimensionality
    '''
    ## Test Params
    batch = torch.FloatTensor(batch_size,
                              img_colors,
                              img_width,
                              img_height)
    batch.normal_(0,1)

    ##Test
    net = network(z_dim,base_net(),on_gpu=False)
    z,mu,logstd = net(Variable(batch))

    assert z.size() == torch.Size([batch_size,z_dim])
    assert mu.size() == torch.Size([batch_size,z_dim])
    assert logstd.size() == torch.Size([batch_size,z_dim])
