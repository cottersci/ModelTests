import pytest

import torch
from torch.autograd import Variable

from pytorch_utils import tests

import conv
import linear

#
#Networks to test
#
G_networks = [ conv.Net_G, linear.Net_G]
D_networks = [ conv.Net_D, linear.Net_D]

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
def test_weights_learn_G(network,z_dim,batch_size):
    '''
        All paramaters in generator networks update during learning.
    '''
    ## Test Params
    batch = torch.FloatTensor(batch_size,z_dim).normal_(0.1)

    ##Test
    net = network(z_dim)
    tests.is_learning(net,batch)

@pytest.mark.parametrize("network", G_networks)
def test_G_output_dimensions(network,z_dim,batch_size,img_colors,img_width,img_height):
    '''
        Generator networks have correct output dimensionality
    '''
    ## Test Params
    batch = torch.FloatTensor(batch_size,z_dim).normal_(0,1)

    ##Test
    net = network(z_dim)
    res = net(Variable(batch))

    assert res.size() == (batch_size,img_colors,img_width,img_height)

@pytest.mark.parametrize("network", D_networks)
def test_weights_learn_D(network,batch_size,img_colors,img_width,img_height):
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
    net = network()
    tests.is_learning(net,batch)

@pytest.mark.parametrize("network", D_networks)
def test_D_output_dimensions(network,z_dim,batch_size,img_colors,img_width,img_height):
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
    net = network()
    res = net(Variable(batch))

    assert res.size() == torch.Size([batch_size,1])
